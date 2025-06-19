import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Union
import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import numpy as np
import os
import math

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class LoRAExpert(nn.Module):
    def __init__(self, in_features, out_features, r, init_orthogonal=False):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        if init_orthogonal:
            nn.init.orthogonal_(self.lora_A)

    def forward(self, x, dropout, scaling):
        out = x @ self.lora_A.T
        out = dropout(out)
        out = out @ self.lora_B.T
        return out * scaling


class COLORADataCollator:
    def __init__(self, tokenizer, task_to_id: Dict[str, int], mlm=False):
        self.tokenizer = tokenizer
        self.task_to_id = task_to_id
        self.mlm = mlm

    def __call__(self, features):
        task_ids = []
        for f in features:
            task_id = f.pop("task_id", "default")
            if task_id not in self.task_to_id:
                task_id = "default"
            task_ids.append(self.task_to_id[task_id])

        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["task_ids"] = torch.tensor(task_ids, dtype=torch.long)
        return batch


class COLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        task_names: List[str],
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.task_names = task_names
        self.task_to_id = {name: i for i, name in enumerate(task_names)}
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.task_experts = nn.ModuleDict(
            {
                task: LoRAExpert(
                    base_layer.in_features,
                    base_layer.out_features,
                    r,
                    init_orthogonal=False,
                )
                for task in task_names
            }
        )

        self.shared_lora_A = nn.Parameter(torch.randn(r, base_layer.in_features) * 0.01)
        self.shared_lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))
        nn.init.orthogonal_(self.shared_lora_A)

        self.task_embeddings = nn.Parameter(
            torch.zeros(len(task_names), base_layer.in_features)
        )
        nn.init.orthogonal_(self.task_embeddings)

        self.collaboration_weight = nn.Parameter(torch.tensor(0.5))
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, task_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        base_out = self.base_layer(x)

        # Shared LoRA output
        shared_out = x @ self.shared_lora_A.T
        shared_out = self.lora_dropout(shared_out)
        shared_out = shared_out @ self.shared_lora_B.T
        shared_out *= self.scaling

        # Task-specific output with proper routing
        if task_ids is not None and len(task_ids.shape) > 0:
            # Batch processing with known task_ids
            task_out = torch.zeros_like(shared_out)

            for i, task_name in enumerate(self.task_names):
                # Find samples that belong to this task
                task_mask = (task_ids == i).float()  # [batch_size]

                if task_mask.sum() > 0:
                    # Only compute for samples of this task
                    expert_out = self.task_experts[task_name](
                        x, self.lora_dropout, self.scaling
                    )
                    # Apply mask: [batch_size, 1, 1] * [batch_size, seq_len, hidden_dim]
                    task_mask_expanded = task_mask.view(-1, 1, 1).expand_as(expert_out)
                    task_out += task_mask_expanded * expert_out
        else:
            # Flatten for routing computation
            x_flat = x.view(-1, x.size(-1))  # [batch_size * seq_len, hidden_dim]

            # Compute routing scores
            attn_scores = torch.matmul(
                x_flat, self.task_embeddings.T
            )  # [batch*seq, num_tasks]
            routing_scores = F.softmax(attn_scores, dim=-1)

            # Weighted combination of expert outputs
            task_out = torch.zeros_like(
                x_flat @ self.shared_lora_A.T @ self.shared_lora_B.T
            )

            for i, task_name in enumerate(self.task_names):
                expert_out_flat = self.task_experts[task_name](
                    x_flat, self.lora_dropout, self.scaling
                ).view(-1, task_out.size(-1))

                weights = routing_scores[:, i : i + 1]  # [batch*seq, 1]
                task_out += weights * expert_out_flat

            # Reshape back
            task_out = task_out.view(batch_size, seq_len, -1)

        # Collaboration between shared and task-specific
        collab_weight = torch.sigmoid(self.collaboration_weight)
        lora_out = collab_weight * shared_out + (1 - collab_weight) * task_out

        return base_out + lora_out

    def merge_and_freeze(self, routing_stats: Optional[Dict] = None):
        with torch.no_grad():
            # 1. Merge shared LoRA
            shared_weight = (self.shared_lora_B @ self.shared_lora_A) * self.scaling

            # 2. Compute expert weights
            expert_weights = []
            for task in self.task_names:
                A = self.task_experts[task].lora_A
                B = self.task_experts[task].lora_B
                expert_weights.append((B @ A) * self.scaling)

            # 3. Use routing statistics if available, otherwise equal weighting
            if routing_stats:
                alpha = [
                    routing_stats.get(task, 1.0 / len(self.task_names))
                    for task in self.task_names
                ]
                # Normalize
                total = sum(alpha)
                alpha = [a / total for a in alpha]
            else:
                alpha = [1.0 / len(self.task_names)] * len(self.task_names)

            # 4. Weighted average of expert weights
            avg_expert_weight = sum(
                alpha[i] * expert_weights[i] for i in range(len(expert_weights))
            )

            # 5. Combine with shared LoRA
            collab_weight = torch.sigmoid(self.collaboration_weight).item()
            merged_weight = (
                collab_weight * shared_weight + (1 - collab_weight) * avg_expert_weight
            )

            # 6. Merge into base layer
            self.base_layer.weight.data += merged_weight.to(
                self.base_layer.weight.dtype
            )

        # 7. Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False


class CoLAOrthogonalityLoss(nn.Module):
    def __init__(self, lambda_orth: float = 0.01, lambda_collab: float = 0.001):
        super().__init__()
        self.lambda_orth = lambda_orth
        self.lambda_collab = lambda_collab

    def forward(self, model) -> torch.Tensor:
        orth_loss = 0.0
        collab_loss = 0.0
        count = 0

        for name, module in model.named_modules():
            if isinstance(module, COLoRALinear):
                # Orthogonality loss for shared LoRA
                A_shared = module.shared_lora_A
                A_gram = torch.matmul(A_shared, A_shared.T)
                I = torch.eye(
                    A_shared.size(0), device=A_shared.device, dtype=A_shared.dtype
                )
                orth_loss += torch.norm(A_gram - I, p="fro") ** 2

                # Orthogonality loss for task experts
                for task in module.task_names:
                    A_task = module.task_experts[task].lora_A
                    A_gram = torch.matmul(A_task, A_task.T)
                    I = torch.eye(
                        A_task.size(0), device=A_task.device, dtype=A_task.dtype
                    )
                    orth_loss += torch.norm(A_gram - I, p="fro") ** 2

                # Collaboration diversity loss
                for i, task1 in enumerate(module.task_names):
                    for j, task2 in enumerate(module.task_names[i + 1 :], i + 1):
                        A1 = module.task_experts[task1].lora_A
                        A2 = module.task_experts[task2].lora_A
                        similarity = torch.norm(torch.matmul(A1, A2.T), p="fro") ** 2
                        collab_loss += similarity

                count += 1

        orth_loss_avg = orth_loss / count if count > 0 else 0.0
        collab_loss_avg = collab_loss / count if count > 0 else 0.0

        orth_loss_avg = torch.clamp(orth_loss_avg, max=10.0)
        collab_loss_avg = torch.clamp(collab_loss_avg, max=10.0)

        total_loss = (
            self.lambda_orth * orth_loss_avg + self.lambda_collab * collab_loss_avg
        )

        return total_loss, orth_loss_avg, collab_loss_avg


def apply_cola_orthogonal_lora(
    model, task_names, target_modules=None, r=8, lora_alpha=16, lora_dropout=0.05
):
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    modified_modules = []

    def replace_module(parent, name, module):
        if isinstance(module, nn.Linear) and any(
            target in name for target in target_modules
        ):
            # Replace with COLoRALinear
            setattr(
                parent,
                name,
                COLoRALinear(module, task_names, r, lora_alpha, lora_dropout),
            )
            modified_modules.append(f"{parent.__class__.__name__}.{name}")
            return True
        return False

    def apply_recursive(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            if not replace_module(module, name, child):
                apply_recursive(child, full_name)

    apply_recursive(model)
    print(f"Applied COLoRA to {len(modified_modules)} modules for tasks: {task_names}")

    return model


class COLoRAModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(
        self, input_ids, attention_mask=None, labels=None, task_ids=None, **kwargs
    ):
        # Store task_ids for COLoRA layers
        self._current_task_ids = task_ids

        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

        return outputs

    def named_modules(self):
        return self.base_model.named_modules()

    def parameters(self):
        return self.base_model.parameters()

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)


class CoLAOLoRATrainer(Trainer):
    def __init__(
        self,
        lambda_orth: float = 0.01,
        lambda_collab: float = 0.001,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_orth = lambda_orth
        self.lambda_collab = lambda_collab
        self.loss_fn = CoLAOrthogonalityLoss(lambda_orth, lambda_collab)
        self.step_count = 0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        task_ids = inputs.get("task_ids", None)  # Don't pop, keep for model

        # Monkey patch the forward method to pass task_ids to COLoRA layers
        original_forward_methods = {}

        def patch_colora_forward(module, task_ids):
            if isinstance(module, COLoRALinear):
                original_forward = module.forward

                def new_forward(x, task_ids_param=None):
                    return original_forward(
                        x, task_ids if task_ids_param is None else task_ids_param
                    )

                module.forward = new_forward
                return original_forward
            return None

        # Apply patches
        for name, module in model.named_modules():
            original = patch_colora_forward(module, task_ids)
            if original:
                original_forward_methods[name] = original

        try:
            # Forward pass
            outputs = model(**inputs)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                labels = inputs.get("labels", None)
                if labels is None:
                    raise ValueError(
                        "Labels not found in inputs for manual loss computation."
                    )
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # COLoRA regularization
            additional_loss, orth_loss, collab_loss = self.loss_fn(model)
            total_loss = loss + additional_loss

            # Logging
            if self.state.global_step % 5 == 0:
                collab_weight_val = -1
                for name, module in model.named_modules():
                    if isinstance(module, COLoRALinear):
                        collab_weight_val = torch.sigmoid(
                            module.collaboration_weight
                        ).item()
                        break
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[Step {self.state.global_step}] "
                    f"LM Loss: {loss:.4f} | "
                    f"Orth: {orth_loss:.6f} | "
                    f"Collab: {collab_loss:.6f} | "
                    f"Total CLoRA Loss: {additional_loss:.6f} | "
                    f"Collab Weight: {collab_weight_val:.4f} | "
                    f"LR: {current_lr:.2e}"
                )

        finally:
            # Restore original forward methods
            for name, module in model.named_modules():
                if name in original_forward_methods:
                    module.forward = original_forward_methods[name]

        return (total_loss, outputs) if return_outputs else total_loss


def train_olora(
    jsonl_path: str,
    adapter_name: str,
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    output_dir: str = "./colora_output",
    r: int = 8,
    max_seq_len: int = 256,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 5e-5,
    lambda_orth: float = 0.01,
    lambda_collab: float = 0.001,
    device: str = "mps",
):
    # Load and process data
    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    raw_data = load_jsonl(jsonl_path)

    # Auto-detect task names
    for item in raw_data:
        item["task_id"] = item.get("task", "default")
    task_names = list(set(item["task_id"] for item in raw_data))

    if "default" not in task_names:
        task_names.append("default")

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.shuffle(seed=42)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = model.to(device)

    # Apply COLoRA
    print("Applying CoLA + Orthogonal LoRA...")
    alpha = r * 2
    print(f"Using r={r}, alpha={alpha}")
    model = apply_cola_orthogonal_lora(
        model,
        task_names=task_names,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Tokenize dataset
    def tokenize(example):
        prompt = example["instruction"]
        response = example["output"]

        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        full = tokenizer(
            prompt + response,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors=None,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        labels = input_ids.copy()
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "task_id": example["task_id"],
        }

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=["task", "instruction", "output", "task_id"],
        batched=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{adapter_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        learning_rate=lr,
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_grad_norm=0.5,
    )

    # Data collator
    task_to_id = {name: idx for idx, name in enumerate(task_names)}
    data_collator = COLORADataCollator(tokenizer, task_to_id=task_to_id, mlm=False)

    # Initialize trainer
    print("Starting CoLA + O-LoRA training...")
    trainer = CoLAOLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        lambda_orth=lambda_orth,
        lambda_collab=lambda_collab,
    )

    # Train
    trainer.train()

    # Save adapter
    def save_colora_adapter(model, save_path, task_names):
        os.makedirs(save_path, exist_ok=True)

        lora_state_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, COLoRALinear):
                # Save shared LoRA
                lora_state_dict[f"{name}.shared_lora_A"] = module.shared_lora_A.data
                lora_state_dict[f"{name}.shared_lora_B"] = module.shared_lora_B.data

                # Save task-specific LoRAs
                for task in task_names:
                    lora_state_dict[f"{name}.task_experts.{task}.lora_A"] = (
                        module.task_experts[task].lora_A.data
                    )
                    lora_state_dict[f"{name}.task_experts.{task}.lora_B"] = (
                        module.task_experts[task].lora_B.data
                    )

                # Save collaboration weight and task embeddings
                lora_state_dict[f"{name}.collaboration_weight"] = (
                    module.collaboration_weight.data
                )
                lora_state_dict[f"{name}.task_embeddings"] = module.task_embeddings.data

        torch.save(lora_state_dict, f"{save_path}/cola_olora_weights.pt")

        # Save config
        config = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": 0.05,
            "task_names": task_names,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }
        with open(f"{save_path}/cola_olora_config.json", "w") as f:
            json.dump(config, f, indent=2)

    save_colora_adapter(model, f"{output_dir}/{adapter_name}", task_names)
    tokenizer.save_pretrained(f"{output_dir}/{adapter_name}")

    print(f"COLoRA adapter '{adapter_name}' saved to {output_dir}/{adapter_name}")

    return model, tokenizer


def merge_colora_adapters(model, routing_stats=None):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, COLoRALinear):
            module.merge_and_freeze(routing_stats)
            count += 1
    print(f"Merged and froze {count} COLoRALinear modules.")


def run_cola_chain(
    data_path: str,
    base_model_path: str,
    output_root: str = "./colora_output",
    adapter_prefix: str = "cola_adapter",
    chain_length: int = 3,
    max_seq_len: int = 16,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 5e-4,
    lambda_orth: float = 0.01,
    lambda_collab: float = 0.001,
    device: str = "mps",
):
    current_model_path = base_model_path
    r = 8

    for round_id in range(1, chain_length + 1):
        print(f"Starting COLA round {round_id}/{chain_length}")
        adapter_name = f"{adapter_prefix}_round{round_id}"
        output_dir = f"{output_root}/{adapter_name}"

        # Fine-tune one COLA round
        model, tokenizer = train_olora(
            jsonl_path=data_path,
            adapter_name=adapter_name,
            base_model=current_model_path,
            output_dir=output_root,
            r=r,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            lambda_orth=lambda_orth,
            lambda_collab=lambda_collab,
            device=device,
        )

        merge_colora_adapters(model)
        print(f"Saving full model for round {round_id}...")
        full_save_path = f"{output_root}/{adapter_name}_full"
        model.save_pretrained(full_save_path)
        tokenizer.save_pretrained(full_save_path)
        current_model_path = full_save_path

        r = max(2, r // 2)

    print(
        f"\n✅ Done chaining {chain_length} COLA rounds. Final model at: {current_model_path}"
    )
    return current_model_path, tokenizer


if __name__ == "__main__":
    final_model_path, tokenizer = run_cola_chain(
        data_path="./data/data_1k_16token.jsonl",
        base_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        adapter_prefix="dev_support_colora",
        chain_length=2,
        max_seq_len=16,
        batch_size=4,
        epochs=2,
        lr=1e-5,
        # lr=5e-5,
        # lr=1e-4,
        lambda_orth=0.01,
        lambda_collab=0.001,
        device="mps",
    )
