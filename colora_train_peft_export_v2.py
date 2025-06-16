import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import json, os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from copy import deepcopy
from typing import List, Dict
from peft import PeftModel

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


# === Tách adapter từng task theo chuẩn PEFT ===
def extract_single_task_adapter(
    trained_model,  # model đã huấn luyện (chứa CoLAOrthogonalLoRALinear)
    task_name,
    target_modules,
    save_path,
    base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
):
    print(f"\nExtracting adapter for task: {task_name}")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # Load lại model gốc chưa bị chỉnh sửa
    clean_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    peft_model = get_peft_model(clean_model, lora_config)

    # Gán A, B từ model đã train vào PEFT adapter
    for name, module in trained_model.named_modules():
        if isinstance(module, CoLAOrthogonalLoRALinear):
            if name not in dict(peft_model.named_modules()):
                continue
            peft_layer = dict(peft_model.named_modules())[name]
            A = module.task_experts[task_name].lora_A
            B = module.task_experts[task_name].lora_B
            peft_layer.lora_A.default.weight.data.copy_(A)
            peft_layer.lora_B.default.weight.data.copy_(B)

    peft_model.save_pretrained(save_path)
    print(f"Saved PEFT adapter for [{task_name}] → {save_path}")


# === Define core LoRA components ===
class LoRAExpert(nn.Module):
    def __init__(self, in_features, out_features, r, init_orthogonal=True):
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
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, features):
        task_ids = [f.pop("task_id") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["task_id"] = task_ids
        return batch


class CoLAOrthogonalLoRALinear(nn.Module):
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
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.task_experts = nn.ModuleDict(
            {
                task: LoRAExpert(base_layer.in_features, base_layer.out_features, r)
                for task in task_names
            }
        )

        self.shared_lora_A = nn.Parameter(torch.randn(r, base_layer.in_features) * 0.01)
        self.shared_lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))
        nn.init.orthogonal_(self.shared_lora_A)

        self.task_router = nn.Linear(base_layer.in_features, len(task_names))
        self.collaboration_weight = nn.Parameter(torch.tensor(0.5))
        self.lora_dropout = (
            nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        )

    def orthogonalize_weights(self):
        with torch.no_grad():
            Q, _ = torch.linalg.qr(self.shared_lora_A.detach().cpu().T)
            self.shared_lora_A.data = Q.T[: self.r].to(self.shared_lora_A.device)

            Q, _ = torch.linalg.qr(self.shared_lora_B.detach().cpu())
            self.shared_lora_B.data = Q[:, : self.r].to(self.shared_lora_B.device)

            for task in self.task_names:
                A = self.task_experts[task].lora_A
                B = self.task_experts[task].lora_B

                Q, _ = torch.linalg.qr(A.detach().cpu().T)
                self.task_experts[task].lora_A.data = Q.T[: self.r].to(A.device)

                Q, _ = torch.linalg.qr(B.detach().cpu())
                self.task_experts[task].lora_B.data = Q[:, : self.r].to(B.device)

    def forward(self, x: torch.Tensor, task_id: Optional[str] = None) -> torch.Tensor:
        base_out = self.base_layer(x)

        shared_out = x @ self.shared_lora_A.T
        shared_out = self.lora_dropout(shared_out)
        shared_out = shared_out @ self.shared_lora_B.T
        shared_out *= self.scaling

        if task_id and task_id in self.task_experts:
            task_A = self.task_experts[task_id].lora_A
            task_B = self.task_experts[task_id].lora_B
            task_out = x @ task_A.T
            task_out = self.lora_dropout(task_out)
            task_out = task_out @ task_B.T
            task_out *= self.scaling
            collab_weight = torch.sigmoid(self.collaboration_weight)
            lora_out = collab_weight * shared_out + (1 - collab_weight) * task_out
        else:
            routing_scores = F.softmax(self.task_router(x.mean(dim=-2)), dim=-1)
            task_out = 0
            for i, task in enumerate(self.task_names):
                task_A = self.task_experts[task].lora_A
                task_B = self.task_experts[task].lora_B
                expert_out = x @ task_A.T
                expert_out = self.lora_dropout(expert_out)
                expert_out = expert_out @ task_B.T
                expert_out *= self.scaling
                task_out += routing_scores[:, i : i + 1, None] * expert_out
            collab_weight = torch.sigmoid(self.collaboration_weight)
            lora_out = collab_weight * shared_out + (1 - collab_weight) * task_out

        return base_out + lora_out


class CoLAOrthogonalityLoss(nn.Module):
    """
    Loss function for CoLA + O-LoRA with additional collaboration constraints
    """

    def __init__(self, lambda_orth: float = 0.01, lambda_collab: float = 0.001):
        super().__init__()
        self.lambda_orth = lambda_orth
        self.lambda_collab = lambda_collab

    def forward(self, model) -> torch.Tensor:
        orth_loss = 0.0
        collab_loss = 0.0
        count = 0

        for name, module in model.named_modules():
            if isinstance(module, CoLAOrthogonalLoRALinear):
                # Orthogonality loss for shared LoRA
                A_shared = module.shared_lora_A
                A_gram = torch.matmul(A_shared, A_shared.T)
                I = torch.eye(
                    A_shared.size(0), device=A_shared.device, dtype=A_shared.dtype
                )
                orth_loss += torch.norm(A_gram - I, p="fro") ** 2

                B_shared = module.shared_lora_B
                B_gram = torch.matmul(B_shared.T, B_shared)
                I = torch.eye(
                    B_shared.size(1), device=B_shared.device, dtype=B_shared.dtype
                )
                orth_loss += torch.norm(B_gram - I, p="fro") ** 2

                # Orthogonality loss for task-specific LoRAs
                for task in module.task_names:
                    A_task = module.task_experts[task].lora_A
                    A_gram = torch.matmul(A_task, A_task.T)
                    I = torch.eye(
                        A_task.size(0), device=A_task.device, dtype=A_task.dtype
                    )
                    orth_loss += torch.norm(A_gram - I, p="fro") ** 2

                    B_task = module.task_experts[task].lora_B
                    B_gram = torch.matmul(B_task.T, B_task)
                    I = torch.eye(
                        B_task.size(1), device=B_task.device, dtype=B_task.dtype
                    )
                    orth_loss += torch.norm(B_gram - I, p="fro") ** 2

                # Collaboration loss - encourage diversity between task experts
                for i, task1 in enumerate(module.task_names):
                    for j, task2 in enumerate(module.task_names[i + 1 :], i + 1):
                        A1 = module.task_experts[task1].lora_A
                        A2 = module.task_experts[task2].lora_A

                        # Encourage orthogonality between different task experts
                        similarity = torch.norm(torch.matmul(A1, A2.T), p="fro") ** 2
                        collab_loss += similarity

                count += 1

        total_loss = 0
        if count > 0:
            total_loss += self.lambda_orth * orth_loss / count
            if collab_loss > 0:
                total_loss += self.lambda_collab * collab_loss / count

        return total_loss


def apply_cola_orthogonal_lora(
    model, task_names, target_modules=None, r=8, lora_alpha=16, lora_dropout=0.05
):
    """
    Apply CoLA + Orthogonal LoRA to specified modules
    """
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
            # Replace with CoLAOrthogonalLoRALinear
            setattr(
                parent,
                name,
                CoLAOrthogonalLoRALinear(
                    module, task_names, r, lora_alpha, lora_dropout
                ),
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
    print(
        f"Applied CoLA + O-LoRA to {len(modified_modules)} modules for tasks: {task_names}"
    )

    return model


class CoLAOLoRATrainer(Trainer):
    """
    Custom Trainer for CoLA + O-LoRA
    """

    def __init__(
        self,
        lambda_orth: float = 0.01,
        lambda_collab: float = 0.001,
        orthogonalize_freq: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_orth = lambda_orth
        self.lambda_collab = lambda_collab
        self.orthogonalize_freq = orthogonalize_freq
        self.loss_fn = CoLAOrthogonalityLoss(lambda_orth, lambda_collab)
        self.step_count = 0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        task_id = inputs.pop("task_id", None) if "task_id" in inputs else None
        inputs = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Forward pass
        if task_id is not None:

            def forward_with_task(input_ids, attention_mask=None, **kwargs):
                return model.forward(input_ids, attention_mask=attention_mask, **kwargs)

            outputs = forward_with_task(**inputs)
        else:
            outputs = model(**inputs)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            # fallback: tự tính loss nếu model không trả về loss
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            labels = inputs.get("labels", None)
            if labels is None:
                raise ValueError(
                    "Labels not found in inputs for manual loss computation."
                )
            loss_fct = nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # CoLA + O-LoRA regularization
        additional_loss = self.loss_fn(model)
        total_loss = loss + additional_loss

        if self.state.global_step % 50 == 0:
            print(
                f"Step {self.state.global_step}: LM Loss: {loss:.4f}, CoLA+O-LoRA Loss: {additional_loss:.4f}"
            )

        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items):
        loss = super().training_step(model, inputs, num_items)
        self.step_count += 1
        if self.step_count % self.orthogonalize_freq == 0:
            self._orthogonalize_model_weights(model)
        return loss

    def _orthogonalize_model_weights(self, model):
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, CoLAOrthogonalLoRALinear):
                module.orthogonalize_weights()
                count += 1
        if count > 0:
            print(f"Orthogonalized {count} CoLA+O-LoRA modules")


def train_cola_olora(
    jsonl_path: str,
    base_model: str,
    output_dir: str,
    max_seq_len: int,
    batch_size: int,
    epochs: int,
    lr: float,
    lambda_orth: float,
    lambda_collab: float,
    orthogonalize_freq: int,
    device: str = "mps",
):
    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float32
    )
    model = model.to(device)

    # Load + format dataset
    with open(jsonl_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    task_names = list(set(item.get("task", "default") for item in raw_data))

    def format_instruction(sample):
        return {
            "text": f"{sample['instruction']}\n{sample['output']}<|endoftext|>",
            "task_id": sample.get("task", "default"),
        }

    formatted_data = [format_instruction(d) for d in raw_data]
    dataset = Dataset.from_list(formatted_data)

    def tokenize(examples):
        tok = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        tok["labels"] = tok["input_ids"].clone()
        tok["task_id"] = examples["task_id"]
        return tok

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Apply CoLA adapter
    model = apply_cola_orthogonal_lora(
        model, task_names, r=8, lora_alpha=16, lora_dropout=0.05
    )

    # Setup Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        logging_steps=10,
        learning_rate=lr,
        weight_decay=0.01,
        fp16=False,
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        # max_steps=60,
    )

    trainer = CoLAOLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=COLORADataCollator(tokenizer),
        lambda_orth=lambda_orth,
        lambda_collab=lambda_collab,
        orthogonalize_freq=orthogonalize_freq,
    )

    trainer.train()
    for name, module in model.named_modules():
        if isinstance(module, CoLAOrthogonalLoRALinear):
            module.orthogonalize_weights()

    return model, tokenizer


# === Chạy huấn luyện 1 vòng CoLA cho 1 task ===
def train_cola_round(
    train_cola_olora_fn,
    task: str,
    rank: int,
    base_model_path: str,
    output_dir: str,
    jsonl_path: str,
    max_seq_len: int = 16,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 3e-4,
    lambda_orth: float = 0.01,
    lambda_collab: float = 0.001,
    orthogonalize_freq: int = 20,
    device: str = "mps",
):
    print(f"\nTraining task: {task} (rank={rank}) from base: {base_model_path}")

    model, tokenizer = train_cola_olora_fn(
        jsonl_path=jsonl_path,
        base_model=base_model_path,
        output_dir=output_dir,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lambda_orth=lambda_orth,
        lambda_collab=lambda_collab,
        orthogonalize_freq=orthogonalize_freq,
        device=device,
    )

    return model, tokenizer


# === Merge adapter sau mỗi vòng ===
def merge_adapter(base_model_path, adapter_path):
    print(f"Merging adapter: {adapter_path}")

    # Load base model gốc chưa chỉnh sửa
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True
    )

    # Load adapter PEFT đã lưu
    model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge LoRA vào backbone và trả về mô hình thường
    merged_model = model_with_adapter.merge_and_unload()
    return merged_model


# === Chạy toàn bộ chuỗi huấn luyện CoLA ===
def run_cola_chain(
    train_cola_olora_fn,
    jsonl_path: str,
    cola_steps: List[Dict],
):
    model = None
    tokenizer = None

    for step in cola_steps:
        base_model_path = step["base_model_path"]
        output_dir = step["output_dir"]
        adapter_path = step["adapter_path"]
        task = step["task"]
        rank = step["rank"]

        # Train + apply CoLA
        model, tokenizer = train_cola_round(
            train_cola_olora_fn=train_cola_olora_fn,
            task=task,
            rank=rank,
            base_model_path=base_model_path,
            output_dir=output_dir,
            jsonl_path=jsonl_path,
        )

        # (Tuỳ chọn) Tách adapter riêng ra nếu cần dùng lại ở nơi khác
        extract_single_task_adapter(
            trained_model=model,
            task_name=task,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            save_path=adapter_path,
            base_model_name=base_model_path,
        )
        model = merge_adapter(base_model_path, adapter_path)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"Round {step['round']} complete → model ready for next round.")

    # Cuối cùng, lưu model đã tích hợp tất cả adapter
    final_path = os.path.join(cola_steps[-1]["output_dir"], "final_model")
    print(f"\nSaving final merged model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("Finished all CoLA rounds.")


if __name__ == "__main__":
    # Run the entire CoLA training chain
    run_cola_chain(
        train_cola_olora_fn=train_cola_olora,
        jsonl_path="./data/data_100.jsonl",
        # jsonl_path="./data/colora_1k_64token.jsonl",
        cola_steps=[
            {
                "round": 1,
                "task": "support",
                "rank": 8,
                "base_model_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "output_dir": "./colora_output/round_1",
                "adapter_path": "./colora_output/round_1/peft_adapters/support_adapter",
            },
            {
                "round": 2,
                "task": "dev",
                "rank": 4,
                "base_model_path": "./colora_output/round_1",
                "output_dir": "./colora_output/round_2",
                "adapter_path": "./colora_output/round_2/peft_adapters/dev_adapter",
            },
        ],
    )
