import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
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
from peft import prepare_model_for_kbit_training
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


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


class OrthogonalLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_dropout = (
            nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        )

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(r, base_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))
        nn.init.orthogonal_(self.lora_A)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = x @ self.lora_A.T
        lora_out = self.lora_dropout(lora_out)
        lora_out = lora_out @ self.lora_B.T
        lora_out *= self.scaling
        return base_out + lora_out


class OrthogonalityLoss(nn.Module):
    def __init__(self, lambda_orth: float = 0.001):  # Giảm từ 0.01 xuống 0.001
        super().__init__()
        self.lambda_orth = lambda_orth

    def forward(self, model) -> torch.Tensor:
        orth_loss = 0.0
        count = 0

        for name, module in model.named_modules():
            if isinstance(module, OrthogonalLoRALinear):
                A = module.lora_A
                A_gram = torch.matmul(A, A.T)
                I_A = torch.eye(A_gram.size(0), device=A.device, dtype=A.dtype)
                orth_loss += F.mse_loss(A_gram, I_A, reduction="sum")
                count += 1

        if count > 0:
            return self.lambda_orth * (orth_loss / count)
        return torch.tensor(0.0, device=next(model.parameters()).device)


def apply_orthogonal_lora(
    model, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None
):
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ]

    def replace_module(parent, name, module):
        if isinstance(module, nn.Linear) and any(
            target in name for target in target_modules
        ):
            setattr(
                parent, name, OrthogonalLoRALinear(module, r, lora_alpha, lora_dropout)
            )

    def apply_recursive(module):
        for name, child in module.named_children():
            replace_module(module, name, child)
            apply_recursive(child)

    apply_recursive(model)
    return model


class CoLAOLoRATrainer(Trainer):
    def __init__(self, lambda_orth: float = 0.001, *args, **kwargs):  # Giảm lambda
        super().__init__(*args, **kwargs)
        self.loss_fn = OrthogonalityLoss(lambda_orth)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Shift labels để align với generation task
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        orth_loss = self.loss_fn(model)
        total_loss = lm_loss + orth_loss

        print(f"lm_loss: {lm_loss.item()}, orth_loss: {orth_loss.item()}")

        return (total_loss, outputs) if return_outputs else total_loss


def train_cola_olora(
    jsonl_path: str,
    adapter_name: str,
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    output_dir: str = "./colora_output",
    max_seq_len: int = 256,
    batch_size: int = 4,
    epochs: int = 3,
    lr: float = 5e-5,
    lambda_orth: float = 0.001,
    device: str = "cpu",
):
    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    raw_data = load_jsonl(jsonl_path)

    def format_instruction(sample):
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{sample['output']}<|eot_id|>"
        return {"text": text}

    print("Loading and formatting data...")
    formatted_data = [format_instruction(d) for d in raw_data]
    dataset = Dataset.from_list(formatted_data)
    # Load model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Thiết lập tokenizer đúng cách
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Quan trọng cho training

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model = model.to(device)

    model = apply_orthogonal_lora(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
        ],
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Tokenize dataset với label masking đúng
    def tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )

        # Tạo labels và mask phần input (chỉ train trên output)
        labels = tokenized["input_ids"].clone()

        for i, input_ids in enumerate(labels):
            assistant_start = None
            for j in range(len(input_ids) - 1):
                if tokenizer.decode(input_ids[j : j + 10]).find("assistant") != -1:
                    assistant_start = j + 10  # Skip header
                    break

            if assistant_start is not None:
                labels[i][:assistant_start] = -100

        tokenized["labels"] = labels
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training arguments với cài đặt conservative hơn
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{adapter_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=50,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        learning_rate=lr,
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        max_grad_norm=1.0,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print("Starting O-LoRA training...")
    trainer = CoLAOLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        lambda_orth=lambda_orth,
    )

    trainer.train()

    # Save adapter
    def save_cola_olora_adapter(model, save_path):
        os.makedirs(save_path, exist_ok=True)

        lora_state_dict = {}
        for name, module in model.named_modules():
            if isinstance(module, OrthogonalLoRALinear):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.data

        torch.save(lora_state_dict, f"{save_path}/lora_weights.pt")

        config = {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj",
            ],
        }

        with open(f"{save_path}/cola_olora_config.json", "w") as f:
            json.dump(config, f, indent=2)

    save_cola_olora_adapter(model, f"{output_dir}/{adapter_name}")
    tokenizer.save_pretrained(f"{output_dir}/{adapter_name}")

    print(f"CO-LoRA adapter '{adapter_name}' saved to {output_dir}/{adapter_name}")

    return model, tokenizer


if __name__ == "__main__":
    model, tokenizer = train_cola_olora(
        jsonl_path="./data/data_100.jsonl",
        adapter_name="dev_support_colora",
        lambda_orth=0.001,
        epochs=30,
        lr=1e-5,
        batch_size=4,
        max_seq_len=32,
    )
