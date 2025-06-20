from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback
import torch
import os
import warnings
import math

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")


def alpha_strategy(r):
    c = 8
    return int(round(math.sqrt(r) * c))


def orthogonal_loss(A):
    AtA = torch.einsum("ik,jk->ij", A, A)
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    return torch.sum((AtA - I) ** 2)


class OrthLoRATrainer(Trainer):
    def __init__(self, *args, orth_lambda=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.orth_lambda = orth_lambda

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        main_loss = outputs.loss
        current_epoch = getattr(self.state, "epoch", 0.0)
        total_epochs = getattr(self.args, "num_train_epochs", 1)
        threshold = getattr(self.args, "threshold", 2.0)

        orth_loss = 0.0
        if main_loss.item() < threshold:
            lambda_orth = self.orth_lambda * (current_epoch / total_epochs)
            ck_orth = ["q_proj", "v_proj", "k_proj", "gate_proj"]
            for name, module in model.named_modules():
                if any(key in name for key in ck_orth) and hasattr(module, "lora_A"):
                    lora_A = module.lora_A
                    if isinstance(lora_A, torch.nn.ModuleDict):
                        for _, sub_A in lora_A.items():
                            orth_loss += orthogonal_loss(sub_A.weight)
                    else:
                        orth_loss += orthogonal_loss(lora_A.weight)
        else:
            lambda_orth = 0.0

        total_loss = main_loss + lambda_orth * orth_loss
        print(f"Main loss: {main_loss.item():.2f} Orth loss: {float(orth_loss):.6f}")

        return (total_loss, outputs) if return_outputs else total_loss


def training_using_cola(
    dataset_path="./data/data_1000.jsonl",
    model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    r_list=[16, 8, 4],
    batch_size=2,
    num_epochs=5,
    orth_lambda=0.1,
    tokenizer_len=32,
    warmup_ratio=0.03,
    learning_rate=5e-5,
    device="mps",
    output_dir="./colora_output",
    base_adapter_name="colora",
):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=dataset_path)["train"].train_test_split(
        test_size=0.1
    )

    def tokenize(example):
        formatted = f"### Câu hỏi:\n{example['instruction']}\n\n### Trả lời:\n{example['output']}{tokenizer.eos_token}"
        return tokenizer(
            formatted, truncation=True, padding="max_length", max_length=tokenizer_len
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    model_checkpoint = model_base
    adapter_names = []

    for round_idx, r in enumerate(r_list):
        print(f"\n=== Vòng {round_idx + 1} | r = {r} ===")

        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        # Freeze các adapter cũ
        if adapter_names:
            frozen_count = 0
            for name, param in model.named_parameters():
                if any(aname in name for aname in adapter_names):
                    param.requires_grad = False
                    frozen_count += 1

        alpha = alpha_strategy(r)
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model = model.to(device)

        adapter_name = f"{base_adapter_name}_r{r}"
        adapter_names.append(adapter_name)

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{adapter_name}",
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            logging_steps=20,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            report_to="none",
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,
        )
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.001,
        )

        trainer = OrthLoRATrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            orth_lambda=orth_lambda,
            callbacks=[early_stopping],
        )

        trainer.train()

        model = model.merge_and_unload()

        merged_ckpt_dir = f"{output_dir}/merged_model_{r}"
        os.makedirs(merged_ckpt_dir, exist_ok=True)
        model.save_pretrained(merged_ckpt_dir)
        tokenizer.save_pretrained(merged_ckpt_dir)
        print(f"Đã lưu mô hình merged tại: {merged_ckpt_dir}")

        model_checkpoint = merged_ckpt_dir


if __name__ == "__main__":
    training_using_cola(
        dataset_path="./data/data.jsonl",
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        r_list=[16, 8, 4],
        batch_size=2,
        num_epochs=3,
        orth_lambda=0.5,
        tokenizer_len=128,
        warmup_ratio=0.04,
        learning_rate=2e-5,
        # learning_rate=5e-5,
        device="mps",
        output_dir="./colora_output",
        base_adapter_name="colora",
    )
