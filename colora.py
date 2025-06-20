from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
import torch
import os


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def orthogonal_loss(A):
    AtA = torch.matmul(A, A.T)  # [r, r]
    I = torch.eye(A.size(0), device=A.device)
    return torch.norm(AtA - I, p="fro") ** 2


class OrthLoRATrainer(Trainer):
    def __init__(self, *args, orth_lambda=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.orth_lambda = orth_lambda

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        main_loss = outputs.loss

        lambda_orth = (
            self.args.orth_lambda if hasattr(self.args, "orth_lambda") else 0.1
        )
        orth_loss = 0.0

        ck_orth = ["q_proj", "v_proj"]
        for name, module in model.named_modules():
            if any(key in name for key in ck_orth) and hasattr(module, "lora_A"):
                lora_A = module.lora_A
                if isinstance(lora_A, torch.nn.ModuleDict):
                    for name, sub_A in lora_A.items():
                        orth_loss += orthogonal_loss(sub_A.weight)
                else:
                    orth_loss += orthogonal_loss(lora_A.weight)

        total_loss = main_loss + lambda_orth * orth_loss

        print(f"Main loss: {main_loss.item():.2f}, Orth loss: {orth_loss.item():.2f}")
        return (total_loss, outputs) if return_outputs else total_loss


def training_using_cola(
    dataset_path: str = "./data/data_100.jsonl",
    model_base: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    r_list: list = [16, 8, 4],
    batch_size: int = 4,
    num_epochs: int = 3,
    orth_lambda: float = 0.01,
    tokenizer_len: int = 64,
    warmup_ratio: float = 0.03,
    learning_rate: float = 1e-4,
    device: str = "mps",
    output_dir: str = "./colora_output",
    base_adapter_name: str = "colora",
):
    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset once
    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset["train"].train_test_split(test_size=0.1)

    def tokenize(example):
        formatted = f"### Câu hỏi:\n{example['instruction']}\n\n### Trả lời:\n{example['output']}"
        return tokenizer(
            formatted, truncation=True, padding="max_length", max_length=tokenizer_len
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    # Start with base model
    model_checkpoint = model_base

    for round_idx, r in enumerate(r_list):
        print(f"\n=== Vòng {round_idx + 1} | r = {r} ===")

        # Load model (từ mô hình gốc hoặc mô hình đã merge trước đó)
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        model.resize_token_embeddings(len(tokenizer))

        # Apply new LoRA config
        alpha = r * 2
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

        # Define training args
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{base_adapter_name}_r{r}",
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            logging_steps=20,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            save_strategy="no",
            report_to="none",
            fp16=False,
        )

        # Train
        trainer = OrthLoRATrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            orth_lambda=orth_lambda,
        )

        trainer.train()
        model = model.merge_and_unload()

        # Lưu checkpoint đã merge
        merged_ckpt_dir = f"{output_dir}/merged_model_{r}"
        os.makedirs(merged_ckpt_dir, exist_ok=True)
        model.save_pretrained(merged_ckpt_dir)
        tokenizer.save_pretrained(merged_ckpt_dir)
        print(f"Đã lưu mô hình merged tại: {merged_ckpt_dir}")

        # Cập nhật đường dẫn checkpoint cho vòng tiếp theo
        model_checkpoint = merged_ckpt_dir

    print("\nHoàn tất huấn luyện theo chiến lược CoLoRA.")


if __name__ == "__main__":
    training_using_cola(
        dataset_path="./data/data_1000.jsonl",
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        r_list=[16, 8],
        batch_size=2,
        num_epochs=3,
        orth_lambda=0.1,
        tokenizer_len=32,
        learning_rate=5e-5,
        device="mps",
        output_dir="./colora_output",
        base_adapter_name="colora",
    )
