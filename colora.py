from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
import torch
import os
import warnings
import math

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")


def alpha_strategy(r):
    # return r * 2
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
        # print(f"Main loss: {main_loss.item():.2f} Orth loss: {orth_loss.item():.6f}")
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
        formatted = f"### Câu hỏi:\n{example['instruction']}\n\n### Trả lời:\n{example['output']}"
        return tokenizer(
            formatted, truncation=True, padding="max_length", max_length=tokenizer_len
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    model_checkpoint = model_base
    adapter_names = []

    for round_idx, r in enumerate(r_list):
        print(f"\n=== Vòng {round_idx + 1} | r = {r} ===")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
        # model.resize_token_embeddings(len(tokenizer))

        # Freeze các adapter cũ (nếu có)
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
            save_strategy="no",
            report_to="none",
            fp16=False,
        )

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

        model.save_pretrained(f"{output_dir}/{adapter_name}")
        tokenizer.save_pretrained(f"{output_dir}/{adapter_name}")
        model_checkpoint = f"{output_dir}/{adapter_name}"

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model = get_peft_model(model, lora_config)  # apply cấu trúc lại để merge
    model = model.merge_and_unload()

    final_path = f"{output_dir}/merged_model_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"\nĐã huấn luyện xong và merge tại: {final_path}")


if __name__ == "__main__":
    training_using_cola(
        dataset_path="./data/data_1000.jsonl",
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        r_list=[16, 8],
        batch_size=2,
        num_epochs=5,
        orth_lambda=0.5,
        tokenizer_len=32,
        warmup_ratio=0.03,
        learning_rate=5e-5,
        device="mps",
        output_dir="./colora_output",
        base_adapter_name="colora",
    )
