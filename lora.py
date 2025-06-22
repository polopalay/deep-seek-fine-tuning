from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling
import torch
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


def training_using_lora(
    dataset_path: str = "./data/data_100.jsonl",
    model_base: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    r: int = 8,
    batch_size: int = 4,
    num_epochs: int = 3,
    output_dir: str = "./colora_output",
    adapter_name: str = "deepseek_lora_adapter",
):
    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset["train"].train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        formatted = f"### Câu hỏi:\n{example['instruction']}\n\n### Trả lời:\n{example['output']}{tokenizer.eos_token}"
        return tokenizer(
            formatted, truncation=True, padding="max_length", max_length=128
        )

    model = AutoModelForCausalLM.from_pretrained(model_base)
    model.resize_token_embeddings(len(tokenizer))
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
    tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    # 5. Cấu hình training
    training_args = TrainingArguments(
        output_dir="./results_lora",
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=1,
        num_train_epochs=num_epochs,
        logging_steps=20,
        # save_strategy="epoch",
        # evaluation_strategy="epoch",
        report_to="none",
        fp16=False,
    )

    # 6. Huấn luyện
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    model.save_pretrained(f"{output_dir}/{adapter_name}")
    tokenizer.save_pretrained(f"{output_dir}/{adapter_name}")


if __name__ == "__main__":
    training_using_lora(
        dataset_path="./data/data_100.jsonl",
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        r=16,
        batch_size=1,
        num_epochs=1,
        output_dir="./colora_output",
        adapter_name="lora_adapter",
    )
