from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os
import json
import math

def alpha_strategy(r):
    c = 8
    return int(round(math.sqrt(r) * c))


def training_using_lora(
    data_path="./data/data_1000.jsonl",
    model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    r=8,
    learning_rate=2e-4,
    num_epochs=8,
    batch_size=2,
    tokenizer_len=128,
    warmup_ratio=0.1,
    output_dir="./lora_output",
    adapter_name="lora_r8",
    device="mps",
):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=data_path)["train"].train_test_split(
        test_size=0.1
    )

    def tokenize(data):
        formatted = tokenizer.apply_chat_template(
            data["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer(
            formatted,
            truncation=True,
            padding=True,
            max_length=tokenizer_len,
        )

    tokenized = dataset.map(tokenize, remove_columns=["messages"])

    model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha_strategy(r),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config).to(device)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{adapter_name}",
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=100,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        report_to="none",
        save_strategy="no",
        fp16=False,
        max_grad_norm=(8 / r) if r < 8 else None,
    )

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
        data_path="data/data.jsonl",
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        r=16,
        learning_rate=5e-4,
        num_epochs=20,
        batch_size=2,
        tokenizer_len=128,
        warmup_ratio=0.1,
        output_dir="output",
        adapter_name="lora",
        device="mps",
    )
