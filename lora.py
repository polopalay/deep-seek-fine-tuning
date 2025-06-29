import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import Dataset
import json

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "/content/drive/MyDrive/data.jsonl"
NUM_EPOCHS_PER_LOOP = 2
BATCH_SIZE = 1
OUTPUT_DIR = "./cola_outputs"
DEVICE = "cuda"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token


with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

dataset = Dataset.from_list(raw_data).train_test_split(test_size=0.1)


def tokenize(example):
    formatted = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(
        formatted,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized = {}
for split in dataset:
    tokenized[split] = dataset[split].map(
        tokenize,
        remove_columns=["messages"],
        batched=False,
    )

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model).to(DEVICE)

for loop in range(2):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/loop{loop+1}",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
    )

    trainer.train()
    model.merge_and_unload()

model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
