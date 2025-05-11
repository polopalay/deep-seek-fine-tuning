from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset_path = "data.jsonl"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # MPS không hỗ trợ bfloat16/fp16 ổn định
    device_map={"": device},
)

# LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Tùy model mà chọn khác
    bias="none",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Load dataset
dataset = load_dataset("json", data_files=dataset_path)["train"]


def formatting_func(example):
    prompt = f"<|user|>\n{example['instruction']}\n{example['input']}\n<|assistant|>\n{example['output']}"
    return {"text": prompt}


dataset = dataset.map(formatting_func)


# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"], truncation=True, padding="max_length", max_length=512
    )


dataset = dataset.map(tokenize, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./deepseek-lora-output",
    per_device_train_batch_size=1,  # nhỏ lại để tránh tràn RAM
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    evaluation_strategy="no",
    fp16=False,  # ❌ TẮT fp16
    bf16=False,  # ❌ TẮT bf16
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
