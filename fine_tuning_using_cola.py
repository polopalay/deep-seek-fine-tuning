import os
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 🔧 Cấu hình
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
data_path = "deepseek_invoice_cola_vi_2k.jsonl"  # Đổi tên nếu khác
device = "cpu"

# 🚀 Load tokenizer và model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model = prepare_model_for_kbit_training(model)  # Cho phép LoRA chạy nhẹ
model = model.to(device)

# 🧩 Cấu hình LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


# 📚 Load và chuẩn hóa dữ liệu
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


raw_data = load_jsonl(data_path)


# Chuyển dữ liệu về format CoLA cho huấn luyện
def format_instruction(sample):
    prompt = f"### Câu hỏi:\n{sample['instruction']}\n\n### Trả lời:"
    return {"text": f"{prompt} {sample['output']}"}


formatted_data = [format_instruction(d) for d in raw_data]
dataset = Dataset.from_list(formatted_data)


# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"], truncation=True, max_length=128, padding="max_length"
    )


tokenized_dataset = dataset.map(tokenize)

# 🏋️ Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir="./deepseek_lora_invoice_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    logging_steps=20,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
    report_to="none",
    resume_from_checkpoint=True,
)

# 🧠 Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 🔥 Bắt đầu huấn luyện
trainer.train()
model.save_pretrained("deepseek_cola_invoice_adapter")
