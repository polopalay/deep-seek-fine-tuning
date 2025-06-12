import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel

# Cấu hình mô hình và dữ liệu
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
data_path = "deepseek_finetune_invoice_10k.jsonl"
output_dir = "./deepseek_lora_invoice_cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Đặt token kết thúc làm padding token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # CPU chỉ hỗ trợ float32
)
model = model.to("cpu")

# Cấu hình LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Load dataset JSONL
dataset = load_dataset("json", data_files=data_path, split="train")


# Tiền xử lý
def tokenize(example):
    prompt = example["instruction"]
    if example.get("input"):
        prompt += "\n" + example["input"]
    prompt += "\n" + example["output"]
    tokens = tokenizer(prompt, padding="max_length", max_length=128, truncation=True)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# Cấu hình huấn luyện
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # CPU nên để thấp
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=False,  # CPU không hỗ trợ fp16
    evaluation_strategy="no",
    report_to="none",
    resume_from_checkpoint=True,  # Bỏ qua nếu không cần resume
)

# Huấn luyện
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # resume_from_checkpoint=True,
)

trainer.train()
