import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# **1. Chọn thiết bị (Mac M1/M2 hỗ trợ MPS)**
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# **2. Tắt giới hạn VRAM trên Mac MPS**
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# **3. Load dataset**
dataset_path = "lora_general_1k.jsonl"
dataset = load_dataset("json", data_files=dataset_path, split="train")

# **4. Chọn model**
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# **5. Load tokenizer**
tokenizer = AutoTokenizer.from_pretrained(model_name)

# **6. Load model (KHÔNG dùng 4-bit quantization)**
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

# **7. Áp dụng LoRA để fine-tune nhanh hơn**
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
# model = model.to(device)


# **8. Tokenization**
def tokenize_function(examples):
    # Nối từng cặp instruction + input
    combined_inputs = [
        instr + "\n" + inp
        for instr, inp in zip(examples["instruction"], examples["input"])
    ]

    # Tokenize đầu vào và đầu ra
    model_inputs = tokenizer(
        combined_inputs,
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    labels = tokenizer(
        examples["output"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )["input_ids"]

    model_inputs["labels"] = labels
    return model_inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# **9. Chia train/test**
split_datasets = tokenized_datasets.train_test_split(train_size=0.8, seed=42)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# **10. Cấu hình Training**
training_args = TrainingArguments(
    output_dir="./deepseek_finetuned",
    per_device_train_batch_size=1,  # nhỏ nhất để tránh nghẽn RAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # tăng batch hiệu dụng lên 2
    num_train_epochs=1,  # huấn luyện 1 vòng đủ để test chất lượng
    max_steps=300,  # giới hạn chỉ train 50 step
    logging_steps=5,  # in log thường xuyên để theo dõi
    save_strategy="no",  # không lưu model giữa chừng
    evaluation_strategy="no",  # bỏ eval để tiết kiệm thời gian
    logging_dir="./logs",
    remove_unused_columns=False,  # bắt buộc với PEFT
    optim="adamw_torch",  # native optimizer của PyTorch
    report_to="none",  # tắt WandB/huggingface logging
    fp16=False,  # MPS không hỗ trợ fp16
    bf16=False,
)


# **11. Khởi tạo Trainer**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# **12. Bắt đầu huấn luyện**
trainer.train()

# **13. Lưu model sau khi train**
model.save_pretrained("deepseek_finetuned_model")
tokenizer.save_pretrained("deepseek_finetuned_model")

# **14. Test mô hình sau khi train**
from transformers import pipeline

qa_pipeline = pipeline("text-generation", model="deepseek_finetuned_model")

question = "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?"
context = "Lâm Bá Kiệt"
result = qa_pipeline(question + " " + context, return_full_text=False)
result = print(result[0]["generated_text"])
print(result)
