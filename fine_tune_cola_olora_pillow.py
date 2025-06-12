import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util

# Cấu hình thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer và model
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = prepare_model_for_kbit_training(model)
model.to(device)

# Prompt pool người dùng định nghĩa trước
prompt_pool = [
    "Tạo báo cáo tổng hợp công việc.",
    "Đề xuất nhiệm vụ phù hợp theo ngữ cảnh.",
    "Hướng dẫn nhân viên mới xử lý công việc.",
]

# Load sentence transformer để chọn prompt tốt nhất
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
prompt_embeddings = emb_model.encode(prompt_pool, convert_to_tensor=True)


def select_prompt(question):
    query_embedding = emb_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, prompt_embeddings)
    best_idx = scores.argmax()
    return prompt_pool[best_idx]


# Load dữ liệu
dataset = load_dataset("json", data_files="data.jsonl", split="train")
splits = dataset.train_test_split(test_size=0.66, seed=42)
task1 = splits["train"]
splits = splits["test"].train_test_split(test_size=0.5, seed=42)
task2, task3 = splits["train"], splits["test"]
tasks = [task1, task2, task3]
task_names = ["task_summary", "task_suggest", "task_guide"]

for i, task in enumerate(tasks):
    print(f"=== Training task {i + 1}: {task_names[i]} ===")

    def preprocess(example):
        prompt = select_prompt(example["instruction"])
        input_text = prompt + "\n" + example["instruction"] + "\n" + example["input"]
        return {
            "input_ids": tokenizer(
                input_text, truncation=True, padding="max_length", max_length=128
            )["input_ids"],
            "labels": tokenizer(
                example["output"], truncation=True, padding="max_length", max_length=128
            )["input_ids"],
        }

    tokenized_dataset = task.map(preprocess)

    # Cấu hình LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_lora = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=f"./results_{task_names[i]}",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model_lora,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    adapter_dir = f"adapter_{task_names[i]}"
    model_lora.save_pretrained(adapter_dir, safe_serialization=False)
    tokenizer.save_pretrained(adapter_dir)

model = model_lora.merge_and_unload()

# Lưu mô hình đã merge adapter (không còn dùng PEFT)
model.save_pretrained("combined_cola_olora_pillow_full", safe_serialization=False)
tokenizer.save_pretrained("combined_cola_olora_pillow_full")
print("✅ Đã huấn luyện và lưu mô hình kết hợp CoLA + O-LoRA + PILLOW")
