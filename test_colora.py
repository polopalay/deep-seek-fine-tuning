import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

final_model_dir = "./colora_output/merged_model_16/"
device = "mps"
max_new_tokens = 128

# Danh sách các prompt
prompt_list = [
    "### Câu hỏi:\nDoanh nghiệp FDI có dùng HĐĐT?\n\n### Trả lời:\n",
    "### Câu hỏi:\nLỗi ERR:11 là gì?\n\n### Trả lời:\n",
    "### Câu hỏi:\nSửa sai số lượng hàng hóa?\n\n### Trả lời:\n",
    "### Câu hỏi:\nERR:1 là lỗi gì?\n\n### Trả lời:\n",
    "### Câu hỏi:\nHóa đơn có thể bị từ chối?\n\n### Trả lời:\n",
    "### Câu hỏi:\n1+1=?\n\n### Trả lời:\n",
]

# Load model & tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
model = AutoModelForCausalLM.from_pretrained(final_model_dir).to(device)
model.eval()

# Xử lý từng prompt
for i, prompt in enumerate(prompt_list):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== Output {i+1} ===")
    print(generated_text)
