import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os

final_model_dir = "./cola_output/merged_model_4"  #
prompt = "### Câu hỏi:\nLỗi ERR:1 là gì?\n\n### Trả lời:\n"
max_new_tokens = 128
device = "mps"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
model = AutoModelForCausalLM.from_pretrained(final_model_dir).to(device)
model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(device)

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

print("\n=== Output ===")
print(generated_text)
