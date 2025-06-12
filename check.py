from transformers import AutoTokenizer
import json
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

token_lengths = []

with open("data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        prompt = f"{item.get('instruction', '')}\n{item.get('input', '')}\n{item.get('output', '')}"
        tokens = tokenizer(prompt, return_tensors="pt")
        token_lengths.append(len(tokens["input_ids"][0]))

print("📊 Thống kê độ dài token:")
print("Min:", min(token_lengths))
print("Max:", max(token_lengths))
print("Mean:", np.mean(token_lengths))
print("Median:", np.median(token_lengths))
print("90th percentile:", np.percentile(token_lengths, 90))
print("95th percentile:", np.percentile(token_lengths, 95))
print("99th percentile:", np.percentile(token_lengths, 99))
