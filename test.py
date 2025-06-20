from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

max_tokens = 0
max_line = None
max_index = -1

with open("data/data.jsonl", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        tokens_instruction = len(tokenizer.encode(obj["instruction"]))
        tokens_output = len(tokenizer.encode(obj["output"]))
        tokens_function = len(tokenizer.encode(obj["function"]))
        sum_tokens = tokens_instruction + tokens_output + tokens_function

        if sum_tokens > max_tokens:
            max_tokens = sum_tokens
            max_line = obj
            max_index = idx

print(f"Dòng có tổng token lớn nhất: {max_index}")
print(f"Token count: {max_tokens}")
print(f"Nội dung dòng:")
print(json.dumps(max_line, ensure_ascii=False, indent=2))
