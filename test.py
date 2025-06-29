from transformers import AutoTokenizer
import json
import numpy as np  # để tính median

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-V2", trust_remote_code=True
)

# Load dữ liệu
with open("./data/data.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]


def count_tokens(example):
    if "messages" in example:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return len(tokenizer.encode(text))
    elif "instruction" in example and "output" in example:
        return len(tokenizer.encode(example["instruction"] + "\n" + example["output"]))
    return 0


# Tính token length
token_lengths = [count_tokens(e) for e in data]

# Phân nhóm
jira = token_lengths[:550]
faq = token_lengths[550:1000]
chatgpt = token_lengths[1000:]


def print_stats(name, group):
    print(f"{name}:")
    print(f"  Trung bình: {sum(group)/len(group):.2f}")
    print(f"  Trung vị  : {np.median(group):.2f}")
    print(f"  Dài nhất  : {max(group)}")
    print()


print_stats("Jira", jira)
print_stats("FAQ", faq)
print_stats("ChatGPT", chatgpt)
