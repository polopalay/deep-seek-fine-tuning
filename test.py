from transformers import AutoTokenizer
import json

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


token_lengths = [count_tokens(e) for e in data]
jira = token_lengths[:550]
faq = token_lengths[550:1000]
chatgpt = token_lengths[1000:]

print("Jira:", sum(jira) / len(jira))
print("FAQ:", sum(faq) / len(faq))
print("ChatGPT:", sum(chatgpt) / len(chatgpt))
