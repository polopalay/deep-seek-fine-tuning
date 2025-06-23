from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
print(tokenizer.special_tokens_map)
print(tokenizer.all_special_tokens)
