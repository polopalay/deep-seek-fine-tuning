from datasets import load_dataset
import json

dataset = load_dataset("taidng/UIT-ViQuAD2.0", split="train[:1000]")

converted_data = []
for row in dataset:
    if row.get("answers", {}).get("text"):
        answer = row["answers"]["text"][0]
    elif row.get("plausible_answers", {}).get("text"):
        answer = row["plausible_answers"]["text"][0]
    else:
        continue

    entry = {
        "instruction": row["question"],
        "input": row["context"],
        "output": answer,
    }
    converted_data.append(entry)

with open("lora_general_1k.jsonl", "w", encoding="utf-8") as f:
    for item in converted_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
