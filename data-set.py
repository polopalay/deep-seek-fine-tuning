import json
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("taidng/UIT-ViQuAD2.0", split="train").to_pandas()


def extract_answer(row):
    if (
        isinstance(row["answers"], dict)
        and "text" in row["answers"]
        and len(row["answers"]["text"]) > 0
    ):
        return row["answers"]["text"][0]
    elif (
        isinstance(row["plausible_answers"], dict)
        and "text" in row["plausible_answers"]
        and len(row["plausible_answers"]["text"]) > 0
    ):
        return row["plausible_answers"]["text"][0]
    else:
        return None


dataset["answer"] = dataset.apply(extract_answer, axis=1)
dataset = dataset.dropna(subset=["answer"])

output_data = []
for _, row in dataset.iterrows():
    entry = {
        "instruction": "Trả lời câu hỏi dựa trên đoạn văn sau:",
        "input": f"Câu hỏi: {row['question']}\n\nVăn bản: {row['context']}",
        "output": row["answer"],
    }
    output_data.append(entry)

with open("lora_general_stage1.jsonl", "w", encoding="utf-8") as f:
    for item in output_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
