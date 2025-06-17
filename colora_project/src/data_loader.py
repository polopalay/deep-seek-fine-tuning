import json
from datasets import Dataset


def load_and_format_data(path):
    def format(sample):
        return {
            "text": f"{sample['instruction']}\n{sample['output']}",
            "task_id": sample.get("task", "default"),
        }

    with open(path, "r") as f:
        data = [json.loads(l) for l in f]
    formatted = [format(d) for d in data]
    return Dataset.from_list(formatted), list(set(d["task_id"] for d in formatted))
