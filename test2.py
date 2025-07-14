import json

file_path = "data/data.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except Exception as e:
            print(f"Lỗi ở dòng {i}: {e}")
