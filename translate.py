from deep_translator import GoogleTranslator
import json

translated_data = []
with open("data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        translated_item = {
            "instruction": GoogleTranslator(source="en", target="vi").translate(
                item["instruction"]
            ),
            "input": GoogleTranslator(source="en", target="vi").translate(
                item["input"]
            ),
            "output": GoogleTranslator(source="en", target="vi").translate(
                item["output"]
            ),
        }
        translated_data.append(translated_item)

with open("data_vi.jsonl", "w", encoding="utf-8") as f:
    for item in translated_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
