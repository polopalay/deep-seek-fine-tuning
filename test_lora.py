from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json
import random
import os


def load_random_questions(jsonl_path, n_questions=5):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    all_questions = [item["messages"][0]["content"].strip() for item in data]
    return random.sample(all_questions, min(n_questions, len(all_questions)))


def test_multiple_adapters(
    base_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    adapter_dirs=["./output/colora_r8", "./output/colora_r6", "./output/colora_r4"],
    jsonl_path="./data/data.jsonl",
    n_questions=5,
    device="mps",
    max_new_tokens=128,
):
    questions = load_random_questions(jsonl_path, n_questions)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16
    ).to(device)

    model = PeftModel.from_pretrained(base_model, adapter_dirs[0])
    model = model.to(device)

    for adapter_path in adapter_dirs[1:]:
        adapter_name = os.path.basename(adapter_path)
        model.load_adapter(adapter_path, adapter_name=adapter_name, is_trainable=False)

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for q in questions:
        chat = [{"role": "user", "content": q}]
        input_ids = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        answer = full_output.replace(prompt_text, "").strip()

        print(f"Q: {q}\nA: {answer}\n{'-'*60}")


test_multiple_adapters(
    adapter_dirs=[
        "./output/colora_r8",
        # "./output/colora_r6",
    ],
    n_questions=10,
)
