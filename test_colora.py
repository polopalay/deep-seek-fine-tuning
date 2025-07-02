from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
from sentence_transformers import SentenceTransformer, util

model_sim = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

similarity_threshold = 0.8


def load_random_questions(jsonl_path, n_questions=5):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    all_qas = []
    for item in data:
        q = item["messages"][0]["content"].strip()
        a = item["messages"][1]["content"].strip()
        all_qas.append((q, a))

    return random.sample(all_qas, min(n_questions, len(all_qas)))


def test_merged_model(
    model_path="./output/colora/",
    jsonl_path="./data/data.jsonl",
    n_questions=5,
    device="mps",
    max_new_tokens=64,
):
    questions = load_random_questions(jsonl_path, n_questions)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for q, expected in questions:
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
        # answer = answer.split(".")[0].strip()
        sim = util.cos_sim(
            model_sim.encode(answer, convert_to_tensor=True),
            model_sim.encode(expected, convert_to_tensor=True),
        ).item()

        is_correct = sim >= similarity_threshold

        print(
            f"Q: {q}\nA: {answer}\nSimilarity: {sim:.2f} → {'✓' if is_correct else '✗'}\n{'-'*60}"
        )


test_merged_model(model_path="./output/colora/", n_questions=40)
