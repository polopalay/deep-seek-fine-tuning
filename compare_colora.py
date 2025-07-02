from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
from sentence_transformers import SentenceTransformer, util
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

model_sim = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
similarity_threshold = 0.8


def load_random_questions(jsonl_path, n_questions=5):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    all_qas = [
        (item["messages"][0]["content"].strip(), item["messages"][1]["content"].strip())
        for item in data
    ]
    return random.sample(all_qas, min(n_questions, len(all_qas)))


def run_inference(model, tokenizer, question, expected, device, max_new_tokens=64):
    chat = [{"role": "user", "content": question}]
    input_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    answer = full_output.replace(prompt_text, "").strip()
    sim = util.cos_sim(
        model_sim.encode(answer, convert_to_tensor=True),
        model_sim.encode(expected, convert_to_tensor=True),
    ).item()

    return answer, sim


def test_merged_model(
    jsonl_path="./data/data.jsonl", n_questions=5, device="mps", max_new_tokens=64
):
    questions = load_random_questions(jsonl_path, n_questions)

    path_lora = "./output/lora/"
    t_lora = AutoTokenizer.from_pretrained(path_lora)
    if t_lora.pad_token is None:
        t_lora.pad_token = t_lora.eos_token
    lora = (
        AutoModelForCausalLM.from_pretrained(path_lora, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    path_colora = "./output/colora/"
    t_colora = AutoTokenizer.from_pretrained(path_colora)
    if t_colora.pad_token is None:
        t_colora.pad_token = t_colora.eos_token
    colora = (
        AutoModelForCausalLM.from_pretrained(path_colora, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    for q, expected in questions:
        a_lora, sim_lora = run_inference(
            lora, t_lora, q, expected, device, max_new_tokens
        )
        a_colora, sim_colora = run_inference(
            colora, t_colora, q, expected, device, max_new_tokens
        )

        # print(f"Lora:\nQ: {q}\nA: {a_lora}\n{sim_lora:.2f}\n{'-'*60}")
        # print(f"COLora:\nQ: {q}\nA: {a_colora}\n{sim_colora:.2f}\n{'-'*60}")
        print(f"Lora:{sim_lora:.2f}")
        print(f"COLora:{sim_colora:.2f}\n{'-'*60}")


test_merged_model(n_questions=100)
