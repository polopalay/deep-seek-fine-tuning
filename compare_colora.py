import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from transformers.utils import logging

logging.set_verbosity_error()
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
    jsonl_path="./data/data.jsonl",
    n_questions=5,
    device="mps",
    max_new_tokens=64,
    path_lora="./output/lora/",
    path_colora="./output/colora/",
    similarity_threshold=0.8,
    compare_mode="score",  # hoặc "match"
    show_answer=True,
    save_path=None,
    random_seed=42,
):
    random.seed(random_seed)
    questions = load_random_questions(jsonl_path, n_questions)

    # Load Lora model
    t_lora = AutoTokenizer.from_pretrained(path_lora)
    if t_lora.pad_token is None:
        t_lora.pad_token = t_lora.eos_token
    lora = (
        AutoModelForCausalLM.from_pretrained(path_lora, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    # Load COLoRA model
    t_colora = AutoTokenizer.from_pretrained(path_colora)
    if t_colora.pad_token is None:
        t_colora.pad_token = t_colora.eos_token
    colora = (
        AutoModelForCausalLM.from_pretrained(path_colora, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    result_list = []

    for idx, (q, expected) in enumerate(questions):
        a_lora, sim_lora = run_inference(
            lora, t_lora, q, expected, device, max_new_tokens
        )
        a_colora, sim_colora = run_inference(
            colora, t_colora, q, expected, device, max_new_tokens
        )

        match_lora = sim_lora >= similarity_threshold
        match_colora = sim_colora >= similarity_threshold

        if show_answer:
            print(f"\n[{idx+1}] Câu hỏi: {q}")
            print(
                f"[Lora]\n→ {a_lora}\n→ Similarity: {sim_lora:.2f} → {'v' if match_lora else 'x'}"
            )
            print(
                f"[COLoRA]\n→ {a_colora}\n→ Similarity: {sim_colora:.2f} → {'v' if match_colora else 'x'}"
            )
        else:
            print(f"[{idx+1}] Lora: {sim_lora:.2f}, COLoRA: {sim_colora:.2f}")

        result_list.append(
            {
                "question": q,
                "expected": expected,
                "lora": {"answer": a_lora, "similarity": sim_lora, "match": match_lora},
                "colora": {
                    "answer": a_colora,
                    "similarity": sim_colora,
                    "match": match_colora,
                },
            }
        )

    total = len(result_list)
    if compare_mode == "score":
        avg_lora = sum(r["lora"]["similarity"] for r in result_list) / total
        avg_colora = sum(r["colora"]["similarity"] for r in result_list) / total
        print(f"\nAvg Similarity → Lora: {avg_lora:.3f}, COLoRA: {avg_colora:.3f}")
    else:
        acc_lora = sum(r["lora"]["match"] for r in result_list) / total
        acc_colora = sum(r["colora"]["match"] for r in result_list) / total
        print(
            f"\nAccuracy (>{similarity_threshold}) → Lora: {acc_lora:.2%}, COLoRA: {acc_colora:.2%}"
        )

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)
        print(f"\nKết quả đã lưu tại: {save_path}")


test_merged_model(
    jsonl_path="./data/data.jsonl",
    n_questions=100,
    device="mps",
    max_new_tokens=64,
    path_lora="./output/lora/",
    path_colora="./output/olora/",
    similarity_threshold=0.75,
    compare_mode="match",
    show_answer=False,
    save_path="./compare_result.json",
)
