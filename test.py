import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util


# =============================
# CONFIG
# =============================
MODEL_PATH = "./output/solora"
DATA_PATH = "./data/data.jsonl"
DEVICE = "mps"  # "cuda" or "cpu" if needed
TOP_K = 5
MAX_NEW_TOKENS = 64
SIM_THRESHOLD = 0.8


# =============================
# LOAD MODELS
# =============================
print("Loading models...")
sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()


# =============================
# UTILS
# =============================
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_top_k_answers(prompt, tokenizer, model, k=5, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        num_return_sequences=k,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]


def eval_example(question, gt_answer, k=5):
    generated = generate_top_k_answers(question, tokenizer, model, k=k)
    scores = [
        util.cos_sim(
            sim_model.encode(gt_answer, convert_to_tensor=True),
            sim_model.encode(ans, convert_to_tensor=True),
        ).item()
        for ans in generated
    ]
    hits = [1 if score >= SIM_THRESHOLD else 0 for score in scores]

    precision_at_k = sum(hits) / k
    recall_at_k = sum(hits) / 1  # 1 ground truth
    ndcg = sum(
        [hit / torch.log2(torch.tensor(rank + 2.0)) for rank, hit in enumerate(hits)]
    ).item()
    map_k = sum(
        [sum(hits[: i + 1]) / (i + 1) if hits[i] == 1 else 0 for i in range(k)]
    ) / max(1, sum(hits))
    mrr = 1 / (hits.index(1) + 1) if 1 in hits else 0

    return {
        "P@K": precision_at_k,
        "R@K": recall_at_k,
        "NDCG@K": ndcg,
        "MAP@K": map_k,
        "MRR": mrr,
        "topK": generated,
        "scores": scores,
        "hit@K": sum(hits),
    }


# =============================
# MAIN TEST FUNCTION
# =============================
def evaluate_model(data_path):
    data = load_jsonl(data_path)
    results = []

    print(f"Evaluating on {len(data)} samples...")
    for item in data:
        q = item["messages"][0]["content"]
        a = item["messages"][1]["content"]
        result = eval_example(q, a, k=TOP_K)
        result["question"] = q
        result["ground_truth"] = a
        results.append(result)

    print("\n==== AVERAGE METRICS ====")
    print(f"Precision@{TOP_K}: {np.mean([r['P@K'] for r in results]):.4f}")
    print(f"Recall@{TOP_K}: {np.mean([r['R@K'] for r in results]):.4f}")
    print(f"NDCG@{TOP_K}: {np.mean([r['NDCG@K'] for r in results]):.4f}")
    print(f"MAP@{TOP_K}: {np.mean([r['MAP@K'] for r in results]):.4f}")
    print(f"MRR: {np.mean([r['MRR'] for r in results]):.4f}")

    return results


# =============================
# RUN
# =============================
if __name__ == "__main__":
    results = evaluate_model(DATA_PATH)

    # Optionally save the results
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
