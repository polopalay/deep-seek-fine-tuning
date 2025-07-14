import torch
from bert_score import score as bert_score_fn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)
from sentence_transformers import SentenceTransformer, util

from rouge_score import rouge_scorer
from detoxify import Detoxify
import nltk
from collections import Counter
import json
import numpy as np
import random
import os
import sacrebleu

# nltk.download("punkt")

# Load all required models
bert_sim = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tox_model = Detoxify("original")
similarity_threshold = 0.8
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
# bleu = evaluate.load("bleu")


def tokenize_and_get_bleu(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score / 100.0


def exact_match(pred, ref):
    return pred.strip().lower() == ref.strip().lower()


def distinct_n(responses, n):
    all_ngrams = [
        tuple(tokens[i : i + n])
        for r in responses
        for tokens in [r.split()]
        for i in range(len(tokens) - n + 1)
    ]
    return len(set(all_ngrams)) / max(1, len(all_ngrams))


def evaluate_model(
    model_path="./output/solora",
    jsonl_path="./data/data.jsonl",
    device="cuda",
    n_questions=20,
    max_new_tokens=64,
    seed=42,
):
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    samples = random.sample(data, min(n_questions, len(data)))
    refs, preds, sims, ems, rouges, lens = [], [], [], [], [], []

    for sample in samples:
        q = sample["messages"][0]["content"].strip()
        a = sample["messages"][1]["content"].strip()

        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            q,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(output[0], skip_special_tokens=True)

        # Metrics
        refs.append(a)
        preds.append(gen)
        lens.append(len(gen.split()))
        ems.append(exact_match(gen, a))
        sims.append(
            util.cos_sim(
                bert_sim.encode(gen, convert_to_tensor=True),
                bert_sim.encode(a, convert_to_tensor=True),
            )[0][0].item()
        )
        rouges.append(rouge.score(a, gen)["rougeL"].fmeasure)
        print("=" * 80)
        print(f"[Q]: {q}")
        print(f"[Model Output ]: {gen}")

    bleu4 = tokenize_and_get_bleu(preds, refs)
    bert_p, bert_r, bert_f1 = bert_score_fn(
        preds, refs, lang="en", rescale_with_baseline=True
    )

    bert_p = bert_p.cpu().numpy()
    bert_r = bert_r.cpu().numpy()
    bert_f1 = bert_f1.cpu().numpy()

    toxicity = np.mean([tox_model.predict(g)["toxicity"] for g in preds])
    distinct_1 = distinct_n(preds, 1)
    distinct_2 = distinct_n(preds, 2)
    dup_percent = (1 - len(set(preds)) / len(preds)) * 100

    return {
        "Perplexity": "Need LMHead model",  # optional to add
        "BLEU-4": round(bleu4, 4),
        "ROUGE-L": round(np.mean(rouges), 4),
        "BERTScore-P": round(np.mean(bert_p), 4),
        "BERTScore-R": round(np.mean(bert_r), 4),
        "BERTScore-F1": round(np.mean(bert_f1), 4),
        "Exact Match (%)": round(np.mean(ems) * 100, 2),
        "Avg Cosine Similarity": round(np.mean(sims), 4),
        "Toxicity Score": round(toxicity, 4),
        "Avg Length": round(np.mean(lens), 2),
        "Min Length": np.min(lens),
        "Max Length": np.max(lens),
        "Distinct-1": round(distinct_1, 4),
        "Distinct-2": round(distinct_2, 4),
        "% Duplicate Responses": round(dup_percent, 2),
    }


from pprint import pprint

# print("Evaluating LoRA model...")
# result_lora = evaluate_model(
# model_path="./output/lora",
# jsonl_path="./data/data-test.jsonl",
# device="mps",
# n_questions=50,
# )


# pprint(result_lora)

print("Evaluating So-LoRA model...")
result_solora = evaluate_model(
    model_path="./output/solora",
    jsonl_path="./data/data-test.jsonl",
    device="cpu",
    n_questions=50,
)


pprint(result_solora)
