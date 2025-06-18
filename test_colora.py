import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_cola_olora(
    test_jsonl_path: str,
    model_path: str,
    max_seq_len: int = 256,
    device: str = "mps",
    num_samples: int = 5,
):
    # Load tokenizer và mô hình đã merge adapter
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    # Load test samples
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    samples = raw_data[:num_samples]

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        expected_output = sample.get("output", "")
        prompt = f"{instruction}\n"

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                do_sample=True,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_output = (
            generated_text[len(prompt) :].split("<|endoftext|>")[0].strip()
        )

        print("=" * 50)
        print(f"[Sample {i+1}]")
        print(f"Instruction:\n{instruction}")
        print(f"Expected Output:\n{expected_output}")
        print(f"Generated Output:\n{generated_output}")


if __name__ == "__main__":
    test_cola_olora(
        test_jsonl_path="./data/test_data.jsonl",
        model_path="./colora_output/dev_support_colora_round2_merged",
        num_samples=5,
    )
