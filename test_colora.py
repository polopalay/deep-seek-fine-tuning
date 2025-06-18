import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from colora_train_chain_apply import apply_cola_orthogonal_lora


def test_cola_olora(
    test_jsonl_path: str,
    adapter_path: str,
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_len: int = 256,
    device: str = "mps",
    num_samples: int = 5,
):
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float32
    )
    model = model.to(device)

    # Load adapter weights
    adapter_weights = torch.load(
        f"{adapter_path}/cola_olora_weights.pt", map_location=device
    )
    config_path = f"{adapter_path}/cola_olora_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Apply CoLA + O-LoRA
    model = apply_cola_orthogonal_lora(
        model,
        task_names=config["task_names"],
        target_modules=config["target_modules"],
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )
    model = model.to(device)
    model.load_state_dict(adapter_weights, strict=False)
    model.eval()

    # Load test samples
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    samples = raw_data[:num_samples]

    for i, sample in enumerate(samples):
        instruction = sample["instruction"]
        task_id = sample.get("task", "default")
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
        print(f"[Sample {i+1}] - Task ID: {task_id}")
        print(f"Instruction:\n{instruction}")
        print(f"Expected Output:\n{expected_output}")
        print(f"Generated Output:\n{generated_output}")


if __name__ == "__main__":
    test_cola_olora(
        test_jsonl_path="./data/test_data.jsonl",
        adapter_path="./colora_output/dev_support_colora/",
        num_samples=5,
    )
