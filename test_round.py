import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

from colora_train_chain_apply_soft_only_a import (
    apply_cola_orthogonal_lora,
)


def test_colora_model(
    test_jsonl_path: str,
    adapter_path: str,
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_len: int = 256,
    device: str = "mps",
    num_samples: int = 5,
):
    # Load config
    config_path = f"{adapter_path}/cola_olora_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    task_names = config["task_names"]
    r = config.get("r", 8)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.05)
    target_modules = config.get("target_modules")

    # Load tokenizer và model gốc
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)

    # Apply COLoRA modules
    model = apply_cola_orthogonal_lora(
        model,
        task_names=task_names,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model = model.to(device)

    # Load adapter weights
    print("Loading adapter weights...")
    state_dict = torch.load(
        f"{adapter_path}/cola_olora_weights.pt", map_location=device
    )

    for name, param in model.named_parameters():
        if name in state_dict:
            with torch.no_grad():
                param.copy_(state_dict[name].to(param.device))

    model.eval()

    # Load test samples
    print("Loading test samples...")
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]

    samples = data[:num_samples]

    for i, sample in enumerate(samples):
        prompt = sample["instruction"]
        task_id = sample.get("task", "default")
        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        ).input_ids.to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                # attention_mask=attention_mask,
                max_new_tokens=16,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[{i+1}] Task: {task_id}")
        print(f"Instruction: {prompt}")
        print(f"Generated: {decoded}\n")


# Example usage
if __name__ == "__main__":
    test_colora_model(
        test_jsonl_path="./data/test_data.jsonl",
        adapter_path="./colora_output/dev_support_colora_round1",
        device="mps",
        num_samples=6,
        max_seq_len=16,
    )
