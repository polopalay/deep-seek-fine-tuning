import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training
from fine_tuning_using_colora import apply_cola_orthogonal_lora


def load_model_and_tokenizer(model_dir, device="cpu"):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float32
    )
    model = prepare_model_for_kbit_training(model)

    # Load adapter config
    with open(os.path.join(model_dir, "cola_olora_config.json"), "r") as f:
        adapter_config = json.load(f)

    # Apply adapter architecture
    model = apply_cola_orthogonal_lora(
        model,
        task_names=adapter_config["task_names"],
        target_modules=adapter_config["target_modules"],
        r=adapter_config["r"],
        lora_alpha=adapter_config["lora_alpha"],
        lora_dropout=adapter_config["lora_dropout"],
    )

    # Load LoRA weights
    adapter_weights = torch.load(
        os.path.join(model_dir, "cola_olora_weights.pt"), map_location=device
    )
    model.load_state_dict(adapter_weights, strict=False)

    return model.to(device), tokenizer


def test_task_specific(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt) :].strip()


if __name__ == "__main__":
    model_dir = "./colora_output/dev_support_colora"
    device = "cpu"

    print(f"Loading model on device: {device}")
    model, tokenizer = load_model_and_tokenizer(model_dir, device=device)
    # Test thử với dữ liệu cơ bản
    prompt_support = "[SUPPORT] 1+1 bằng bao nhiêu?"
    answer_support = test_task_specific(model, tokenizer, prompt_support)
    print(f"\n[SUPPORT] → {answer_support}")

    # Test thử với prompt support
    prompt_support = (
        "[SUPPORT] Tôi nhập sai tên khách hàng trên hóa đơn, bây giờ tôi cần làm gì?"
    )
    answer_support = test_task_specific(model, tokenizer, prompt_support)
    print(f"\n[SUPPORT] → {answer_support}")

    # Test thử với prompt dev
    prompt_dev = "[DEV] Làm sao cấu hình endpoint nhận hóa đơn trong .NET Core?"
    answer_dev = test_task_specific(model, tokenizer, prompt_dev)
    print(f"\n[DEV] → {answer_dev}")
