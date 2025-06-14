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
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"🧠 Loading model on device: {device}")
    model, tokenizer = load_model_and_tokenizer(model_dir, device=device)

    prompts = {
        "🔧 DEV": "### Câu hỏi:\n[DEV] Hàm ImportAndPublishInv trả về lỗi ERR:99 nghĩa là gì?\n\n### Trả lời:",
        "📞 SUPPORT": "### Câu hỏi:\n[SUPPORT] Làm sao để reset mật khẩu?\n\n### Trả lời:",
    }

    for tag, prompt in prompts.items():
        response = test_task_specific(model, tokenizer, prompt)
        print(f"{tag} Response: {response}\n")
