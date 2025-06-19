from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def test_colora_model(
    model_path: str = "./colora_output/dev_support_colora_round1_full",
    prompt: str = "Class hoá đơn là gì?",
    device: str = "mps",
    max_new_tokens: int = 64,
):
    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[INPUT]: {prompt}")
    print(f"[OUTPUT]: {output_text}\n")

    return output_text


test_colora_model(
    model_path="./colora_output/dev_support_colora_round1_full",
    prompt="Làm sao để cấu hình CORS cho ứng dụng web?",
)
