import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./colora_output/round_2/final_model"
DEVICE = "cpu"


def load_model(model_path: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def test_model_response(model, tokenizer, prompts, device, max_new_tokens: int = 64):
    for prompt in prompts:
        input_text = prompt["instruction"]

        full_input = f"{input_text}\n"
        inputs = tokenizer(full_input, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=max_new_tokens,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded[len(full_input) :].strip()

        print(f">>> Prompt: {input_text}")
        print(f">>> Response: {response}")


if __name__ == "__main__":
    model, tokenizer = load_model(MODEL_PATH, DEVICE)

    test_prompts = [
        {"instruction": "ERR:1", "task_id": "dev"},
        {"instruction": "Làm sao để hủy hóa đơn?", "task_id": "support"},
        {"instruction": "1+1=", "task_id": "support"},
    ]

    test_model_response(model, tokenizer, test_prompts, DEVICE)
