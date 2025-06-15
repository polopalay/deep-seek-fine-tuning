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
        task_id = prompt.get("task_id", None)

        full_input = f"{input_text}\n"
        inputs = tokenizer(full_input, return_tensors="pt").to(device)

        with torch.no_grad():
            if task_id and "task_id" in model.forward.__code__.co_varnames:
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=max_new_tokens,
                    task_id=task_id,
                )
            else:
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
        {"instruction": "Làm sao để cấu hình HTTPS cho máy chủ?", "task_id": "dev"},
        {"instruction": "Cần hỗ trợ ký số hóa đơn như thế nào?", "task_id": "support"},
        {"instruction": "Cách xử lý lỗi khi server trả về 500?", "task_id": "dev"},
        {"instruction": "Làm sao tôi liên hệ bộ phận kỹ thuật?", "task_id": "support"},
    ]

    test_model_response(model, tokenizer, test_prompts, DEVICE)
