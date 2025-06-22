import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def test_lora_model(
    prompts,
    model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    adapter_path="./colora_output/lora_adapter",
    max_new_tokens=64,
    device="mps",
):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(model_base)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model = model.to(device)

    results = []
    for prompt in prompts:
        input_text = f"### Câu hỏi:\n{prompt}\n\n### Trả lời:\n"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode output, bỏ phần prompt đầu
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("### Trả lời:\n")[-1].strip()
        results.append(answer)
    return results


if __name__ == "__main__":
    prompts = [
        "Hóa đơn điện tử là gì?",
        "Làm thế nào để đăng ký tài khoản VNPT Invoice?",
        "Lỗi ERR:21 là gì?",
        "Lỗi ERR:1 là gì?",
        "1+1 là bao nhiêu?",
    ]
    answers = test_lora_model(
        prompts,
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        adapter_path="./colora_output/lora_adapter/",
    )
    for q, a in zip(prompts, answers):
        print(f"Câu hỏi: {q}\nTrả lời: {a}\n")
