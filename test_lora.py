from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


def test_model(
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    adapter_path: str = "my_falcon_lora_adapter",
    prompt: str = "Lỗi ERR:1 là gì?",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    formatted_prompt = f"### Câu hỏi:\n{prompt}\n\n### Trả lời:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n📌 **Kết quả mô hình:**\n")
    print(output_text[len(formatted_prompt) :].strip())


if __name__ == "__main__":
    test_model(
        base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        adapter_path="colora_output/deepseek_lora_adapter",
        prompt="Class của hoá đơn tên là gì?",
        max_new_tokens=100,
    )
