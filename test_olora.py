from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


def generate_answer(
    model_path: str,
    base_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    instruction: str = "",
    max_new_tokens: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, model_path).to(device)

    model.eval()
    prompt = f"### Câu hỏi:\n{instruction}\n\n### Trả lời:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n==== KẾT QUẢ SINH ====\n")
    print(decoded)
    print("\n=======================")


if __name__ == "__main__":
    model_path = "./colora_output/deepseek_lora_adapter"
    inruction3 = "Tôi bị lỗi ERR:1?"
    generate_answer(model_path=model_path, instruction=inruction3)
