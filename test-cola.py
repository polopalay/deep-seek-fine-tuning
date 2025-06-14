from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def test_model(
    prompt: str,
    base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    adapter_path="deepseek_cola_invoice_adapter",
    max_new_tokens=100,
):
    # Load tokenizer và base model
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Format prompt giống lúc huấn luyện
    full_prompt = f"### Câu hỏi:\n{prompt}\n\n### Trả lời:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # Sinh câu trả lời
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("📌 Kết quả:")
    print(output_text.replace(full_prompt, "").strip())


test_model("Một cộng một bằng mấy?")
test_model("Bạn là ai?")
test_model("Khách hỏi cách tìm hóa đơn theo mã tra cứu")
