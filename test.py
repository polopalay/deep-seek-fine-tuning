from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch


def test_model(model_path: str, instruction: str, input_text: str = ""):
    # Load tokenizer và mô hình đã fine-tune
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    ).to("cpu")

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Ghép prompt đúng định dạng huấn luyện
    prompt = instruction
    if input_text:
        prompt += "\n" + input_text
    prompt += "\n"

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Sinh kết quả
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Loại bỏ prompt khỏi phần output
    generated = output_text[len(prompt) :].strip()
    print("📌 Kết quả:")
    print(generated)


test_model(
    model_path="./deepseek_lora_invoice_cpu/checkpoint-7500",
    instruction="Muốn biết quy trình xuất hóa đơn theo NĐ 123",
    input_text="",
)
