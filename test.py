from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch


def test_model(
    base_model_path, lora_ckpt_path, instruction, input_text="", max_new_tokens=128
):
    device = "cpu"

    # Load tokenizer và mô hình gốc
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32
    )
    base_model = base_model.to(device)

    # Nạp LoRA đã fine-tune
    model = PeftModel.from_pretrained(base_model, lora_ckpt_path)
    model = model.to(device)
    model.eval()

    # Tạo prompt
    prompt = f"<|user|>\n{instruction}\n{input_text}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Sinh output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Cắt phần prompt khỏi kết quả trả lời
    return result.replace(prompt, "").strip()


output = test_model(
    base_model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    lora_ckpt_path="./checkpoints/deepseek-lora-cpu",  # đường dẫn bạn đã huấn luyện
    instruction="Tạo báo cáo tổng hợp công việc",
    input_text="Hôm nay có 3 nhân viên nghỉ việc và tồn kho giảm",
)

print("💡 Output:", output)
