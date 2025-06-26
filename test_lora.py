from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch

def test_lora_model(
    adapter_path: str = "./colora_output/colora_r16",
    model_base: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    prompt: str = "Làm sao biết công ty chưa đăng ký chứng thư số khi gọi GetCertInfo?",
    device: str = "mps"
):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()

    messages = [
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n🧪 Output:")
    print(output_text.split(prompt)[-1].strip())


if __name__ == "__main__":
    test_lora_model()
