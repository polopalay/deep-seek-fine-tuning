from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_base = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = "./colora_output/colora_r16"

tokenizer = AutoTokenizer.from_pretrained(model_base)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.to(device)

chat = [{"role": "user", "content": "Thay thế hóa đơn có cần báo cáo lại không?"}]

input_ids = tokenizer.apply_chat_template(
    chat, add_generation_prompt=True, return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        generation_config=GenerationConfig.from_pretrained(model_base),
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
