from transformers import AutoModelForCausalLM, AutoTokenizer
from src.colora_modules import apply_cola_orthogonal_lora


def build_model(base_model, task_names, device):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", trust_remote_code=True
    )
    model = model.to(device)
    model = apply_cola_orthogonal_lora(model, task_names=task_names)
    return model, tokenizer
