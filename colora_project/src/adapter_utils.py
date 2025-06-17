def save_colora_adapter(model, save_path, task_names):
    os.makedirs(save_path, exist_ok=True)

    lora_state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, COLoRALinear):
            # Save shared LoRA
            lora_state_dict[f"{name}.shared_lora_A"] = module.shared_lora_A.data
            lora_state_dict[f"{name}.shared_lora_B"] = module.shared_lora_B.data

            # Save task-specific LoRAs
            for task in task_names:
                lora_state_dict[f"{name}.task_experts.{task}.lora_A"] = (
                    module.task_experts[task].lora_A.data
                )
                lora_state_dict[f"{name}.task_experts.{task}.lora_B"] = (
                    module.task_experts[task].lora_B.data
                )

            # Save collaboration weight and router
            lora_state_dict[f"{name}.collaboration_weight"] = (
                module.collaboration_weight.data
            )

    torch.save(lora_state_dict, f"{save_path}/cola_olora_weights.pt")

    # Save config
    config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "task_names": task_names,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    }
    with open(f"{save_path}/cola_olora_config.json", "w") as f:
        json.dump(config, f, indent=2)


def convert_to_peft_adapter(
    trained_model, task_name: str, base_model_path: str, save_path: str
):
    print(f"Extracting PEFT adapter for task: {task_name}")

    # Load lại mô hình gốc (chưa LoRA)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    peft_model = get_peft_model(base_model, lora_config)

    for name, module in trained_model.named_modules():
        if isinstance(module, COLoRALinear):
            try:
                peft_layer = dict(peft_model.named_modules())[name]
            except KeyError:
                continue
            A = module.task_experts[task_name].lora_A
            B = module.task_experts[task_name].lora_B
            peft_layer.lora_A.default.weight.data.copy_(A)
            peft_layer.lora_B.default.weight.data.copy_(B)

    peft_model.save_pretrained(save_path)
    print(f"Saved PEFT adapter at {save_path}")


def merge_peft_adapter(base_model_path: str, adapter_path: str, save_path: str):
    base = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(save_path)
    print(f"Merged adapter saved at {save_path}")
