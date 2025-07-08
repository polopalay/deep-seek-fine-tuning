from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os
import json
import math


def alpha_strategy(r):
    c = 8
    return int(round(math.sqrt(r) * c))


def orthogonal_loss_a(A):
    AtA = A @ A.T
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    return torch.sum((AtA - I) ** 2)


def orthogonal_loss_between_a(A_now, A_list_prev):
    compatible = [A_prev for A_prev in A_list_prev if A_prev.shape == A_now.shape]
    if not compatible:
        return torch.tensor(0.0, device=A_now.device, dtype=A_now.dtype)
    A_prev_stack = torch.stack(compatible)
    sim = torch.matmul(A_now, A_prev_stack.transpose(1, 2))
    sim_sq = sim.pow(2).sum()
    return sim_sq


class OrthLoRATrainer(Trainer):
    def __init__(
        self,
        *args,
        lambda_internal=0.01,
        lambda_external=0.01,
        prev_A_list=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_internal = lambda_internal
        self.lambda_external = lambda_external
        self.prev_A_list = prev_A_list if prev_A_list is not None else []

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        main_loss = outputs.loss

        internal_loss = 0.0
        external_loss = 0.0
        ck_orth = ["q_proj", "v_proj", "k_proj"]
        for name, module in model.named_modules():
            if any(key in name for key in ck_orth):
                if hasattr(module, "lora_A"):
                    lora_A = module.lora_A
                    if isinstance(lora_A, torch.nn.ModuleDict):
                        for _, sub_A in lora_A.items():
                            internal_loss += orthogonal_loss_a(sub_A.weight)
                            external_loss += orthogonal_loss_between_a(
                                sub_A.weight, self.prev_A_list
                            )
                    else:
                        internal_loss += orthogonal_loss_a(lora_A.weight)
                        external_loss += orthogonal_loss_between_a(
                            lora_A.weight, self.prev_A_list
                        )

        total_loss = (
            main_loss
            + self.lambda_internal * internal_loss
            + self.lambda_external * external_loss
        )
        return (total_loss, outputs) if return_outputs else total_loss


def training_using_olora(
    data_path="./data/data.jsonl",
    model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    r=8,
    learning_rate=2e-4,
    num_epochs=8,
    batch_size=2,
    tokenizer_len=128,
    warmup_ratio=0.1,
    lambda_internal=0.001,
    lambda_external=1.0,
    output_dir="./lora_output",
    adapter_name="lora_r8",
    device="mps",
):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=data_path)["train"].train_test_split(
        test_size=0.1
    )

    def tokenize(data):
        formatted = tokenizer.apply_chat_template(
            data["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer(
            formatted,
            truncation=True,
            padding=True,
            max_length=tokenizer_len,
        )

    tokenized = dataset.map(tokenize, remove_columns=["messages"])

    model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha_strategy(r),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config).to(device)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{adapter_name}",
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=100,
        warmup_ratio=warmup_ratio,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        report_to="none",
        save_strategy="no",
        fp16=False,
    )

    prev_A_list = []
    for name, module in model.named_modules():
        if any(key in name for key in ["q_proj", "v_proj", "k_proj"]):
            if hasattr(module, "weight"):
                prev_A_list.append(module.weight.detach().to(device))

    trainer = OrthLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        lambda_internal=lambda_internal,
        lambda_external=lambda_external,
        prev_A_list=prev_A_list,
    )

    trainer.train()

    model.save_pretrained(f"{output_dir}/{adapter_name}")
    tokenizer.save_pretrained(f"{output_dir}/{adapter_name}")


if __name__ == "__main__":
    training_using_olora(
        data_path="data/data.jsonl",
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        r=16,
        learning_rate=5e-4,
        num_epochs=24,
        batch_size=2,
        tokenizer_len=108,
        warmup_ratio=0.1,
        lambda_internal=0.001,
        lambda_external=1.0,
        output_dir="output",
        adapter_name="olora",
        device="mps",
    )
