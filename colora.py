from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch
import os
import warnings
import math
import gc
import json

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
warnings.filterwarnings("ignore")


def alpha_strategy(r):
    c = 8
    return int(round(math.sqrt(r) * c))


def orthogonal_loss_a(A):
    AtA = A @ A.T
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    return torch.sum((AtA - I) ** 2)


def orthogonal_loss_between_a(A_now, A_list_prev):
    if not A_list_prev:
        return torch.tensor(0.0, device=A_now.device, dtype=A_now.dtype)

    compatible = [A_prev for A_prev in A_list_prev if A_prev.shape == A_now.shape]

    if not compatible:
        return torch.tensor(0.0, device=A_now.device, dtype=A_now.dtype)

    compatible = [
        A_prev.to(device=A_now.device, dtype=A_now.dtype) for A_prev in compatible
    ]

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
        ck_orth = [
            "q_proj",
            "v_proj",
        ]
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


def load_lora_A_matrices(adapter_paths, device):
    matrices = []
    for path in adapter_paths:
        adapter = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16),
            path,
        )
        for name, module in adapter.named_modules():
            if any(key in name for key in ["q_proj", "v_proj"]):
                if hasattr(module, "lora_A"):
                    lora_A = module.lora_A
                    if isinstance(lora_A, torch.nn.ModuleDict):
                        for _, sub_A in lora_A.items():
                            matrices.append(sub_A.weight.to(device))
                    else:
                        matrices.append(lora_A.weight.to(device))


def training_using_cola(
    dataset_path="./data/data_1000.jsonl",
    model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    r_list=[16, 8, 4],
    lambdas_internal=[0.5, 0.0, 0.0],
    lambdas_external=[0.0, 0.5, 0.1],
    epoch_list=[3, 5, 7],
    batch_size=2,
    tokenizer_len=32,
    warmup_ratio=0.03,
    learning_rates=[2e-5, 1e-5, 5e-6],
    device="mps",
    output_dir="./colora_output",
    base_adapter_name="colora",
):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    dataset = Dataset.from_list(raw_data).train_test_split(test_size=0.1)

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

    adapter_names = []

    for round_idx, r in enumerate(r_list):
        torch.mps.empty_cache()
        print(f"\n=== Vòng {round_idx + 1} | r = {r} ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_base, torch_dtype=torch.float16
        )
        alpha = alpha_strategy(r)
        lambda_internal = lambdas_internal[round_idx]
        lambda_external = lambdas_external[round_idx]
        learning_rate = learning_rates[round_idx]
        num_epochs = epoch_list[round_idx]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                # "up_proj",
                # "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config).to(device)
        for adapter_name in adapter_names:
            model.load_adapter(
                f"{output_dir}/{adapter_name}",
                adapter_name=adapter_name,
                is_trainable=False,
            )

        adapter_name = f"{base_adapter_name}_r{r}"
        adapter_names.append(adapter_name)

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
            # max_steps=1,
        )

        prev_A_list = []
        if round_idx > 0:
            prev_adapter_paths = [f"{output_dir}/{name}" for name in adapter_names[:-1]]
            prev_A_list = load_lora_A_matrices(prev_adapter_paths, device=device)

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
        gc.disable()
        trainer.train()
        gc.enable()

        if round_idx == len(r_list) - 1:
            model = model.merge_and_unload()
            merged_ckpt_dir = f"{output_dir}/colora_final"
            os.makedirs(merged_ckpt_dir, exist_ok=True)
            model.save_pretrained(merged_ckpt_dir)
            tokenizer.save_pretrained(merged_ckpt_dir)
        else:
            model.save_pretrained(f"{output_dir}/{adapter_name}")

        del trainer
        del prev_A_list
        del model
        # del tokenizer
        gc.collect()
        print(
            f"[DEBUG] Memory allocated: {torch.mps.current_allocated_memory() // 1024**2} MB"
        )
        torch.mps.empty_cache()


if __name__ == "__main__":
    training_using_cola(
        dataset_path="data/data.jsonl",
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        r_list=[8, 6, 4],
        lambdas_internal=[0.01, 0.001, 0.0],
        lambdas_external=[0.0, 0.01, 0.001],
        learning_rates=[1e-4, 5e-5, 2e-5],
        epoch_list=[3, 5, 7],
        batch_size=2,
        tokenizer_len=128,
        warmup_ratio=0.1,
        device="mps",
        output_dir="colora_output",
        base_adapter_name="colora",
    )
