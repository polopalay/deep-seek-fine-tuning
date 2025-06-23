from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import DataCollatorForLanguageModeling
import torch
import os
import warnings
import math
import gc

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
warnings.filterwarnings("ignore")


def alpha_strategy(r):
    c = 8
    return int(round(math.sqrt(r) * c))


def orthogonal_loss_a(A):
    AtA = torch.einsum("ik,jk->ij", A, A)
    I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    return torch.sum((AtA - I) ** 2).float()


def orthogonal_loss_between_a(A_now, A_list_prev):
    loss = 0.0
    for A_prev in A_list_prev:
        sim = torch.matmul(A_now, A_prev.T)
        loss += torch.sum(sim**2).float()
    return loss


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
        ck_orth = ["q_proj", "v_proj", "k_proj", "gate_proj"]
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
            if any(key in name for key in ["q_proj", "v_proj", "k_proj", "gate_proj"]):
                if hasattr(module, "lora_A"):
                    lora_A = module.lora_A
                    if isinstance(lora_A, torch.nn.ModuleDict):
                        for _, sub_A in lora_A.items():
                            matrices.append(sub_A.weight.to(device))
                    else:
                        matrices.append(lora_A.weight.to(device))
    return matrices


def training_using_cola(
    dataset_path="./data/data_1000.jsonl",
    model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    r_list=[16, 8, 4],
    lambdas_internal=[0.5, 0.0, 0.0],
    lambdas_external=[0.0, 0.5, 0.1],
    batch_size=2,
    num_epochs=5,
    tokenizer_len=32,
    warmup_ratio=0.03,
    learning_rate=5e-5,
    device="mps",
    output_dir="./colora_output",
    base_adapter_name="colora",
):
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    )

    dataset = load_dataset("json", data_files=dataset_path)["train"].train_test_split(
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
            padding="max_length",
            max_length=tokenizer_len,
        )

    tokenized = dataset.map(tokenize, remove_columns=["messages"])

    adapter_names = []

    for round_idx, r in enumerate(r_list):
        print(f"\n=== Vòng {round_idx + 1} | r = {r} ===")
        if os.path.exists(f"{output_dir}/{base_adapter_name}_r{r}"):
            print(f"Đã có adapter {base_adapter_name}_r{r}, bỏ qua vòng này.")
            continue
        torch.mps.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_base, torch_dtype=torch.float16
        )
        alpha = alpha_strategy(r)
        lambda_internal = lambdas_internal[round_idx]
        lambda_external = lambdas_external[round_idx]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model = model.to(device)
        # model = model.to(device).to(torch.float16)
        for adapter_name in adapter_names:
            model.load_adapter(
                f"{output_dir}/{adapter_name}",
                adapter_name=adapter_name,
                is_trainable=False,
            )

        adapter_name = f"{base_adapter_name}_r{r}"
        # model.add_adapter(adapter_name, lora_config)
        adapter_names.append(adapter_name)

        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{adapter_name}",
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            logging_steps=100,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            report_to="none",
            save_strategy="no",
            load_best_model_at_end=False,
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

        trainer.train()
        if round_idx == len(r_list) - 1:
            model = model.merge_and_unload()
            merged_ckpt_dir = f"{output_dir}/merged_model_final"
            os.makedirs(merged_ckpt_dir, exist_ok=True)
            model.save_pretrained(merged_ckpt_dir)
            tokenizer.save_pretrained(merged_ckpt_dir)
        else:
            model.save_pretrained(f"{output_dir}/{adapter_name}")

        del model
        gc.collect()
        torch.mps.empty_cache()


if __name__ == "__main__":
    training_using_cola(
        dataset_path="./data/data.jsonl",
        model_base="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        r_list=[16, 8, 4],
        lambdas_internal=[0.2, 0.1, 0.0],
        lambdas_external=[0.0, 0.1, 0.2],
        batch_size=2,
        num_epochs=3,
        tokenizer_len=128,
        warmup_ratio=0.04,
        learning_rate=2e-5,
        # learning_rate=5e-5,
        device="mps",
        output_dir="./colora_output",
        base_adapter_name="colora",
    )
