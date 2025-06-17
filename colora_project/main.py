from src.data_loader import load_and_format_data
from src.model_builder import build_model
from src.training import train_model
from src.adapter_utils import (
    save_colora_adapter,
    convert_to_peft_adapter,
    merge_peft_adapter,
)
import yaml


def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset, task_names = load_and_format_data(cfg["jsonl_path"])
    model, tokenizer = build_model(cfg["base_model"], task_names, cfg["device"])
    model = train_model(model, tokenizer, dataset, cfg)
    save_colora_adapter(model, f"{cfg['output_dir']}/{cfg['adapter_name']}", task_names)
    convert_to_peft_adapter(
        model, "support", cfg["base_model"], "./peft_adapter_support"
    )
    merge_peft_adapter(
        cfg["base_model"], "./peft_adapter_support", "./merged_support_model"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
