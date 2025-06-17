from transformers import TrainingArguments
from src.colora_modules import CoLAOLoRATrainer, COLORADataCollator


def train_model(model, tokenizer, dataset, config):
    args = TrainingArguments(**config["training_args"])
    trainer = CoLAOLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=args,
        data_collator=COLORADataCollator(tokenizer),
        lambda_orth=config["lambda_orth"],
        lambda_collab=config["lambda_collab"],
        orthogonalize_freq=config["orthogonalize_freq"],
    )
    trainer.train()
    return model
