"""Train a DrugBank Stage-II LoRA from one fixed triplet JSONL for 20 epochs."""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from llm2vec import LLM2Vec
from llm2vec.loss.HardNegativeNLLLoss import HardNegativeNLLLoss
from peft import LoraConfig, get_peft_model
from torch.utils.data import SequentialSampler
from transformers import Trainer, TrainingArguments, set_seed

from embed import MAX_LENGTH, SEPARATOR

PROMPT = "Given the description of one drug, retrieve the more similar description of the same drug.; "
GLOBAL_BATCH = 64


def training_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if any(not isinstance(row.get(key), str) or not row[key].strip()
           for row in rows for key in ("query", "positive", "negative")):
        raise ValueError("Every triplet must contain query, positive and negative text")
    order = list(range(len(rows)))
    rng = random.Random(42)
    rng.shuffle(order)
    batches = [order[start:start + GLOBAL_BATCH] for start in range(0, len(order), GLOBAL_BATCH)
               if start + GLOBAL_BATCH <= len(order)]
    rng.shuffle(batches)
    if not batches:
        raise ValueError(f"Need at least {GLOBAL_BATCH} triplets")
    return [rows[index] for batch in batches for index in batch]


class TripletTrainer(Trainer):
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        query, positive, negative = [model(features) for features in inputs["triplets"]]
        loss = HardNegativeNLLLoss(scale=50)(query, positive, negative)
        return (loss, {"embeddings": query}) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        self.accelerator.unwrap_model(self.model).save(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default="z-cao/DrugSpace-mntp-eval-8B")
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        parser.error("Run with torchrun --nproc_per_node=2 to keep global batch size 64")
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error("Use a new, empty output directory")
    set_seed(42)
    dataset = training_rows(args.triplets)
    model = LLM2Vec.from_pretrained(
        args.base_model, enable_bidirectional=True, pooling_mode="mean",
        max_length=MAX_LENGTH, doc_max_length=MAX_LENGTH, skip_instruction=True,
        torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    )
    model.model = get_peft_model(model.model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))

    def collate(rows):
        return {"triplets": [model.tokenize([
            (PROMPT if key == "query" else "") + SEPARATOR + row[key] for row in rows
        ]) for key in ("query", "positive", "negative")]}

    training_args = TrainingArguments(
        output_dir=args.output, per_device_train_batch_size=32, num_train_epochs=20,
        learning_rate=1e-4, warmup_steps=300, gradient_accumulation_steps=1,
        seed=42, remove_unused_columns=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, save_strategy="no", logging_steps=50,
        report_to="none", optim="adamw_torch", lr_scheduler_type="linear",
        weight_decay=0.0, max_grad_norm=1.0,
    )
    trainer = TripletTrainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate)
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        print(f"Saved final adapter after {trainer.state.global_step} steps to {args.output}")


if __name__ == "__main__":
    main()
