#!/usr/bin/env python
"""Generic LoRA training entry point for MLLMs.

This is intentionally model-agnostic. For Qwen2.5-VL, InternVL, or LLaVA, you may need to
adapt the processor/collator according to the exact Hugging Face model implementation.

Recommended practical use:
1. Use this file as the template for text-only fields and LoRA setup.
2. Replace `build_messages` and `DataCollatorForVisionLanguage` with your model's official
   fine-tuning recipe if necessary.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from PIL import Image


class JsonlVLDataset(Dataset):
    """JSONL schema:

    {"image": "/path/to/image.jpg", "question": "...", "answer": "..."}

    For multiple-choice tasks, put the final answer string in `answer` and include choices in question.
    """

    def __init__(self, jsonl_path: str | Path):
        self.rows = []
        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        return {
            "image": image,
            "question": row["question"],
            "answer": row["answer"],
        }


def build_messages(question: str, answer: str) -> str:
    # Simple supervised format. Replace with official chat template if needed.
    return f"USER: <image>\n{question}\nASSISTANT: {answer}"


@dataclass
class DataCollatorForVisionLanguage:
    processor: Any
    max_length: int = 2048

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [build_messages(f["question"], f["answer"]) for f in features]
        images = [f["image"] for f in features]
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = JsonlVLDataset(args.train_jsonl)
    collator = DataCollatorForVisionLanguage(processor=processor, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=not args.bf16,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
