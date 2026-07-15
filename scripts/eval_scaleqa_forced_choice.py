#!/usr/bin/env python3
"""Evaluate Qwen2.5-VL on scale forced-choice examples.

The script supports the existing synthetic scale dataset and the Scale-QA stress
CSV/JSONL format.  It asks the model to choose between candidate A and B and
scores whether the first recognizable choice matches the gold label.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapters", nargs="+", default=["base=NONE"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    return parser.parse_args()


def read_rows(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
    else:
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                if max_samples is not None and len(rows) >= max_samples:
                    break
    return rows


def get_field(row: dict[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return str(row[name])
    return default


def make_prompt(row: dict[str, Any]) -> str:
    if row.get("forced_choice_prompt"):
        return str(row["forced_choice_prompt"])
    question = get_field(row, "question", "query")
    a = get_field(row, "candidate_a")
    b = get_field(row, "candidate_b")
    return (
        f"{question}\n"
        "Choose the answer that matches the scale used by the chart and the question.\n"
        f"A. {a}\n"
        f"B. {b}\n"
        "Answer with only A or B."
    )


def resolve_image(row: dict[str, Any], image_root: Path) -> Image.Image:
    image_path = get_field(row, "image_path", "image")
    if image_path.startswith("hf://"):
        raise ValueError(f"Image path is a dataset reference, not a local file: {image_path}")
    path = Path(image_path)
    if not path.is_absolute():
        path = image_root / path
    return Image.open(path).convert("RGB")


def parse_choice(text: str) -> str:
    t = text.strip().upper()
    m = re.search(r"\b([AB])\b", t)
    if m:
        return m.group(1)
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    return ""


def generate_choice(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def summarize(rows: list[dict[str, Any]], output: Path) -> None:
    buckets: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        adapter = row["adapter"]
        split = row.get("pair_direction") or row.get("scale_type") or "all"
        for key in [(adapter, "all"), (adapter, split)]:
            buckets.setdefault(key, []).append(int(row["score"]))

    summary_rows = []
    for (adapter, split), scores in sorted(buckets.items()):
        summary_rows.append(
            {
                "adapter": adapter,
                "split": split,
                "n": len(scores),
                "accuracy": sum(scores) / len(scores) if scores else 0.0,
            }
        )

    summary_path = output.with_suffix(output.suffix + ".summary.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["adapter", "split", "n", "accuracy"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary: {summary_path}")
    for row in summary_rows:
        if row["split"] == "all":
            print(row)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.input), args.max_samples)

    processor_kwargs: dict[str, Any] = {"local_files_only": True}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, **processor_kwargs)
    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    specs = []
    for item in args.adapters:
        name, path = item.split("=", 1)
        specs.append((name, path))

    out_rows: list[dict[str, Any]] = []
    image_root = Path(args.image_root)

    peft_model = None
    loaded_adapters = set()
    for adapter_name, adapter_path in specs:
        print(f"=== Evaluating {adapter_name} ===")
        if adapter_path == "NONE":
            active_model = model
        else:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
                loaded_adapters.add(adapter_name)
            elif adapter_name not in loaded_adapters:
                peft_model.load_adapter(adapter_path, adapter_name=adapter_name)
                loaded_adapters.add(adapter_name)
            peft_model.set_adapter(adapter_name)
            peft_model.eval()
            active_model = peft_model

        for i, row in enumerate(rows):
            image = resolve_image(row, image_root)
            prompt = make_prompt(row)
            prediction = generate_choice(active_model, processor, image, prompt, args.max_new_tokens)
            choice = parse_choice(prediction)
            label = get_field(row, "label").upper()
            score = int(choice == label)
            out_row = {
                "adapter": adapter_name,
                "row_idx": i,
                "source_idx": get_field(row, "source_idx", "idx", default=str(i)),
                "question": get_field(row, "question", "query"),
                "candidate_a": get_field(row, "candidate_a"),
                "candidate_b": get_field(row, "candidate_b"),
                "label": label,
                "prediction": prediction,
                "choice": choice,
                "score": score,
                "pair_direction": get_field(row, "pair_direction"),
                "scale_type": get_field(row, "scale_type"),
                "question_type": get_field(row, "question_type"),
            }
            out_rows.append(out_row)
            if (i + 1) % 50 == 0:
                partial = [int(r["score"]) for r in out_rows if r["adapter"] == adapter_name]
                print(adapter_name, i + 1, "acc", sum(partial) / len(partial))

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved detailed: {output}")
    summarize(out_rows, output)


if __name__ == "__main__":
    main()
