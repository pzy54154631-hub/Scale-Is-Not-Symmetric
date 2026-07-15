#!/usr/bin/env python3
"""Free-form evaluation on synthetic scale chart examples.

Unlike the forced-choice script, this asks the original chart question and lets
the model generate a short answer.  The evaluator then checks whether the answer
matches the gold value, the deliberately scaled alternative, or neither.  This
is useful for mining naturally expressed scale inconsistencies from a larger
synthetic stress pool.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/scale_synth/scale_synth.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapters", nargs="+", default=["base=NONE"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--prompt-mode", choices=["plain", "scale"], default="plain")
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    return parser.parse_args()


def read_rows(path: Path, max_samples: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def extract_number(text: Any) -> float | None:
    s = str(text).replace(",", "").replace("−", "-").strip()
    nums = NUM_RE.findall(s)
    if len(nums) != 1:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


def relaxed_numeric_match(pred: Any, target: Any) -> bool:
    p = extract_number(pred)
    t = extract_number(target)
    if p is None or t is None:
        return str(pred).strip().lower() == str(target).strip().lower()
    tol = max(0.05 * abs(t), 1e-3)
    return abs(p - t) <= tol


def scaled_alternative(row: dict[str, str]) -> str:
    label = row["label"].upper()
    return row["candidate_b"] if label == "A" else row["candidate_a"]


def classify_prediction(pred: str, row: dict[str, str]) -> str:
    if relaxed_numeric_match(pred, row["answer"]):
        return "gold_match"
    if relaxed_numeric_match(pred, scaled_alternative(row)):
        return "scaled_negative_match"
    return "other"


def make_prompt(row: dict[str, str], prompt_mode: str) -> str:
    if prompt_mode == "scale":
        return (
            row["question"]
            + " Answer briefly with the final value only. Preserve the chart's numeric scale: "
            + "if the chart uses decimal rates such as 0.65, answer as 0.65 rather than 65 or 65%; "
            + "if the chart uses percentages, answer in percentage units."
        )
    return row["question"] + " Answer briefly with the final value only."


def resolve_image(row: dict[str, str], image_root: Path) -> Image.Image:
    path = Path(row["image_path"])
    if not path.is_absolute():
        path = image_root / path
    return Image.open(path).convert("RGB")


def generate_answer(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
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


def summarize(out_rows: list[dict[str, Any]], output: Path) -> None:
    buckets: dict[tuple[str, str], dict[str, int]] = {}
    for row in out_rows:
        adapter = row["adapter"]
        splits = ["all", row["scale_type"], row["question_type"]]
        for split in splits:
            key = (adapter, split)
            buckets.setdefault(key, {"n": 0, "gold_match": 0, "scaled_negative_match": 0, "other": 0})
            buckets[key]["n"] += 1
            buckets[key][row["outcome"]] += 1

    summary_rows = []
    for (adapter, split), vals in sorted(buckets.items()):
        n = vals["n"]
        summary_rows.append(
            {
                "adapter": adapter,
                "split": split,
                "n": n,
                "gold_match_rate": vals["gold_match"] / n if n else 0.0,
                "scaled_negative_match_rate": vals["scaled_negative_match"] / n if n else 0.0,
                "other_rate": vals["other"] / n if n else 0.0,
                **vals,
            }
        )

    summary_path = output.with_suffix(output.suffix + ".summary.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
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
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    base_model.eval()

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
            active_model = base_model
        else:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base_model, adapter_path, adapter_name=adapter_name)
                loaded_adapters.add(adapter_name)
            elif adapter_name not in loaded_adapters:
                peft_model.load_adapter(adapter_path, adapter_name=adapter_name)
                loaded_adapters.add(adapter_name)
            peft_model.set_adapter(adapter_name)
            peft_model.eval()
            active_model = peft_model

        for i, row in enumerate(rows):
            image = resolve_image(row, image_root)
            pred = generate_answer(active_model, processor, image, make_prompt(row, args.prompt_mode), args.max_new_tokens)
            outcome = classify_prediction(pred, row)
            out_rows.append(
                {
                    "adapter": adapter_name,
                    "idx": row["idx"],
                    "image_path": row["image_path"],
                    "chart_type": row["chart_type"],
                    "scale_type": row["scale_type"],
                    "question_type": row["question_type"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "scaled_negative": scaled_alternative(row),
                    "prediction": pred,
                    "outcome": outcome,
                    "score": int(outcome == "gold_match"),
                    "prompt_mode": args.prompt_mode,
                }
            )
            if (i + 1) % 50 == 0:
                local = [r for r in out_rows if r["adapter"] == adapter_name]
                gold = sum(1 for r in local if r["outcome"] == "gold_match") / len(local)
                scaled = sum(1 for r in local if r["outcome"] == "scaled_negative_match") / len(local)
                print(adapter_name, i + 1, "gold", gold, "scaled_neg", scaled)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved detailed: {output}")
    summarize(out_rows, output)


if __name__ == "__main__":
    main()
