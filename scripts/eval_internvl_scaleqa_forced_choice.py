#!/usr/bin/env python3
"""Evaluate InternVL-style chat models on scale forced-choice examples."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/scale_synth/scale_synth.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/InternVL2_5-1B")
    parser.add_argument("--model-name", default="internvl25_1b")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-num-tiles", type=int, default=4)
    return parser.parse_args()


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=4, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file: str, image_root: Path, input_size=448, max_num=4):
    path = Path(image_file)
    if not path.is_absolute():
        path = image_root / path
    image = Image.open(path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    return torch.stack([transform(img) for img in images])


def read_rows(path: Path, max_samples: int | None) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def make_prompt(row: dict[str, str]) -> str:
    if row.get("forced_choice_prompt"):
        return "<image>\n" + row["forced_choice_prompt"]
    return (
        "<image>\n"
        f"{row['question']}\n"
        "Choose the answer that matches the scale used by the chart and the question.\n"
        f"A. {row['candidate_a']}\n"
        f"B. {row['candidate_b']}\n"
        "Answer with only A or B."
    )


def parse_choice(text: str) -> str:
    t = text.strip().upper()
    match = re.search(r"\b([AB])\b", t)
    if match:
        return match.group(1)
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    return ""


def summarize(out_rows: list[dict[str, Any]], output: Path) -> None:
    buckets: dict[str, list[int]] = {}
    for row in out_rows:
        for split in ["all", row.get("pair_direction", ""), row.get("scale_type", ""), row.get("question_type", "")]:
            if not split:
                continue
            buckets.setdefault(split, []).append(int(row["score"]))

    rows = []
    for split, scores in sorted(buckets.items()):
        rows.append(
            {
                "model": out_rows[0]["model"] if out_rows else "",
                "split": split,
                "n": len(scores),
                "accuracy": sum(scores) / len(scores) if scores else 0.0,
            }
        )
    summary_path = output.with_suffix(output.suffix + ".summary.csv")
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary: {summary_path}")
    for row in rows:
        if row["split"] == "all":
            print(row)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.input), args.max_samples)
    image_root = Path(args.image_root)

    print("Loading InternVL tokenizer/model...")
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=False,
        tokenizer_type="qwen2",
    )
    model = AutoModel.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        pixel_values = load_image(
            row["image_path"], image_root=image_root, max_num=args.max_num_tiles
        ).to(torch.bfloat16).cuda()
        pred = model.chat(tokenizer, pixel_values, make_prompt(row), generation_config)
        choice = parse_choice(pred)
        label = row["label"].upper()
        out_rows.append(
            {
                "model": args.model_name,
                "idx": row.get("idx") or row.get("stress_id") or i,
                "scale_type": row.get("scale_type") or row.get("scale_factor", ""),
                "question_type": row.get("question_type") or "",
                "pair_direction": row.get("pair_direction", ""),
                "question": row["question"],
                "candidate_a": row["candidate_a"],
                "candidate_b": row["candidate_b"],
                "label": label,
                "prediction": pred,
                "choice": choice,
                "score": int(choice == label),
            }
        )
        if (i + 1) % 50 == 0:
            acc = sum(r["score"] for r in out_rows) / len(out_rows)
            print(args.model_name, i + 1, "acc", acc)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved detailed: {output}")
    summarize(out_rows, output)


if __name__ == "__main__":
    main()
