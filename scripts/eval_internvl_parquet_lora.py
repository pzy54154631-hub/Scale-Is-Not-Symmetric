#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchvision.transforms as T
from peft import PeftModel
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/InternVL2_5-1B")
    ap.add_argument("--adapters", nargs="+", default=["base=NONE"])
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--max-samples", type=int, default=None)
    return ap.parse_args()


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


def load_image(obj: Any) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, dict):
        if obj.get("bytes") is not None:
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path"):
            return Image.open(obj["path"]).convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")
    raise TypeError(type(obj))


def image_to_pixels(image: Image.Image, max_num: int = 4) -> torch.Tensor:
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(image, image_size=448, max_num=max_num, use_thumbnail=True)
    return torch.stack([transform(img) for img in images])


def norm(text: Any) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9.\-/% ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_number(text: Any) -> float | None:
    match = NUM_RE.search(str(text).replace(",", "").replace("−", "-"))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def relaxed_score(pred: Any, answers: Any) -> int:
    if hasattr(answers, "tolist"):
        ans_list = answers.tolist()
    elif isinstance(answers, (list, tuple)):
        ans_list = list(answers)
    else:
        ans_list = [answers]
    pred_n = norm(pred)
    for ans in ans_list:
        ans_n = norm(ans)
        if pred_n == ans_n:
            return 1
        p_num = extract_number(pred_n)
        a_num = extract_number(ans_n)
        if p_num is not None and a_num is not None and abs(p_num - a_num) <= max(0.05 * abs(a_num), 1e-3):
            return 1
        if p_num is None and a_num is None and len(ans_n) >= 3 and (ans_n in pred_n or pred_n in ans_n):
            return 1
    return 0


def first_answer(ans: Any) -> str:
    if hasattr(ans, "tolist"):
        ans = ans.tolist()
    if isinstance(ans, (list, tuple)) and ans:
        return str(ans[0])
    return str(ans)


def chat_model(model):
    if hasattr(model, "chat"):
        return model
    base = getattr(model, "base_model", None)
    if base is not None:
        if hasattr(base, "chat"):
            return base
        inner = getattr(base, "model", None)
        if inner is not None and hasattr(inner, "chat"):
            return inner
    return model


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    if args.max_samples:
        df = df.iloc[: args.max_samples].copy()

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=False,
        tokenizer_type="qwen2",
    )
    base_model = AutoModel.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    generation_config = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    specs = [item.split("=", 1) for item in args.adapters]
    rows = []
    peft_model = None
    for name, path in specs:
        print("=== Evaluating", name, path, "===")
        if path == "NONE":
            model = base_model
        else:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base_model, path, adapter_name=name, is_trainable=False)
            else:
                peft_model.load_adapter(path, adapter_name=name, is_trainable=False)
            peft_model.set_adapter(name)
            peft_model.eval()
            model = peft_model
        correct = 0
        for i, ex in df.iterrows():
            pixel_values = image_to_pixels(load_image(ex["image"])).to(torch.bfloat16).cuda()
            question = "<image>\n" + str(ex["query"]) + " Answer briefly with the final value only."
            pred = chat_model(model).chat(tokenizer, pixel_values, question, generation_config)
            score = relaxed_score(pred, ex["label"])
            correct += score
            rows.append(
                {
                    "adapter": name,
                    "idx": int(i),
                    "question": str(ex["query"]),
                    "answers": json.dumps([first_answer(ex["label"])], ensure_ascii=False),
                    "prediction": str(pred).splitlines()[0].strip(),
                    "score": score,
                    "scale_error_subtype": ex.get("scale_error_subtype", ""),
                }
            )
            if (len([r for r in rows if r["adapter"] == name]) % 50) == 0:
                n = len([r for r in rows if r["adapter"] == name])
                print(name, n, "acc", correct / n, flush=True)
        print("SUMMARY", name, correct, "/", len(df), correct / max(1, len(df)))

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name, _ in specs:
        vals = [int(r["score"]) for r in rows if r["adapter"] == name]
        summary[name] = sum(vals) / max(1, len(vals))
    output.with_suffix(output.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
