#!/usr/bin/env python3
"""Evaluate Qwen2.5-VL on local ChartQA parquet files.

This script is intentionally separate from the original HF-loader evaluation
path so that validation/test runs can be reproduced offline from cached
ChartQA parquet shards. It supports plain and scale-focused prompts plus
optional PEFT adapters.
"""

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
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapters", nargs="+", default=["base=NONE"])
    parser.add_argument("--prompt-mode", choices=["plain", "scale"], default="plain")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    return parser.parse_args()


def first_answer(ans: Any) -> str:
    if isinstance(ans, str):
        try:
            ans = json.loads(ans)
        except Exception:
            return ans
    if isinstance(ans, (list, tuple)) and ans:
        return str(ans[0])
    if hasattr(ans, "tolist"):
        vals = ans.tolist()
        if vals:
            return str(vals[0])
    return str(ans)


def norm(text: Any) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9.\-/% ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_number(text: Any) -> float | None:
    s = str(text).replace(",", "").replace("−", "-")
    match = NUM_RE.search(s)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def has_number(text: Any) -> bool:
    return NUM_RE.search(str(text)) is not None


def relaxed_score(pred: Any, answers: Any) -> int:
    if isinstance(answers, str):
        try:
            ans_list = json.loads(answers)
        except Exception:
            ans_list = [answers]
    elif hasattr(answers, "tolist"):
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
        if p_num is not None and a_num is not None:
            tol = max(0.05 * abs(a_num), 1e-3)
            if abs(p_num - a_num) <= tol:
                return 1
            continue

        if not has_number(pred_n) and not has_number(ans_n):
            if len(ans_n) >= 3 and (ans_n in pred_n or pred_n in ans_n):
                return 1

    return 0


def clean_prediction(text: str) -> str:
    raw = str(text).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ["final_answer", "answer", "computed_answer"]:
                if key in obj:
                    return str(obj[key]).strip()
    except Exception:
        pass
    match = re.search(r"FINAL\s*:\s*(.+)", raw, flags=re.I)
    if match:
        return match.group(1).strip().splitlines()[0]
    return raw.splitlines()[0].strip()


def load_image(obj: Any) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, dict):
        data = obj.get("bytes")
        if data is not None:
            return Image.open(io.BytesIO(data)).convert("RGB")
        path = obj.get("path")
        if path:
            return Image.open(path).convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")
    raise TypeError(f"Unsupported image object: {type(obj)}")


def make_prompt(question: str, prompt_mode: str) -> str:
    if prompt_mode == "scale":
        return (
            "Read the chart carefully and answer the question.\n"
            "Pay special attention to the original chart scale, axis tick labels, "
            "value labels, legend, and units. Preserve the scale shown in the chart. "
            "If the chart value is shown as 59, answer 59, not 0.59. "
            "If the chart value is shown as 0.59, answer 0.59, not 59. "
            "Do not convert percentages to decimals or decimals to percentages unless the question explicitly asks.\n"
            "Return only the final short answer.\n"
            f"Question: {question}"
        )
    return f"{question} Answer briefly."


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
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return clean_prediction(output_text)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()

    processor_kwargs: dict[str, Any] = {"local_files_only": True}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        processor_kwargs["max_pixels"] = args.max_pixels

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, **processor_kwargs)
    print("Loading model...")
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

    rows: list[dict[str, Any]] = []
    extra_fields = [
        c
        for c in [
            "candidate_id",
            "source_split",
            "source_row_idx",
            "review_label",
            "scale_error_subtype",
            "scale_factors",
            "scale_directions",
        ]
        if c in df.columns
    ]
    peft_model = None
    loaded_adapters: set[str] = set()
    for adapter_name, adapter_path in specs:
        print(f"=== Evaluating {adapter_name} ({args.prompt_mode}) ===")
        if adapter_path == "NONE":
            model = base_model
        else:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    adapter_name=adapter_name,
                    local_files_only=True,
                )
                loaded_adapters.add(adapter_name)
            elif adapter_name not in loaded_adapters:
                peft_model.load_adapter(
                    adapter_path,
                    adapter_name=adapter_name,
                    is_trainable=False,
                    local_files_only=True,
                )
                loaded_adapters.add(adapter_name)
            peft_model.set_adapter(adapter_name)
            peft_model.eval()
            model = peft_model

        correct = 0
        for i, ex in df.iterrows():
            image = load_image(ex["image"])
            question = str(ex["query"])
            answers = ex["label"]
            prompt = make_prompt(question, args.prompt_mode)
            pred = generate_answer(model, processor, image, prompt, args.max_new_tokens)
            score = relaxed_score(pred, answers)
            correct += score
            row = {
                    "adapter": adapter_name,
                    "prompt_mode": args.prompt_mode,
                    "idx": int(i),
                    "question": question,
                    "answers": json.dumps([first_answer(answers)], ensure_ascii=False),
                    "prediction": pred,
                    "score": score,
                }
            for field in extra_fields:
                row[field] = ex[field]
            rows.append(row)
            if (len(rows) % 50) == 0:
                local = [r for r in rows if r["adapter"] == adapter_name]
                acc = sum(int(r["score"]) for r in local) / len(local)
                print(adapter_name, len(local), "acc", acc)
        print("SUMMARY", adapter_name, correct, "/", len(df), correct / max(1, len(df)))

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "adapter",
                "prompt_mode",
                "idx",
                "question",
                "answers",
                "prediction",
                "score",
            ]
            + extra_fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for adapter_name, _ in specs:
        vals = [int(r["score"]) for r in rows if r["adapter"] == adapter_name]
        summary[adapter_name] = sum(vals) / max(1, len(vals))
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", output)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
