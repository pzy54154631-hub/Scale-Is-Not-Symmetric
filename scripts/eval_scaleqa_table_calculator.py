#!/usr/bin/env python
"""DePlot-style table extraction plus calculator baseline for ScaleQA-Natural.

This baseline deliberately separates chart reading from numerical reasoning:
1. ask the VLM to extract a compact table/axis-unit description;
2. ask the language model side to solve from that text and emit CALC/FINAL;
3. safely evaluate CALC expressions when possible.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import operator
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", re.I)


def to_list(x):
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = str(x).strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            y = fn(s)
            return y if isinstance(y, list) else [y]
        except Exception:
            pass
    return [s]


def norm(x):
    s = str(x).strip().lower().replace("−", "-").replace(",", "")
    s = s.replace("%", " percent ")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;\"'[](){}")


def nums(x):
    out = []
    for m in NUM_RE.findall(str(x).lower().replace(",", "").replace("−", "-")):
        try:
            out.append(float(m))
        except Exception:
            pass
    return out


def close(p, g):
    return abs(p - g) <= (1e-4 if abs(g) < 1e-12 else max(0.05 * abs(g), 1e-3))


def score(pred, answers):
    pn = norm(pred)
    ans = [a for a in to_list(answers) if str(a).strip()]
    ansn = [norm(a) for a in ans]
    if any(pn == a for a in ansn):
        return 1
    for p in nums(pred):
        for a in ans:
            for g in nums(a):
                if close(p, g):
                    return 1
    for a in ansn:
        if len(a) >= 3 and len(pn) >= 3 and (a in pn or pn in a):
            return 1
    return 0


def clean_answer(x):
    s = str(x).strip()
    for pat in [r"FINAL\s*:\s*(.+)", r"answer\s*[:：]\s*(.+)"]:
        m = re.search(pat, s, flags=re.I | re.S)
        if m:
            s = m.group(1).strip()
            break
    s = s.splitlines()[0].strip().strip(" .,:;\"'[](){}")
    if len(s.split()) > 12:
        xs = nums(s)
        if xs:
            v = xs[-1]
            return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.6g}"
    return s


def normalize_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return image


def gen_image(model, processor, image, prompt, max_new_tokens):
    image = normalize_image(image)
    msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msg)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def gen_text(model, processor, prompt, max_new_tokens):
    msg = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def table_prompt(question):
    return (
        "Extract the chart into structured evidence for answering one question.\n"
        "Preserve numbers, tick labels, axis units, legends, and percent signs exactly as shown.\n"
        "Do not rescale values. Do not convert decimals to percentages or percentages to decimals.\n"
        "Return compact JSON with keys: title, x_axis, y_axis, units, ticks, legend, data_points, notes.\n"
        "Use null when unknown. Keep it short but include all visible numbers needed for the question.\n\n"
        f"Question: {question}"
    )


def solve_prompt(evidence, question):
    return (
        "You are solving a chart question from extracted chart evidence.\n"
        "Use only the evidence below. Preserve the chart's displayed scale and unit.\n"
        "If arithmetic is needed, write CALC: followed by a Python-style numeric expression using only numbers and + - * / parentheses.\n"
        "Then write FINAL: followed by one short answer. If no arithmetic is needed, set CALC to the final number.\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question}\n"
        "CALC:"
    )


SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_expr(expr):
    expr = expr.strip().splitlines()[0]
    expr = re.sub(r"[^0-9eE\.\+\-\*\/\(\) ]", "", expr)
    if not expr or len(expr) > 120:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
            return SAFE_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
            right = ev(node.right)
            if isinstance(node.op, ast.Div) and abs(right) < 1e-12:
                raise ZeroDivisionError
            return SAFE_OPS[type(node.op)](ev(node.left), right)
        raise ValueError(type(node).__name__)

    try:
        val = ev(ast.parse(expr, mode="eval"))
    except Exception:
        return None
    if not math.isfinite(val):
        return None
    return str(int(round(val))) if abs(val - round(val)) < 1e-6 else f"{val:.6g}"


def parse_calc(raw):
    text = str(raw)
    m = re.search(r"CALC\s*:\s*(.+?)(?:FINAL\s*:|$)", text, flags=re.I | re.S)
    if not m:
        # raw often begins immediately after CALC: because the prompt ends there.
        m = re.search(r"^(.+?)(?:FINAL\s*:|$)", text, flags=re.I | re.S)
    if not m:
        return None
    return safe_eval_expr(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapters/chartqa_answer_only_3000_r16")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.max_samples is not None:
        df = df.iloc[: args.max_samples].copy()
    if args.limit is not None:
        df = df.iloc[args.start : args.start + args.limit].copy()
    elif args.start:
        df = df.iloc[args.start :].copy()

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, device_map="auto", local_files_only=True
    )
    if args.adapter and args.adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    rows = []
    for pos, (idx, ex) in enumerate(df.iterrows(), 1):
        image = ex["image"]
        q = str(ex.get("query", ex.get("question", "")))
        answers = ex.get("label", ex.get("answers", ""))
        cid = ex.get("candidate_id", idx)
        subtype = ex.get("scale_error_subtype", "")
        directions = ex.get("scale_directions", "")
        evidence = gen_image(model, processor, image, table_prompt(q), 384)
        raw = gen_text(model, processor, solve_prompt(evidence, q), 96)
        final_pred = clean_answer(raw)
        calc_pred = parse_calc(raw)
        calc_or_final = calc_pred or final_pred
        records = [
            ("table_reason", final_pred, raw, score(final_pred, answers)),
            ("table_calculator", calc_pred or "", raw, score(calc_pred or "", answers) if calc_pred else 0),
            ("table_calc_or_reason", calc_or_final, raw, score(calc_or_final, answers)),
        ]
        for adapter, pred, raw_text, sc in records:
            rows.append(
                {
                    "adapter": adapter,
                    "idx": idx,
                    "candidate_id": cid,
                    "question": q,
                    "answers": json.dumps(to_list(answers), ensure_ascii=False),
                    "prediction": pred,
                    "score": sc,
                    "raw": raw_text,
                    "table_text": evidence,
                    "scale_error_subtype": subtype,
                    "scale_directions": directions,
                }
            )
        if pos % 10 == 0:
            tmp = pd.DataFrame(rows)
            print(pos, tmp.groupby("adapter")["score"].mean().to_dict(), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    summary = {
        "overall": res.groupby("adapter")["score"].mean().to_dict(),
        "by_subtype": res.groupby(["adapter", "scale_error_subtype"])["score"].mean().reset_index().to_dict("records"),
        "n_examples": int(res["candidate_id"].nunique()),
    }
    out.with_suffix(out.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
