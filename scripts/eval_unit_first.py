#!/usr/bin/env python
"""Unit-first decoding and unit-aware sampling for chart numeric QA.

Runs, per example, on the same answer-tuned backbone used elsewhere in the repo:

  direct        : plain short-answer prompt (baseline)
  self_check    : the self-check prompt (strongest fixed baseline on ScaleQA-473)
  unit_first    : model reads (value, unit-as-displayed, where it read it) and the
                  answer convention implied by the question; a deterministic rule
                  then converts. The model never chooses the output scale itself.
  blind_k       : k stochastic samples with the direct prompt (pass@k and
                  majority vote), optional
  unit_vote     : the k samples parsed into (value, unit) and voted within the
                  unit bucket chosen by the unit_first read, optional

Outputs the same CSV schema as eval_scaleqa_parquet_tta_prompts_v2.py so the
existing analysis scripts (fix/break, subtype, bootstrap) apply unchanged.

Usage (ScaleQA-Natural-473 parquet or a ChartQA split parquet):
  python scripts/eval_unit_first.py --parquet data/scaleqa473.parquet \
      --output results/eval/unit_first_scaleqa473.csv --samples 20

Nothing here trains anything; a 7B model in bf16 fits a 24-32 GB GPU.
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd
import torch

from eval_scaleqa_parquet_tta_prompts_v2 import (NUM_RE, clean, gen, norm, normalize_image, nums, score, to_list)
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DIRECT = "Answer briefly.\nQuestion: {question}"
SELF_CHECK = ("Answer the chart question, then silently check whether the number you wrote is in the scale the chart "
              "displays (percent sign, decimal fraction, thousands, millions). Return only the final short answer.\nQuestion: {question}")
UNIT_FIRST = (
    "You will answer a chart question in two steps. Do not convert any number.\n"
    "Step 1. Read the value exactly as the chart displays it and report where you read it.\n"
    "Step 2. State the answer convention the question asks for.\n"
    "Reply with exactly these four lines and nothing else:\n"
    "VALUE: <the number exactly as displayed, e.g. 61 or 0.61 or 3.2>\n"
    "UNIT: <one of: percent, decimal_fraction, thousand, million, billion, currency, none>\n"
    "SOURCE: <one of: data_label, axis_tick, legend, title, computed>\n"
    "FORMAT: <one of: percent_number, decimal_fraction, raw, same_as_chart>\n"
    "Question: {question}")

UNIT_FACTOR = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def parse_unit_first(raw: str):
    f = {}
    for key in ("VALUE", "UNIT", "SOURCE", "FORMAT"):
        m = re.search(rf"{key}\s*:\s*([^\n]+)", raw, flags=re.I)
        f[key.lower()] = m.group(1).strip().lower() if m else ""
    v = nums(f["value"])
    f["value_num"] = v[0] if v else None
    f["value_has_pct"] = "%" in f["value"]
    return f


def convert(f, question: str):
    """Deterministic scale decision. Returns (answer_string, rule_used)."""
    v = f["value_num"]
    if v is None:
        return "", "no_value"
    unit, fmt = f["unit"], f["format"]
    q = question.lower()
    asks_percent = ("percent" in q or "%" in q or "percentage" in q)
    # Rule 1: chart shows percent, question asks percent or same -> keep displayed number.
    if unit == "percent" or f["value_has_pct"]:
        if fmt == "decimal_fraction":
            return fmt_num(v / 100), "pct_to_decimal"
        return fmt_num(v), "keep_percent_number"
    # Rule 2: chart shows a decimal fraction (0-1), question asks percent -> x100.
    if unit == "decimal_fraction" and (fmt == "percent_number" or (fmt == "same_as_chart" and asks_percent)):
        if 0 <= abs(v) <= 1.5:
            return fmt_num(v * 100), "decimal_to_pct"
    # Rule 3: scaled units (thousand/million/billion): keep displayed unless the
    # question explicitly asks for a raw value ("in dollars", "how many people").
    if unit in UNIT_FACTOR:
        if fmt == "raw" or re.search(r"\b(in dollars|in units|how many (people|units|individuals))\b", q):
            return fmt_num(v * UNIT_FACTOR[unit]), f"{unit}_to_raw"
        return fmt_num(v), f"keep_{unit}"
    return fmt_num(v), "keep_displayed"


def fmt_num(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.6g}"


def sample(model, processor, image, prompt, k, temperature, max_new_tokens):
    image = normalize_image(image)
    msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msg)
    inputs = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k_: v.to(device) if hasattr(v, "to") else v for k_, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                             top_p=0.95, num_return_sequences=k)
    L = inputs["input_ids"].shape[1]
    raws = processor.batch_decode([o[L:] for o in out], skip_special_tokens=True)
    return [clean(r) for r in raws]


def unit_bucket(pred: str):
    return "percent" if "%" in pred else ("fraction" if (nums(pred) and 0 <= abs(nums(pred)[0]) <= 1.5) else "number")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapters/chartqa_answer_only_3000_r16")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--samples", type=int, default=0, help="k blind samples for pass@k / unit-aware vote (0 = off)")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.max_samples:
        df = df.iloc[: args.max_samples].copy()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    if args.adapter and args.adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    rows = []
    def add(name, idx, cid, q, answers, pred, raw, subtype, extra=None):
        r = {"adapter": name, "idx": idx, "candidate_id": cid, "question": q, "answers": json.dumps(to_list(answers), ensure_ascii=False),
             "prediction": pred, "raw": raw, "score": score(pred, answers), "scale_error_subtype": subtype}
        if extra: r.update(extra)
        rows.append(r)

    for idx, ex in df.iterrows():
        image, q, answers = ex["image"], str(ex["query"]), ex["label"]
        cid, subtype = ex.get("candidate_id", idx), ex.get("scale_error_subtype", "")
        p, raw = gen(model, processor, image, DIRECT.format(question=q), args.max_new_tokens); add("direct", idx, cid, q, answers, p, raw, subtype)
        p, raw = gen(model, processor, image, SELF_CHECK.format(question=q), args.max_new_tokens); add("self_check", idx, cid, q, answers, p, raw, subtype)
        _, raw = gen(model, processor, image, UNIT_FIRST.format(question=q), 64)
        f = parse_unit_first(raw); ans, rule = convert(f, q)
        add("unit_first", idx, cid, q, answers, ans, raw, subtype, {"uf_unit": f["unit"], "uf_source": f["source"], "uf_format": f["format"], "uf_rule": rule})
        if args.samples > 0:
            preds = sample(model, processor, image, DIRECT.format(question=q), args.samples, args.temperature, args.max_new_tokens)
            scs = [score(x, answers) for x in preds]
            add("blind_pass_at_k", idx, cid, q, answers, "", json.dumps(preds, ensure_ascii=False), subtype, {"score": max(scs)})
            rows[-1]["score"] = max(scs)
            maj = Counter(norm(x) for x in preds if norm(x)).most_common(1)
            maj_pred = next((x for x in preds if norm(x) == maj[0][0]), preds[0]) if maj else preds[0]
            add("blind_majority", idx, cid, q, answers, maj_pred, "", subtype)
            # unit-aware vote: restrict to samples whose bucket matches the unit-first read, then majority.
            want = "percent" if (f["unit"] == "percent" or f["value_has_pct"]) else ("fraction" if f["unit"] == "decimal_fraction" else "number")
            pool = [x for x in preds if unit_bucket(x) == want] or preds
            c = Counter(norm(x) for x in pool if norm(x)).most_common(1)
            uv = next((x for x in pool if norm(x) == c[0][0]), pool[0]) if c else pool[0]
            add("unit_vote", idx, cid, q, answers, uv, json.dumps(pool, ensure_ascii=False), subtype, {"uf_bucket": want})
        n = len({r["idx"] for r in rows})
        if n % 25 == 0:
            d = pd.DataFrame(rows); print("done", n, d.groupby("adapter")["score"].mean().round(4).to_dict(), flush=True)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows); res.to_csv(out, index=False)
    # paired fix/break of each method vs direct
    piv = res.pivot_table(index="candidate_id", columns="adapter", values="score")
    fb = {}
    for m in piv.columns:
        if m == "direct": continue
        fb[m] = {"acc": float(piv[m].mean()), "fix": float(((piv["direct"] == 0) & (piv[m] == 1)).mean()), "break": float(((piv["direct"] == 1) & (piv[m] == 0)).mean())}
    summary = {"n": int(len(piv)), "direct": float(piv["direct"].mean()), "methods": fb,
               "unit_first_rules": res.query("adapter=='unit_first'")["uf_rule"].value_counts().to_dict()}
    out.with_suffix(out.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
