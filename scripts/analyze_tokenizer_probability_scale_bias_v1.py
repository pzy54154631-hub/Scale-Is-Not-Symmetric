#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

OUT = Path("results/analysis/tokenizer_probability_scale_bias_v1")
OUT.mkdir(parents=True, exist_ok=True)

PARQ = Path("results/scaleqa_natural/image_assisted_v3/scaleqa_natural_520_image_assisted.parquet")
CAND = Path("results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
LOGPROB_FILES = [
    Path("results/eval/20260521_scaleqa520_logprob_verifier_answer_tuned.csv"),
    Path("results/eval/20260521_scaleqa520_logprob_verifier_large_seed1.csv"),
    Path("results/eval/20260521_scaleqa520_logprob_verifier_base.csv"),
]
MODEL_DIR = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_num(x):
    if pd.isna(x):
        return np.nan
    m = NUM_RE.search(str(x).replace(",", "").replace("−", "-"))
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except ValueError:
        return np.nan


def norm_answers(x):
    if isinstance(x, list):
        return x
    try:
        y = ast.literal_eval(str(x))
        return y if isinstance(y, list) else [y]
    except Exception:
        return [x]


def tok_features(tok, s: str):
    text = str(s)
    ids = tok.encode(text, add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    return {
        "text": text,
        "n_tokens": len(ids),
        "n_chars": len(text),
        "has_decimal": "." in text,
        "has_percent": "%" in text,
        "starts_zero_dot": bool(re.match(r"\\s*[-+]?0\\.", text)),
        "tokens": pieces,
    }


def scale_variants(v: float):
    if not np.isfinite(v):
        return {}
    return {
        "gold": f"{v:g}",
        "x10": f"{v*10:g}",
        "x100": f"{v*100:g}",
        "div10": f"{v/10:g}",
        "div100": f"{v/100:g}",
        "percent_suffix": f"{v:g}%",
    }


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    df = pd.read_parquet(PARQ)
    cand = pd.read_csv(CAND)
    meta = cand.drop_duplicates("candidate_id").set_index("candidate_id")

    rows = []
    for _, ex in df.iterrows():
        cid = str(ex.get("candidate_id"))
        answers = norm_answers(ex.get("label", ex.get("answers", "")))
        nums = [parse_num(a) for a in answers]
        nums = [x for x in nums if np.isfinite(x)]
        if not nums:
            continue
        gold = nums[0]
        direction = str(ex.get("scale_directions", meta.loc[cid].get("scale_directions", "") if cid in meta.index else ""))
        subtype = str(ex.get("scale_error_subtype", meta.loc[cid].get("scale_error_subtype", "") if cid in meta.index else ""))
        for name, text in scale_variants(gold).items():
            feat = tok_features(tok, text)
            rows.append(
                {
                    "candidate_id": cid,
                    "scale_directions": direction,
                    "scale_error_subtype": subtype,
                    "variant": name,
                    **{k: v for k, v in feat.items() if k != "tokens"},
                    "tokens_json": json.dumps(feat["tokens"], ensure_ascii=False),
                }
            )
    tdf = pd.DataFrame(rows)
    tdf.to_csv(OUT / "tokenizer_variant_features.csv", index=False)
    variant_summary = (
        tdf.groupby(["variant"])
        .agg(n=("candidate_id", "count"), mean_tokens=("n_tokens", "mean"), median_tokens=("n_tokens", "median"), mean_chars=("n_chars", "mean"), decimal_rate=("has_decimal", "mean"), zero_dot_rate=("starts_zero_dot", "mean"))
        .reset_index()
    )
    variant_summary.to_csv(OUT / "tokenizer_variant_summary.csv", index=False)
    dir_summary = (
        tdf.groupby(["scale_directions", "variant"])
        .agg(n=("candidate_id", "count"), mean_tokens=("n_tokens", "mean"), decimal_rate=("has_decimal", "mean"), zero_dot_rate=("starts_zero_dot", "mean"))
        .reset_index()
    )
    dir_summary.to_csv(OUT / "tokenizer_by_direction_summary.csv", index=False)

    # Parse logprob candidate outputs: raw contains candidate list for oracle rows and logprob dicts for verifier rows.
    log_rows = []
    for path in LOGPROB_FILES:
        if not path.exists():
            continue
        dfp = pd.read_csv(path)
        source = path.stem.replace(".csv", "")
        for _, r in dfp.iterrows():
            adapter = str(r.get("adapter", ""))
            if "oracle" in adapter:
                continue
            raw = str(r.get("raw", ""))
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            logps = obj.get("logprobs") if isinstance(obj, dict) else None
            chosen = obj.get("letter") if isinstance(obj, dict) else None
            if not isinstance(logps, dict):
                continue
            vals = {k: float(v) for k, v in logps.items() if isinstance(v, (int, float))}
            if not vals:
                continue
            sorted_vals = sorted(vals.items(), key=lambda x: x[1], reverse=True)
            margin = sorted_vals[0][1] - sorted_vals[1][1] if len(sorted_vals) > 1 else np.nan
            log_rows.append(
                {
                    "source": source,
                    "adapter": adapter,
                    "candidate_id": r.get("candidate_id"),
                    "scale_directions": r.get("scale_directions"),
                    "scale_error_subtype": r.get("scale_error_subtype"),
                    "score": int(r.get("score", 0)),
                    "chosen_letter": chosen,
                    "top_letter": sorted_vals[0][0],
                    "top_logp": sorted_vals[0][1],
                    "margin": margin,
                    "entropy": float(-sum(math.exp(v) * v for v in vals.values() if np.isfinite(v))),
                    "raw_logps": json.dumps(vals),
                }
            )
    ldf = pd.DataFrame(log_rows)
    if not ldf.empty:
        ldf.to_csv(OUT / "logprob_letter_bias_rows.csv", index=False)
        bias_summary = (
            ldf.groupby(["source", "adapter", "scale_directions"])
            .agg(n=("score", "count"), acc=("score", "mean"), mean_margin=("margin", "mean"), top_A_rate=("top_letter", lambda x: float(np.mean(np.asarray(x) == "A"))), top_B_rate=("top_letter", lambda x: float(np.mean(np.asarray(x) == "B"))), top_C_rate=("top_letter", lambda x: float(np.mean(np.asarray(x) == "C"))), top_D_rate=("top_letter", lambda x: float(np.mean(np.asarray(x) == "D"))))
            .reset_index()
        )
        bias_summary.to_csv(OUT / "logprob_bias_by_direction.csv", index=False)

    summary = {
        "n_examples": int(df.shape[0]),
        "tokenizer_variant_summary": variant_summary.to_dict("records"),
        "directional_tokenizer_summary": dir_summary.to_dict("records")[:40],
        "logprob_bias_available": bool(not ldf.empty),
        "logprob_bias_rows": int(ldf.shape[0]) if not ldf.empty else 0,
    }
    (OUT / "tokenizer_probability_scale_bias_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
