#!/usr/bin/env python3
"""Summarize ChartQA eval CSVs on scale diagnostic subsets.

The input format is the project-standard evaluation CSV with one row per
adapter/example and columns: adapter, idx, question, answers, prediction, score.
The script defines strict scale errors with respect to the chosen base adapter:
base is wrong and its single numeric prediction differs from the single numeric
gold answer by approximately 100x.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-strict-cases", required=True)
    parser.add_argument("--base-adapter", default="base")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def nums(text) -> list[float]:
    out = []
    s = str(text).lower().replace(",", "").replace("−", "-")
    for item in NUM_RE.findall(s):
        try:
            out.append(float(item))
        except ValueError:
            pass
    return out


def one_number(text) -> float | None:
    values = nums(text)
    return values[0] if len(values) == 1 else None


def first_answer(ans) -> str:
    try:
        obj = ast.literal_eval(str(ans))
        if isinstance(obj, list) and obj:
            return str(obj[0])
    except Exception:
        pass
    return str(ans)


def factor100(a, b) -> bool:
    x = one_number(a)
    y = one_number(b)
    if x is None or y is None or abs(x) < 1e-12 or abs(y) < 1e-12:
        return False
    ratio = abs(x / y)
    return (0.008 <= ratio <= 0.0125) or (80 <= ratio <= 125)


def bootstrap_delta(base_scores: np.ndarray, other_scores: np.ndarray, n_boot: int, rng: np.random.Generator):
    if len(base_scores) == 0:
        return float("nan"), float("nan"), float("nan")
    delta = other_scores.mean() - base_scores.mean()
    if n_boot <= 0:
        return float(delta), float("nan"), float("nan")
    n = len(base_scores)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.integers(0, n, size=n)
        vals[i] = other_scores[sample].mean() - base_scores[sample].mean()
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(delta), float(lo), float(hi)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input).drop_duplicates(["adapter", "idx"]).copy()
    base = df[df["adapter"] == args.base_adapter].copy()
    if base.empty:
        raise ValueError(f"Base adapter not found: {args.base_adapter}")
    base["gold"] = base["answers"].apply(first_answer)
    base["gold_scale_error"] = [
        factor100(pred, gold) and int(score) == 0
        for pred, gold, score in zip(base["prediction"], base["gold"], base["score"])
    ]
    strict_ids = set(base.loc[base["gold_scale_error"], "idx"].astype(int))

    strict_cases = base.loc[base["gold_scale_error"], ["idx", "question", "answers", "prediction", "score", "gold"]]
    Path(args.output_strict_cases).parent.mkdir(parents=True, exist_ok=True)
    strict_cases.to_csv(args.output_strict_cases, index=False)

    base_scores = base.set_index("idx")["score"].astype(float)
    rows = []
    rng = np.random.default_rng(args.seed)
    for adapter in df["adapter"].unique():
        sub = df[df["adapter"] == adapter].copy().set_index("idx")
        common = sorted(set(base_scores.index) & set(sub.index))
        for split_name, ids in [
            ("all", common),
            ("strict", [i for i in common if int(i) in strict_ids]),
            ("non_strict", [i for i in common if int(i) not in strict_ids]),
        ]:
            if not ids:
                continue
            b = base_scores.loc[ids].to_numpy(dtype=float)
            s = sub.loc[ids, "score"].to_numpy(dtype=float)
            delta, lo, hi = bootstrap_delta(b, s, args.bootstrap, rng)
            rows.append(
                {
                    "adapter": adapter,
                    "split": split_name,
                    "n": len(ids),
                    "accuracy": float(s.mean()),
                    "base_accuracy": float(b.mean()),
                    "delta": delta,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "correct": int(s.sum()),
                }
            )

    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved:", args.output_summary)
    print("Saved:", args.output_strict_cases)


if __name__ == "__main__":
    main()
