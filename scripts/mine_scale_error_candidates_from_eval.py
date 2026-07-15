#!/usr/bin/env python3
"""Mine natural scale-error candidates from evaluation CSV files.

The script is intentionally high-recall. It scans model prediction CSVs that
contain question, answers, prediction, and score columns, then extracts cases
where an incorrect numeric prediction is close to a 10x/100x/1000x transform of
the gold answer. The output is for human annotation, not final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
TARGETS = [0.001, 0.01, 0.1, 10.0, 100.0, 1000.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-candidates", required=True)
    parser.add_argument("--output-merged", required=True)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def first_answer(value: Any) -> str:
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, list) and obj:
                return str(obj[0])
        except Exception:
            pass
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return str(value)


def extract_number(value: Any) -> float | None:
    text = str(value).replace(",", "").replace("−", "-")
    match = NUM_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def count_numbers(value: Any) -> int:
    return len(NUM_RE.findall(str(value).replace(",", "").replace("−", "-")))


def factor_label(factor: float) -> str:
    if factor < 1:
        return f"1/{int(round(1 / factor))}x"
    return f"{int(round(factor))}x"


def near_factor(ratio: float, tolerance: float) -> float | None:
    if ratio <= 0 or not math.isfinite(ratio):
        return None
    best = min(TARGETS, key=lambda t: abs(math.log(ratio / t)))
    if best / (1 + tolerance) <= ratio <= best * (1 + tolerance):
        return best
    return None


def infer_split(path: Path) -> str:
    name = path.name.lower()
    if "test" in name:
        return "test"
    if "train" in name:
        return "train_or_train_subset"
    if "val" in name or "full" in name:
        return "validation/dev"
    return "unknown"


def infer_dataset(path: Path) -> str:
    name = path.name.lower()
    if "chartqa" in name:
        return "ChartQA"
    if "textvqa" in name:
        return "TextVQA"
    return "unknown"


def infer_priority(factor: str, question: str, n_gold: int, n_pred: int) -> str:
    q = question.lower()
    if factor in {"100x", "1/100x"} and n_gold == 1 and n_pred == 1:
        return "A_scale_100x_single_number"
    if factor in {"10x", "1/10x", "1000x", "1/1000x"} and n_gold == 1 and n_pred == 1:
        return "B_scale_10x_1000x_single_number"
    if any(k in q for k in ["percent", "percentage", "%", "rate", "share", "million", "thousand", "billion"]):
        return "C_scale_keyword_ratio"
    return "D_broad_ratio_candidate"


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    files = [Path(x) for x in args.inputs]
    if args.max_files is not None:
        files = files[: args.max_files]

    for path in files:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        required = {"idx", "question", "answers", "prediction", "score"}
        if not required.issubset(df.columns):
            continue
        adapter_col = "adapter" if "adapter" in df.columns else None
        dataset = infer_dataset(path)
        split = infer_split(path)
        for _, r in df.iterrows():
            try:
                score = int(r["score"])
            except Exception:
                continue
            if score != 0:
                continue
            gold = first_answer(r["answers"])
            pred = str(r["prediction"])
            g = extract_number(gold)
            p = extract_number(pred)
            if g is None or p is None or g == 0 or p == 0:
                continue
            if 1500 <= abs(g) <= 2200:
                continue
            ratio = abs(p / g)
            factor = near_factor(ratio, args.tolerance)
            if factor is None:
                continue
            factor_s = factor_label(factor)
            direction = "prediction_too_large" if abs(p) > abs(g) else "prediction_too_small"
            n_gold = count_numbers(gold)
            n_pred = count_numbers(pred)
            question = str(r["question"])
            adapter = str(r[adapter_col]) if adapter_col else path.stem
            rows.append(
                {
                    "source_file": str(path),
                    "dataset": dataset,
                    "split": split,
                    "idx": int(r["idx"]),
                    "adapter_or_model": adapter,
                    "question": question,
                    "gold_answer": gold,
                    "prediction": pred,
                    "gold_number": g,
                    "prediction_number": p,
                    "abs_pred_over_gold": ratio,
                    "scale_factor": factor_s,
                    "scale_direction": direction,
                    "num_gold_numbers": n_gold,
                    "num_prediction_numbers": n_pred,
                    "priority": infer_priority(factor_s, question, n_gold, n_pred),
                }
            )

    out = pd.DataFrame(rows)
    Path(args.output_candidates).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_candidates, index=False, quoting=csv.QUOTE_MINIMAL)

    if out.empty:
        out.to_csv(args.output_merged, index=False)
        return

    def join_unique(vals: pd.Series, limit: int = 12) -> str:
        unique = []
        for v in vals:
            s = str(v)
            if s not in unique:
                unique.append(s)
            if len(unique) >= limit:
                break
        return " | ".join(unique)

    merged = (
        out.sort_values(["priority", "dataset", "split", "idx"])
        .groupby(["dataset", "split", "idx", "question", "gold_answer"], as_index=False)
        .agg(
            candidate_occurrences=("prediction", "size"),
            adapters_or_models=("adapter_or_model", join_unique),
            predictions=("prediction", join_unique),
            scale_factors=("scale_factor", join_unique),
            scale_directions=("scale_direction", join_unique),
            priorities=("priority", join_unique),
            source_files=("source_file", join_unique),
        )
    )
    priority_order = {
        "A_scale_100x_single_number": 0,
        "B_scale_10x_1000x_single_number": 1,
        "C_scale_keyword_ratio": 2,
        "D_broad_ratio_candidate": 3,
    }
    merged["priority_rank"] = merged["priorities"].map(
        lambda x: min(priority_order.get(p, 9) for p in str(x).split(" | "))
    )
    merged = merged.sort_values(["priority_rank", "candidate_occurrences"], ascending=[True, False])
    merged = merged.drop(columns=["priority_rank"])
    merged.to_csv(args.output_merged, index=False, quoting=csv.QUOTE_MINIMAL)
    print("raw_candidates", len(out))
    print("merged_candidates", len(merged))
    print(out["priority"].value_counts().to_string())


if __name__ == "__main__":
    main()
