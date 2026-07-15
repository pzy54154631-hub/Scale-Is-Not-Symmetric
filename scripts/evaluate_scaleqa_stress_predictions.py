#!/usr/bin/env python3
"""Score free-form model predictions on a Scale-QA stress-test file.

This evaluator joins an existing ChartQA-style prediction CSV by source index
and measures whether the model prediction matches the gold answer, a deliberately
scaled negative answer, or neither.  It is intentionally lightweight: it can
reuse already completed ChartQA runs without launching a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-jsonl", required=True)
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-col", default="prediction")
    return parser.parse_args()


def extract_number(text: Any) -> float | None:
    s = str(text).replace(",", "").replace("−", "-").strip()
    nums = NUM_RE.findall(s)
    if len(nums) != 1:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


def normalize_text(text: Any) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9.\-/%]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def relaxed_numeric_match(pred: Any, target: Any) -> bool:
    p = extract_number(pred)
    t = extract_number(target)
    if p is None or t is None:
        return normalize_text(pred) == normalize_text(target)
    tol = max(0.05 * abs(t), 1e-3)
    return abs(p - t) <= tol


def read_stress(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_predictions(path: Path, prediction_col: str) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if prediction_col not in reader.fieldnames:
            candidates = [c for c in reader.fieldnames or [] if "pred" in c.lower()]
            if not candidates:
                raise ValueError(f"No prediction column found in {path}")
            prediction_col = candidates[0]
        for row in reader:
            idx = int(row["idx"])
            row["_prediction_col"] = prediction_col
            out[idx].append(row)
    return out


def add_count(bucket: dict[str, Any], outcome: str) -> None:
    bucket["n"] += 1
    bucket[outcome] += 1


def finalize(bucket: dict[str, Any]) -> dict[str, Any]:
    n = bucket["n"]
    out = dict(bucket)
    for key in ["gold_match", "scaled_negative_match", "other"]:
        out[f"{key}_rate"] = out[key] / n if n else 0.0
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stress = read_stress(Path(args.stress_jsonl))
    preds_by_idx = read_predictions(Path(args.prediction_csv), args.prediction_col)

    detailed_path = output_dir / "scaleqa_stress_freeform_detailed.csv"
    summary_path = output_dir / "scaleqa_stress_freeform_summary.csv"
    summary_json_path = output_dir / "scaleqa_stress_freeform_summary.json"

    summary: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "gold_match": 0,
            "scaled_negative_match": 0,
            "other": 0,
        }
    )
    detailed_rows: list[dict[str, Any]] = []

    for item in stress:
        idx = int(item["source_idx"])
        for pred_row in preds_by_idx.get(idx, []):
            adapter = pred_row.get("adapter", "unknown")
            pred = pred_row[pred_row["_prediction_col"]]
            if relaxed_numeric_match(pred, item["gold_value"]):
                outcome = "gold_match"
            elif relaxed_numeric_match(pred, item["negative_answer"]):
                outcome = "scaled_negative_match"
            else:
                outcome = "other"

            add_count(summary[(adapter, "all")], outcome)
            add_count(summary[(adapter, item["pair_direction"])], outcome)
            detailed_rows.append(
                {
                    "stress_id": item["stress_id"],
                    "source_idx": idx,
                    "adapter": adapter,
                    "pair_direction": item["pair_direction"],
                    "scale_factor": item["scale_factor"],
                    "gold_value": item["gold_value"],
                    "negative_answer": item["negative_answer"],
                    "prediction": pred,
                    "outcome": outcome,
                }
            )

    with detailed_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()) if detailed_rows else [])
        if detailed_rows:
            writer.writeheader()
            writer.writerows(detailed_rows)

    summary_rows = []
    for (adapter, split), bucket in sorted(summary.items()):
        row = {"adapter": adapter, "split": split}
        row.update(finalize(bucket))
        summary_rows.append(row)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    summary_json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"Saved detailed: {detailed_path}")
    print(f"Saved summary:  {summary_path}")
    print(f"Saved JSON:     {summary_json_path}")
    for row in summary_rows:
        if row["split"] == "all":
            print(
                row["adapter"],
                "n=", row["n"],
                "gold=", f"{row['gold_match_rate']:.4f}",
                "scaled_neg=", f"{row['scaled_negative_match_rate']:.4f}",
                "other=", f"{row['other_rate']:.4f}",
            )


if __name__ == "__main__":
    main()
