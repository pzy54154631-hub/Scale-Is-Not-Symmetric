#!/usr/bin/env python3
"""Analyze tokenizer treatment of gold and scaled numeric answers.

This is a cheap mechanistic probe for the scale-asymmetry story.  It does not
claim causality; it reports whether scaled alternatives such as ``61`` and
``0.61`` receive different token lengths or segmentations under the backbone
tokenizer.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-jsonl", required=True)
    parser.add_argument("--tokenizer", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", default="results/analysis/tokenizer_scale_bias")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=True)
    rows = read_jsonl(Path(args.stress_jsonl))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {
            "gold_len": [],
            "negative_len": [],
            "delta_len": [],
            "gold_chars": [],
            "negative_chars": [],
        }
    )

    for row in rows:
        gold = str(row["gold_value"])
        neg = str(row["negative_answer"])
        gold_ids = tok.encode(gold, add_special_tokens=False)
        neg_ids = tok.encode(neg, add_special_tokens=False)
        gold_tokens = tok.convert_ids_to_tokens(gold_ids)
        neg_tokens = tok.convert_ids_to_tokens(neg_ids)
        delta = len(neg_ids) - len(gold_ids)
        key = (row["pair_direction"], row["scale_factor"])
        buckets[key]["gold_len"].append(len(gold_ids))
        buckets[key]["negative_len"].append(len(neg_ids))
        buckets[key]["delta_len"].append(delta)
        buckets[key]["gold_chars"].append(len(gold))
        buckets[key]["negative_chars"].append(len(neg))
        detail_rows.append(
            {
                "stress_id": row["stress_id"],
                "pair_direction": row["pair_direction"],
                "scale_factor": row["scale_factor"],
                "gold_value": gold,
                "negative_answer": neg,
                "gold_token_len": len(gold_ids),
                "negative_token_len": len(neg_ids),
                "delta_token_len": delta,
                "gold_tokens": " ".join(gold_tokens),
                "negative_tokens": " ".join(neg_tokens),
            }
        )

    detail_path = output_dir / "tokenizer_scale_bias_detail.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()) if detail_rows else [])
        if detail_rows:
            writer.writeheader()
            writer.writerows(detail_rows)

    summary_rows = []
    for (direction, factor), vals in sorted(buckets.items()):
        summary_rows.append(
            {
                "pair_direction": direction,
                "scale_factor": factor,
                "n": len(vals["gold_len"]),
                "mean_gold_token_len": mean(vals["gold_len"]),
                "mean_negative_token_len": mean(vals["negative_len"]),
                "mean_delta_negative_minus_gold": mean(vals["delta_len"]),
                "mean_gold_chars": mean(vals["gold_chars"]),
                "mean_negative_chars": mean(vals["negative_chars"]),
            }
        )

    summary_path = output_dir / "tokenizer_scale_bias_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved detail:  {detail_path}")
    print(f"Saved summary: {summary_path}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
