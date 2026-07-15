#!/usr/bin/env python3
"""Build a ScaleQA-Natural draft parquet from annotated candidates.

The annotated candidate CSV is grouped by question and gold answer.  This
script maps the accepted candidates back to local ChartQA parquet rows so that
the resulting file can be evaluated by the existing VLM scripts with images.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--parquets", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--labels", nargs="+", default=["yes"])
    parser.add_argument(
        "--require-source-contains",
        default=None,
        help="Optional substring that must appear in annotation source_files.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def first_answer(value: Any) -> str:
    if isinstance(value, str):
        try:
            obj = json.loads(value)
            if isinstance(obj, list) and obj:
                return str(obj[0])
        except Exception:
            return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return str(value)


def key(question: Any, answer: Any) -> tuple[str, str]:
    return normalize_text(question), normalize_text(first_answer(answer))


def main() -> None:
    args = parse_args()
    if len(args.parquets) != len(args.splits):
        raise ValueError("--parquets and --splits must have the same length")

    ann = pd.read_csv(args.annotations)
    ann = ann[ann["review_label"].isin(args.labels)].copy()
    if args.require_source_contains is not None:
        mask = ann["source_files"].fillna("").astype(str).str.contains(
            args.require_source_contains, regex=False
        )
        ann = ann[mask].copy()

    target: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in ann.iterrows():
        k = key(row["question"], row["gold_answer"])
        # Keep the first annotation if duplicate question/gold rows appear.
        target.setdefault(k, row.to_dict())

    rows: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, str]] = set()
    duplicate_matches: dict[str, int] = {}
    for parquet, split in zip(args.parquets, args.splits):
        df = pd.read_parquet(parquet)
        for row_idx, ex in df.iterrows():
            k = key(ex["query"], ex["label"])
            if k not in target:
                continue
            ann_row = target[k]
            cid = str(ann_row["candidate_id"])
            duplicate_matches[cid] = duplicate_matches.get(cid, 0) + 1
            # Keep one image row per accepted candidate.  Preference is given
            # by the order of --parquets/--splits supplied by the caller.
            if k in matched_keys:
                continue
            matched_keys.add(k)
            item = ex.to_dict()
            item["candidate_id"] = cid
            item["source_split"] = split
            item["source_row_idx"] = int(row_idx)
            item["review_label"] = ann_row.get("review_label", "")
            item["codex_confidence"] = ann_row.get("codex_confidence", "")
            item["scale_error_subtype"] = ann_row.get("scale_error_subtype", "")
            item["scale_factors"] = ann_row.get("scale_factors", "")
            item["scale_directions"] = ann_row.get("scale_directions", "")
            item["representative_predictions"] = ann_row.get("representative_predictions", "")
            rows.append(item)

    out = pd.DataFrame(rows)
    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_parquet, index=False)

    # Lightweight CSV manifest without image bytes.
    manifest_cols = [
        "candidate_id",
        "source_split",
        "source_row_idx",
        "query",
        "label",
        "scale_error_subtype",
        "scale_factors",
        "scale_directions",
        "representative_predictions",
    ]
    out[[c for c in manifest_cols if c in out.columns]].to_csv(args.output_csv, index=False)

    unmatched = sorted(set(target) - matched_keys)
    summary = {
        "annotation_rows_after_filter": int(len(ann)),
        "unique_annotation_keys": int(len(target)),
        "matched_rows": int(len(out)),
        "unmatched_keys": int(len(unmatched)),
        "source_split_counts": out["source_split"].value_counts().to_dict()
        if len(out)
        else {},
        "subtype_counts": out["scale_error_subtype"].value_counts().to_dict()
        if len(out)
        else {},
        "duplicate_match_candidate_count": int(
            sum(1 for v in duplicate_matches.values() if v > 1)
        ),
    }
    Path(args.output_summary).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
