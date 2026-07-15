#!/usr/bin/env python3
"""Summarize ScaleQA-Natural evaluation CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def summarize_file(path: Path) -> list[dict[str, object]]:
    df = pd.read_csv(path)
    if "adapter" not in df.columns:
        df["adapter"] = path.stem
    if "prompt_mode" not in df.columns:
        df["prompt_mode"] = "unknown"
    rows = []
    for (adapter, prompt_mode), grp in df.groupby(["adapter", "prompt_mode"], dropna=False):
        rows.append(
            {
                "file": str(path),
                "adapter": str(adapter),
                "prompt_mode": str(prompt_mode),
                "n": int(len(grp)),
                "accuracy": float(grp["score"].astype(int).mean()) if len(grp) else 0.0,
            }
        )
        if "scale_error_subtype" in grp.columns:
            for subtype, sub in grp.groupby("scale_error_subtype", dropna=False):
                rows.append(
                    {
                        "file": str(path),
                        "adapter": str(adapter),
                        "prompt_mode": str(prompt_mode),
                        "subset": str(subtype),
                        "n": int(len(sub)),
                        "accuracy": float(sub["score"].astype(int).mean()) if len(sub) else 0.0,
                    }
                )
    return rows


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return ""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(format(value, floatfmt))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for item in args.inputs:
        path = Path(item)
        if path.exists() and path.stat().st_size > 0:
            rows.extend(summarize_file(path))

    overall = [r for r in rows if "subset" not in r]
    by_subset = [r for r in rows if "subset" in r]
    payload = {"overall": overall, "by_subset": by_subset}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# ScaleQA-Natural Results", "", "## Overall", ""]
    if overall:
        odf = pd.DataFrame(overall)
        lines.append(markdown_table(odf))
    lines.extend(["", "## By Subtype", ""])
    if by_subset:
        sdf = pd.DataFrame(by_subset)
        lines.append(markdown_table(sdf))
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
