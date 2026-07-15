#!/usr/bin/env python3
"""Build compact summary tables for scale-stress supplementary experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path


OUT_DIR = Path("results/scaleqa_stress")

DISPLAY_NAMES = {
    "base": "Qwen2.5-VL-7B base",
    "large_k300_seed3": "Qwen2.5-VL-7B large-pos adapter",
    "small_k300_seed1": "Qwen2.5-VL-7B small-pos adapter",
    "qwen25vl3b_base": "Qwen2.5-VL-3B base",
    "internvl25_1b": "InternVL2.5-1B base",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def read_forced(path: Path, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row.get("adapter") or row.get("model") or label
            if method in {"base", "qwen25vl3b_base", "internvl25_1b"}:
                method = label
            method = DISPLAY_NAMES.get(str(method), method)
            n = int(row["n"])
            value = float(row["accuracy"])
            correct = round(value * n)
            lo, hi = wilson(correct, n)
            rows.append(
                {
                    "benchmark": "synthetic_forced_choice",
                    "model_or_method": method,
                    "split": row["split"],
                    "n": n,
                    "correct": correct,
                    "metric": "accuracy",
                    "value": value,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return rows


def read_freeform(path: Path, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] not in {"all", "decimal", "percent"}:
                continue
            method = row.get("adapter") or row.get("model") or label
            if method == "base":
                method = label
            method = DISPLAY_NAMES.get(str(method), method)
            n = int(row["n"])
            gold = int(row["gold_match"])
            scaled = int(row["scaled_negative_match"])
            gold_lo, gold_hi = wilson(gold, n)
            scaled_lo, scaled_hi = wilson(scaled, n)
            rows.append(
                {
                    "benchmark": "synthetic_free_form",
                    "model_or_method": method,
                    "split": row["split"],
                    "n": n,
                    "gold_match": gold,
                    "scaled_negative_match": scaled,
                    "other": int(row["other"]),
                    "gold_match_rate": float(row["gold_match_rate"]),
                    "gold_ci95_low": gold_lo,
                    "gold_ci95_high": gold_hi,
                    "scaled_negative_match_rate": float(row["scaled_negative_match_rate"]),
                    "scaled_neg_ci95_low": scaled_lo,
                    "scaled_neg_ci95_high": scaled_hi,
                    "other_rate": float(row["other_rate"]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_forced_latex(rows: list[dict[str, object]]) -> None:
    all_rows = [row for row in rows if row["split"] == "all"]
    path = Path("paper/tables/synthetic_forced_choice_cross_backbone.tex")
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model / method & $N$ & Acc. & 95\\% CI low & 95\\% CI high \\\\\n")
        f.write("\\midrule\n")
        for row in all_rows:
            f.write(
                f"{row['model_or_method']} & {row['n']} & {float(row['value']):.3f} "
                f"& {float(row['ci95_low']):.3f} & {float(row['ci95_high']):.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def read_detailed_forced(path: Path, benchmark: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["benchmark"] = benchmark
        row["method"] = DISPLAY_NAMES.get(row.get("adapter") or row.get("model", ""), row.get("adapter") or row.get("model", ""))
    return rows


def aggregate_detailed_forced(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        method = row["method"]
        for split in ("all", row.get("pair_direction", ""), row.get("scale_type", "")):
            if split:
                buckets.setdefault((method, split), []).append(int(row["score"]))

    out: list[dict[str, object]] = []
    for (method, split), scores in sorted(buckets.items()):
        n = len(scores)
        correct = sum(scores)
        lo, hi = wilson(correct, n)
        out.append(
            {
                "benchmark": "chartqa_scale_stress_forced_choice",
                "model_or_method": method,
                "split": split,
                "n": n,
                "correct": correct,
                "metric": "accuracy",
                "value": correct / n if n else 0.0,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    return out


def write_chartqa_forced_latex(rows: list[dict[str, object]]) -> None:
    methods = []
    by_key = {}
    for row in rows:
        method = row["model_or_method"]
        if method not in methods:
            methods.append(method)
        by_key[(method, row["split"])] = row

    preferred = [
        "Qwen2.5-VL-7B base",
        "Qwen2.5-VL-7B large-pos adapter",
        "Qwen2.5-VL-3B base",
        "InternVL2.5-1B base",
    ]
    methods = [m for m in preferred if m in methods] + [m for m in methods if m not in preferred]
    path = Path("paper/tables/chartqa_scale_stress_forced_choice.tex")
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Model / method & $N$ & All & Large-pos. & Small-pos. \\\\\n")
        f.write("\\midrule\n")
        for method in methods:
            all_row = by_key.get((method, "all"))
            large_row = by_key.get((method, "large_positive"))
            small_row = by_key.get((method, "small_positive"))
            if not all_row:
                continue
            f.write(
                f"{method} & {all_row['n']} & {float(all_row['value']):.3f} "
                f"& {float(large_row['value']):.3f} & {float(small_row['value']):.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def write_freeform_latex(rows: list[dict[str, object]]) -> None:
    all_rows = [row for row in rows if row["split"] == "all"]
    path = Path("paper/tables/synthetic_freeform_cross_backbone.tex")
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        f.write("Backbone & $N$ & Gold & Scaled neg. & Other \\\\\n")
        f.write("\\midrule\n")
        for row in all_rows:
            f.write(
                f"{row['model_or_method']} & {row['n']} & {float(row['gold_match_rate']):.3f} "
                f"& {float(row['scaled_negative_match_rate']):.3f} & {float(row['other_rate']):.3f} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    forced: list[dict[str, object]] = []
    forced.extend(
        read_forced(
            OUT_DIR / "scale_synth_forced_choice_1000_base_seed3.csv.summary.csv",
            "Qwen2.5-VL-7B base",
        )
    )
    forced.extend(
        read_forced(
            OUT_DIR / "scale_synth_forced_choice_1000_small_seed1.csv.summary.csv",
            "Qwen2.5-VL-7B small-pos adapter",
        )
    )
    forced.extend(
        read_forced(
            OUT_DIR / "scale_synth_forced_choice_1000_qwen25vl3b_base.csv.summary.csv",
            "Qwen2.5-VL-3B base",
        )
    )
    forced.extend(
        read_forced(
            OUT_DIR / "scale_synth_forced_choice_1000_internvl25_1b.csv.summary.csv",
            "InternVL2.5-1B base",
        )
    )
    dedup: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()
    for row in forced:
        key = (row["benchmark"], row["model_or_method"], row["split"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
    forced = dedup

    write_csv(
        OUT_DIR / "synthetic_forced_choice_cross_backbone_summary.csv",
        forced,
        [
            "benchmark",
            "model_or_method",
            "split",
            "n",
            "correct",
            "metric",
            "value",
            "ci95_low",
            "ci95_high",
        ],
    )
    write_forced_latex(forced)

    freeform: list[dict[str, object]] = []
    freeform.extend(
        read_freeform(
            OUT_DIR / "scale_synth_freeform_1000_base_large_seed3.csv.summary.csv",
            "Qwen2.5-VL-7B base",
        )
    )
    freeform.extend(
        read_freeform(
            OUT_DIR / "scale_synth_freeform_1000_qwen25vl3b_base.csv.summary.csv",
            "Qwen2.5-VL-3B base",
        )
    )
    freeform.extend(
        read_freeform(
            OUT_DIR / "scale_synth_freeform_1000_internvl25_1b.csv.summary.csv",
            "InternVL2.5-1B base",
        )
    )
    write_csv(
        OUT_DIR / "synthetic_freeform_cross_backbone_summary.csv",
        freeform,
        [
            "benchmark",
            "model_or_method",
            "split",
            "n",
            "gold_match",
            "scaled_negative_match",
            "other",
            "gold_match_rate",
            "gold_ci95_low",
            "gold_ci95_high",
            "scaled_negative_match_rate",
            "scaled_neg_ci95_low",
            "scaled_neg_ci95_high",
            "other_rate",
        ],
    )
    write_freeform_latex(freeform)

    chartqa_forced_raw: list[dict[str, str]] = []
    chartqa_forced_raw.extend(
        read_detailed_forced(
            OUT_DIR / "chartqa_val_images_1000_forced_choice_qwen7_base_large.csv",
            "chartqa_scale_stress_forced_choice",
        )
    )
    chartqa_forced_raw.extend(
        read_detailed_forced(
            OUT_DIR / "chartqa_val_images_1000_forced_choice_qwen3_base.csv",
            "chartqa_scale_stress_forced_choice",
        )
    )
    chartqa_forced_raw.extend(
        read_detailed_forced(
            OUT_DIR / "chartqa_val_images_1000_forced_choice_internvl25_1b.csv",
            "chartqa_scale_stress_forced_choice",
        )
    )
    chartqa_forced = aggregate_detailed_forced(chartqa_forced_raw)
    write_csv(
        OUT_DIR / "chartqa_scale_stress_forced_choice_cross_backbone_summary.csv",
        chartqa_forced,
        [
            "benchmark",
            "model_or_method",
            "split",
            "n",
            "correct",
            "metric",
            "value",
            "ci95_low",
            "ci95_high",
        ],
    )
    write_chartqa_forced_latex(chartqa_forced)

    print("FORCED")
    for row in forced:
        if row["split"] == "all":
            print(row)
    print("FREEFORM")
    for row in freeform:
        if row["split"] == "all":
            print(row)
    print("CHARTQA FORCED")
    for row in chartqa_forced:
        if row["split"] == "all":
            print(row)


if __name__ == "__main__":
    main()
