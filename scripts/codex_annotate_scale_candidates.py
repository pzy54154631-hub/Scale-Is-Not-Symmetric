#!/usr/bin/env python3
"""Fill scale-error review labels for a candidate workbook/CSV.

This is a conservative annotation pass intended to reduce human workload.  It
labels clear 100x/1-100x numeric scale mismatches as benchmark candidates and
routes ambiguous 10x/1000x, count, year, or nonnumeric cases to review.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
try:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    HAS_OPENPYXL = True
except ModuleNotFoundError:
    HAS_OPENPYXL = False


NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
PERCENT_TERMS = re.compile(r"%|percent|percentage|rate|share|proportion", re.I)
VALUE_TERMS = re.compile(
    r"value|bar|line|point|average|difference|sum|total|max|min|highest|lowest|largest|smallest|"
    r"rightmost|leftmost|median|ratio|employment|death rate|support|responsibility|worried|concerned",
    re.I,
)
COUNT_YEAR_TERMS = re.compile(
    r"which year|what year|how many years|how many data points|data points|time period|2nd biggest|second biggest piece|"
    r"how many points|how many females|how many people|how many consumers|how much money|operating income|sponsorship|population",
    re.I,
)
NON_SCALE_HINTS = re.compile(r"oppose\s*:|favor\s*-|;|from \d{4}|for \d+ years", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-xlsx", required=True)
    return parser.parse_args()


def extract_number(value: Any) -> float | None:
    match = NUM_RE.search(str(value).replace(",", "").replace("−", "-"))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def numeric_count(value: Any) -> int:
    return len(NUM_RE.findall(str(value).replace(",", "").replace("−", "-")))


def first_prediction(value: Any) -> str:
    text = str(value)
    return text.split("|", 1)[0].strip()


def has_only_factor(factors: str, allowed: set[str]) -> bool:
    found = {x.strip() for x in str(factors).split("|") if x.strip()}
    return bool(found) and found.issubset(allowed)


def annotate(row: pd.Series) -> tuple[str, float, str, str, str]:
    q = str(row.get("question", ""))
    gold = str(row.get("gold_answer", ""))
    preds = str(row.get("representative_predictions", ""))
    pred0 = first_prediction(preds)
    priorities = str(row.get("priorities", ""))
    factors = str(row.get("scale_factors", ""))
    directions = str(row.get("scale_directions", ""))
    g = extract_number(gold)
    p = extract_number(pred0)
    g_count = numeric_count(gold)
    p_count = numeric_count(pred0)
    pure_100 = has_only_factor(factors, {"100x", "1/100x"})
    pure_minor = has_only_factor(factors, {"10x", "1/10x", "1000x", "1/1000x"})
    is_a = "A_scale_100x_single_number" in priorities
    text = f"{q} {gold} {preds}"

    if g is None or p is None or g_count != 1:
        return (
            "not_scale",
            0.95,
            "not_benchmark",
            "nonnumeric_or_multi_gold",
            "Gold answer is nonnumeric or contains multiple numeric fields.",
        )
    if NON_SCALE_HINTS.search(text) or (p_count > 1 and not is_a):
        return (
            "not_scale",
            0.85,
            "not_benchmark",
            "multi_field_or_explanation",
            "Prediction is a structured/multi-field answer or explanatory text rather than a single scale mismatch.",
        )

    if is_a and pure_100 and p_count == 1:
        if PERCENT_TERMS.search(q):
            subtype = "decimal_to_percent" if "prediction_too_large" in directions else "percent_to_decimal"
            return (
                "yes",
                0.97,
                "include",
                subtype,
                "Clear 100x/1-100x mismatch on a percentage/rate/share question.",
            )
        if VALUE_TERMS.search(q) and not COUNT_YEAR_TERMS.search(q):
            subtype = "value_scaled_up" if "prediction_too_large" in directions else "value_scaled_down"
            return (
                "yes",
                0.90,
                "include",
                subtype,
                "Single numeric answer differs from gold by 100x/1-100x on a value/aggregate chart question.",
            )
        if COUNT_YEAR_TERMS.search(q):
            return (
                "needs_image",
                0.62,
                "review",
                "count_or_unit_ambiguous_100x",
                "100x relation exists, but the question wording suggests counts, years, money, or population; verify chart units before inclusion.",
            )
        return (
            "uncertain",
            0.65,
            "review",
            "untyped_100x",
            "100x relation exists, but wording does not clearly indicate a scale-sensitive value.",
        )

    if pure_minor:
        if PERCENT_TERMS.search(q) and p_count == 1:
            return (
                "needs_image",
                0.64,
                "review",
                "minor_factor_percentage_candidate",
                "10x/1000x relation on a percentage-like question; requires chart verification.",
            )
        return (
            "needs_image",
            0.55,
            "review",
            "minor_factor_or_unit_candidate",
            "10x/1000x relation may reflect units, missing digits, or arithmetic rather than strict scale error.",
        )

    return (
        "uncertain",
        0.50,
        "review",
        "mixed_factor_candidate",
        "Multiple factors or broad ratio evidence; keep for optional review, not automatic benchmark inclusion.",
    )


def write_sheet(ws, df: pd.DataFrame, fill_color: str) -> None:
    fill = PatternFill("solid", fgColor=fill_color)
    for col_idx, col in enumerate(df.columns, start=1):
        cell = ws.cell(1, col_idx, col)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = fill
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        for col_idx, col in enumerate(df.columns, start=1):
            ws.cell(row_idx, col_idx, row[col])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    widths = {
        "candidate_id": 16,
        "review_label": 16,
        "codex_confidence": 16,
        "benchmark_action": 18,
        "scale_error_subtype": 26,
        "question": 72,
        "gold_answer": 16,
        "representative_predictions": 46,
        "codex_reason": 70,
        "review_notes": 35,
        "source_files": 80,
        "all_models": 60,
    }
    for idx, col in enumerate(df.columns, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = widths.get(col, 20)
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    if "review_label" in df.columns:
        col_idx = list(df.columns).index("review_label") + 1
        col = get_column_letter(col_idx)
        dv = DataValidation(type="list", formula1='"yes,not_scale,uncertain,needs_image,duplicate"', allow_blank=False)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{max(2, len(df) + 1)}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    labels = df.apply(annotate, axis=1, result_type="expand")
    labels.columns = [
        "review_label",
        "codex_confidence",
        "benchmark_action",
        "scale_error_subtype",
        "codex_reason",
    ]
    # Replace empty review columns from the prefilled package.
    for col in labels.columns:
        df[col] = labels[col]
    preferred = [
        "candidate_id",
        "review_label",
        "codex_confidence",
        "benchmark_action",
        "scale_error_subtype",
        "question",
        "gold_answer",
        "representative_predictions",
        "scale_factors",
        "scale_directions",
        "priorities",
        "codex_reason",
        "review_notes",
        "suggested_label",
        "splits",
        "datasets",
        "occurrences",
        "distinct_models",
        "base_like_occurrences",
        "base_like_models",
        "all_models",
        "source_files",
        "auto_reason",
    ]
    remaining = [c for c in df.columns if c not in preferred]
    df = df[[c for c in preferred if c in df.columns] + remaining]
    label_rank = {"yes": 0, "needs_image": 1, "uncertain": 2, "not_scale": 3, "duplicate": 4}
    df["_rank"] = df["review_label"].map(label_rank).fillna(9)
    df = df.sort_values(
        ["_rank", "codex_confidence", "base_like_occurrences", "occurrences"],
        ascending=[True, False, False, False],
    ).drop(columns=["_rank"])

    out_csv = Path(args.output_csv)
    out_xlsx = Path(args.output_xlsx)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary = {
        "total_candidates": len(df),
        "label_counts": df["review_label"].value_counts().to_dict(),
        "action_counts": df["benchmark_action"].value_counts().to_dict(),
        "subtype_counts": df["scale_error_subtype"].value_counts().to_dict(),
    }
    if not HAS_OPENPYXL:
        print(summary)
        print("openpyxl is not installed; wrote CSV only:", out_csv)
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    lines = [
        ("Purpose", "Codex-filled annotation pass for natural scale-error candidates."),
        ("review_label=yes", "High-confidence benchmark candidate; still recommended to spot-check chart image before final release."),
        ("review_label=needs_image", "Numeric relation is plausible but chart units/count wording must be verified."),
        ("review_label=uncertain", "Keep for optional second pass."),
        ("review_label=not_scale", "Exclude from benchmark."),
        ("benchmark_action=include", "Candidate can be used for the ScaleQA-Natural draft after spot-checking."),
        ("Caution", "This pass uses question/gold/prediction evidence; it does not visually inspect every chart."),
    ]
    for r, (k, v) in enumerate(lines, start=1):
        ws.cell(r, 1, k).font = Font(bold=True)
        ws.cell(r, 2, v).alignment = Alignment(wrap_text=True)
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 110

    sheets = [
        ("include_yes", df[df["review_label"] == "yes"], "1F6F43"),
        ("needs_image", df[df["review_label"] == "needs_image"], "8A5A00"),
        ("uncertain", df[df["review_label"] == "uncertain"], "666666"),
        ("not_scale", df[df["review_label"] == "not_scale"], "7A2E2E"),
        ("all_annotated", df, "1F4E79"),
    ]
    for name, sub, color in sheets:
        ws_sub = wb.create_sheet(name)
        write_sheet(ws_sub, sub, color)

    ws_sum = wb.create_sheet("summary")
    row = 1
    for key, val in summary.items():
        ws_sum.cell(row, 1, key).font = Font(bold=True)
        ws_sum.cell(row, 2, str(val))
        row += 1
    ws_sum.column_dimensions["A"].width = 26
    ws_sum.column_dimensions["B"].width = 120

    wb.save(out_xlsx)
    print(summary)
    print("saved", out_csv)
    print("saved", out_xlsx)


if __name__ == "__main__":
    main()
