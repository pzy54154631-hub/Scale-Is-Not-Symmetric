#!/usr/bin/env python3
"""Codex-side annotation pass for ScaleQA-Natural expansion candidates.

The goal is to expand the reviewed ScaleQA-Natural core without polluting it
with obvious non-scale cases.  The pass is intentionally conservative: clear
single-number 100x/1-100x mismatches on percentage/value chart questions are
labeled ``yes``; year/date/count/unit-ambiguous cases are routed to review.
"""

from __future__ import annotations

import argparse
import json
import math
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


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", re.I)
PERCENT_TERMS = re.compile(r"%|percent|percentage|rate|share|proportion|ratio", re.I)
VALUE_TERMS = re.compile(
    r"value|bar|graph|chart|line|point|average|mean|median|difference|sum|total|"
    r"highest|lowest|largest|smallest|second largest|segment|data|decrease|increase",
    re.I,
)
DATE_TERMS = re.compile(r"\b(year|date|month|when|last time|time period|fiscal year)\b", re.I)
COUNT_UNIT_TERMS = re.compile(
    r"how many people|how many consumers|how many jobs|how many females|population|"
    r"people died|died from|killed|murder|income|revenue|money|sponsorship|"
    r"operating income|companies|employees|employed|forecast|demand|imports?",
    re.I,
)
MULTI_FIELD_HINTS = re.compile(r"\[|\]|\||:|;|\n|approximately|shown in the chart|includes both", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v1-review-csv", required=True)
    p.add_argument("--v2-new-csv", required=True)
    p.add_argument("--core-csv", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x).strip())


def numbers(x: Any) -> list[float]:
    out = []
    for m in NUM_RE.findall(str(x).replace(",", "").replace("−", "-")):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def first_prediction(x: Any) -> str:
    s = str(x)
    # Existing files separate representative predictions with pipes.
    return s.split("|", 1)[0].strip()


def factor100_relation(gold: Any, pred: Any) -> tuple[bool, str, float | None]:
    gs = numbers(gold)
    ps = numbers(pred)
    if len(gs) != 1 or len(ps) != 1:
        return False, "", None
    g, p = gs[0], ps[0]
    if abs(g) < 1e-12 or abs(p) < 1e-12:
        return False, "", None
    ratio = abs(p / g)
    if 80 <= ratio <= 125:
        return True, "prediction_too_large", ratio
    if 0.008 <= ratio <= 0.0125:
        return True, "prediction_too_small", 1.0 / ratio
    return False, "", ratio


def looks_like_year(gold: Any, question: str) -> bool:
    ns = numbers(gold)
    if len(ns) != 1:
        return False
    g = ns[0]
    return 1800 <= g <= 2100 and DATE_TERMS.search(question) is not None


def infer_subtype(direction: str, gold: Any, pred: Any, question: str) -> str:
    g = numbers(gold)[0]
    p = numbers(pred)[0]
    if direction == "prediction_too_large":
        if abs(g) <= 2 <= abs(p) or PERCENT_TERMS.search(question):
            return "decimal_to_percent"
        return "value_scaled_up"
    if abs(p) <= 2 <= abs(g) or PERCENT_TERMS.search(question):
        return "percent_to_decimal"
    return "value_scaled_down"


def annotate_row(row: pd.Series) -> dict[str, Any]:
    q = norm_text(row.get("question", ""))
    gold = norm_text(row.get("gold_answer", ""))
    pred = first_prediction(row.get("representative_predictions", ""))
    text = f"{q} {gold} {pred}"
    ok100, direction, factor = factor100_relation(gold, pred)
    distinct_models = int(float(row.get("distinct_models", 0) or 0))
    occurrences = int(float(row.get("occurrences", 0) or 0))

    if not ok100:
        return {
            "review_label": "not_scale",
            "codex_confidence": 0.92,
            "benchmark_action": "exclude",
            "scale_error_subtype": str(row.get("scale_error_subtype", "not_100x")),
            "codex_reason": f"Not a clean single-number 100x/1-100x mismatch (factor={factor}).",
        }

    if looks_like_year(gold, q):
        return {
            "review_label": "not_scale",
            "codex_confidence": 0.88,
            "benchmark_action": "exclude",
            "scale_error_subtype": "year_or_date_not_scale",
            "codex_reason": "Gold looks like a year/date answer; factor relation is likely accidental.",
        }

    if MULTI_FIELD_HINTS.search(str(row.get("representative_predictions", ""))) and len(numbers(pred)) != 1:
        return {
            "review_label": "needs_image",
            "codex_confidence": 0.62,
            "benchmark_action": "review",
            "scale_error_subtype": "multi_field_or_explanation",
            "codex_reason": "Representative prediction is multi-field or explanatory; image/question context should be checked.",
        }

    subtype = infer_subtype(direction, gold, pred, q)

    if PERCENT_TERMS.search(q):
        return {
            "review_label": "yes",
            "codex_confidence": 0.95,
            "benchmark_action": "include",
            "scale_error_subtype": subtype,
            "codex_reason": "Clear 100x decimal/percent-like mismatch on a percentage/rate/share question.",
        }

    if VALUE_TERMS.search(q) and not COUNT_UNIT_TERMS.search(q):
        return {
            "review_label": "yes",
            "codex_confidence": 0.91,
            "benchmark_action": "include",
            "scale_error_subtype": subtype,
            "codex_reason": "Clear single-number 100x mismatch on a chart value/aggregate question.",
        }

    if COUNT_UNIT_TERMS.search(q):
        # Multi-model recurrence is useful, but still not enough to release
        # without image/unit verification.
        return {
            "review_label": "needs_image",
            "codex_confidence": 0.66 if distinct_models >= 2 or occurrences >= 3 else 0.58,
            "benchmark_action": "review",
            "scale_error_subtype": "count_or_unit_ambiguous_100x",
            "codex_reason": "100x relation is plausible, but count/unit wording requires chart-unit verification.",
        }

    if distinct_models >= 2 and occurrences >= 3:
        return {
            "review_label": "yes",
            "codex_confidence": 0.86,
            "benchmark_action": "include",
            "scale_error_subtype": subtype,
            "codex_reason": "Clean 100x single-number mismatch recurring across multiple models; accepted conservatively.",
        }

    return {
        "review_label": "uncertain",
        "codex_confidence": 0.58,
        "benchmark_action": "review",
        "scale_error_subtype": subtype,
        "codex_reason": "Clean 100x relation, but question wording is not specific enough for automatic inclusion.",
    }


def annotate(df: pd.DataFrame, source: str) -> pd.DataFrame:
    labels = df.apply(annotate_row, axis=1, result_type="expand")
    out = df.copy()
    for c in labels.columns:
        out[c] = labels[c]
    out["annotation_source"] = source
    order = {"yes": 0, "needs_image": 1, "uncertain": 2, "not_scale": 3}
    out["_rank"] = out["review_label"].map(order).fillna(9)
    sort_cols = ["_rank", "codex_confidence"]
    ascending = [True, False]
    for c in ["distinct_models", "occurrences", "priority_score"]:
        if c in out.columns:
            sort_cols.append(c)
            ascending.append(False)
    return out.sort_values(sort_cols, ascending=ascending).drop(columns=["_rank"])


def qg_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["question"].map(lambda x: norm_text(x).lower())
        + " || "
        + df["gold_answer"].map(lambda x: norm_text(x).lower())
    )


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    if not HAS_OPENPYXL:
        return
    wb = Workbook()
    wb.remove(wb.active)
    fills = {
        "yes": "1F6F43",
        "needs_image": "8A5A00",
        "uncertain": "666666",
        "not_scale": "7A2E2E",
        "all": "1F4E79",
    }
    for name, df in sheets.items():
        ws = wb.create_sheet(name[:31])
        fill = PatternFill("solid", fgColor=fills.get(name.split("_")[0], "1F4E79"))
        for j, col in enumerate(df.columns, start=1):
            cell = ws.cell(1, j, col)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            ws.column_dimensions[get_column_letter(j)].width = {
                "question": 72,
                "representative_predictions": 48,
                "codex_reason": 74,
                "source_files": 80,
            }.get(col, 20)
        for i, (_, row) in enumerate(df.iterrows(), start=2):
            for j, col in enumerate(df.columns, start=1):
                ws.cell(i, j, row[col])
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        if "review_label" in df.columns:
            col = get_column_letter(list(df.columns).index("review_label") + 1)
            dv = DataValidation(type="list", formula1='"yes,needs_image,uncertain,not_scale,duplicate"', allow_blank=False)
            ws.add_data_validation(dv)
            dv.add(f"{col}2:{col}{max(2, len(df)+1)}")
    wb.save(path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v1 = pd.read_csv(args.v1_review_csv)
    v2 = pd.read_csv(args.v2_new_csv)
    core = pd.read_csv(args.core_csv)
    if "gold_answer" not in core.columns and "label" in core.columns:
        core["gold_answer"] = core["label"]
    if "question" not in core.columns and "query" in core.columns:
        core["question"] = core["query"]

    v1_ann = annotate(v1, "v1_review_queue_self")
    v2_ann = annotate(v2, "v2_new_strict_self")

    v1_csv = out_dir / "scaleqa_natural_v1_review_queue_CODEX_SELF_ANNOTATED.csv"
    v2_csv = out_dir / "scaleqa_natural_v2_new_strict_CODEX_SELF_ANNOTATED.csv"
    v1_xlsx = out_dir / "scaleqa_natural_v1_review_queue_CODEX_SELF_ANNOTATED.xlsx"
    v2_xlsx = out_dir / "scaleqa_natural_v2_new_strict_CODEX_SELF_ANNOTATED.xlsx"
    v1_ann.to_csv(v1_csv, index=False)
    v2_ann.to_csv(v2_csv, index=False)
    write_xlsx(v1_xlsx, {
        "yes": v1_ann[v1_ann["review_label"] == "yes"],
        "needs_image": v1_ann[v1_ann["review_label"] == "needs_image"],
        "uncertain": v1_ann[v1_ann["review_label"] == "uncertain"],
        "not_scale": v1_ann[v1_ann["review_label"] == "not_scale"],
        "all": v1_ann,
    })
    write_xlsx(v2_xlsx, {
        "yes": v2_ann[v2_ann["review_label"] == "yes"],
        "needs_image": v2_ann[v2_ann["review_label"] == "needs_image"],
        "uncertain": v2_ann[v2_ann["review_label"] == "uncertain"],
        "not_scale": v2_ann[v2_ann["review_label"] == "not_scale"],
        "all": v2_ann,
    })

    core = core.copy()
    core["annotation_source"] = "core_reviewed_yes"
    core["review_label"] = "yes"
    core["_qg"] = qg_key(core)
    existing = set(core["_qg"])

    additions = []
    for ann in [v1_ann, v2_ann]:
        yes = ann[ann["review_label"] == "yes"].copy()
        yes["_qg"] = qg_key(yes)
        yes = yes[~yes["_qg"].isin(existing)].copy()
        existing |= set(yes["_qg"])
        additions.append(yes)
    add_df = pd.concat(additions, ignore_index=True) if additions else pd.DataFrame()

    # Harmonize columns for an expanded release candidate.
    preferred = [
        "candidate_id",
        "review_label",
        "codex_confidence",
        "benchmark_action",
        "scale_error_subtype",
        "scale_factors",
        "scale_directions",
        "question",
        "gold_answer",
        "representative_predictions",
        "splits",
        "datasets",
        "occurrences",
        "distinct_models",
        "models",
        "adapters_or_models",
        "source_splits",
        "source_files",
        "codex_reason",
        "annotation_source",
    ]
    expanded = pd.concat([core.drop(columns=["_qg"], errors="ignore"), add_df.drop(columns=["_qg"], errors="ignore")], ignore_index=True, sort=False)
    cols = [c for c in preferred if c in expanded.columns] + [c for c in expanded.columns if c not in preferred]
    expanded = expanded[cols]
    expanded_csv = out_dir / "scaleqa_natural_expanded_core_403_plus_self_annotated_yes.csv"
    expanded.to_csv(expanded_csv, index=False)

    summary = {
        "core_input_n": int(len(core)),
        "v1_review_input_n": int(len(v1)),
        "v1_label_counts": {str(k): int(v) for k, v in v1_ann["review_label"].value_counts().items()},
        "v2_new_input_n": int(len(v2)),
        "v2_label_counts": {str(k): int(v) for k, v in v2_ann["review_label"].value_counts().items()},
        "new_yes_added_after_question_gold_dedup": int(len(add_df)),
        "expanded_core_n": int(len(expanded)),
        "expanded_by_source": {str(k): int(v) for k, v in expanded["annotation_source"].value_counts(dropna=False).items()},
        "outputs": {
            "v1_annotated_csv": str(v1_csv),
            "v1_annotated_xlsx": str(v1_xlsx) if HAS_OPENPYXL else None,
            "v2_annotated_csv": str(v2_csv),
            "v2_annotated_xlsx": str(v2_xlsx) if HAS_OPENPYXL else None,
            "expanded_core_csv": str(expanded_csv),
        },
        "caution": "Self-annotation is conservative but should be described as Codex-assisted unless manually spot-checked by the author.",
    }
    summary_path = out_dir / "scaleqa_natural_self_annotation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
