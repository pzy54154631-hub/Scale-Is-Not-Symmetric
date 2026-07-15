#!/usr/bin/env python
"""Fine-grained ScaleQA error taxonomy for direction/regime analysis."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def norm(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).lower()


def question_pattern(q: object) -> str:
    s = norm(q)
    if any(w in s for w in ["average", "mean"]):
        return "average"
    if any(w in s for w in ["difference", "how much more", "how much less", "more than", "less than"]):
        return "difference"
    if any(w in s for w in ["ratio", "times", "divide", "a:b"]):
        return "ratio"
    if any(w in s for w in ["percent", "percentage", "%"]):
        return "percent"
    if any(w in s for w in ["highest", "largest", "maximum", "max", "greatest"]):
        return "max"
    if any(w in s for w in ["lowest", "smallest", "minimum", "min", "least"]):
        return "min"
    if any(w in s for w in ["sum", "total", "combined"]):
        return "sum"
    if any(w in s for w in ["how many", "number of", "count"]):
        return "count"
    if any(w in s for w in ["what is", "what's", "find", "value"]):
        return "lookup"
    return "other"


def visual_mention(q: object, ctx: object) -> str:
    s = norm(q) + " " + norm(ctx)
    if "pie" in s:
        return "pie-mentioned"
    if "line" in s:
        return "line-mentioned"
    if "bar" in s:
        return "bar-mentioned"
    if "legend" in s or "series" in s:
        return "legend-or-series"
    return "unspecified"


def unit_class(ctx: object, q: object = "") -> str:
    s = norm(ctx) + " " + norm(q)
    if "%" in s or "percent" in s or "percentage" in s:
        return "percent-axis-or-question"
    if any(w in s for w in ["million", "millions", "billion", "billions", "thousand", "thousands"]):
        return "large-unit-word"
    if any(w in s for w in ["ratio", "a:b", "per ", "rate"]):
        return "ratio-or-rate"
    if any(w in s for w in ["usd", "$", "dollar", "euro", "cost", "price"]):
        return "currency"
    if any(w in s for w in ["kg", "kilogram", "km", "meter", "metre", "ton", "gram"]):
        return "physical-unit"
    return "no-explicit-unit"


def num_bucket(x: object) -> str:
    nums = []
    for m in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", str(x).replace(",", "")):
        try:
            nums.append(abs(float(m)))
        except Exception:
            pass
    if not nums:
        return "no-number"
    v = nums[0]
    if v == 0:
        return "zero"
    if v < 0.01:
        return "<0.01"
    if v < 0.1:
        return "0.01-0.1"
    if v < 1:
        return "0.1-1"
    if v < 10:
        return "1-10"
    if v < 100:
        return "10-100"
    if v < 1000:
        return "100-1000"
    return ">=1000"


def summarize(df: pd.DataFrame, group_cols: list[str], out: Path):
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("candidate_id", "nunique"),
            answer_tuned_acc=("answer_tuned_score", "mean"),
            large_mean_acc=("large_mean_score", "mean"),
            self_check_acc=("self_check_score", "mean"),
            fix_rate=("large_fix", "mean"),
            break_rate=("large_break", "mean"),
            oracle_rate=("candidate_oracle_score", "mean"),
        )
        .reset_index()
        .sort_values(["n", "fix_rate"], ascending=[False, False])
    )
    agg.to_csv(out, index=False)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-large-csv", required=True)
    ap.add_argument("--tta-csv", required=True)
    ap.add_argument("--verifier-csv", required=True)
    ap.add_argument("--chart-context-csv", required=True)
    ap.add_argument("--out-dir", default="results/analysis/fine_error_taxonomy")
    ap.add_argument("--name", default="scaleqa520")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    al = pd.read_csv(args.answer_large_csv)
    # Collapse large seeds per example.
    ans = al[al["adapter"].eq("answer_tuned")].drop_duplicates("candidate_id").copy()
    large = al[al["adapter"].str.startswith("large_seed", na=False)].groupby("candidate_id")["score"].mean().rename("large_mean_score")
    ans = ans.merge(large, on="candidate_id", how="left")
    ans = ans.rename(columns={"score": "answer_tuned_score"})
    ans["large_fix"] = ((ans["answer_tuned_score"] < 0.5) & (ans["large_mean_score"] > 0.5)).astype(float)
    ans["large_break"] = ((ans["answer_tuned_score"] > 0.5) & (ans["large_mean_score"] < 0.5)).astype(float)

    tta = pd.read_csv(args.tta_csv)
    self_check = tta[tta["adapter"].eq("self_check")][["candidate_id", "score", "prediction"]].rename(
        columns={"score": "self_check_score", "prediction": "self_check_prediction"}
    )
    ans = ans.merge(self_check, on="candidate_id", how="left")

    ver = pd.read_csv(args.verifier_csv)
    oracle = ver[ver["adapter"].eq("candidate_oracle")][["candidate_id", "score"]].rename(
        columns={"score": "candidate_oracle_score"}
    )
    ans = ans.merge(oracle, on="candidate_id", how="left")

    ctx = pd.read_csv(args.chart_context_csv)
    ans = ans.merge(ctx[["candidate_id", "chart_context"]], on="candidate_id", how="left")
    ans["question_pattern"] = ans["question"].map(question_pattern)
    ans["unit_class"] = [unit_class(c, q) for c, q in zip(ans["chart_context"], ans["question"])]
    ans["visual_mention"] = [visual_mention(q, c) for q, c in zip(ans["question"], ans["chart_context"])]
    ans["gold_bucket"] = ans["answers"].map(num_bucket)
    ans["base_pred_bucket"] = ans["prediction"].map(num_bucket)

    detail = out_dir / f"{args.name}_fine_error_detail.csv"
    ans.to_csv(detail, index=False)
    outputs = {"detail": str(detail)}
    for group in [
        ["scale_error_subtype"],
        ["scale_directions"],
        ["question_pattern"],
        ["unit_class"],
        ["visual_mention"],
        ["gold_bucket"],
        ["scale_error_subtype", "question_pattern"],
        ["scale_error_subtype", "unit_class"],
        ["scale_directions", "unit_class"],
    ]:
        key = "_x_".join(group)
        path = out_dir / f"{args.name}_by_{key}.csv"
        summarize(ans, group, path)
        outputs[key] = str(path)

    summary = {
        "name": args.name,
        "n": int(ans["candidate_id"].nunique()),
        "overall": {
            "answer_tuned_acc": float(ans["answer_tuned_score"].mean()),
            "large_mean_acc": float(ans["large_mean_score"].mean()),
            "self_check_acc": float(ans["self_check_score"].mean()),
            "large_fix_rate": float(ans["large_fix"].mean()),
            "large_break_rate": float(ans["large_break"].mean()),
            "candidate_oracle": float(ans["candidate_oracle_score"].mean()),
        },
        "counts": {
            "question_pattern": ans["question_pattern"].value_counts().to_dict(),
            "unit_class": ans["unit_class"].value_counts().to_dict(),
            "visual_mention": ans["visual_mention"].value_counts().to_dict(),
        },
        "files": outputs,
    }
    summary_json = out_dir / f"{args.name}_fine_error_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = out_dir / f"{args.name}_fine_error_summary.md"
    lines = [f"# Fine error taxonomy: {args.name}", "", json.dumps(summary["overall"], indent=2), "", "## Files"]
    for k, v in outputs.items():
        lines.append(f"- {k}: {v}")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
