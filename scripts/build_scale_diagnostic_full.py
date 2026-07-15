import ast
import re
from pathlib import Path

import pandas as pd


def nums(x):
    s = str(x).lower().replace(",", "").replace("−", "-")
    out = []
    for n in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", s):
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def first_answer(ans):
    try:
        obj = ast.literal_eval(str(ans))
        if isinstance(obj, list) and obj:
            return str(obj[0])
    except Exception:
        pass
    return str(ans)


def one_number(x):
    ns = nums(x)
    return ns[0] if len(ns) == 1 else None


def factor100(a, b):
    x = one_number(a)
    y = one_number(b)
    if x is None or y is None:
        return False
    if abs(x) < 1e-12 or abs(y) < 1e-12:
        return False
    r = abs(x / y)
    return (0.008 <= r <= 0.0125) or (80 <= r <= 125)


def numeric_single(x):
    return one_number(x) is not None


def has_scale_keyword(q):
    q = str(q).lower()
    keys = [
        "percentage", "percent", "%", "rate", "ratio",
        "average", "mean", "median", "sum", "total",
        "difference", "greater", "less", "times",
        "value", "number", "how many", "how much",
    ]
    return any(k in q for k in keys)


path = "results/eval/chartqa_real_contrastive_seeds_full.csv"
if not Path(path).exists():
    path = "results/eval/chartqa_real_contrastive_seeds_1500.csv"

df = pd.read_csv(path).drop_duplicates(["adapter", "idx"]).copy()

pred_col = "prediction" if "prediction" in df.columns else "pred"
if pred_col not in df.columns:
    pred_col = [c for c in df.columns if "pred" in c.lower()][0]

base = df[df["adapter"] == "base"].copy()
base = base[["idx", "question", "answers", pred_col, "score"]].rename(
    columns={pred_col: "base_pred", "score": "base_score"}
)

base["gold"] = base["answers"].apply(first_answer)
base["gold_numeric"] = base["gold"].apply(numeric_single)
base["base_numeric"] = base["base_pred"].apply(numeric_single)
base["scale_keyword"] = base["question"].apply(has_scale_keyword)
base["gold_scale_error"] = [
    factor100(p, g) and int(s) == 0
    for p, g, s in zip(base["base_pred"], base["gold"], base["base_score"])
]

seed_names = sorted([a for a in df["adapter"].unique() if "seed" in str(a)])
merged = base.copy()

for seed in seed_names:
    s = df[df["adapter"] == seed][["idx", pred_col, "score"]].rename(
        columns={pred_col: f"{seed}_pred", "score": f"{seed}_score"}
    )
    merged = merged.merge(s, on="idx", how="left")
    merged[f"{seed}_factor100_with_base"] = [
        factor100(a, b)
        for a, b in zip(merged[f"{seed}_pred"], merged["base_pred"])
    ]
    merged[f"{seed}_fixes_base"] = (
        (merged["base_score"] == 0) & (merged[f"{seed}_score"] == 1)
    )
    merged[f"{seed}_breaks_base"] = (
        (merged["base_score"] == 1) & (merged[f"{seed}_score"] == 0)
    )

factor_cols = [f"{s}_factor100_with_base" for s in seed_names]
fix_cols = [f"{s}_fixes_base" for s in seed_names]
break_cols = [f"{s}_breaks_base" for s in seed_names]

merged["num_seed_factor100_disagree"] = merged[factor_cols].sum(axis=1) if factor_cols else 0
merged["num_seed_fixes"] = merged[fix_cols].sum(axis=1) if fix_cols else 0
merged["num_seed_breaks"] = merged[break_cols].sum(axis=1) if break_cols else 0

# 更大的 diagnostic split：数字题 + scale keyword + 至少一个 seed 和 base 出现 100x disagreement
merged["scale_conflict_diagnostic"] = (
    merged["gold_numeric"]
    & merged["scale_keyword"]
    & (merged["num_seed_factor100_disagree"] > 0)
)

# 更强版本：base 本来错，至少一个 seed 修回来
merged["recoverable_scale_conflict"] = (
    merged["scale_conflict_diagnostic"]
    & (merged["base_score"] == 0)
    & (merged["num_seed_fixes"] > 0)
)

# over-correction：base 本来对，至少一个 seed 弄错，且和 base 100x disagreement
merged["overcorrection_risk"] = (
    merged["scale_conflict_diagnostic"]
    & (merged["base_score"] == 1)
    & (merged["num_seed_breaks"] > 0)
)

Path("results/diagnostics").mkdir(parents=True, exist_ok=True)
merged.to_csv("results/diagnostics/chartqa_full_scale_diagnostic.csv", index=False)

summary = {
    "n_total": len(merged),
    "n_gold_numeric": int(merged["gold_numeric"].sum()),
    "n_scale_keyword": int(merged["scale_keyword"].sum()),
    "n_strict_gold_scale_error": int(merged["gold_scale_error"].sum()),
    "n_scale_conflict_diagnostic": int(merged["scale_conflict_diagnostic"].sum()),
    "n_recoverable_scale_conflict": int(merged["recoverable_scale_conflict"].sum()),
    "n_overcorrection_risk": int(merged["overcorrection_risk"].sum()),
}

print("\n=== Scale Diagnostic Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

print("\n=== Strict gold-scale errors ===")
print(
    merged[merged["gold_scale_error"]][
        ["idx", "question", "gold", "base_pred", "base_score", "num_seed_fixes"]
    ].head(30).to_string(index=False)
)

print("\n=== Recoverable scale conflicts ===")
print(
    merged[merged["recoverable_scale_conflict"]][
        ["idx", "question", "gold", "base_pred", "base_score", "num_seed_fixes"]
    ].head(30).to_string(index=False)
)

print("\n=== Over-correction risks ===")
print(
    merged[merged["overcorrection_risk"]][
        ["idx", "question", "gold", "base_pred", "base_score", "num_seed_breaks"]
    ].head(30).to_string(index=False)
)

print("\nSaved: results/diagnostics/chartqa_full_scale_diagnostic.csv")
