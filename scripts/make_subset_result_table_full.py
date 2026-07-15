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


def is_numeric_single(x):
    return one_number(x) is not None


def is_factor100(a, b):
    x = one_number(a)
    y = one_number(b)
    if x is None or y is None:
        return False
    if abs(x) < 1e-12 or abs(y) < 1e-12:
        return False
    r = abs(x / y)
    return (0.008 <= r <= 0.0125) or (80 <= r <= 125)


def has_scale_keyword(q):
    q = str(q).lower()
    keys = [
        "percentage", "percent", "%", "rate", "ratio",
        "average", "mean", "median", "sum", "total",
        "difference", "greater", "less", "times",
        "value", "number", "how many", "how much",
    ]
    return any(k in q for k in keys)


eval_path = "results/eval/chartqa_real_contrastive_seeds_full.csv"
df = pd.read_csv(eval_path).drop_duplicates(["adapter", "idx"]).copy()

pred_col = "prediction" if "prediction" in df.columns else "pred"
if pred_col not in df.columns:
    pred_col = [c for c in df.columns if "pred" in c.lower()][0]

base = df[df["adapter"] == "base"].copy()
base = base[["idx", "question", "answers", pred_col, "score"]].rename(
    columns={pred_col: "base_pred", "score": "base_score"}
)

base["gold"] = base["answers"].apply(first_answer)
base["gold_numeric"] = base["gold"].apply(is_numeric_single)
base["scale_keyword"] = base["question"].apply(has_scale_keyword)
base["strict_gold_scale_error"] = [
    is_factor100(pred, gold) and int(score) == 0
    for pred, gold, score in zip(base["base_pred"], base["gold"], base["base_score"])
]

seed_names = sorted([a for a in df["adapter"].unique() if "seed" in str(a)])

merged = base.copy()

for seed in seed_names:
    s = df[df["adapter"] == seed][["idx", "score"]].rename(
        columns={"score": f"{seed}_score"}
    )
    merged = merged.merge(s, on="idx", how="left")

seed_score_cols = [f"{s}_score" for s in seed_names]
merged["ours_mean"] = merged[seed_score_cols].mean(axis=1)
merged["ours_std"] = merged[seed_score_cols].std(axis=1)

rows = []

def add_split(name, mask):
    sub = merged[mask].copy()
    if len(sub) == 0:
        return
    per_seed = [sub[c].mean() for c in seed_score_cols]
    rows.append({
        "Split": name,
        "Size": len(sub),
        "Base": sub["base_score"].mean(),
        "Ours": sum(per_seed) / len(per_seed),
        "Std": pd.Series(per_seed).std(),
        "Delta": sum(per_seed) / len(per_seed) - sub["base_score"].mean(),
    })

add_split("All examples", merged.index == merged.index)
add_split("Numeric-answer examples", merged["gold_numeric"])
add_split("Scale-keyword numeric examples", merged["gold_numeric"] & merged["scale_keyword"])
add_split("Strict gold-scale errors", merged["strict_gold_scale_error"])
add_split("Non-gold-scale examples", ~merged["strict_gold_scale_error"])

out = pd.DataFrame(rows)

print("\n=== Full ChartQA subset result table ===")
print(out.round(4).to_string(index=False))

Path("paper/tables").mkdir(parents=True, exist_ok=True)
out.to_csv("paper/tables/chartqa_full_subset_results.csv", index=False)
Path("paper/tables/chartqa_full_subset_results.tex").write_text(
    out.round(4).to_latex(index=False, escape=False)
)

print("\nSaved:")
print("paper/tables/chartqa_full_subset_results.csv")
print("paper/tables/chartqa_full_subset_results.tex")
