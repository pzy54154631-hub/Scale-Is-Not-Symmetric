import ast
import re
from pathlib import Path

import numpy as np
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


def bootstrap_delta(base_scores, ours_scores, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    base_scores = np.asarray(base_scores, dtype=float)
    ours_scores = np.asarray(ours_scores, dtype=float)
    n = len(base_scores)

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(ours_scores[idx].mean() - base_scores[idx].mean())

    deltas = np.asarray(deltas)
    return {
        "delta": float(ours_scores.mean() - base_scores.mean()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "p_delta_le_0": float((deltas <= 0).mean()),
    }


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
m = base.copy()

for seed in seed_names:
    s = df[df["adapter"] == seed][["idx", "score"]].rename(columns={"score": f"{seed}_score"})
    m = m.merge(s, on="idx", how="left")

seed_cols = [f"{s}_score" for s in seed_names]
m["ours_mean_score"] = m[seed_cols].mean(axis=1)

splits = {
    "All examples": m.index == m.index,
    "Numeric-answer examples": m["gold_numeric"],
    "Scale-keyword numeric examples": m["gold_numeric"] & m["scale_keyword"],
    "Strict gold-scale errors": m["strict_gold_scale_error"],
    "Non-gold-scale examples": ~m["strict_gold_scale_error"],
}

rows = []
for name, mask in splits.items():
    sub = m[mask].copy()
    if len(sub) == 0:
        continue

    stat = bootstrap_delta(sub["base_score"], sub["ours_mean_score"])
    rows.append({
        "split": name,
        "n": len(sub),
        "base": sub["base_score"].mean(),
        "ours": sub["ours_mean_score"].mean(),
        "delta": stat["delta"],
        "ci_low": stat["ci_low"],
        "ci_high": stat["ci_high"],
        "p_delta_le_0": stat["p_delta_le_0"],
    })

out = pd.DataFrame(rows)

print("\n=== Paired bootstrap on ChartQA-full ===")
print(out.round(5).to_string(index=False))

Path("paper/tables").mkdir(parents=True, exist_ok=True)
out.to_csv("paper/tables/bootstrap_chartqa_full.csv", index=False)
Path("paper/tables/bootstrap_chartqa_full.tex").write_text(
    out.round(5).to_latex(index=False, escape=False)
)

print("\nSaved:")
print("paper/tables/bootstrap_chartqa_full.csv")
print("paper/tables/bootstrap_chartqa_full.tex")
