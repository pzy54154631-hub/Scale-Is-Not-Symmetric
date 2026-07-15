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


def is_factor100(a, b):
    x = one_number(a)
    y = one_number(b)
    if x is None or y is None:
        return False
    if abs(x) < 1e-12 or abs(y) < 1e-12:
        return False
    r = abs(x / y)
    return (0.008 <= r <= 0.0125) or (80 <= r <= 125)


df = pd.read_csv("results/eval/chartqa_real_contrastive_seeds_full.csv")
df = df.drop_duplicates(["adapter", "idx"])

pred_col = "prediction" if "prediction" in df.columns else "pred"
if pred_col not in df.columns:
    pred_col = [c for c in df.columns if "pred" in c.lower()][0]

base = df[df["adapter"] == "base"].copy()
base = base[["idx", "question", "answers", pred_col, "score"]].rename(
    columns={pred_col: "base_pred", "score": "base_score"}
)
base["gold"] = base["answers"].apply(first_answer)
base["strict_gold_scale_error"] = [
    is_factor100(pred, gold) and int(score) == 0
    for pred, gold, score in zip(base["base_pred"], base["gold"], base["base_score"])
]

seed_names = sorted([a for a in df["adapter"].unique() if "seed" in str(a)])
m = base.copy()

for seed in seed_names:
    s = df[df["adapter"] == seed][["idx", pred_col, "score"]].rename(
        columns={pred_col: f"{seed}_pred", "score": f"{seed}_score"}
    )
    m = m.merge(s, on="idx", how="left")
    m[f"{seed}_fix"] = (m["base_score"] == 0) & (m[f"{seed}_score"] == 1)
    m[f"{seed}_break"] = (m["base_score"] == 1) & (m[f"{seed}_score"] == 0)

fix_cols = [f"{s}_fix" for s in seed_names]
break_cols = [f"{s}_break" for s in seed_names]

m["n_seed_fix"] = m[fix_cols].sum(axis=1)
m["n_seed_break"] = m[break_cols].sum(axis=1)

rows = []

def add(name, mask):
    sub = m[mask].copy()
    if len(sub) == 0:
        return
    rows.append({
        "split": name,
        "size": len(sub),
        "base_acc": sub["base_score"].mean(),
        "mean_seed_fix_count": sub["n_seed_fix"].mean(),
        "mean_seed_break_count": sub["n_seed_break"].mean(),
        "any_seed_fix_rate": (sub["n_seed_fix"] > 0).mean(),
        "all_seed_fix_rate": (sub["n_seed_fix"] == len(seed_names)).mean(),
        "any_seed_break_rate": (sub["n_seed_break"] > 0).mean(),
        "all_seed_break_rate": (sub["n_seed_break"] == len(seed_names)).mean(),
    })

add("All examples", m.index == m.index)
add("Base-wrong examples", m["base_score"] == 0)
add("Base-correct examples", m["base_score"] == 1)
add("Strict gold-scale errors", m["strict_gold_scale_error"])
add("Non-gold-scale examples", ~m["strict_gold_scale_error"])

out = pd.DataFrame(rows)

print("\n=== Seed consistency summary ===")
print(out.round(4).to_string(index=False))

Path("paper/tables").mkdir(parents=True, exist_ok=True)
out.to_csv("paper/tables/seed_consistency_full.csv", index=False)
Path("paper/tables/seed_consistency_full.tex").write_text(
    out.round(4).to_latex(index=False, escape=False)
)

# clean examples fixed by all seeds
fixed_all = m[m["n_seed_fix"] == len(seed_names)].copy()
broken_all = m[m["n_seed_break"] == len(seed_names)].copy()

cols = ["idx", "question", "gold", "base_pred", "base_score", "n_seed_fix", "n_seed_break"]
fixed_all[cols].to_csv("results/eval/fixed_by_all_seeds_full.csv", index=False)
broken_all[cols].to_csv("results/eval/broken_by_all_seeds_full.csv", index=False)

print("\nFixed by all seeds:", len(fixed_all))
print(fixed_all[cols].head(20).to_string(index=False))

print("\nBroken by all seeds:", len(broken_all))
print(broken_all[cols].head(20).to_string(index=False))

print("\nSaved:")
print("paper/tables/seed_consistency_full.csv")
print("results/eval/fixed_by_all_seeds_full.csv")
print("results/eval/broken_by_all_seeds_full.csv")
