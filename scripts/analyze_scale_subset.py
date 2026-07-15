import re
import pandas as pd
from pathlib import Path

def nums(x):
    s = str(x).lower().replace(",", "").replace("−", "-")
    out = []
    for n in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", s):
        try:
            out.append(float(n))
        except Exception:
            pass
    return out

def one(x):
    ns = nums(x)
    return ns[0] if len(ns) == 1 else None

def factor100(a, b):
    x = one(a)
    y = one(b)
    if x is None or y is None or abs(y) < 1e-12:
        return False
    ratio = abs(x / y)
    return 80 <= ratio <= 125 or 0.008 <= ratio <= 0.0125

base_path = "results/eval/chartqa_scale_unit_verifier_logprob_500_comp_final.csv"
base = pd.read_csv(base_path).drop_duplicates("idx")

scale_idx = set()
for _, r in base.iterrows():
    if factor100(r["base_pred"], r["comp_pred"]):
        scale_idx.add(int(r["idx"]))

print("scale-conflict subset size:", len(scale_idx))

files = [
    "results/eval/chartqa_scale_contrastive_s400_500.csv",
    "results/eval/chartqa_scale_contrastive_real_s300_lr1e6_500.csv",
    "results/eval/chartqa_scale_contrastive_bal_s600_lr5e7_500.csv",
    "results/eval/chartqa_scale_contrastive_large_s300_lr5e7_500.csv",
]

rows = []

# base
b = base.copy()
b["is_scale"] = b["idx"].astype(int).isin(scale_idx)
rows.append({
    "method": "base",
    "overall": b["base_score"].mean(),
    "scale_subset": b[b["is_scale"]]["base_score"].mean(),
    "non_scale": b[~b["is_scale"]]["base_score"].mean(),
})

for f in files:
    p = Path(f)
    if not p.exists():
        continue
    df = pd.read_csv(p).drop_duplicates("idx")
    df["is_scale"] = df["idx"].astype(int).isin(scale_idx)
    method = p.stem.replace("chartqa_", "")
    rows.append({
        "method": method,
        "overall": df["score"].mean(),
        "scale_subset": df[df["is_scale"]]["score"].mean(),
        "non_scale": df[~df["is_scale"]]["score"].mean(),
    })

out = pd.DataFrame(rows)
print(out.round(4).to_string(index=False))
out.to_csv("results/eval/scale_subset_summary.csv", index=False)
print("\nSaved: results/eval/scale_subset_summary.csv")
