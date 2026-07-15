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
    s = str(ans)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and obj:
            return str(obj[0])
    except Exception:
        pass
    return s

def one_number(x):
    ns = nums(x)
    if len(ns) != 1:
        return None
    return ns[0]

def is_gold_scale_error(pred, gold):
    p = one_number(pred)
    g = one_number(gold)
    if p is None or g is None:
        return False
    if abs(p) < 1e-12 or abs(g) < 1e-12:
        return False

    ratio = abs(p / g)

    # pred = 0.61, gold = 61  OR  pred = 61, gold = 0.61
    return (0.008 <= ratio <= 0.0125) or (80 <= ratio <= 125)

base_path = "results/eval/chartqa_scale_unit_verifier_logprob_500_comp_final.csv"
base = pd.read_csv(base_path).drop_duplicates("idx").copy()
base["gold"] = base["answers"].apply(first_answer)

# base 本身犯了 100x scale 错误的样本
base["gold_scale_error"] = [
    is_gold_scale_error(p, g) and int(s) == 0
    for p, g, s in zip(base["base_pred"], base["gold"], base["base_score"])
]

scale_idx = set(base.loc[base["gold_scale_error"], "idx"].astype(int))

print("gold-scale-error subset size:", len(scale_idx))
print("idx:", sorted(scale_idx))

rows = []

def add_row(method, df, score_col):
    df = df.drop_duplicates("idx").copy()
    df["is_gold_scale_error"] = df["idx"].astype(int).isin(scale_idx)

    rows.append({
        "method": method,
        "overall": df[score_col].mean(),
        "gold_scale_error_subset": df[df["is_gold_scale_error"]][score_col].mean() if len(scale_idx) else None,
        "non_gold_scale_error": df[~df["is_gold_scale_error"]][score_col].mean(),
    })

add_row("base", base.rename(columns={"base_score": "score"}), "score")

files = [
    "results/eval/chartqa_scale_contrastive_s400_500.csv",
    "results/eval/chartqa_scale_contrastive_real_s300_lr1e6_500.csv",
]

files += [str(p) for p in sorted(Path("results/eval").glob("chartqa_scale_contrastive_real_s300_lr1e6_seed*_500.csv"))]

for f in files:
    p = Path(f)
    if not p.exists():
        continue
    df = pd.read_csv(p)
    method = p.stem.replace("chartqa_", "")
    add_row(method, df, "score")

out = pd.DataFrame(rows)
print("\n=== Gold scale error subset summary ===")
print(out.round(4).to_string(index=False))

out.to_csv("results/eval/gold_scale_error_subset_summary.csv", index=False)
print("\nSaved: results/eval/gold_scale_error_subset_summary.csv")
