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

    # pred = 0.61, gold = 61
    # pred = 61, gold = 0.61
    return (0.008 <= ratio <= 0.0125) or (80 <= ratio <= 125)


base_path = "results/eval/chartqa_scale_unit_verifier_logprob_500_comp_final.csv"
base = pd.read_csv(base_path).drop_duplicates("idx").copy()

base["gold"] = base["answers"].apply(first_answer)
base["gold_scale_error"] = [
    is_gold_scale_error(pred, gold) and int(score) == 0
    for pred, gold, score in zip(base["base_pred"], base["gold"], base["base_score"])
]

scale_idx = set(base.loc[base["gold_scale_error"], "idx"].astype(int))

print("gold-scale-error subset size:", len(scale_idx))
print("idx:", sorted(scale_idx))

rows = []


def summarize(method, idx_series, score_series):
    tmp = pd.DataFrame({
        "idx": idx_series.astype(int).values,
        "score": pd.to_numeric(score_series, errors="coerce").values,
    }).drop_duplicates("idx")

    tmp["is_gold_scale_error"] = tmp["idx"].isin(scale_idx)

    overall = float(tmp["score"].mean())
    scale_score = float(tmp.loc[tmp["is_gold_scale_error"], "score"].mean()) if len(scale_idx) else float("nan")
    nonscale_score = float(tmp.loc[~tmp["is_gold_scale_error"], "score"].mean())

    rows.append({
        "method": method,
        "overall": overall,
        "gold_scale_error_subset": scale_score,
        "non_gold_scale_error": nonscale_score,
        "gold_scale_n": int(tmp["is_gold_scale_error"].sum()),
        "non_gold_scale_n": int((~tmp["is_gold_scale_error"]).sum()),
    })


# base
summarize(
    "base",
    base["idx"],
    base["base_score"],
)

# methods
files = [
    "results/eval/chartqa_scale_contrastive_s400_500.csv",
    "results/eval/chartqa_scale_contrastive_real_s300_lr1e6_500.csv",
    "results/eval/chartqa_scale_contrastive_bal_s600_lr5e7_500.csv",
    "results/eval/chartqa_scale_contrastive_large_s300_lr5e7_500.csv",
]

files += [
    str(p)
    for p in sorted(Path("results/eval").glob("chartqa_scale_contrastive_real_s300_lr1e6_seed*_500.csv"))
]

for f in files:
    p = Path(f)
    if not p.exists():
        continue

    df = pd.read_csv(p).drop_duplicates("idx").copy()
    method = p.stem.replace("chartqa_", "")

    summarize(
        method,
        df["idx"],
        df["score"],
    )

out = pd.DataFrame(rows)

# 强制转成 numeric，防止混入字符串
for col in ["overall", "gold_scale_error_subset", "non_gold_scale_error"]:
    out[col] = pd.to_numeric(out[col], errors="coerce")

print("\n=== Clean gold-scale-error subset summary ===")
print(out.round(4).to_string(index=False))

out.to_csv("results/eval/gold_scale_error_subset_summary_clean.csv", index=False)
print("\nSaved: results/eval/gold_scale_error_subset_summary_clean.csv")

seed = out[out["method"].str.contains("seed", na=False)].copy()

if len(seed):
    print("\n=== Seed mean ===")
    for col in ["overall", "gold_scale_error_subset", "non_gold_scale_error"]:
        print(
            col,
            "mean =",
            round(float(seed[col].mean()), 4),
            "std =",
            round(float(seed[col].std()), 4),
        )

    base_row = out[out["method"] == "base"].iloc[0]

    print("\n=== Mean delta over base ===")
    print("overall_delta =", round(float(seed["overall"].mean() - base_row["overall"]), 4))
    print("gold_scale_delta =", round(float(seed["gold_scale_error_subset"].mean() - base_row["gold_scale_error_subset"]), 4))
    print("non_scale_delta =", round(float(seed["non_gold_scale_error"].mean() - base_row["non_gold_scale_error"]), 4))
