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


def is_gold_scale_error(pred, gold):
    p = one_number(pred)
    g = one_number(gold)
    if p is None or g is None:
        return False
    if abs(p) < 1e-12 or abs(g) < 1e-12:
        return False

    r = abs(p / g)
    return (0.008 <= r <= 0.0125) or (80 <= r <= 125)


base_path = "results/eval/chartqa_scale_unit_verifier_logprob_500_comp_final.csv"
base = pd.read_csv(base_path).drop_duplicates("idx").copy()
base["gold"] = base["answers"].apply(first_answer)
base["gold_scale_error"] = [
    is_gold_scale_error(pred, gold) and int(score) == 0
    for pred, gold, score in zip(base["base_pred"], base["gold"], base["base_score"])
]
scale_idx = set(base.loc[base["gold_scale_error"], "idx"].astype(int))

rows = []


def add_base():
    tmp = base[["idx", "base_score"]].rename(columns={"base_score": "score"}).copy()
    tmp["idx"] = tmp["idx"].astype(int)
    tmp["is_gold_scale_error"] = tmp["idx"].isin(scale_idx)

    rows.append({
        "method": "Base",
        "overall": tmp["score"].mean(),
        "std": None,
        "gold_scale_error": tmp[tmp["is_gold_scale_error"]]["score"].mean(),
        "non_scale": tmp[~tmp["is_gold_scale_error"]]["score"].mean(),
        "n_runs": 1,
        "note": "original base adapter",
    })


def summarize_single(method, path, note=""):
    p = Path(path)
    if not p.exists():
        print(f"[missing] {method}: {path}")
        return

    df = pd.read_csv(p).copy()
    if "adapter" in df.columns:
        df = df.drop_duplicates(["adapter", "idx"])
    else:
        df = df.drop_duplicates("idx")

    if "score" not in df.columns:
        print(f"[skip] no score column: {path}")
        return

    df["idx"] = df["idx"].astype(int)
    df["is_gold_scale_error"] = df["idx"].isin(scale_idx)

    rows.append({
        "method": method,
        "overall": df["score"].mean(),
        "std": None,
        "gold_scale_error": df[df["is_gold_scale_error"]]["score"].mean(),
        "non_scale": df[~df["is_gold_scale_error"]]["score"].mean(),
        "n_runs": 1,
        "note": note,
    })


def summarize_seed_group(method, pattern, note=""):
    files = sorted(Path("results/eval").glob(pattern))
    if not files:
        print(f"[missing seed group] {method}: {pattern}")
        return

    per_run = []
    for p in files:
        df = pd.read_csv(p).copy()
        if "adapter" in df.columns:
            df = df.drop_duplicates(["adapter", "idx"])
        else:
            df = df.drop_duplicates("idx")

        df["idx"] = df["idx"].astype(int)
        df["is_gold_scale_error"] = df["idx"].isin(scale_idx)

        per_run.append({
            "file": p.name,
            "overall": df["score"].mean(),
            "gold_scale_error": df[df["is_gold_scale_error"]]["score"].mean(),
            "non_scale": df[~df["is_gold_scale_error"]]["score"].mean(),
        })

    r = pd.DataFrame(per_run)

    rows.append({
        "method": method,
        "overall": r["overall"].mean(),
        "std": r["overall"].std(),
        "gold_scale_error": r["gold_scale_error"].mean(),
        "non_scale": r["non_scale"].mean(),
        "n_runs": len(files),
        "note": note,
    })


add_base()

summarize_single(
    "Scale-aware prompt",
    "results/eval/chartqa_scaleprompt_base_500.csv",
    "prompt-only baseline",
)

summarize_single(
    "All contrastive",
    "results/eval/chartqa_scale_contrastive_s400_500.csv",
    "mixed contrastive pairs",
)

summarize_single(
    "Real-only contrastive",
    "results/eval/chartqa_scale_contrastive_real_s300_lr1e6_500.csv",
    "single run",
)

summarize_seed_group(
    "Real-only contrastive, 5 seeds",
    "chartqa_scale_contrastive_real_s300_lr1e6_seed*_500.csv",
    "main method",
)

summarize_single(
    "Balanced contrastive",
    "results/eval/chartqa_scale_contrastive_bal_s600_lr5e7_500.csv",
    "balanced pairs",
)

summarize_single(
    "Large-only contrastive",
    "results/eval/chartqa_scale_contrastive_large_s300_lr5e7_500.csv",
    "large-only pairs",
)

# direct SFT candidates, skip automatically if missing
direct_sft_files = [
    ("Direct SFT s800", "results/eval/chartqa_scale_final_s800_500.csv"),
    ("Direct SFT s1500", "results/eval/chartqa_scale_final_s1500_500.csv"),
    ("Direct SFT s800 alt", "results/eval/chartqa_scale_final_s800.csv"),
    ("Direct SFT s1500 alt", "results/eval/chartqa_scale_final_s1500.csv"),
]

for name, path in direct_sft_files:
    if Path(path).exists():
        summarize_single(name, path, "direct supervised fine-tuning")

out = pd.DataFrame(rows)

base_overall = out[out["method"] == "Base"]["overall"].iloc[0]
base_scale = out[out["method"] == "Base"]["gold_scale_error"].iloc[0]
base_non = out[out["method"] == "Base"]["non_scale"].iloc[0]

out["overall_delta"] = out["overall"] - base_overall
out["gold_scale_delta"] = out["gold_scale_error"] - base_scale
out["non_scale_delta"] = out["non_scale"] - base_non

cols = [
    "method",
    "overall",
    "std",
    "overall_delta",
    "gold_scale_error",
    "gold_scale_delta",
    "non_scale",
    "non_scale_delta",
    "n_runs",
    "note",
]

out = out[cols]

print("\n=== Ablation Table on ChartQA-500 ===")
print(out.round(4).to_string(index=False))

Path("results/paper_tables").mkdir(parents=True, exist_ok=True)
out.to_csv("results/paper_tables/ablation_chartqa500.csv", index=False)
Path("results/paper_tables/ablation_chartqa500.tex").write_text(
    out.round(4).to_latex(index=False, escape=False)
)

print("\nSaved:")
print("results/paper_tables/ablation_chartqa500.csv")
print("results/paper_tables/ablation_chartqa500.tex")
