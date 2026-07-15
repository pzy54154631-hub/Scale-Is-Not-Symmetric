import ast
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

def first_answer(ans):
    try:
        obj = ast.literal_eval(str(ans))
        if isinstance(obj, list) and obj:
            return str(obj[0])
    except Exception:
        pass
    return str(ans)

def one(x):
    ns = nums(x)
    return ns[0] if len(ns) == 1 else None

def is_factor100(a, b):
    x, y = one(a), one(b)
    if x is None or y is None or abs(y) < 1e-12:
        return False
    r = abs(x / y)
    return (0.008 <= r <= 0.0125) or (80 <= r <= 125)

df = pd.read_csv("results/eval/chartqa_real_contrastive_seeds_1500.csv")
df = df.drop_duplicates(["adapter", "idx"])

pred_col = "prediction" if "prediction" in df.columns else "pred"
if pred_col not in df.columns:
    pred_col = [c for c in df.columns if "pred" in c.lower()][0]

base = df[df["adapter"] == "base"].copy()
ours = df[df["adapter"] == "real_seed5"].copy()

m = base[["idx", "question", "answers", pred_col, "score"]].rename(
    columns={pred_col: "base_pred", "score": "base_score"}
).merge(
    ours[["idx", pred_col, "score"]].rename(
        columns={pred_col: "ours_pred", "score": "ours_score"}
    ),
    on="idx"
)

m["gold"] = m["answers"].apply(first_answer)

# Base wrong because base is 100x away from gold; ours correct.
scale_wins = m[
    (m["base_score"] == 0)
    & (m["ours_score"] == 1)
    & m.apply(lambda r: is_factor100(r["base_pred"], r["gold"]), axis=1)
].copy()

# Base correct; ours becomes 100x away from gold.
scale_losses = m[
    (m["base_score"] == 1)
    & (m["ours_score"] == 0)
    & m.apply(lambda r: is_factor100(r["ours_pred"], r["gold"]), axis=1)
].copy()

cols = ["idx", "question", "gold", "base_pred", "ours_pred", "base_score", "ours_score"]

print("\n=== Clean scale wins ===")
print(scale_wins[cols].to_string(index=False))

print("\n=== Scale over-correction losses ===")
print(scale_losses[cols].to_string(index=False))

scale_wins[cols].to_csv("results/eval/paper_scale_wins.csv", index=False)
scale_losses[cols].to_csv("results/eval/paper_scale_losses.csv", index=False)

# LaTeX friendly small table
case_rows = pd.concat([
    scale_wins[cols].head(4).assign(type="Recovered scale error"),
    scale_losses[cols].head(2).assign(type="Over-correction failure"),
])

case_rows = case_rows[["type", "idx", "question", "gold", "base_pred", "ours_pred"]]
case_rows.to_csv("results/eval/paper_case_table.csv", index=False)
Path("results/eval/paper_case_table.tex").write_text(
    case_rows.to_latex(index=False, escape=False)
)

print("\nSaved:")
print("results/eval/paper_scale_wins.csv")
print("results/eval/paper_scale_losses.csv")
print("results/eval/paper_case_table.csv")
print("results/eval/paper_case_table.tex")
