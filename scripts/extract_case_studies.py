import pandas as pd
from pathlib import Path

base_file = "results/eval/chartqa_real_contrastive_seeds_1500.csv"
df = pd.read_csv(base_file).drop_duplicates(["adapter", "idx"])

base = df[df["adapter"] == "base"].copy()
seed = df[df["adapter"] == "real_seed5"].copy()  # seed5 score 稳，可以先用它做 case

pred_col = "prediction" if "prediction" in df.columns else "pred"
if pred_col not in df.columns:
    pred_col = [c for c in df.columns if "pred" in c.lower()][0]

m = base[["idx", "question", "answers", pred_col, "score"]].rename(
    columns={pred_col: "base_pred", "score": "base_score"}
).merge(
    seed[["idx", pred_col, "score"]].rename(
        columns={pred_col: "ours_pred", "score": "ours_score"}
    ),
    on="idx"
)

wins = m[(m["base_score"] == 0) & (m["ours_score"] == 1)].copy()
losses = m[(m["base_score"] == 1) & (m["ours_score"] == 0)].copy()
stable = m[(m["base_score"] == 1) & (m["ours_score"] == 1)].copy()

wins.to_csv("results/eval/case_wins_seed5.csv", index=False)
losses.to_csv("results/eval/case_losses_seed5.csv", index=False)
stable.head(50).to_csv("results/eval/case_stable_seed5.csv", index=False)

print("wins:", len(wins))
print(wins[["idx", "question", "answers", "base_pred", "ours_pred"]].head(20).to_string(index=False))

print("\nlosses:", len(losses))
print(losses[["idx", "question", "answers", "base_pred", "ours_pred"]].head(20).to_string(index=False))

print("\nSaved:")
print("results/eval/case_wins_seed5.csv")
print("results/eval/case_losses_seed5.csv")
print("results/eval/case_stable_seed5.csv")
