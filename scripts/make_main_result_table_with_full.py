import pandas as pd
from pathlib import Path

rows = []

# Existing compact results from your saved table
old = pd.read_csv("results/paper_tables/main_result_table.csv")
for _, r in old.iterrows():
    rows.append({
        "Dataset": r["Dataset"],
        "Base": r["Base Overall"],
        "Ours": r["Ours Overall"],
        "Std": r["Std"],
        "Delta": r["Delta"],
    })

# Full ChartQA
df = pd.read_csv("results/eval/chartqa_real_contrastive_seeds_full.csv")
scores = df.groupby("adapter")["score"].mean()
base = scores["base"]
seeds = scores[[x for x in scores.index if "seed" in x]]

rows.append({
    "Dataset": "ChartQA-full",
    "Base": base,
    "Ours": seeds.mean(),
    "Std": seeds.std(),
    "Delta": seeds.mean() - base,
})

out = pd.DataFrame(rows)

# Remove duplicate old full if exists
out = out.drop_duplicates("Dataset", keep="last")

print(out.round(4).to_string(index=False))

out.to_csv("paper/tables/main_result_table_with_full.csv", index=False)
Path("paper/tables/main_result_table_with_full.tex").write_text(
    out.round(4).to_latex(index=False, escape=False)
)

print("\nSaved:")
print("paper/tables/main_result_table_with_full.csv")
print("paper/tables/main_result_table_with_full.tex")
