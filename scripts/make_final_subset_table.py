import pandas as pd
from pathlib import Path

df = pd.read_csv("results/eval/gold_scale_error_subset_large_pos_full_summary.csv")

base = df[df["method"] == "base"].iloc[0]
seeds = df[df["method"].str.contains("seed", na=False)]

rows = [
    {
        "Split": "All examples",
        "Base": base["overall"],
        "Ours": seeds["overall"].mean(),
        "Std": seeds["overall"].std(),
        "Delta": seeds["overall"].mean() - base["overall"],
        "N": int(base["gold_scale_n"] + base["non_gold_scale_n"]),
    },
    {
        "Split": "Strict gold-scale errors",
        "Base": base["gold_scale_error_subset"],
        "Ours": seeds["gold_scale_error_subset"].mean(),
        "Std": seeds["gold_scale_error_subset"].std(),
        "Delta": seeds["gold_scale_error_subset"].mean() - base["gold_scale_error_subset"],
        "N": int(base["gold_scale_n"]),
    },
    {
        "Split": "Non-gold-scale examples",
        "Base": base["non_gold_scale_error"],
        "Ours": seeds["non_gold_scale_error"].mean(),
        "Std": seeds["non_gold_scale_error"].std(),
        "Delta": seeds["non_gold_scale_error"].mean() - base["non_gold_scale_error"],
        "N": int(base["non_gold_scale_n"]),
    },
]

out = pd.DataFrame(rows)
print(out.round(4).to_string(index=False))

Path("paper/tables").mkdir(parents=True, exist_ok=True)
out.to_csv("paper/tables/final_subset_table.csv", index=False)
Path("paper/tables/final_subset_table.tex").write_text(
    out.round(4).to_latex(index=False, escape=False)
)

print("\nSaved:")
print("paper/tables/final_subset_table.csv")
print("paper/tables/final_subset_table.tex")
