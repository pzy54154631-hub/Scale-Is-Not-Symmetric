import pandas as pd
from pathlib import Path

def load_summary(path, dataset):
    df = pd.read_csv(path)
    base = df[df["method"] == "base"].iloc[0]
    seeds = df[df["method"].str.contains("seed", na=False)].copy()

    return {
        "Dataset": dataset,
        "Base Overall": base["overall"],
        "Ours Overall": seeds["overall"].mean(),
        "Std": seeds["overall"].std(),
        "Delta": seeds["overall"].mean() - base["overall"],
        "Base Gold-Scale": base["gold_scale_error_subset"],
        "Ours Gold-Scale": seeds["gold_scale_error_subset"].mean(),
        "Gold-Scale Delta": seeds["gold_scale_error_subset"].mean() - base["gold_scale_error_subset"],
        "Base Non-Scale": base["non_gold_scale_error"],
        "Ours Non-Scale": seeds["non_gold_scale_error"].mean(),
        "Non-Scale Delta": seeds["non_gold_scale_error"].mean() - base["non_gold_scale_error"],
    }

rows = []

paths = [
    ("results/eval/gold_scale_error_subset_summary_clean.csv", "ChartQA-500"),
    ("results/eval/gold_scale_error_subset_1000_summary.csv", "ChartQA-1000"),
    ("results/eval/gold_scale_error_subset_1500_summary.csv", "ChartQA-1500"),
]

for path, name in paths:
    if Path(path).exists():
        rows.append(load_summary(path, name))

# TextVQA
text_base = pd.read_csv("results/eval/textvqa_base_500.csv")["score"].mean()
text_scores = []
for p in sorted(Path("results/eval").glob("textvqa_real_contrastive_seed*_500.csv")):
    text_scores.append(pd.read_csv(p)["score"].mean())

if text_scores:
    rows.append({
        "Dataset": "TextVQA-500",
        "Base Overall": text_base,
        "Ours Overall": sum(text_scores) / len(text_scores),
        "Std": pd.Series(text_scores).std(),
        "Delta": sum(text_scores) / len(text_scores) - text_base,
        "Base Gold-Scale": None,
        "Ours Gold-Scale": None,
        "Gold-Scale Delta": None,
        "Base Non-Scale": None,
        "Ours Non-Scale": None,
        "Non-Scale Delta": None,
    })

out = pd.DataFrame(rows)

for c in out.columns:
    if c != "Dataset":
        out[c] = pd.to_numeric(out[c], errors="coerce")

print(out.round(4).to_string(index=False))

out.to_csv("results/eval/main_result_table.csv", index=False)

latex = out.round(4).to_latex(index=False, escape=False)
Path("results/eval/main_result_table.tex").write_text(latex)

print("\nSaved:")
print("results/eval/main_result_table.csv")
print("results/eval/main_result_table.tex")
