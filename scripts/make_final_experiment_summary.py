import pandas as pd
from pathlib import Path

Path("paper/tables").mkdir(parents=True, exist_ok=True)

lines = []
lines.append("# Final Experiment Summary\n")

# 1. ChartQA final
chart = pd.read_csv("results/eval/chartqa_final_large_pos_only_k300_full.csv")
chart_scores = chart.groupby("adapter")["score"].mean()
chart_base = chart_scores["base"]
chart_seeds = chart_scores[[x for x in chart_scores.index if "seed" in x]]

lines.append("## Main ChartQA-full Result\n")
lines.append(f"- Base: {chart_base:.4f}")
lines.append(f"- Ours mean: {chart_seeds.mean():.4f}")
lines.append(f"- Ours std: {chart_seeds.std():.4f}")
lines.append(f"- Delta: {chart_seeds.mean() - chart_base:.4f}\n")

# 2. TextVQA safety
text_path = Path("results/eval/textvqa_final_large_pos_only_k300_500.csv")
if text_path.exists():
    text = pd.read_csv(text_path)
    text_scores = text.groupby("adapter")["score"].mean()
    text_base = text_scores["base"]
    text_seeds = text_scores[[x for x in text_scores.index if "seed" in x]]

    lines.append("## TextVQA-500 Safety Check\n")
    lines.append(f"- Base: {text_base:.4f}")
    lines.append(f"- Ours mean: {text_seeds.mean():.4f}")
    lines.append(f"- Ours std: {text_seeds.std():.4f}")
    lines.append(f"- Delta: {text_seeds.mean() - text_base:.4f}\n")

# 3. Subset
subset_path = Path("paper/tables/final_subset_table.csv")
if subset_path.exists():
    subset = pd.read_csv(subset_path)
    lines.append("## ChartQA Subset Analysis\n")
    lines.append(subset.round(4).to_markdown(index=False))
    lines.append("\n")

# 4. Pair-mode ablation
pair_path = Path("results/eval/chartqa_pairmode_ablation_500.csv")
if pair_path.exists():
    pair = pd.read_csv(pair_path)
    pair_out = pair.groupby("adapter")["score"].mean().sort_values(ascending=False).reset_index()
    pair_out.columns = ["Method", "Accuracy"]
    pair_out.to_csv("paper/tables/pair_mode_ablation_500.csv", index=False)
    Path("paper/tables/pair_mode_ablation_500.tex").write_text(
        pair_out.round(4).to_latex(index=False, escape=False)
    )

    lines.append("## Pair-mode Ablation on ChartQA-500\n")
    lines.append(pair_out.round(4).to_markdown(index=False))
    lines.append("\n")

# 5. Size curve
size_path = Path("results/eval/chartqa_size_curve_realonly_500.csv")
if size_path.exists():
    size = pd.read_csv(size_path)
    size_out = size.groupby("adapter")["score"].mean().sort_values(ascending=False).reset_index()
    size_out.columns = ["Method", "Accuracy"]
    size_out.to_csv("paper/tables/size_curve_realonly_500.csv", index=False)
    Path("paper/tables/size_curve_realonly_500.tex").write_text(
        size_out.round(4).to_latex(index=False, escape=False)
    )

    lines.append("## Real-only Data-size Curve on ChartQA-500\n")
    lines.append(size_out.round(4).to_markdown(index=False))
    lines.append("\n")

Path("paper/final_experiment_summary.md").write_text("\n".join(lines))

print("\n".join(lines))
print("\nSaved:")
print("paper/final_experiment_summary.md")
print("paper/tables/pair_mode_ablation_500.csv")
print("paper/tables/pair_mode_ablation_500.tex")
print("paper/tables/size_curve_realonly_500.csv")
print("paper/tables/size_curve_realonly_500.tex")
