import pandas as pd
from pathlib import Path

inp = "results/paper_tables/ablation_chartqa500.csv"
df = pd.read_csv(inp)

# 只保留论文里需要的列
df = df[[
    "method",
    "overall",
    "overall_delta",
    "gold_scale_error",
    "gold_scale_delta",
    "non_scale",
    "non_scale_delta",
    "n_runs",
]]

# 方法名改短
rename = {
    "Base": "Base",
    "Scale-aware prompt": "Scale-aware prompt",
    "All contrastive": "Mixed contrastive",
    "Real-only contrastive": "Real contrastive",
    "Real-only contrastive, 5 seeds": "Real contrastive (5 seeds)",
    "Balanced contrastive": "Balanced contrastive",
    "Large-only contrastive": "Large-only contrastive",
    "Direct SFT s800": "Direct SFT (800)",
    "Direct SFT s1500": "Direct SFT (1500)",
}
df["method"] = df["method"].map(lambda x: rename.get(x, x))

# 转成百分比形式
for c in [
    "overall",
    "overall_delta",
    "gold_scale_error",
    "gold_scale_delta",
    "non_scale",
    "non_scale_delta",
]:
    df[c] = pd.to_numeric(df[c], errors="coerce") * 100

# 排序
order = [
    "Base",
    "Scale-aware prompt",
    "Direct SFT (800)",
    "Direct SFT (1500)",
    "Mixed contrastive",
    "Balanced contrastive",
    "Large-only contrastive",
    "Real contrastive",
    "Real contrastive (5 seeds)",
]
df["order"] = df["method"].apply(lambda x: order.index(x) if x in order else 999)
df = df.sort_values("order").drop(columns=["order"])

# 输出 CSV
Path("paper/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("paper/tables/ablation_chartqa500_paper.csv", index=False)

# 手写更干净的 LaTeX
lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrrrrrr}")
lines.append(r"\toprule")
lines.append(r"Method & Overall & $\Delta$ & Gold-scale & $\Delta$ & Non-scale & $\Delta$ \\")
lines.append(r"\midrule")

for _, r in df.iterrows():
    method = r["method"]
    overall = f"{r['overall']:.2f}" if pd.notna(r["overall"]) else "--"
    od = f"{r['overall_delta']:+.2f}" if pd.notna(r["overall_delta"]) else "--"
    gs = f"{r['gold_scale_error']:.2f}" if pd.notna(r["gold_scale_error"]) else "--"
    gsd = f"{r['gold_scale_delta']:+.2f}" if pd.notna(r["gold_scale_delta"]) else "--"
    ns = f"{r['non_scale']:.2f}" if pd.notna(r["non_scale"]) else "--"
    nsd = f"{r['non_scale_delta']:+.2f}" if pd.notna(r["non_scale_delta"]) else "--"

    if "Real contrastive (5 seeds)" in method:
        method = r"\textbf{Real contrastive (5 seeds)}"
        overall = r"\textbf{" + overall + r"}"
        od = r"\textbf{" + od + r"}"

    lines.append(f"{method} & {overall} & {od} & {gs} & {gsd} & {ns} & {nsd} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{Ablation results on ChartQA-500. Gold-scale denotes strict 100$\times$ scale-error cases where the base model predicts an answer at the wrong numerical scale.}")
lines.append(r"\label{tab:ablation}")
lines.append(r"\end{table}")

Path("paper/tables/ablation_chartqa500_paper.tex").write_text("\n".join(lines))

print(df.round(2).to_string(index=False))
print("\nSaved:")
print("paper/tables/ablation_chartqa500_paper.csv")
print("paper/tables/ablation_chartqa500_paper.tex")
