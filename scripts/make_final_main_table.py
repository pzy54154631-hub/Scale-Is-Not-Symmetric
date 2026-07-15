import pandas as pd
from pathlib import Path

rows = []

def add_eval(name, path, seed_keyword):
    df = pd.read_csv(path)
    scores = df.groupby("adapter")["score"].mean()
    base = scores["base"]
    seeds = scores[[x for x in scores.index if seed_keyword in x]]
    rows.append({
        "Dataset": name,
        "Base": base,
        "Ours": seeds.mean(),
        "Std": seeds.std(),
        "Delta": seeds.mean() - base,
    })

add_eval(
    "ChartQA-full",
    "results/eval/chartqa_final_large_pos_only_k300_full.csv",
    "seed",
)

if Path("results/eval/textvqa_final_large_pos_only_k300_500.csv").exists():
    add_eval(
        "TextVQA-500",
        "results/eval/textvqa_final_large_pos_only_k300_500.csv",
        "seed",
    )

out = pd.DataFrame(rows)
print(out.round(4).to_string(index=False))

Path("paper/tables").mkdir(parents=True, exist_ok=True)
out.to_csv("paper/tables/final_main_table.csv", index=False)
Path("paper/tables/final_main_table.tex").write_text(
    out.round(4).to_latex(index=False, escape=False)
)

print("\nSaved:")
print("paper/tables/final_main_table.csv")
print("paper/tables/final_main_table.tex")
