import pandas as pd
from pathlib import Path

path = "results/diagnostics/chartqa_full_scale_diagnostic.csv"
df = pd.read_csv(path)

rows = []

def add(name, mask):
    sub = df[mask].copy()
    rows.append({
        "Split": name,
        "Size": len(sub),
        "Base Acc.": sub["base_score"].mean() if len(sub) else None,
        "Recoverable by seeds": sub["num_seed_fixes"].gt(0).mean() if len(sub) else None,
        "Over-correction risk": sub["num_seed_breaks"].gt(0).mean() if len(sub) else None,
    })

add("All examples", df.index == df.index)
add("Numeric-answer examples", df["gold_numeric"])
add("Scale-keyword numeric examples", df["gold_numeric"] & df["scale_keyword"])
add("Strict gold-scale errors", df["gold_scale_error"])
add("Scale-conflict diagnostic", df["scale_conflict_diagnostic"])
add("Recoverable scale conflicts", df["recoverable_scale_conflict"])
add("Over-correction risks", df["overcorrection_risk"])

out = pd.DataFrame(rows)

print(out.round(4).to_string(index=False))

Path("paper/tables").mkdir(parents=True, exist_ok=True)
out.to_csv("paper/tables/scale_diagnostic_table.csv", index=False)
Path("paper/tables/scale_diagnostic_table.tex").write_text(
    out.round(4).to_latex(index=False, escape=False)
)

print("\nSaved:")
print("paper/tables/scale_diagnostic_table.csv")
print("paper/tables/scale_diagnostic_table.tex")
