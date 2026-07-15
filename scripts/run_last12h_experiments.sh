set -e

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BEST=$(python - <<'PY'
import json
print(json.load(open("results/eval/sfg_cem_best.json"))["path"])
PY
)

TRAIN_LOG=results/eval/chartqa_scale_unit_verifier_logprob_train1500_comp_final.csv

echo "BEST=$BEST"
echo "TRAIN_LOG=$TRAIN_LOG"

# ============================================================
# 1. Full pair-mode ablation, one seed
# ============================================================
echo "===== Full pair-mode ablation ====="

python scripts/eval_tiny_adapters_csr.py \
  --task chartqa \
  --split val \
  --max-samples 999999 \
  --output results/eval/chartqa_pairmode_ablation_full.csv \
  --adapters \
  base=$BEST \
  large_pos_only_k300=outputs/adapters/mirage_scale_large_pos_only_k300_seed1 \
  all_k300=outputs/adapters/mirage_scale_all_k300_seed1 \
  balanced_k300=outputs/adapters/mirage_scale_balanced_scale_k300_seed1 \
  real_only_k300=outputs/adapters/mirage_scale_real_only_k300_seed1 \
  small_pos_only_k300=outputs/adapters/mirage_scale_small_pos_only_k300_seed1

# ============================================================
# 2. Large-pos-only data-size curve
# ============================================================
echo "===== Train large-pos-only size curve ====="

for K in 25 50 100 200 400 600 800; do
  OUT=outputs/adapters/mirage_scale_large_pos_only_k${K}_curve_seed1

  if [ "$K" = "300" ]; then
    continue
  fi

  if [ -d "$OUT" ]; then
    echo "Skip existing $OUT"
    continue
  fi

  echo "Training large_pos_only K=$K"

  python scripts/train_scale_contrastive_final.py \
    --base-adapter "$BEST" \
    --train-logprob "$TRAIN_LOG" \
    --output-adapter "$OUT" \
    --pair-mode large_pos_only \
    --max-pairs-per-kind $K \
    --steps 300 \
    --lr 1e-6 \
    --seed 1
done

# ============================================================
# 3. Eval large-pos-only size curve on ChartQA-1500
# ============================================================
echo "===== Eval large-pos-only size curve on 1500 ====="

python scripts/eval_tiny_adapters_csr.py \
  --task chartqa \
  --split val \
  --max-samples 1500 \
  --output results/eval/chartqa_large_pos_size_curve_1500.csv \
  --adapters \
  base=$BEST \
  large_k25=outputs/adapters/mirage_scale_large_pos_only_k25_curve_seed1 \
  large_k50=outputs/adapters/mirage_scale_large_pos_only_k50_curve_seed1 \
  large_k100=outputs/adapters/mirage_scale_large_pos_only_k100_curve_seed1 \
  large_k200=outputs/adapters/mirage_scale_large_pos_only_k200_curve_seed1 \
  large_k300=outputs/adapters/mirage_scale_large_pos_only_k300_seed1 \
  large_k400=outputs/adapters/mirage_scale_large_pos_only_k400_curve_seed1 \
  large_k600=outputs/adapters/mirage_scale_large_pos_only_k600_curve_seed1 \
  large_k800=outputs/adapters/mirage_scale_large_pos_only_k800_curve_seed1

# ============================================================
# 4. Make summary tables
# ============================================================
echo "===== Make final appendix tables ====="

python - <<'PY'
import re
from pathlib import Path
import pandas as pd

Path("paper/tables").mkdir(parents=True, exist_ok=True)

tables = {}

# Pair-mode full
pair_path = Path("results/eval/chartqa_pairmode_ablation_full.csv")
if pair_path.exists():
    df = pd.read_csv(pair_path)
    out = df.groupby("adapter")["score"].mean().sort_values(ascending=False).reset_index()
    out.columns = ["Method", "Accuracy"]
    out["Delta"] = out["Accuracy"] - float(out[out["Method"] == "base"]["Accuracy"].iloc[0])
    out.to_csv("paper/tables/pair_mode_ablation_full.csv", index=False)
    Path("paper/tables/pair_mode_ablation_full.tex").write_text(
        out.round(4).to_latex(index=False, escape=False)
    )
    tables["Pair-mode full"] = out

# Large-pos size curve
curve_path = Path("results/eval/chartqa_large_pos_size_curve_1500.csv")
if curve_path.exists():
    df = pd.read_csv(curve_path)
    score = df.groupby("adapter")["score"].mean()
    base = float(score["base"])

    rows = []
    for name, acc in score.items():
        if name == "base":
            k = 0
        else:
            m = re.search(r"k(\d+)", name)
            k = int(m.group(1)) if m else None
        rows.append({
            "Method": name,
            "Pairs per kind": k,
            "Accuracy": acc,
            "Delta": acc - base,
        })

    out = pd.DataFrame(rows).sort_values("Pairs per kind")
    out.to_csv("paper/tables/large_pos_size_curve_1500.csv", index=False)
    Path("paper/tables/large_pos_size_curve_1500.tex").write_text(
        out.round(4).to_latex(index=False, escape=False)
    )
    tables["Large-pos size curve"] = out

for name, tab in tables.items():
    print("\n" + "=" * 80)
    print(name)
    print(tab.round(4).to_string(index=False))

print("\nSaved tables under paper/tables/")
PY

echo "===== Done last-12h experiments ====="
