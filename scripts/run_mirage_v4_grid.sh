#!/usr/bin/env bash
set -e

TEXT=outputs/adapters/textvqa_answer_only_3000_r16
CHART=outputs/adapters/chartqa_answer_only_3000_r16
PREFIX=outputs/merged/textvqa_chartqa_answer_only_3000_r16_mirage_v4

mkdir -p outputs/merged

run_one () {
  NAME=$1
  shift
  echo "=== MERGE $NAME ==="
  python scripts/merge_mirage_v4_rank16.py \
    --adapter-a "$TEXT" \
    --adapter-b "$CHART" \
    --output "${PREFIX}_${NAME}" \
    --target-r 16 \
    --target-alpha 32 \
    "$@"
}

# 1. Visual MLP conservative: 最接近你目前最好结果，但更稳
run_one vmlp_cons \
  --prior-visual-mlp 1.0 \
  --prior-visual-attn 0.0 \
  --prior-language 0.0 \
  --energy-power 0.5 \
  --tau-cos 0.0 \
  --temp-cos 0.35 \
  --tau-delta 0.50 \
  --temp-delta 0.25 \
  --delta-weight 0.7 \
  --risk-weight 0.6 \
  --dominance-weight 0.4 \
  --gate-sharpness 1.0 \
  --conflict-damping 0.20 \
  --synergy-gain 0.03 \
  --scale-min 0.80 \
  --scale-max 1.05

# 2. Visual MLP risk-driven
run_one vmlp_risk \
  --prior-visual-mlp 1.0 \
  --prior-visual-attn 0.0 \
  --prior-language 0.0 \
  --energy-power 1.0 \
  --tau-cos 0.0 \
  --temp-cos 0.25 \
  --tau-delta 0.45 \
  --temp-delta 0.20 \
  --delta-weight 1.2 \
  --risk-weight 0.8 \
  --dominance-weight 0.2 \
  --gate-sharpness 1.5 \
  --conflict-damping 0.30 \
  --synergy-gain 0.05 \
  --scale-min 0.70 \
  --scale-max 1.10

# 3. Visual MLP dominance-driven
run_one vmlp_dom \
  --prior-visual-mlp 1.0 \
  --prior-visual-attn 0.0 \
  --prior-language 0.0 \
  --energy-power 1.5 \
  --tau-cos 0.0 \
  --temp-cos 0.30 \
  --tau-delta 0.45 \
  --temp-delta 0.25 \
  --delta-weight 0.8 \
  --risk-weight 0.3 \
  --dominance-weight 0.7 \
  --gate-sharpness 1.2 \
  --conflict-damping 0.25 \
  --synergy-gain 0.04 \
  --scale-min 0.75 \
  --scale-max 1.10

# 4. Visual MLP sharp gate
run_one vmlp_sharp \
  --prior-visual-mlp 1.0 \
  --prior-visual-attn 0.0 \
  --prior-language 0.0 \
  --energy-power 1.0 \
  --tau-cos 0.0 \
  --temp-cos 0.20 \
  --tau-delta 0.45 \
  --temp-delta 0.18 \
  --delta-weight 1.0 \
  --risk-weight 0.7 \
  --dominance-weight 0.3 \
  --gate-sharpness 2.5 \
  --conflict-damping 0.30 \
  --synergy-gain 0.04 \
  --scale-min 0.70 \
  --scale-max 1.10

# 5. Visual MLP + weak visual attention
run_one vmlp_vattn \
  --prior-visual-mlp 1.0 \
  --prior-visual-attn 0.3 \
  --prior-language 0.0 \
  --energy-power 1.0 \
  --tau-cos 0.0 \
  --temp-cos 0.30 \
  --tau-delta 0.45 \
  --temp-delta 0.22 \
  --delta-weight 1.0 \
  --risk-weight 0.7 \
  --dominance-weight 0.3 \
  --gate-sharpness 1.5 \
  --conflict-damping 0.25 \
  --synergy-gain 0.04 \
  --scale-min 0.75 \
  --scale-max 1.10

# 6. Visual MLP + weak language
run_one vmlp_lang \
  --prior-visual-mlp 1.0 \
  --prior-visual-attn 0.0 \
  --prior-language 0.3 \
  --energy-power 1.0 \
  --tau-cos 0.0 \
  --temp-cos 0.30 \
  --tau-delta 0.45 \
  --temp-delta 0.22 \
  --delta-weight 1.0 \
  --risk-weight 0.7 \
  --dominance-weight 0.3 \
  --gate-sharpness 1.5 \
  --conflict-damping 0.25 \
  --synergy-gain 0.04 \
  --scale-min 0.75 \
  --scale-max 1.10

# 7. All modules soft
run_one all_soft \
  --prior-visual-mlp 0.7 \
  --prior-visual-attn 0.4 \
  --prior-language 0.3 \
  --prior-other 0.1 \
  --energy-power 0.5 \
  --tau-cos 0.0 \
  --temp-cos 0.40 \
  --tau-delta 0.50 \
  --temp-delta 0.30 \
  --delta-weight 0.5 \
  --risk-weight 0.5 \
  --dominance-weight 0.5 \
  --gate-sharpness 1.0 \
  --conflict-damping 0.15 \
  --synergy-gain 0.03 \
  --scale-min 0.85 \
  --scale-max 1.05

# 8. Visual attention only
run_one vattn_only \
  --prior-visual-mlp 0.0 \
  --prior-visual-attn 1.0 \
  --prior-language 0.0 \
  --energy-power 1.0 \
  --tau-cos 0.0 \
  --temp-cos 0.30 \
  --tau-delta 0.45 \
  --temp-delta 0.22 \
  --delta-weight 1.0 \
  --risk-weight 0.7 \
  --dominance-weight 0.3 \
  --gate-sharpness 1.5 \
  --conflict-damping 0.25 \
  --synergy-gain 0.04 \
  --scale-min 0.75 \
  --scale-max 1.10

echo "All MIRAGE-v4 candidates generated."
