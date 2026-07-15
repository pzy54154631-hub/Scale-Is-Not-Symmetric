#!/usr/bin/env bash
set -euo pipefail

TEXT=outputs/adapters/textvqa_answer_only_3000_r16
CHART=outputs/adapters/chartqa_answer_only_3000_r16
AVG=outputs/merged/textvqa_chartqa_answer_only_3000_r16_average
PREFIX=outputs/merged/textvqa_chartqa_answer_only_3000_r16

make_spf () {
  NAME=$1
  shift

  echo "=== MERGE $NAME ==="
  python scripts/merge_mirage_spf_lite.py \
    --text "$TEXT" \
    --chart "$CHART" \
    --average "$AVG" \
    --output "${PREFIX}_${NAME}" \
    "$@"
}

# 保守：接近 spf_safe
make_spf spf_calib_a \
  --base-lambda 0.07 \
  --lambda-min 0.02 \
  --lambda-max 0.09 \
  --constant-weight 0.50 \
  --evidence-weight 0.20 \
  --private-weight 0.20 \
  --shared-weight 0.10

# 稍微强化 ChartQA evidence
make_spf spf_calib_b \
  --base-lambda 0.08 \
  --lambda-min 0.02 \
  --lambda-max 0.10 \
  --constant-weight 0.40 \
  --evidence-weight 0.35 \
  --private-weight 0.15 \
  --shared-weight 0.10

# 强化 private residual
make_spf spf_calib_c \
  --base-lambda 0.09 \
  --lambda-min 0.03 \
  --lambda-max 0.11 \
  --constant-weight 0.35 \
  --evidence-weight 0.25 \
  --private-weight 0.30 \
  --shared-weight 0.10

# 强化 shared-private contrast
make_spf spf_calib_d \
  --base-lambda 0.09 \
  --lambda-min 0.03 \
  --lambda-max 0.12 \
  --constant-weight 0.25 \
  --evidence-weight 0.25 \
  --private-weight 0.25 \
  --shared-weight 0.25

# 更激进，但仍然限制上限
make_spf spf_calib_e \
  --base-lambda 0.10 \
  --lambda-min 0.04 \
  --lambda-max 0.12 \
  --constant-weight 0.35 \
  --evidence-weight 0.30 \
  --private-weight 0.25 \
  --shared-weight 0.10

# 深层视觉 MLP 稍微多给一点
make_spf spf_calib_f \
  --base-lambda 0.09 \
  --lambda-min 0.02 \
  --lambda-max 0.12 \
  --constant-weight 0.35 \
  --evidence-weight 0.30 \
  --private-weight 0.25 \
  --shared-weight 0.10 \
  --shallow-scale 0.90 \
  --middle-scale 1.00 \
  --deep-scale 1.10

echo "=== EVAL ChartQA-200 ==="
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python scripts/eval_tiny_adapters.py \
  --task chartqa \
  --split val \
  --max-samples 200 \
  --output results/eval/chartqa_spf_calib_grid_200.csv \
  --adapters \
  average="${AVG}" \
  vmlp_chart060="${PREFIX}_vmlp_chartw_060" \
  pas_group_stable="${PREFIX}_pas_group_stable" \
  spf_safe="${PREFIX}_spf_safe" \
  spf_calib_a="${PREFIX}_spf_calib_a" \
  spf_calib_b="${PREFIX}_spf_calib_b" \
  spf_calib_c="${PREFIX}_spf_calib_c" \
  spf_calib_d="${PREFIX}_spf_calib_d" \
  spf_calib_e="${PREFIX}_spf_calib_e" \
  spf_calib_f="${PREFIX}_spf_calib_f"

echo "=== EVAL TextVQA-200 ==="
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python scripts/eval_tiny_adapters.py \
  --task textvqa \
  --split validation \
  --max-samples 200 \
  --output results/eval/textvqa_spf_calib_grid_200.csv \
  --adapters \
  average="${AVG}" \
  vmlp_chart060="${PREFIX}_vmlp_chartw_060" \
  pas_group_stable="${PREFIX}_pas_group_stable" \
  spf_safe="${PREFIX}_spf_safe" \
  spf_calib_a="${PREFIX}_spf_calib_a" \
  spf_calib_b="${PREFIX}_spf_calib_b" \
  spf_calib_c="${PREFIX}_spf_calib_c" \
  spf_calib_d="${PREFIX}_spf_calib_d" \
  spf_calib_e="${PREFIX}_spf_calib_e" \
  spf_calib_f="${PREFIX}_spf_calib_f"

echo "Done."
