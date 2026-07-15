#!/usr/bin/env bash
set -euo pipefail

REPO="/root/autodl-tmp/.autodl/mirage_csr"
cd "${REPO}"

INPUT="results/annotation_mining/scaleqa_natural_candidates_round2_after_internvl_train_PREFILLED.csv"
OUTPUT_CSV="results/annotation_mining/scaleqa_natural_candidates_round2_after_internvl_train_CODEX_ANNOTATED.csv"
OUTPUT_XLSX="results/annotation_mining/scaleqa_natural_candidates_round2_after_internvl_train_CODEX_ANNOTATED.xlsx"
LOG="results/annotation_mining/final_codex_annotation.log"

while [[ ! -f "${INPUT}" ]]; do
  sleep 60
done

/root/miniconda3/envs/mirage/bin/python scripts/codex_annotate_scale_candidates.py \
  --input-csv "${INPUT}" \
  --output-csv "${OUTPUT_CSV}" \
  --output-xlsx "${OUTPUT_XLSX}" > "${LOG}" 2>&1

ARCHIVE="outputs/experiment_archives/20260519_D_final_codex_annotation_after_internvl_train.tar.gz"
FILES=(
  scripts/codex_annotate_scale_candidates.py
  "${INPUT}"
  "${OUTPUT_CSV}"
  "${OUTPUT_XLSX}"
  "${LOG}"
)
EXISTING=()
for f in "${FILES[@]}"; do
  if [[ -e "${f}" ]]; then
    EXISTING+=("${f}")
  fi
done
tar -czf "${ARCHIVE}" "${EXISTING[@]}"
echo "archive ${ARCHIVE}"
