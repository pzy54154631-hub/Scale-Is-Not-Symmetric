#!/usr/bin/env bash
set -euo pipefail

REPO="/root/autodl-tmp/.autodl/mirage_csr"
cd "${REPO}"

mkdir -p results/annotation_mining outputs/experiment_archives

WATCH_PID="${1:-}"
if [[ -n "${WATCH_PID}" ]]; then
  while kill -0 "${WATCH_PID}" >/dev/null 2>&1; do
    sleep 60
  done
fi

RAW="results/annotation_mining/chartqa_all_eval_scale_candidates_raw_after_internvl_train.csv"
MERGED="results/annotation_mining/chartqa_all_eval_scale_candidates_merged_after_internvl_train.csv"
XLSX="results/annotation_mining/scaleqa_natural_candidates_round2_after_internvl_train_PREFILLED.xlsx"
CSV="results/annotation_mining/scaleqa_natural_candidates_round2_after_internvl_train_PREFILLED.csv"
SUMMARY="results/annotation_mining/scaleqa_round2_after_internvl_train_summary.json"

/root/miniconda3/envs/mirage/bin/python scripts/mine_scale_error_candidates_from_eval.py \
  --inputs results/eval/chartqa*.csv \
  --output-candidates "${RAW}" \
  --output-merged "${MERGED}" \
  --tolerance 0.25

/root/miniconda3/envs/mirage/bin/python scripts/build_scale_annotation_workbook.py \
  --raw-candidates "${RAW}" \
  --output-xlsx "${XLSX}" \
  --output-csv "${CSV}" \
  --max-review 800

/root/miniconda3/envs/mirage/bin/python - <<'PY'
from pathlib import Path
import json
import pandas as pd

raw = pd.read_csv("results/annotation_mining/chartqa_all_eval_scale_candidates_raw_after_internvl_train.csv")
merged = pd.read_csv("results/annotation_mining/chartqa_all_eval_scale_candidates_merged_after_internvl_train.csv")
review = pd.read_csv("results/annotation_mining/scaleqa_natural_candidates_round2_after_internvl_train_PREFILLED.csv")
summary = {
    "raw_occurrences": int(len(raw)),
    "merged_split_idx_candidates": int(len(merged)),
    "distinct_question_gold_candidates": int(len(review)),
    "suggested_label_counts": {str(k): int(v) for k, v in review["suggested_label"].value_counts().items()},
    "base_like_candidate_count": int((review["base_like_occurrences"] > 0).sum()),
    "new_internvl_train_files": sorted(str(p) for p in Path("results/eval").glob("chartqa_train_shard*_internvl25_1b_plain.csv")),
}
Path("results/annotation_mining/scaleqa_round2_after_internvl_train_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

ARCHIVE="outputs/experiment_archives/20260519_D_large_scale_annotation_mining_after_internvl_train.tar.gz"
FILES=(
  scripts/mine_scale_error_candidates_from_eval.py
  scripts/build_scale_annotation_workbook.py
  scripts/run_annotation_mining_postprocess.sh
  results/annotation_mining/internvl_train_shards_eval.log
  results/annotation_mining/chartqa_existing_eval_scale_candidates_raw.csv
  results/annotation_mining/chartqa_existing_eval_scale_candidates_merged.csv
  "${RAW}"
  "${MERGED}"
  "${XLSX}"
  "${CSV}"
  "${SUMMARY}"
  results/eval/chartqa_train_shard1_internvl25_1b_plain.csv
  results/eval/chartqa_train_shard1_internvl25_1b_plain.csv.summary.json
  results/eval/chartqa_train_shard2_internvl25_1b_plain.csv
  results/eval/chartqa_train_shard2_internvl25_1b_plain.csv.summary.json
  results/eval/chartqa_train_shard3_internvl25_1b_plain.csv
  results/eval/chartqa_train_shard3_internvl25_1b_plain.csv.summary.json
)
EXISTING=()
for f in "${FILES[@]}"; do
  if [[ -e "${f}" ]]; then
    EXISTING+=("${f}")
  fi
done
tar -czf "${ARCHIVE}" "${EXISTING[@]}"

echo "archive ${ARCHIVE}"
