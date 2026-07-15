#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/.autodl/mirage_csr
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mirage

mkdir -p logs results/eval results/analysis outputs/experiment_archives

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

MODEL_DIR=/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
ADAPTERS=(
  base=NONE
  answer_tuned=outputs/adapters/chartqa_answer_only_3000_r16
  large_pos_seed1=outputs/adapters/20260520_corrected_pairmode_large_pos_true_k300_seed1
  small_pos_seed1=outputs/adapters/20260520_corrected_pairmode_small_pos_true_k300_seed1
  balanced_seed1=outputs/adapters/20260520_corrected_pairmode_balanced_true_k300_seed1
)

run_eval() {
  local parquet="$1"
  local output="$2"
  echo "=== $(date) ==="
  echo "Evaluating $parquet -> $output"
  /root/miniconda3/envs/mirage/bin/python scripts/eval_chartqa_parquet_qwen_prompt.py \
    --model-dir "$MODEL_DIR" \
    --parquet "$parquet" \
    --output "$output" \
    --adapters "${ADAPTERS[@]}" \
    --prompt-mode plain \
    --max-new-tokens 24 \
    --max-pixels 1003520
}

run_eval \
  results/external/20260521_chartbench_external_internvl8b/chartbench_numeric500.parquet \
  results/eval/20260522_chartbench500_qwen7_external_suite.csv

run_eval \
  results/external/20260521_mathvista_local_external_internvl8b/mathvista_chart_table_numeric300.parquet \
  results/eval/20260522_mathvista243_qwen7_external_suite.csv

run_eval \
  results/external/20260521_infographicvqa_external_internvl8b/infographicvqa_numeric300.parquet \
  results/eval/20260522_infographicvqa300_qwen7_external_suite.csv

/root/miniconda3/envs/mirage/bin/python - <<'PY'
from __future__ import annotations

import json
import tarfile
from pathlib import Path

summary_files = [
    Path("results/eval/20260522_chartbench500_qwen7_external_suite.csv.summary.json"),
    Path("results/eval/20260522_mathvista243_qwen7_external_suite.csv.summary.json"),
    Path("results/eval/20260522_infographicvqa300_qwen7_external_suite.csv.summary.json"),
]
summary = {}
for path in summary_files:
    if path.exists():
        summary[path.name] = json.loads(path.read_text())
    else:
        summary[path.name] = {"missing": True}

out = Path("results/analysis/20260522_qwen7_external_suite_summary.json")
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

archive = Path("outputs/experiment_archives/20260522_qwen7_external_suite.tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    for p in [
        "results/eval/20260522_chartbench500_qwen7_external_suite.csv",
        "results/eval/20260522_chartbench500_qwen7_external_suite.csv.summary.json",
        "results/eval/20260522_mathvista243_qwen7_external_suite.csv",
        "results/eval/20260522_mathvista243_qwen7_external_suite.csv.summary.json",
        "results/eval/20260522_infographicvqa300_qwen7_external_suite.csv",
        "results/eval/20260522_infographicvqa300_qwen7_external_suite.csv.summary.json",
        str(out),
        "logs/run_20260522_qwen7_external_suite.log",
        "logs/run_20260522_qwen7_external_suite.sh",
    ]:
        path = Path(p)
        if path.exists():
            tar.add(str(path))
print("SUMMARY_JSON=" + str(out))
print("ARCHIVE=" + str(archive))
PY

date
