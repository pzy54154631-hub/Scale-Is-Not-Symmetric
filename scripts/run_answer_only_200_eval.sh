#!/usr/bin/env bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mirage

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/datasets
export MODEL_DIR=/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

mkdir -p results/eval logs

echo "=== Running TextVQA answer-only 200 eval ==="
python scripts/eval_tiny_adapters.py \
  --task textvqa \
  --split validation \
  --max-samples 200 \
  --output results/eval/textvqa_answer_only_200_eval.csv \
  --adapters \
  base=NONE \
  textvqa=outputs/adapters/textvqa_answer_only_1000 \
  chartqa=outputs/adapters/chartqa_answer_only_1000 \
  average=outputs/merged/textvqa_chartqa_answer_only_average \
  task_arithmetic=outputs/merged/textvqa_chartqa_answer_only_task_arithmetic \
  ties=outputs/merged/textvqa_chartqa_answer_only_ties \
  dare=outputs/merged/textvqa_chartqa_answer_only_dare \
  mirage=outputs/merged/textvqa_chartqa_answer_only_mirage

echo "=== Running ChartQA answer-only 200 eval ==="
python scripts/eval_tiny_adapters.py \
  --task chartqa \
  --split val \
  --max-samples 200 \
  --output results/eval/chartqa_answer_only_200_eval.csv \
  --adapters \
  base=NONE \
  textvqa=outputs/adapters/textvqa_answer_only_1000 \
  chartqa=outputs/adapters/chartqa_answer_only_1000 \
  average=outputs/merged/textvqa_chartqa_answer_only_average \
  task_arithmetic=outputs/merged/textvqa_chartqa_answer_only_task_arithmetic \
  ties=outputs/merged/textvqa_chartqa_answer_only_ties \
  dare=outputs/merged/textvqa_chartqa_answer_only_dare \
  mirage=outputs/merged/textvqa_chartqa_answer_only_mirage

echo "=== Done ==="
cat results/eval/textvqa_answer_only_200_eval.summary.json
cat results/eval/chartqa_answer_only_200_eval.summary.json
