#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mirage
cd /root/autodl-tmp/.autodl/mirage_csr

mkdir -p logs results/eval results/analysis outputs/experiment_archives

date
echo "[1/4] Qwen PlotQA numeric-500 answer/scale"
python scripts/eval_chartqa_parquet_qwen_prompt.py \
  --parquet datasets/external_plotqa/plotqa_numeric_500.parquet \
  --output results/eval/20260520_plotqa_numeric500_qwen7_answer_large.csv \
  --adapters \
    answer_tuned=outputs/adapters/chartqa_answer_only_3000_r16 \
    large_pos_seed1=outputs/adapters/20260520_corrected_pairmode_large_pos_true_k300_seed1 \
  --prompt-mode plain \
  --max-new-tokens 24

echo "[2/4] Qwen PlotQA numeric-500 scale prompt base"
python scripts/eval_chartqa_parquet_qwen_prompt.py \
  --parquet datasets/external_plotqa/plotqa_numeric_500.parquet \
  --output results/eval/20260520_plotqa_numeric500_qwen7_base_scaleprompt.csv \
  --adapters base=NONE \
  --prompt-mode scale \
  --max-new-tokens 24

echo "[3/4] Qwen PlotQA numeric-500 TTA answer-tuned"
python scripts/eval_scaleqa_parquet_tta_prompts.py \
  --parquet datasets/external_plotqa/plotqa_numeric_500.parquet \
  --output results/eval/20260520_plotqa_numeric500_qwen7_answer_tuned_tta.csv \
  --adapter outputs/adapters/chartqa_answer_only_3000_r16 \
  --max-new-tokens 32

echo "[4/4] InternVL PlotQA numeric-500 base"
python scripts/eval_chartqa_parquet_internvl.py \
  --parquet datasets/external_plotqa/plotqa_numeric_500.parquet \
  --output results/eval/20260520_plotqa_numeric500_internvl25_1b_plain.csv \
  --model-dir /root/autodl-tmp/models/InternVL2_5-1B \
  --model-name internvl25_1b_plotqa \
  --max-new-tokens 24

python - <<'PY'
import json
import pathlib
import tarfile

files = list(pathlib.Path("results/eval").glob("20260520_plotqa_numeric500*"))
summary = {}
for p in files:
    if p.name.endswith(".summary.json"):
        try:
            summary[p.name] = json.loads(p.read_text())
        except Exception as exc:
            summary[p.name] = str(exc)
    elif p.name.endswith(".summary.csv"):
        summary[p.name] = p.read_text()[:4000]

out = pathlib.Path("results/analysis/20260520_plotqa_external_summary.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

archive = pathlib.Path("outputs/experiment_archives/20260520_plotqa_external_numeric500.tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    tar.add("datasets/external_plotqa/plotqa_numeric_500.parquet")
    for p in files:
        tar.add(str(p))
    tar.add(str(out))

print("SUMMARY_JSON=" + str(out))
print("ARCHIVE=" + str(archive))
PY

date
