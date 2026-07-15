#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate mirage
cd /root/autodl-tmp/.autodl/mirage_csr

mkdir -p logs results/eval results/analysis outputs/adapters outputs/experiment_archives

echo "Waiting for PlotQA external job to finish..."
while pgrep -f "run_20260520_plotqa_external.sh|eval_chartqa_parquet_qwen_prompt.py|eval_scaleqa_parquet_tta_prompts.py|eval_chartqa_parquet_internvl.py" >/dev/null; do
  date
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  sleep 120
done

date
echo "[1/3] Train InternVL large-positive scale LoRA"
python scripts/train_internvl_scale_lora.py \
  --output-adapter outputs/adapters/20260520_internvl25_1b_scale_large_pos_k300_seed1 \
  --pair-mode large_pos \
  --max-pairs 300 \
  --steps 200 \
  --seed 1

echo "[2/3] Train InternVL small-positive scale LoRA"
python scripts/train_internvl_scale_lora.py \
  --output-adapter outputs/adapters/20260520_internvl25_1b_scale_small_pos_k300_seed1 \
  --pair-mode small_pos \
  --max-pairs 300 \
  --steps 200 \
  --seed 1

echo "[3/3] Evaluate InternVL adapters on ScaleQA-Natural-520"
python scripts/eval_internvl_parquet_lora.py \
  --parquet results/scaleqa_natural/image_assisted_v3/scaleqa_natural_520_image_assisted.parquet \
  --output results/eval/20260520_scaleqa520_internvl25_1b_adapter_pairmodes.csv \
  --adapters \
    base=NONE \
    large_pos=outputs/adapters/20260520_internvl25_1b_scale_large_pos_k300_seed1 \
    small_pos=outputs/adapters/20260520_internvl25_1b_scale_small_pos_k300_seed1 \
  --max-new-tokens 24

python - <<'PY'
import json
import pathlib
import tarfile

files = [
    pathlib.Path("results/eval/20260520_scaleqa520_internvl25_1b_adapter_pairmodes.csv"),
    pathlib.Path("results/eval/20260520_scaleqa520_internvl25_1b_adapter_pairmodes.csv.summary.json"),
]
summary = {}
for p in files:
    if p.exists() and p.name.endswith(".summary.json"):
        summary[p.name] = json.loads(p.read_text())
out = pathlib.Path("results/analysis/20260520_internvl_adapter_summary.json")
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

archive = pathlib.Path("outputs/experiment_archives/20260520_internvl25_1b_scale_adapter.tar.gz")
with tarfile.open(archive, "w:gz") as tar:
    for p in files + [out]:
        if p.exists():
            tar.add(str(p))
    for d in [
        pathlib.Path("outputs/adapters/20260520_internvl25_1b_scale_large_pos_k300_seed1"),
        pathlib.Path("outputs/adapters/20260520_internvl25_1b_scale_small_pos_k300_seed1"),
    ]:
        if d.exists():
            tar.add(str(d))
print("SUMMARY_JSON=" + str(out))
print("ARCHIVE=" + str(archive))
PY

date
