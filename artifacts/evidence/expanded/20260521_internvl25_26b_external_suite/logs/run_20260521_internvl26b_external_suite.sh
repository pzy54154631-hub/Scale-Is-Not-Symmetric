#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/.autodl/mirage_csr
export HF_HOME=/root/autodl-tmp/hf_cache_internvl26b
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache_internvl26b
MODEL=/root/autodl-tmp/hf_cache_internvl26b/models--OpenGVLab--InternVL2_5-26B/snapshots/b537a9974b89cb621e9e6b9e7ebe2a904334fff0
run_one() {
  NAME="$1"; PARQ="$2"; OUT="$3"
  echo "=== START ${NAME} $(date -Is) ==="
  if [ -s "${OUT}.summary.json" ]; then
    echo "SKIP existing ${OUT}.summary.json"
    cat "${OUT}.summary.json"
  else
    /root/miniconda3/envs/mirage/bin/python scripts/eval_internvl_parquet_lora.py \
      --parquet "$PARQ" \
      --output "$OUT" \
      --model-dir "$MODEL" \
      --adapters base=NONE \
      --max-new-tokens 24
  fi
  echo "=== DONE ${NAME} $(date -Is) ==="
}
run_one chartbench26 results/external/20260521_chartbench_external_internvl8b/chartbench_numeric500.parquet results/eval/20260521_chartbench500_internvl25_26b_base.csv
run_one mathvista26 results/external/20260521_mathvista_local_external_internvl8b/mathvista_chart_table_numeric300.parquet results/eval/20260521_mathvista300_internvl25_26b_base.csv
run_one infographic26 results/external/20260521_infographicvqa_external_internvl8b/infographicvqa_numeric300.parquet results/eval/20260521_infographicvqa300_internvl25_26b_base.csv
ARCH=outputs/experiment_archives/20260521_internvl25_26b_external_suite.tar.gz
tar -czf "$ARCH" \
  results/eval/20260521_chartbench500_internvl25_26b_base.csv* \
  results/eval/20260521_mathvista300_internvl25_26b_base.csv* \
  results/eval/20260521_infographicvqa300_internvl25_26b_base.csv* \
  logs/run_20260521_internvl26b_external_suite.log \
  logs/run_20260521_internvl26b_external_suite.sh 2>/dev/null || true
echo ARCHIVE=$ARCH
