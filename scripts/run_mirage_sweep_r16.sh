#!/usr/bin/env bash
set -e

mkdir -p outputs/merged

TEXT=outputs/adapters/textvqa_answer_only_3000_r16
CHART=outputs/adapters/chartqa_answer_only_3000_r16

for beta in 2 4 8 16; do
  for gamma in 1 2 4; do
    for epow in 0.0 0.5 1.0; do
      OUT=outputs/merged/textvqa_chartqa_answer_only_3000_r16_mirage_b${beta}_g${gamma}_e${epow}
      echo "Merging $OUT"
      python scripts/merge_adapters.py \
        --method mirage \
        --adapters $TEXT $CHART \
        --output $OUT \
        --mirage-beta $beta \
        --mirage-gamma $gamma \
        --mirage-temperature 1.0 \
        --mirage-energy-power $epow
    done
  done
done
