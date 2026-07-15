#!/usr/bin/env bash
set -euo pipefail

ADAPTERS=(
  outputs/adapters/textvqa
  outputs/adapters/chartqa
  outputs/adapters/docvqa
  outputs/adapters/scienceqa
  outputs/adapters/gqa
  outputs/adapters/mathvista
)

mkdir -p outputs/merged results/conflict figures

python scripts/merge_adapters.py --method average --adapters "${ADAPTERS[@]}" --output outputs/merged/average_6task
python scripts/merge_adapters.py --method task_arithmetic --adapters "${ADAPTERS[@]}" --output outputs/merged/task_arithmetic_6task
python scripts/merge_adapters.py --method ties --adapters "${ADAPTERS[@]}" --output outputs/merged/ties_6task --density 0.5
python scripts/merge_adapters.py --method dare --adapters "${ADAPTERS[@]}" --output outputs/merged/dare_6task --density 0.5 --seed 42
python scripts/merge_adapters.py --method breadcrumbs --adapters "${ADAPTERS[@]}" --output outputs/merged/breadcrumbs_6task --density 0.8

# Raw-factor MIRAGE is kept as an ablation only.
python scripts/merge_adapters.py --method mirage_legacy --adapters "${ADAPTERS[@]}" --output outputs/merged/mirage_legacy_6task --mirage-beta 8.0 --mirage-margin 0.0

# Paper method: canonical SVD + conflict-regularized component weights.
python scripts/merge_adapters.py \
  --method mirage \
  --adapters "${ADAPTERS[@]}" \
  --output outputs/merged/mirage_6task \
  --mirage-beta 8.0 \
  --mirage-gamma 2.0 \
  --mirage-temperature 1.0 \
  --mirage-energy-power 0.5

python scripts/compute_structural_conflict.py \
  --adapters "${ADAPTERS[@]}" \
  --names textvqa chartqa docvqa scienceqa gqa mathvista \
  --output-json results/conflict/structural_conflict_6task.json

python scripts/plot_conflict_heatmap.py \
  --conflict-json results/conflict/structural_conflict_6task.json \
  --output figures/conflict_heatmap.pdf
