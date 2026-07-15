# Scale Is Not Symmetric

Research code and paper materials for **Scale Is Not Symmetric**, a study of directional, regime-dependent scale errors in chart question answering and their relationship to LoRA-based interventions and policy selection.

The repository also contains MIRAGE, an interference-aware rank-wise LoRA merging implementation. It maps LoRA updates into a canonical SVD representation and combines rank components using singular-value energy, cross-task sharedness, conflict, and optional calibration importance.

## Repository layout

```text
configs/       Experiment and merge configurations
docs/          Experiment inventory, status, and analysis notes
paper/         LaTeX source, figures, tables, and compiled paper
results/       Lightweight result summaries used by the paper
scripts/       Data preparation, training, evaluation, and analysis scripts
src/mirage/    Core MIRAGE implementation
```

Large model checkpoints, LoRA adapters, raw datasets, and machine-specific caches are intentionally not tracked.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For a lighter environment without PyTorch in the requirements file:

```bash
pip install -r requirements_no_torch.txt
```

Some training and evaluation scripts additionally require model-specific packages, local datasets, and GPU resources. Review each script's command-line arguments before running it.

## MIRAGE merge example

```bash
python scripts/merge_adapters.py \
  --method mirage \
  --adapters path/to/adapter_a path/to/adapter_b \
  --output outputs/merged/mirage
```

Available baselines include `average`, `ties`, `dare`, and the legacy MIRAGE ablation. Run the command below for the complete interface:

```bash
python scripts/merge_adapters.py --help
```

## Paper

The manuscript source is [`paper/main.tex`](paper/main.tex), and the compiled version is [`paper/main.pdf`](paper/main.pdf). From the `paper/` directory, rebuild it with:

```bash
latexmk -pdf main.tex
```

## Reproducibility notes

Start with the canonical checks:

```bash
make verify     # validate claims, evidence, tables, and checksums
make smoke      # compile all Python entrypoints
make paper      # rebuild the manuscript
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete workflow and [`repro/paper_claims.csv`](repro/paper_claims.csv) for the paper-claim-to-artifact mapping.

- Experiment configurations are under `configs/`.
- Paper-ready tables and lightweight summaries are included where available.
- The original raw archive and large learned weights are not part of this Git repository.
- Several scripts reference external datasets or local prediction files; adapt those paths to your environment.
- Historical run notes in `docs/` may contain machine-specific paths and are retained for provenance.

## Status

This is a cleaned research release assembled from the final project archive. It preserves the runnable source, manuscript, and compact evidence while excluding approximately 42GB of large weights and redundant run artifacts.
