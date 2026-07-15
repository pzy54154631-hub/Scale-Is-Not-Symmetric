# MIRAGE: Modality-Interference-aware Rank-wise LoRA Merging for MLLMs

This repository is a runnable starter kit for an EMNLP-style paper on MLLM LoRA merging.

The intended story:

> Existing MLLM LoRA merging methods treat each adapter as a monolithic task vector. MIRAGE first maps each LoRA update into a canonical SVD-based low-rank space, then shows that interference is layer-wise, rank-wise, and modality-dependent. It merges canonical components according to singular-value energy, cross-task sharedness, conflict, and optional calibration importance.

## Repository Layout

```text
mirage_emnlp/
  paper/                 # LaTeX paper skeleton with empty result tables
  src/mirage/             # merging algorithms and utilities
  scripts/                # train/eval/merge/table scripts
  configs/                # experiment configs
  results/                # put raw evaluation outputs here
  figures/                # generated figures
```

## Recommended Environment

```bash
conda create -n mirage python=3.10 -y
conda activate mirage
pip install -r requirements.txt
```

Optional but recommended:

```bash
pip install flash-attn --no-build-isolation
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

## Minimal Workflow

### 1. Train or collect task LoRA adapters

Recommended task adapters:

```text
TextVQA      -> OCR ability
ChartQA     -> chart reasoning
DocVQA      -> document understanding
ScienceQA   -> science reasoning
GQA         -> general VQA
MathVista   -> visual math reasoning
```

Put adapters under:

```text
outputs/adapters/textvqa/
outputs/adapters/chartqa/
outputs/adapters/docvqa/
outputs/adapters/scienceqa/
outputs/adapters/gqa/
outputs/adapters/mathvista/
```

Each folder should contain PEFT adapter files:

```text
adapter_config.json
adapter_model.safetensors  # or adapter_model.bin
```

### 2. Merge adapters

Average baseline:

```bash
python scripts/merge_adapters.py \
  --method average \
  --adapters outputs/adapters/textvqa outputs/adapters/chartqa outputs/adapters/docvqa outputs/adapters/scienceqa outputs/adapters/gqa outputs/adapters/mathvista \
  --output outputs/merged/average_6task
```

TIES baseline:

```bash
python scripts/merge_adapters.py \
  --method ties \
  --adapters outputs/adapters/textvqa outputs/adapters/chartqa outputs/adapters/docvqa outputs/adapters/scienceqa outputs/adapters/gqa outputs/adapters/mathvista \
  --output outputs/merged/ties_6task \
  --density 0.5
```

DARE baseline:

```bash
python scripts/merge_adapters.py \
  --method dare \
  --adapters outputs/adapters/textvqa outputs/adapters/chartqa outputs/adapters/docvqa outputs/adapters/scienceqa outputs/adapters/gqa outputs/adapters/mathvista \
  --output outputs/merged/dare_6task \
  --density 0.5 \
  --seed 42
```

MIRAGE v2 paper method, canonical SVD + conflict-regularized weighting:

```bash
python scripts/merge_adapters.py \
  --method mirage \
  --adapters outputs/adapters/textvqa outputs/adapters/chartqa outputs/adapters/docvqa outputs/adapters/scienceqa outputs/adapters/gqa outputs/adapters/mathvista \
  --output outputs/merged/mirage_6task \
  --mirage-beta 8.0 \
  --mirage-gamma 2.0 \
  --mirage-temperature 1.0 \
  --mirage-energy-power 0.5
```

Raw-factor MIRAGE legacy ablation:

```bash
python scripts/merge_adapters.py \
  --method mirage_legacy \
  --adapters outputs/adapters/textvqa outputs/adapters/chartqa outputs/adapters/docvqa outputs/adapters/scienceqa outputs/adapters/gqa outputs/adapters/mathvista \
  --output outputs/merged/mirage_legacy_6task \
  --mirage-beta 8.0 \
  --mirage-margin 0.0
```

If you have calibration importance scores:

```bash
python scripts/merge_adapters.py \
  --method mirage \
  --adapters outputs/adapters/textvqa outputs/adapters/chartqa outputs/adapters/docvqa outputs/adapters/scienceqa outputs/adapters/gqa outputs/adapters/mathvista \
  --importance-json results/importance/qwen25vl_7b_6task.json \
  --output outputs/merged/mirage_calibrated_6task
```

### 3. Evaluate merged adapters

This starter uses `lmms-eval` as the default evaluation backend. Check and edit task names in `configs/tasks.yaml` according to your local installation.

```bash
python scripts/eval_lmms_eval.py \
  --model-name qwen2_5_vl \
  --base-model Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter outputs/merged/mirage_6task \
  --tasks textvqa chartqa docvqa scienceqa gqa mathvista \
  --output results/raw/mirage_6task.json
```

### 4. Make result table skeleton

```bash
python scripts/make_tables.py \
  --result-dir results/raw \
  --output paper/tables/results_main.tex
```

Then compile the paper:

```bash
cd paper
latexmk -pdf main.tex
```

## Important Notes

- The code is designed to be a strong experimental scaffold. You still need to verify task names, dataset access, and model-specific evaluation arguments on your machine or cluster.
- MIRAGE defaults to outputting a single PEFT adapter whose rank can grow to `num_adapters * original_rank`. This keeps the merged update exact.
- If you need fixed-rank compression, use `--target-rank`. The v2 code uses a QR + small-SVD trick, so it avoids materializing large dense LoRA update matrices.
- The paper skeleton intentionally leaves all results blank.

## Suggested Main Claim

> MIRAGE achieves state-of-the-art single-adapter MLLM LoRA merging performance across OCR, chart understanding, document QA, general VQA, science reasoning, and visual math reasoning, while revealing layer-wise, rank-wise, and modality-dependent interference among LoRA adapters.
