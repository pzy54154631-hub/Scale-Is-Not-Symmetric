# Reproducibility guide

This repository distinguishes three reproducibility targets instead of treating a 42GB archival snapshot as a runnable package.

## 1. Verify the released evidence

This requires only Python 3.10+ and checks the repository structure, claim matrix, preserved artifacts, paper inputs, checksums, and accidental machine-specific paths in canonical files.

```bash
make verify
```

## 2. Re-run lightweight analysis

Create the environment and run the deterministic smoke test:

```bash
conda env create -f environment.yml
conda activate scale-is-not-symmetric
make smoke
```

The claim-to-script mapping is in `repro/paper_claims.csv`. Archived prediction files are under `artifacts/evidence/`; compact ScaleQA outputs used directly by the release remain under `results/`.

## 3. Re-run model inference and training

Download the external models and datasets listed in `repro/models.yaml`. Set cache and model locations rather than editing scripts:

```bash
export MODEL_DIR=/path/to/Qwen2.5-VL-7B-Instruct
export HF_HOME=/path/to/huggingface-cache
export HF_DATASETS_CACHE=/path/to/datasets-cache
```

The main Qwen experiment uses deterministic decoding (`do_sample=False`, `max_new_tokens=24`). Training details reported in the appendix are rank 16, LoRA alpha 32, dropout 0.05, AdamW, and a fixed seed per run. The final scale adapters use 300 large-positive pairs, 300 steps, learning rate `1e-6`, beta 2.0, and margin 0.10.

Historical scripts are retained for provenance and some still contain old `/root/autodl-tmp` defaults. Prefer an explicit `--model-dir`, `MODEL_DIR`, or cache environment variable. The verifier reports hard-coded paths only as warnings because changing historical experiment code could alter provenance.

## Paper build

```bash
make paper
```

The command uses `latexmk` and writes temporary files to `paper/build/`. A successful build produces `paper/build/main.pdf`.

## What is and is not included

Included:

- manuscript source and compiled PDF;
- core MIRAGE implementation and experiment scripts;
- compact annotation, prediction, selector, verifier, and external-evaluation evidence;
- claim mapping and SHA-256 integrity manifest.

Not duplicated in Git:

- public base-model checkpoints available from their original providers;
- redundant copies of the same LoRA adapters from multiple old workspaces;
- caches, intermediate checkpoints, and machine-specific archives.

This layout supports paper verification and re-analysis while keeping the Git repository usable. Exact full training remains compute-intensive and depends on the third-party model/dataset licenses.

