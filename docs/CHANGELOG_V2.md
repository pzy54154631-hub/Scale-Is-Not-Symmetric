# MIRAGE v2 Changes

This update makes the code align with the stronger mathematical method described in the paper draft.

## New paper-method implementation

`--method mirage` now runs canonical MIRAGE:

1. Convert each LoRA update `Delta W = scale * B @ A` into a canonical SVD representation.
2. Avoid dense update materialization using QR + small SVD.
3. Compute best-match rank-wise sharedness and conflict across adapters.
4. Use a softmax relaxation of a conflict-regularized objective:

```text
alpha ∝ exp((log importance + p log sigma - beta conflict + gamma sharedness) / temperature)
```

5. Reconstruct one deployable merged LoRA adapter.

## Legacy ablation

The old direct raw-factor rank decomposition is still available as:

```bash
--method mirage_legacy
```

Use this for ablations such as `w/o canonical decomposition`.

## New CLI flags

```bash
--mirage-beta            conflict suppression strength
--mirage-gamma           sharedness reward strength
--mirage-temperature     softmax temperature
--mirage-energy-power    singular-value energy exponent
--target-rank            optional fixed-rank recompression
```

## Files changed

```text
src/mirage/lora_utils.py
src/mirage/merge.py
src/mirage/io.py
scripts/merge_adapters.py
scripts/compute_structural_conflict.py
scripts/plot_conflict_heatmap.py
scripts/run_merge_suite.sh
paper/sections/04_method.tex
README.md
```
