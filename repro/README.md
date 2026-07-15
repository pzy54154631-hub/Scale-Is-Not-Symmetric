# Reproduction metadata

- `paper_claims.csv` maps reported claims to the closest canonical script and preserved evidence.
- `models.yaml` records external model and dataset identifiers without assuming machine-specific paths.
- `SHA256SUMS.txt` pins the compact evidence bundle and paper tables included in this release.

The levels used in the claim matrix are:

- `fully_rerunnable`: runnable from repository source with standard tooling.
- `rerunnable_analysis`: recomputable from committed prediction/evidence files after installing the analysis dependencies.
- `archived_predictions`: the exact outputs are preserved, but inference requires separately downloaded models and GPU resources.

