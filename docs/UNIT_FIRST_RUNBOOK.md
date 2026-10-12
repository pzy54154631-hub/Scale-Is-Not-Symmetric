# Unit-first runbook (the experiment that could change the paper's weight)

Goal: a simple, deployable fix that reviewers can understand in one sentence —
the model reads the value and its displayed unit, a deterministic rule picks
the output scale. If it beats self-check by >= 3 points on ScaleQA-473 without
lowering full ChartQA test accuracy, it goes in the paper as the method; if not,
it is one more probe and the paper stays diagnostic.

## Compute
Inference only. Qwen2.5-VL-7B bf16 ~16 GB. On one A100/5090:
- ScaleQA-473, 3 prompts: ~15 min. With --samples 20: ~1.5 h.
- ChartQA test 2500, 3 prompts: ~1 h.
Rentable tonight on any cloud GPU; the script needs the same env as the repo.

## Commands
```bash
# 1. diagnostic set (paired vs direct and self_check, plus pass@20 and unit-aware vote)
python scripts/eval_unit_first.py --parquet <scaleqa473.parquet> \
  --output results/eval/unit_first_scaleqa473.csv --samples 20

# 2. full ChartQA test, safety (no sampling)
python scripts/eval_unit_first.py --parquet <chartqa_test.parquet> \
  --output results/eval/unit_first_chartqa_test.csv

# 3. optional: the prefer-large adapter instead of answer-only, same two commands with --adapter <path>
```

## What to read in the summary.json
- `methods.unit_first.acc` vs `methods.self_check.acc` (ScaleQA-473 self-check reference: 0.5455)
- `methods.unit_first.fix` / `.break` — break must stay low (self-check breaks ~? ; adapter breaks 7.4%)
- `methods.blind_pass_at_k.acc` — should reproduce ~0.79 at k=20
- `methods.unit_vote.acc` — unit-aware vote vs `blind_majority`
- `unit_first_rules` — if almost everything is `keep_displayed`, the model is not emitting units; inspect `raw`.

## Decision
| Outcome on ScaleQA-473 | ChartQA test | Action |
|---|---|---|
| unit_first >= self_check + 3 | not lower than direct | promote to method; add Table in §5; rewrite abstract/intro around it |
| +1 to +3 | not lower | report as strongest training-free policy; keep diagnostic framing |
| <= self_check | any | report in the selector/verifier ladder; no framing change |

## Paper hooks already in place
- Table `selector_main_comparison` has room for a `unit_first` row.
- Fix/break accounting and bootstrap scripts accept the output CSV unchanged.
- `repro/paper_claims.csv`: add C17 with the summary.json path.
