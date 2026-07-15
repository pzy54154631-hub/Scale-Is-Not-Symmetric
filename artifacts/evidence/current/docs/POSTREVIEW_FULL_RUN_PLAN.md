# Post-Review Full Run Plan

Workspace copy:
`/Users/pengziyu/writing/mirage_emnlp_main_POSTREVIEW_20260520_192614`

Goal:
Strengthen the EMNLP/ACL main-paper version with post-review experiments while keeping the previous improved draft intact.

## Estimated Runtime

Best case on one A100-class GPU:
12-18 hours.

Conservative estimate:
18-30 hours.

Risk case:
30-48 hours if a larger backbone has loading issues, missing cache files, or external dataset conversion problems.

## Experiment Blocks

1. Stronger cross-backbone evaluation
- Target: InternVL2.5-8B or another stronger non-Qwen VLM.
- Minimum: base diagnostic on ChartQA test, ScaleQA-Natural-520, PlotQA numeric subset.
- Stronger version: large/small scale LoRA adaptation.
- Estimate:
  - Base diagnostic only: 2-5 hours.
  - Full large/small adapter + eval: 8-16 hours.

2. Held-out scale-regime router
- Train router on ScaleQA-Natural train/dev.
- Evaluate on held-out ScaleQA-Natural, ChartQA strict probes, and PlotQA numeric-500 where possible.
- Compare against fixed large/small/balanced, self-check prompt, and TTA majority.
- Estimate: 0.5-2 hours CPU/GPU-light if prediction files already exist.

3. External non-ChartQA validation
- Expand PlotQA from numeric-500 to larger numeric subset or full numeric subset.
- Optionally add ChartBench/ChartX/InfographicVQA if data is locally available.
- Estimate:
  - PlotQA full numeric: 2-6 hours per model.
  - New dataset conversion + eval: 4-12 hours.

4. Quantitative error analysis
- Compute fix/break/over-correction categories.
- Break down by subtype, answer magnitude, percentage/unit cues, and source split.
- Estimate: 0.5-2 hours CPU.

5. ScaleQA-Natural expansion
- Mine more candidates from additional model outputs or external datasets.
- Run machine prefill, Codex annotation, and author review queue.
- Estimate:
  - Mining/prefill: 1-4 hours.
  - Model inference for new candidates: 4-12 hours depending on source.
  - Human review: variable.

## Recommended Full-Run Order

1. Run router held-out and error analysis first because they are cheap and immediately paper-useful.
2. Run stronger non-Qwen base diagnostics next.
3. If the stronger backbone looks meaningful, run large/small adapter training.
4. Expand PlotQA numeric evaluation while adapters are training or after the first strong-backbone run.
5. Only start ScaleQA-Natural expansion if the above results still leave the paper weak.

## Paper Integration Rule

Do not overwrite the stable or improved draft. Integrate results only into this post-review copy first.
