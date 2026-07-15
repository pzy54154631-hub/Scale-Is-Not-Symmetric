# Experiment Inventory, 2026-05-22

This inventory summarizes the local result archives reviewed for the supplemented EMNLP draft. It is for writing control, not a new paper section.

## Put In Main Text

| Evidence | Best number | Main use |
|---|---:|---|
| ChartQA held-out test, answer-tuned Qwen2.5-VL-7B + large-positive adapter | 89.32 vs 88.60 | The official positive task result; keep framed as modest evidence, not the whole claim. |
| Strict ChartQA test scale-error recovery | 49.52% | Shows the adapter repairs genuine gold-scale errors. |
| ScaleQA-Natural answer-tuned vs global adapter vs self-check | answer-tuned 0.5095; large-positive 0.4951; self-check 0.5514 | Shows global correction is weaker than regime-aware prompting. |
| Best no-leak selector v10b/v11/v13-v17 | 0.5667 +/- 0.0561 | The strongest deployable selector result; modestly beats self-check. |
| Expanded policy/candidate oracles | policy oracle 0.9004; top-8 oracle 0.9175; structured oracle 0.9260 | Core insight: correct candidates often exist, selection is the bottleneck. |
| Qwen7 external suite | ChartBench 0.4820; MathVista 0.6708; InfographicVQA 0.7767, unchanged across adapters | Transfer boundary: phenomenon is more robust than the intervention. |
| InternVL2.5-26B diagnostics | ScaleQA-520 0.4989; PlotQA-500 0.7200; ChartBench 0.4080; MathVista 0.6749; InfographicVQA 0.6300 | Cross-backbone diagnostic evidence, not an adaptation win. |

## Put In Appendix Or Compact Tables

| Evidence | Best number | Main use |
|---|---:|---|
| v8 no-leak selector | 0.5432 +/- 0.0501 | Early clean selector below self-check. |
| v9 expanded selector | 0.5501 +/- 0.0460 | Larger policy pool ties self-check. |
| v13 rationale / v14-v15 candidate likelihood / v16-v17 top-8 selector | all preserve 0.5667 | More evidence does not improve selection. |
| TTA v2 scale-regime prompts | decimal guard 0.5285; TTA oracle 0.7336 | Prompt variants have high oracle but weak deployable choice. |
| Structured verifier top4 | context-letter 0.4841; scale-audit 0.4376; oracle 0.9006 | Smaller candidate pools help little; oracle remains high. |
| Pairwise verifier top8 | 0.5433 | Close to self-check but not enough. |
| Table-calculator baseline | table_calc_or_reason 0.4228 | Strong baseline attempt; weak deployable result. |
| Regime verifier v3 | best deployable 0.4778; oracle 0.8901 | More direct verifier still fails selection. |
| Statistical fusion v1 | 0.5309 +/- 0.0470 | Probabilistic fusion under-switches/underperforms. |
| v21 bucketed Bayesian selector | 0.5239 +/- 0.0480 | Bayesian bucket reliability over-switches. |
| v22 Bayesian-MLP selector | 0.5218 +/- 0.0540 | Hidden-layer selector over-switches despite high oracle. |

## Do Not Emphasize As Wins

- v6 policy patch selector reports 0.6861 but uses a different/unsafe setting; do not present as the main no-leak result.
- v18/v19/v20 were stopped because partial results were not competitive or too slow.
- Qwen7 ScaleQA-stress1000 gives 0.6940 for every variant; this is useful negative transfer evidence, not an adapter improvement.
- InternVL and external numeric datasets are diagnostics. Do not describe them as full benchmark submissions or broad SOTA.

## Main Story Supported By The Inventory

1. Scale inconsistency is directional, sparse in strict official splits, and easier to diagnose than to repair globally.
2. LoRA adapters are useful probes: large-positive helps held-out ChartQA strict scale errors, but global correction is unsafe on ScaleQA-Natural.
3. A deployable no-leak selector gives the best ScaleQA-Natural result, but only modestly above self-check.
4. High oracles across prompts, verifiers, and candidate pools show that answer generation is not the only problem; selecting the correct scale regime is the bottleneck.
5. External and cross-backbone checks support a bounded claim: scale-regime failures transfer more robustly than the current intervention.
