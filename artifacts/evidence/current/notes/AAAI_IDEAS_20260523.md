# AAAI-27 Idea Memo, 2026-05-23

## Timeline Anchor

AAAI-27 full paper deadline: 2026-07-28 AoE. Supplementary material and code deadline: 2026-07-31 AoE.

Official source: https://aaai.org/conference/aaai/aaai-27/

This leaves roughly 9 weeks from 2026-05-23, so the best idea should reuse the existing ScaleQA-Natural / router / verifier infrastructure and add one sharp new mechanism rather than starting a new benchmark from zero.

## Current Base

Existing project story:

- Chart QA scale errors are directional and regime-dependent.
- Global scale correction is unsafe.
- Candidate pools often contain the correct scaled answer.
- The bottleneck is selecting the correct scale regime.
- Best no-leak selector is modestly above self-check, while candidate/policy oracles are around 90-92%.

This is already a strong diagnostic paper. For AAAI, the new idea should move from "we found the bottleneck" to "we propose a principled way to solve or characterize the bottleneck."

## Recommended Main Idea

### Working Title

RegimeLens: Calibrated Counterfactual Selection for Numerical Scale Reliability in Multimodal Chart Reasoning

### One-Sentence Claim

MLLM chart failures are often not answer-generation failures but counterfactual selection failures; by explicitly estimating which scale-regime counterfactual should be trusted, a small calibrated selector can close part of the oracle gap while providing abstention and error diagnosis.

### Core Research Question

Can scale-regime errors in chart QA be reduced by treating candidate answers as counterfactual regimes and learning a calibrated, abstaining selector over regime evidence, rather than training another global correction model?

### Why This Is Better Than "Another Router"

The current router is framed as a lightweight selector. The AAAI version should make the scientific contribution sharper:

1. Counterfactual answer set: base answer, self-check answer, unit/axis guarded answer, large-scale variant, small-scale variant, table-calculated candidate.
2. Regime evidence: question wording, base answer magnitude, OCR/chart context, axis/unit/tick evidence, candidate likelihood, agreement/disagreement patterns.
3. Calibrated selection: not just choose one candidate, but estimate confidence and abstain when evidence is weak.
4. Reliability decomposition: separate generation coverage, selection accuracy, calibration, and abstention utility.

### Method Sketch

For each example, build a candidate set C = {c1, ..., ck}. Each candidate corresponds to a regime transformation or evidence source.

Train a selector s(c | x, q, context, C) using no-leak train/dev/test splits. The selector outputs:

- selected candidate;
- calibrated confidence;
- abstain/keep-base decision;
- predicted regime label for analysis only.

Possible models in increasing ambition:

- Logistic regression / calibrated gradient boosting over engineered evidence.
- Pairwise Bradley-Terry style selector over candidate pairs.
- Conformal selective prediction wrapper around the existing best selector.
- Lightweight cross-encoder using chart context text and candidate explanations, if time permits.

The key is not model size. The key is the evaluation lens: selection coverage, oracle coverage, calibration, and selective risk.

### Primary Experiments

1. ScaleQA-Natural no-leak splits
   - Compare base, self-check, current v10b selector, RegimeLens selector, oracle.
   - Metrics: accuracy, selective accuracy, ECE/Brier score, coverage-risk curve.

2. ChartQA strict scale-error recovery
   - Show whether the selector repairs strict factor errors without globally breaking base-correct cases.
   - Metrics: fix rate, break rate, net accuracy, abstention utility.

3. Cross-backbone diagnostic
   - Reuse Qwen and InternVL predictions already in the workspace.
   - Test whether the learned reliability features transfer as a diagnostic, even if absolute accuracy drops.

4. Candidate coverage vs. selection
   - Report candidate oracle, selector accuracy conditioned on oracle-positive examples, and failures where the correct answer exists but is not selected.

5. Ablations
   - No chart context.
   - No candidate likelihood.
   - No pairwise evidence.
   - No calibration/abstention.
   - Fixed global policy.

### Reviewer-Facing Contributions

1. A counterfactual-regime formulation of scale reliability in chart QA.
2. A calibrated selective selector that improves deployable reliability over self-check/current router.
3. A reliability decomposition showing where chart MLLMs fail: candidate coverage, selection, calibration, or abstention.
4. Cross-backbone evidence that regime features are meaningful beyond one model.

### Failure Modes

- If accuracy gain is small, emphasize selective reliability: high-confidence subset becomes much safer.
- If transfer is weak, frame it as a diagnostic boundary: features explain failure but do not universally transfer.
- If calibration is weak, use conformal prediction or abstention thresholds chosen only on dev.

### Minimum Viable AAAI Paper

The paper is viable if it gets:

- Clearer framing than the EMNLP version.
- A selector that beats self-check or current v10b on at least one no-leak setting.
- A strong selective-risk result even if full-coverage accuracy improves only modestly.
- A clean decomposition table showing the candidate-generation/selection/calibration gap.

## Backup Idea 1

### Working Title

When More Evidence Hurts: Over-Switching and Calibration Failures in Multimodal Chart QA Verification

### Claim

Adding candidates, rationales, table context, and verifier prompts can increase oracle coverage while reducing deployable reliability because the verifier becomes miscalibrated and over-switches away from safe answers.

### Why It May Work

This is strongly supported by the existing negative results: structured verifier, pairwise verifier, Bayesian fusion, TTA prompts, top-8 candidate pools. AAAI reviewers often like careful failure-analysis papers when the evaluation is systematic and general.

### Core Experiment

Construct a verifier-evidence ladder and measure:

- oracle coverage;
- deployable accuracy;
- switch rate;
- false-switch rate;
- calibration error;
- direction-specific asymmetry.

### Risk

It may read too diagnostic unless paired with a useful mitigation such as selective abstention or conservative switching.

## Backup Idea 2

### Working Title

ScaleQA-Stress: Synthetic-to-Natural Stress Testing for Unit and Scale Robustness in Chart Reasoning

### Claim

Controlled chart transformations can expose scale-regime vulnerabilities that are sparse in official benchmarks but predictive of natural failures.

### Method

Generate controlled chart variants from source tables:

- percent vs decimal display;
- thousands/millions axis labels;
- normalized vs raw values;
- log/linear scale where feasible;
- label omission or unit ambiguity.

Evaluate whether models preserve answer convention under transformations.

### Risk

This is more benchmark-heavy and could take too long unless the generation pipeline already works. Use only if the current synthetic scripts are stable.

## My Pick

Pick RegimeLens.

Reason: it reuses the existing ScaleQA-Natural and router infrastructure, converts the current oracle-gap story into a positive method contribution, and gives a natural AAAI angle: reliable AI systems should know when a candidate answer is not just plausible but warranted by the visual/numeric regime.

## Decision Questions

Answering these will determine whether RegimeLens should be framed as a method paper, a reliability paper, or a benchmark-plus-analysis paper.

1. What do we want reviewers to remember as the single new contribution: calibrated selection, selective abstention, or the counterfactual-regime formulation?
2. Which result would make the paper feel like a win even if full-coverage accuracy only improves slightly: better high-confidence accuracy, lower false-switch rate, or cleaner cross-backbone diagnosis?
3. What evidence can be produced fastest from existing artifacts without new expensive inference?
4. If a reviewer says "this is just another router," what precise theorem, metric, decomposition, or experimental contrast proves that it is a reliability formulation instead?

## 2-Week Action Plan

### Week 1

- Freeze current prediction/candidate artifacts.
- Build a unified candidate table with one row per example-candidate pair.
- Add labels for answer correctness, candidate source, regime transformation, and oracle membership.
- Implement calibration metrics: ECE, Brier, coverage-risk curves.
- Reproduce current v10b selector from the unified table.

### Week 2

- Train calibrated selector variants.
- Add conformal or dev-threshold abstention.
- Run ablations over evidence groups.
- Produce the first AAAI-style main table:
  - full coverage accuracy;
  - selective accuracy at 70/80/90% coverage;
  - ECE/Brier;
  - fix/break.

## Paper Skeleton

1. Introduction: scale reliability is a counterfactual selection problem.
2. Problem: define scale regimes, candidate coverage, selector risk, selective risk.
3. Data and candidate generation: ScaleQA-Natural, ChartQA strict probes, external diagnostics.
4. Method: RegimeLens calibrated selector and abstention.
5. Experiments: main accuracy, selective reliability, cross-backbone, ablation.
6. Analysis: over-switching, direction asymmetry, correct-candidate-not-selected failures.
7. Limitations: diagnostic set size, chart-context extraction noise, transfer limits.

## Related Work Anchors To Read/Cite

- ChartQA: https://arxiv.org/abs/2203.10244
- ChartInsights: https://arxiv.org/abs/2405.07001
- ChartQAPro: https://arxiv.org/abs/2504.05506
- MultiChartQA: https://aclanthology.org/2025.naacl-long.566/
- ChartCoder: https://aclanthology.org/2025.acl-long.363/
- Prompting strategies for ChartQA: https://arxiv.org/abs/2603.22288
