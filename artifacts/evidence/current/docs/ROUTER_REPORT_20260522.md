# Router and Scale-Regime Report

Generated: 2026-05-22 morning

This report is the fallback narrative for the current EMNLP/ACL draft. It separates stable results that can be written now from running experiments that may improve the final router number.

## One-Sentence Story

Scale errors in chart QA are not a single calibration problem. They are directional, regime-dependent reliability failures: many candidate systems can generate the right scaled answer, but deployable selectors still struggle to identify the correct scale regime.

## What Is Already Solved

### 1. The paper framing is now safer and stronger

The paper should not sell the global LoRA adapter as the main method. The defensible framing is:

1. scale inconsistency is directional and asymmetric;
2. ScaleQA-Natural is an author-adjudicated diagnostic set for natural scale conflicts;
3. global correction policies are unsafe because they repair one regime while breaking another;
4. regime-aware selection is the real deployment bottleneck.

This turns several negative results into evidence rather than weakness.

### 2. The no-leak router has a real, but modest, positive result

Best stable no-leak selector so far:

| Method | ScaleQA-Natural Mean |
|---|---:|
| self-check prompt | 0.5514 |
| best no-leak selector v10b/v11 | 0.5667 |
| split-selected fixed policy | 0.5274 |
| policy oracle | 0.9004 |

Interpretation: the learned selector does beat the strongest prompt baseline, but only by about +1.5 points. This is enough to claim deployable selection helps, but not enough to claim regime selection is solved.

### 3. The candidate-generation story is very strong

Multiple candidate/verifier pools have very high oracle scores:

| Candidate Pool / Verifier | Deployable Score | Oracle |
|---|---:|---:|
| structured verifier top4 | 0.4841 / 0.4376 | 0.9006 |
| structured candidate oracle | - | 0.9260 |
| policy pool oracle | - | 0.9004 |
| top-8 candidate likelihood/pairwise pool | weak deployable | about 0.9175 |
| v19 candidate-letter interim | about 0.45-0.51 | 0.9260 |

The insight is sharp: the correct answer is often present, but the model fails to select it reliably.

### 4. Cross-backbone evidence is now good enough for diagnostic claims

Stable diagnostic evidence exists across Qwen and InternVL backbones.

| Backbone / Setting | Result |
|---|---:|
| Qwen2.5-VL-7B raw ChartQA test | 0.7804 |
| Qwen2.5-VL-7B answer-tuned ChartQA test | 0.8860 |
| Qwen2.5-VL-7B answer + large-positive scale | 0.8932 |
| Qwen2.5-VL-3B ChartQA test base | 0.8436 |
| Qwen2.5-VL-3B ChartQA test large | 0.8488 |
| InternVL2.5-1B ChartQA test | 0.7524 |
| InternVL2.5-8B ScaleQA-520 base | 0.4672 |
| InternVL2.5-26B ScaleQA-520 base | 0.4989 |
| InternVL2.5-26B PlotQA-500 base | 0.7200 |
| InternVL2.5-26B ChartBench-500 base | 0.4080 |
| InternVL2.5-26B MathVista numeric | 0.6749 |
| InternVL2.5-26B InfographicVQA-300 | 0.6300 |

Recommended wording: the phenomenon transfers more robustly than the intervention. Do not claim the global adapter is universally effective.

### 5. ScaleQA-Natural credibility is mostly addressed

Safe wording:

> ScaleQA-Natural is an author-adjudicated diagnostic set, not a replacement for ChartQA or a broad benchmark.

The current paper can say:

- 520 reviewed/adjudicated items are used for the main diagnostic evaluation;
- expansion candidates remain a review queue until final human approval;
- rejected/uncertain examples are used to clarify inclusion boundaries;
- independent agreement exists according to the user's second review, but report only what is documented in the artifact.

Do not call the expansion a 1000-item benchmark until the approved subset is finalized.

## What Is Still Hard

### 1. Robust regime detection is not solved

Oracle performance remains much higher than deployable performance. This means the key scientific bottleneck is not candidate generation but scale-regime selection.

Current best wording:

> A no-leak selector improves over self-check, but the large oracle gap shows that deployable regime detection remains open.

Avoid:

> We solve scale-regime routing.

### 2. Verifiers and rerankers remain unstable

Structured verifier, table calculator, pairwise verifier, likelihood verifier, and letter-choice verifier all struggle. They often over-trust surface cues such as percent signs, unit words, candidate order, or agreement among wrong candidates.

This should be written as a reliability finding:

> More structure does not automatically yield safer scale correction; a verifier can see the right candidate and still choose the wrong scale regime.

### 3. Global scale adapters do not generalize reliably

The global large-positive adapter gives a small ChartQA test gain, but it is not the best ScaleQA-Natural policy and does not transfer cleanly to InternVL/PlotQA/ChartBench.

Safe wording:

> LoRA adapters are diagnostic probes for measuring directional asymmetry, not the final deployment solution.

### 4. Full external evaluation is limited by cache/network state

InternVL8B/26B full ChartQA test and new Qwen3B ScaleQA/PlotQA runs are blocked by missing local cache and inaccessible remote downloads. The existing results are enough for cross-backbone diagnostics, but not enough to claim complete full-benchmark transfer.

## Running Experiments

### v18 pairwise multi no-leak

Status: stopped early on 2026-05-22 because partial no-leak split results were not competitive with the stable v10b selector.

Early seed results:

| Seed | Selector | Self-check | Axis | Oracle |
|---:|---:|---:|---:|---:|
| 0 | 0.5579 | 0.5158 | 0.5895 | 0.9368 |
| 5 | 0.4842 | 0.5263 | 0.5789 | 0.9158 |
| 10 | 0.5474 | 0.6105 | 0.5684 | 0.9053 |

Partial outcome: useful failure analysis, not a stable SOTA candidate. The logged checkpoints average below the stable v10b selector despite one strong split.

### v19 candidate-letter top8

Status: stopped early on 2026-05-22 after candidate generation and partial selector post-processing.

Interim deployable scores are weak:

| Metric | Score |
|---|---:|
| candidate_majority | 0.5137 |
| letter_axis | 0.4524 |
| letter_scale_guard | 0.4503 |
| candidate_oracle | 0.9260 |

Partial outcome: strong evidence that letter-based verification sees the right answer but does not select reliably. It should be used as bottleneck evidence, not as a main result.

### v20 weight-splitting stacked selector

Status: running CPU.

Purpose: fix the earlier v11-v17 issue where new candlik/pairwise/letter policies were generated but not actually included in the selector policy whitelist. v20 dynamically includes all non-oracle policies and chooses among stack, patch, grouped-answer voting, bucket priors, and Bayesian-smoothed policy weights under no-leak train/dev/test splits.

If v20 exceeds 0.58, it can be promoted as the main deployable router. If not, it still becomes a strong ablation showing that expanding the policy pool alone is insufficient.

### v21 bucketed Bayesian selector

Status: launched.

Purpose: a faster statistical alternative to v20. It estimates policy reliability by question/unit bucket under train/dev splits, then performs Bayesian answer-posterior selection with candidate agreement evidence. It is designed as a readable statistical method for the paper, not a black-box model.

If v21 wins, it gives a clean story: simple Bayesian regime priors + answer consensus can improve scale selection. If it fails, it supports the same bottleneck conclusion.

## Recommended Paper Narrative

### Abstract framing

Use this logic:

1. We study scale inconsistency as a directional reliability failure in chart QA.
2. Scale correction is asymmetric: a policy that fixes one scale regime can break another.
3. ScaleQA-Natural provides an author-adjudicated diagnostic setting for natural scale conflicts.
4. Global adapters are useful diagnostic probes but are not sufficient deployment methods.
5. The bottleneck is regime-aware selection: no-leak selectors slightly improve over self-check, while oracle gaps show large remaining headroom.

### Main contribution order

Recommended order:

1. diagnostic finding: scale errors are directional and asymmetric;
2. dataset contribution: ScaleQA-Natural as a reviewed diagnostic set;
3. controlled interventions: adapters, prompts, structured verifiers, candidate pools;
4. selection result: no-leak selector beats self-check modestly;
5. insight: candidate generation is easier than candidate selection.

### Best sentence to use

> The phenomenon transfers more robustly than the intervention: scale inconsistency appears across backbones and external chart sets, but a single global correction policy does not reliably transfer.

### Second key sentence

> Even when the correct scaled answer is available in the candidate pool, deployable verifiers often fail to select it, making scale-regime detection the central bottleneck.

## What Not To Claim

Do not claim:

- the adapter is the main method;
- ScaleQA-Natural is a full benchmark replacement;
- the router solves regime detection;
- external transfer is strong across all backbones;
- v19/v18 are improvements unless final summaries prove it.

Use:

- diagnostic set;
- reliability probe;
- directional asymmetry;
- no-leak selector;
- oracle gap;
- selection bottleneck.

## Current Bottom Line

The paper is already viable as a diagnostic/reliability paper. The strongest scientific story is not a large accuracy jump; it is the discovery that scale correction is direction-sensitive and that modern MLLM pipelines often generate the correct scaled alternative without being able to choose it reliably.

If v20 or v21 reaches 0.58+, the paper gains a stronger method contribution. If they do not, the paper should still be written around the insight that regime-aware selection is the unsolved bottleneck.
