# Strong-Baselines Revision Snapshot

This folder is the local snapshot for the post-review revision focused on stronger baselines, structured reasoning, and ScaleQA-Natural presentation.

## 2026-05-22 Supplemental Integration Pass

- Created this supplemental copy from the existing writing snapshot rather than rewriting the paper architecture from scratch.
- Integrated recent selector experiments into the existing policy-selection story:
  - v8 no-leak selector: `0.5432`.
  - v9 expanded selector: `0.5501`.
  - v10b no-leak selector: `0.5667`, the first clean selector above self-check (`0.5514`).
  - v11/v13--v17 evidence variants preserve `0.5667` but do not improve it.
  - v21 bucketed Bayesian selector: `0.5239`.
  - v22 Bayesian-MLP selector: `0.5218`.
- Integrated Qwen2.5-VL-7B external diagnostics into the transfer table:
  - ChartBench-500: all variants `0.4820`.
  - MathVista-243: all variants `0.6708`.
  - InfographicVQA-300: all variants `0.7767`.
  - ScaleQA-stress1000: all variants `0.6940`.
- Updated `sections/05_experiments.tex`, `sections/06_analysis.tex`, `sections/08_limitations.tex`, and the selector/transfer tables.
- Compiled `paper/main.pdf` successfully. The paper remains 17 pages total, with main text/references ending on page 8 and appendix starting on page 9.

## 2026-05-22 Writing Sprint Snapshot

- New writing folder: `/Users/pengziyu/writing/mirage_emnlp_main_WRITING_20260522_0905`
- Reframed title, abstract, introduction, method, analysis, conclusion, and limitations around directional scale reliability and regime-aware policy selection.
- Demoted LoRA from the main claimed solution to a diagnostic probe; kept the held-out ChartQA gain as a measured result: 88.60 to 89.32, +0.72 points over answer tuning.
- Promoted the strongest no-leak selector result: 0.5667 versus split-matched self-check 0.5514, while explicitly preserving the large oracle gap: policy-pool 0.9004, top-8 candidate 0.9175, structured oracle 0.9260.
- Added InternVL2.5-26B diagnostic rows to the cross-backbone/external table: ScaleQA-520 0.4989, PlotQA-500 0.7200, ChartBench-500 0.4080, MathVista numeric 0.6749, InfographicVQA-300 0.6300.
- Compiled `paper/main.pdf` successfully with `latexmk`; current PDF has 17 total pages, with references starting on page 8 and appendix starting on page 9.

## Local Paper Snapshot

- Main paper source: `paper/main.tex`
- Compiled PDF: `paper/main.pdf`
- Added/updated tables:
  - `paper/tables/deployable_selector_results.tex`
  - `paper/tables/table_calculator_subtypes.tex`
  - `paper/tables/fine_error_taxonomy.tex`

## Saved Remote Archives

- `remote_archives/20260521_scaleqa_routing_reranker_taxonomy_plus_fast.tar.gz`
- `remote_archives/20260521_scaleqa520_table_calculator.tar.gz`
- `remote_archives/20260521_scaleqa520_structured_verifier.tar.gz`

## Structured Verifier Result

- Remote archive: `/root/autodl-tmp/.autodl/mirage_csr/outputs/experiment_archives/20260521_scaleqa520_structured_verifier.tar.gz`
- `structured_context_letter`: 0.4503
- `structured_scale_audit`: 0.4376
- `structured_candidate_oracle`: 0.9260
- Main interpretation: adding chart context and explicit scaled candidate variants makes the oracle very high, but the deployable verifier strongly prefers the prediction-too-small direction and fails on prediction-too-large cases. This strengthens the regime-selection bottleneck story.

## Current Writing Direction

Position the paper as a diagnostic/reliability study:
scale correction is directional and regime-dependent; global adapters and stronger deployable baselines do not fully solve selection, so reliable scale-regime detection is the central open problem.

## Paper Edits Applied

- Abstract and introduction now mention structured-reasoning baselines and high structured-candidate oracle results.
- ScaleQA-Natural section now describes release fields, source-row image references, and annotation inclusion/exclusion guidelines.
- Analysis now frames candidate generation, table extraction, structured verification, and routing as evidence for the regime-selection bottleneck.
- Limitations now acknowledge that the strong baselines are still lightweight and not specialized chart parsers.
- Compiled `paper/main.pdf` successfully after edits; total PDF length is 15 pages with main text ending before references.

## 2026-05-21 Follow-up

- Added strict-scale prevalence audit from available ChartQA prediction files:
  - `paper/tables/strict_scale_prevalence_available.tex`
  - `results/analysis/prevalence_audit_v3/strict_scale_prevalence_available.csv`
- Added negative/uncertain audit examples:
  - `paper/tables/negative_uncertain_audit_examples.tex`
  - `results/analysis/prevalence_audit_v3/negative_uncertain_audit_examples.csv`
- Added a compact regime-selection pipeline figure to `sections/04_method.tex`.
- Recompiled `paper/main.pdf` successfully; total PDF length is 16 pages, with main text/references ending on page 8.
- Saved archive locally: `remote_archives/20260521_prevalence_negative_audit_v3.tar.gz`.

## Remote Jobs Running/Queued

- Completed: structured verifier top-4 candidate-order run.
  - `structured_context_letter`: `0.4841`
  - `structured_scale_audit`: `0.4376`
  - `structured_candidate_oracle`: `0.9006`
  - Directional failure is sharp: context-letter verifier scores `0.8060` when the prediction is too small, but only `0.1743` when the prediction is too large.
  - Local archive: `remote_archives/20260521_scaleqa520_structured_verifier_top4.tar.gz`
  - Interpretation: reducing the candidate pool improves the deployable verifier over the earlier structured verifier, but it still does not beat self-check. The high oracle and severe direction split strengthen the policy-selection bottleneck story.
- Completed: structured verifier ordered-candidate follow-up.
  - `numeric_asc`: `0.3404`
  - `numeric_desc`: `0.2833`
  - `source_base_last`: `0.4947`
  - Best direction split for `source_base_last`: `0.7974` when the prediction is too small, but only `0.2033` when the prediction is too large.
  - Local archive: `remote_archives/20260521_scaleqa520_structured_verifier_ordered.tar.gz`
  - Interpretation: changing candidate order does not solve verifier selection. The result reinforces the same insight as the top-4 verifier: VLM-based verification is strongly direction/regime biased even when the correct scaled answer is available.
- Completed: no-leak policy selector v8.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_policy_selector_v8_fast_noleak.log`
  - `selector_mean`: `0.5432`
  - `selector_std`: `0.0501`
  - `selfcheck_mean`: `0.5514`
  - `axis_mean`: `0.5411`
  - `best_fixed_split_mean`: `0.5665`
  - `policy_oracle_mean`: `0.7040`
  - Local archive: `remote_archives/20260521_scaleqa_policy_selector_v8_fast_noleak.tar.gz`
  - Interpretation: the clean no-leak selector does not beat self-check; use this as evidence that deployable regime selection remains hard, not as a main-method win.
- Completed: pairwise verifier follow-up.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_scaleqa520_pairwise_verifier.log`
  - `pairwise_default_vs_alt`: `0.5222`
  - Direction split: `0.7716` when the prediction is too small, but only `0.2822` when the prediction is too large.
  - Local archive: `remote_archives/20260521_scaleqa520_pairwise_verifier.tar.gz`
  - Interpretation: pairwise comparison improves over the weaker structured verifiers but still does not beat self-check; it reinforces that verifier decisions remain highly regime-biased.
- Completed: Bayesian policy fusion v1.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_bayesian_policy_fusion_v1.log`
  - `bayes_mean`: `0.5124`
  - `bayes_std`: `0.0515`
  - `selfcheck_mean`: `0.5514`
  - `axis_mean`: `0.5411`
  - `oracle_mean`: `0.8128`
  - Local archive: `remote_archives/20260521_scaleqa_bayesian_policy_fusion_v1.tar.gz`
  - Interpretation: the simple Beta-smoothed fusion underperforms self-check.
- Completed: Bayesian policy fusion v2 after pairwise verifier.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_bayesian_policy_fusion_v2_after_pairwise.log`
  - `bayes_mean`: `0.5149`
  - `bayes_std`: `0.0496`
  - `selfcheck_mean`: `0.5514`
  - `pairwise_mean`: `0.5314`
  - `oracle_mean`: `0.8128`
  - Local archive: `remote_archives/20260521_scaleqa_bayesian_policy_fusion_v2_after_pairwise.tar.gz`
  - Interpretation: adding pairwise verifier evidence slightly helps fusion but still does not beat self-check.
- Running: Bayesian policy fusion v3 with TTA-v2 evidence.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_bayesian_policy_fusion_v3_tta2.log`
  - Remote output: `/root/autodl-tmp/.autodl/mirage_csr/results/scaleqa_natural/bayesian_policy_fusion_v3_tta2/scaleqa520_bayesian_policy_fusion_v1_summary.json`
  - Goal: include pairwise verifier and TTA-v2 prompts in the Beta-smoothed policy-fusion posterior.
- Completed: expanded no-leak policy selector v9.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_policy_selector_v9_expanded_noleak.log`
  - `selector_mean`: `0.5501`
  - `selector_std`: `0.0460`
  - `selfcheck_mean`: `0.5514`
  - `axis_mean`: `0.5411`
  - `best_fixed_split_mean`: `0.5665`
  - `policy_oracle_mean`: `0.8128`
  - Local archive: `remote_archives/20260521_scaleqa_policy_selector_v9_expanded_noleak.tar.gz`
  - Interpretation: adding structured and pairwise verifier candidates greatly raises the oracle ceiling but the no-leak selector still only ties self-check. This strengthens the candidate-available/selection-hard bottleneck.
- Running: expanded no-leak policy selector v10 with TTA-v2 evidence.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_policy_selector_v10_tta2_noleak.log`
  - Remote output: `/root/autodl-tmp/.autodl/mirage_csr/results/scaleqa_natural/policy_selector_v10_tta2_noleak/scaleqa520_policy_selector_v10_tta2_noleak_summary.json`
  - Note: this first version accidentally omitted the TTA-v2 adapter names from the policy list, so its output duplicated v9. A corrected v10b has been launched.
- Running: corrected expanded no-leak policy selector v10b with TTA-v2 evidence.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_policy_selector_v10b_tta2_noleak.log`
  - Remote output: `/root/autodl-tmp/.autodl/mirage_csr/results/scaleqa_natural/policy_selector_v10b_tta2_noleak/scaleqa520_policy_selector_v10b_tta2_noleak_summary.json`
  - Goal: add the scale-regime TTA-v2 predictions to the no-leak candidate pool and test whether a learned selector can exploit the high TTA-v2 oracle.
- Completed: scale-regime TTA prompt v2 on ScaleQA-Natural.
  - Remote log: `/root/autodl-tmp/.autodl/mirage_csr/logs/run_20260521_scaleqa520_tta_v2_scale_regime.log`
  - `decimal_guard`: `0.5285`
  - `axis_unit_audit`: `0.4482`
  - `percent_display`: `0.4440`
  - `regime_check`: `0.4440`
  - `tta_majority`: `0.4440`
  - `tta_oracle`: `0.7336`
  - Local archive: `remote_archives/20260521_scaleqa520_tta_v2_scale_regime.tar.gz`
  - Interpretation: stronger hand-written scale prompts still fail to beat the original self-check baseline, while their oracle is high. This supports the paper's candidate-generation-versus-selection story.

- Completed: InternVL2.5-26B base diagnostic on ScaleQA-Natural-520.
  - Score: `0.4989`
  - Remote archive: `/root/autodl-tmp/.autodl/mirage_csr/outputs/experiment_archives/20260521_scaleqa520_internvl25_26b_base_v2.tar.gz`
- Completed: InternVL2.5-26B base diagnostic on PlotQA numeric-500.
  - Score: `0.7200`
  - Remote archive: `/root/autodl-tmp/.autodl/mirage_csr/outputs/experiment_archives/20260521_plotqa500_internvl25_26b_base.tar.gz`
- Completed: regime-router v5 fast CPU experiment.
  - `router_mean`: `0.5453`
  - `selfcheck_mean`: `0.5514`
  - `best_fixed_split_mean`: `0.5665`
  - `true_regime_oracle_mean`: `0.6425`
  - Interpretation: the learned deployable router still does not beat self-check, but the oracle gap supports the paper's regime-selection bottleneck claim.
  - Local archive: `remote_archives/20260521_scaleqa_regime_router_v5_fast.tar.gz`
- Completed: policy patch selector v6 CPU experiment.
  - `selector_mean`: `0.6861`
  - `selector_std`: `0.0424`
  - `selfcheck_mean`: `0.5514`
  - `axis_mean`: `0.5411`
  - `best_fixed_split_mean`: `0.5665`
  - `policy_oracle_mean`: `0.7040`
  - `switch_rate_mean`: `0.3996`
  - Selected models: RF 31 seeds, ExtraTrees 19 seeds.
  - Selected modes: stack 25 seeds, conservative patch 25 seeds.
  - Important caveat: this version used manifest-derived direction/subtype features and should be treated as an upper-bound/ablation, not a deployable no-leak selector.
  - Local archive: `remote_archives/20260521_scaleqa_policy_patch_selector_v6.tar.gz`
- Completed: InternVL2.5-26B external-suite base diagnostics.
  - ChartBench-500: `0.4080`
  - MathVista chart/table numeric subset: `0.6749`
  - InfographicVQA-300: `0.6300`
  - Remote archive: `/root/autodl-tmp/.autodl/mirage_csr/outputs/experiment_archives/20260521_internvl25_26b_external_suite.tar.gz`
  - Local archive: `remote_archives/20260521_internvl25_26b_external_suite.tar.gz`
  - Interpretation: this provides stronger non-Qwen external diagnostic coverage for the paper. Use conservatively: it supports cross-backbone/external prevalence and difficulty, not adapter generalization.


## 2026-05-21 22:45 CST - Bayesian policy fusion v3 (TTA-v2 evidence)

Completed and copied archive: `remote_archives/20260521_scaleqa_bayesian_policy_fusion_v3_tta2.tar.gz`.

Key ScaleQA-520 split-mean results:
- bayes_mean: 0.5133 (std 0.0556)
- selfcheck_mean: 0.5514
- axis_mean: 0.5411
- direct_mean: 0.5341
- pairwise_mean: 0.5314
- top4_context_mean: 0.4798
- oracle_mean: 0.9004

Interpretation: Bayesian fusion does not beat self-check; useful as a negative result showing that naive probabilistic policy fusion still fails despite a large candidate/policy oracle gap.


## 2026-05-21 22:50 CST - Corrected no-leak policy selector v10b

Completed and copied archive: `remote_archives/20260521_scaleqa_policy_selector_v10b_tta2_noleak.tar.gz`.

Key ScaleQA-520 split-mean results:
- selector_mean: 0.5667 (std 0.0561)
- selfcheck_mean: 0.5514
- axis_mean: 0.5411
- dev_selected_fixed_mean: 0.5274
- best_fixed_split_mean: 0.5714
- policy_oracle_mean: 0.9004
- switch_rate_mean: 0.4891

Interpretation: first no-leak deployable selector that clears self-check on mean, though with high split variance and still far below oracle. This is the strongest candidate for a main-method result if described conservatively as a lightweight regime-aware policy selector.

## 2026-05-21 22:55 CST - Launched statistical policy fusion v1

Started remote job `logs/run_20260521_statistical_policy_fusion_v1.log` with archive target `outputs/experiment_archives/20260521_scaleqa_statistical_policy_fusion_v1.tar.gz`.

Experiment design: no-leak train/dev/test split over ScaleQA-520 IDs; compares hierarchical Beta-Binomial posterior means, Wilson lower-bound conservative scoring, and answer-level weighted voting over the same policy pool used by v10b. Goal is to test whether a more explicit statistical policy selector can improve over v10b (`0.5667`) and self-check (`0.5514`).


## 2026-05-21 23:31 CST - GPU regime verifier v3

Completed and copied archive: `remote_archives/20260521_scaleqa520_regime_verifier_v3.tar.gz`.

Overall ScaleQA-520 image-evaluable results:
- regime_unit_first: 0.4778
- regime_default_safe: 0.4672
- regime_eliminate: 0.4397
- regime_v3_candidate_oracle: 0.8901

By direction, the verifier is highly asymmetric: it is strong on `prediction_too_small` (0.7802-0.8190) but very weak on `prediction_too_large` (0.0747-0.1867). This does not improve the main selector directly, but it is useful evidence that deployable verifiers inherit a direction-specific bias while the candidate pool contains the answer.

## 2026-05-21 23:47 CST - Overnight experiment suite launched

Launched remote GPU/CPU overnight suite.

Active GPU pipeline: `logs/run_20260521_overnight_logprob_v11.log`, PID file `logs/run_20260521_overnight_logprob_v11.pid`.
Stages:
1. Qwen answer-tuned log-prob letter verifier over ScaleQA-520.
2. Qwen large-positive adapter log-prob letter verifier.
3. Qwen base log-prob letter verifier.
4. No-leak policy selector v11 including log-prob outputs plus v3 verifier outputs.
Archive target: `outputs/experiment_archives/20260521_overnight_logprob_v11_selector.tar.gz`.

Active CPU pipeline remains: `logs/run_20260521_statistical_policy_fusion_v1.log`, archive target `outputs/experiment_archives/20260521_scaleqa_statistical_policy_fusion_v1.tar.gz`.

Rationale: log-prob verifier tests whether candidate selection improves when using next-token letter probabilities rather than generated letters; v11 tests whether these candidates improve over v10b (`0.5667`) and self-check (`0.5514`).

## 2026-05-22 05:25 CST - Rationale v13 repaired; base candidate-likelihood v15 launched

Completed and copied archive: `remote_archives/20260522_gpu_rationale_v4_selector_v13.tar.gz`.

Key ScaleQA-520 split-mean selector results:
- selector_mean: 0.5667 (std 0.0561)
- selfcheck_mean: 0.5514
- axis_mean: 0.5411
- policy_oracle_mean: 0.9004
- switch_rate_mean: 0.4891

Rationale/verifier prompts remain weak, but the candidate oracle stays high (~0.890), reinforcing the paper's selection-bottleneck story rather than a verifier-method win.

Current running jobs:
- CPU statistical policy fusion v1 is still running; latest seed-level outputs are mixed and not clearly better than self-check.
- v14 candidate-likelihood selector is in CPU selector stage after finishing answer/large1 likelihood runs.
- Launched v15 base candidate-likelihood follow-up with `--adapter none` after fixing an initial mistaken default-adapter launch. GPU is active on `logs/run_20260522_candidate_likelihood_v5_base_v15.log`.

## 2026-05-22 05:46 CST - Statistical fusion and v14 completed

Completed and copied archives:
- `remote_archives/20260521_scaleqa_statistical_policy_fusion_v1.tar.gz`
- `remote_archives/20260522_after_v13_candidate_likelihood_v14.tar.gz`

Statistical policy fusion v1:
- stat_mean: 0.5309 (std 0.0470)
- selfcheck_mean: 0.5514
- axis_mean: 0.5411
- direct_mean: 0.5341
- pairwise_mean: 0.5314
- decimal_guard_mean: 0.5168
- oracle_mean: 0.9004
- Interpretation: negative result; explicit statistical/Beta-style fusion did not beat self-check.

Candidate-likelihood v14 selector:
- selector_mean: 0.5667 (std 0.0561)
- selfcheck_mean: 0.5514
- axis_mean: 0.5411
- policy_oracle_mean: 0.9004
- switch_rate_mean: 0.4891
- Interpretation: adding answer/large candidate-likelihood evidence does not improve beyond v10b/v13, but preserves the best no-leak selector result.

Current running job:
- v15 base candidate-likelihood selector: base likelihood eval completed (`base:candlik_direct=0.4905`, `scale_guard=0.4271`, `unit_first=0.4292`, candidate oracle `0.8901`); CPU selector is running.

Additional GPU follow-up launched:
- v16 top-8 candidate-likelihood selector, remote log `logs/run_20260522_candidate_likelihood_top8_v16.log`, archive target `outputs/experiment_archives/20260522_candidate_likelihood_top8_v16.tar.gz`.
- Purpose: test whether a larger candidate pool raises the deployable selector beyond v10b/v13/v14 (`0.5667`) or mainly raises only the oracle ceiling.

## 2026-05-22 06:16 CST - v15 base candidate-likelihood selector completed

Completed and copied archive: `remote_archives/20260522_candidate_likelihood_v5_base_v15.tar.gz`.

Key ScaleQA-520 split-mean selector results:
- selector_mean: 0.5667 (std 0.0561)
- selfcheck_mean: 0.5514
- axis_mean: 0.5411
- policy_oracle_mean: 0.9004
- switch_rate_mean: 0.4891

Interpretation: adding base candidate-likelihood evidence again preserves the best no-leak selector result but does not improve beyond v10b/v13/v14.

Current running job:
- v16 top-8 candidate-likelihood is still using the GPU. Latest progress: answer_top8 300/473, with deployable likelihood prompts still weak (`direct` around 0.42, `scale_guard` around 0.32, `unit_first` around 0.34 so far).

## 2026-05-22 08:02 CST - v16 selector stage; v17 pairwise top-8 launched

v16 top-8 candidate-likelihood finished all three GPU likelihood passes and entered CPU selector stage. Top-8 candidate oracle rose to `0.9175`, but practical likelihood prompts remain weak:
- answer_top8 / large1_top8 / base_top8 all show `candlik_direct=0.4503`, `scale_guard=0.3214`, `unit_first=0.3404`.

To keep the GPU busy and test whether a generative comparator benefits from the larger candidate pool, launched:
- v17 pairwise top-8 verifier + selector
- Remote log: `logs/run_20260522_pairwise_top8_v17.log`
- Archive target: `outputs/experiment_archives/20260522_pairwise_top8_v17.tar.gz`

## 2026-05-22 08:17 CST - v17 entered selector; v18 pairwise multi-adapter launched

v17 pairwise top-8 answer-adapter verifier completed its GPU stage and entered CPU selector:
- pairwise_default_vs_alt: `0.5433`

This is below self-check (`0.5514`) and the best no-leak selector (`0.5667`), but is still useful as another selection-bottleneck baseline.

To keep GPU busy, launched v18:
- Pairwise top-8 with large-positive adapter and base model, followed by selector over all top-8 likelihood/pairwise evidence.
- Remote log: `logs/run_20260522_pairwise_multi_v18.log`
- Archive target: `outputs/experiment_archives/20260522_pairwise_multi_v18.tar.gz`

## 2026-05-22 08:48 CST - v16/v17 completed; v19 candidate-letter top-8 launched

Completed and copied archives:
- `remote_archives/20260522_candidate_likelihood_top8_v16.tar.gz`
- `remote_archives/20260522_pairwise_top8_v17.tar.gz`

v16 and v17 both preserve the best no-leak selector result but do not improve it:
- selector_mean: 0.5667 (std 0.0561)
- selfcheck_mean: 0.5514
- policy_oracle_mean: 0.9004

v18 is in CPU selector stage after pairwise multi-adapter GPU passes. To keep GPU active, launched v19 candidate-letter top-8 verifier:
- Remote log: `logs/run_20260522_candidate_letter_top8_v19.log`
- Archive target: `outputs/experiment_archives/20260522_candidate_letter_top8_v19.tar.gz`
- Purpose: test whether asking the model to choose a letter among candidates, rather than emitting/scoring a numeric string, improves deployable selection.
