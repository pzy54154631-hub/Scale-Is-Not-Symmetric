# Revision Notes for ARR October 2026 Resubmission

Previous submission: ARR (EMNLP 2026 cycle), Submission 3292. Meta-review: "Resubmit next cycle" (2). Reviews: U7dc 2.5, y48B 2.5, 9k9v 2.5.

This document is the point-by-point response to paste into the ARR resubmission form. It lists every change made to the manuscript. Items marked **[VERIFY]** correspond to numbers typeset in red in the PDF (`\verify{}` macro); each must be checked against a result file in `artifacts/evidence/` before submission, then the macro is set to identity.

## Summary of changes

1. **Reframed the contribution** as a diagnostic study with one verified targeted effect (abstract, §1, §7). Removed "the obvious fix is tempting and wrong"; claims are now stated as "the unconditional interventions we test do not solve this."
2. **Narrowed scope** explicitly to percentage–decimal, factor-of-100 errors on ChartQA-like data (abstract, §1, §7, Limitations).
3. **Added blinded, construction-independent verification** of strict ChartQA test errors (§3, Appendix "Verified Strict Test Errors"). The ratio rule is now described as a *candidate filter*, not a definition of a scale error.
4. **Added compute-matched controls** (new Table: `compute_matched_controls`): ordinary answer replay, random numeric negatives, non-directional balanced pairs, all continuing the same answer-tuned adapter for 300 steps.
5. **Added blind-generation oracle** (new Table: `blind_oracle`) decomposing candidate coverage by source: model-only samples vs. prompt/adapter candidates vs. deterministic transforms.
6. **Simplified selector presentation** (new Table: `selector_main_comparison`); v8–v17 ladder moved to appendix with an explicit statement that it was developed by repeated held-out-fold evaluation, not nested CV. "No-leak" renamed "label-free" throughout, with its weaker meaning stated. Added the "best fixed policy per split = 57.14 > selector 56.67" caveat.
7. **Annotation pipeline** now reported in three separate stages; κ=0.443 is explicitly automatic-vs-first-pass, and the 100% second-review agreement is explicitly *not* independent initial reliability (§3).
8. **Counts + Wilson 95% CIs** replace over-precise percentages for all small-slice results. All deltas are in percentage points.
9. **Related work** adds ChartQAPro, ChartInsights, ChartMind with a statement of what ScaleQA-Natural adds (directional, fix–break slice).
10. **Naming**: "prefer-large / prefer-small" introduced in prose; tables keep "large-positive / small-positive" with a mapping sentence. "ScaleQA-Natural-473" names the image-evaluable subset. Percent parsing ("0.61%" → 0.61) stated.
11. **Cross-backbone claims** narrowed: they establish that ratio-related errors occur under other backbones and that the adapter does not transfer; they do *not* establish common prevalence or taxonomy. TextVQA-500 described as a small adjacent-task check only.
12. **Limitations and deployment risk** rewritten; explicit recommendation against deploying automatic scale correction.
13. **Oracle definitions** collected in one appendix list.
14. Five tables moved to the appendix to meet the 8-page limit.

## Point-by-point

### Meta-review (AC xrSU)

| Concern | Response |
|---|---|
| Very small number of ChartQA test errors | Verified slice now reported as counts with Wilson CIs; described as diagnostic-slice evidence. Blinded verification added so the slice is independent of ScaleQA-Natural construction. |
| Selector complicated, modest gains | Selector section rewritten around one main comparison; ladder moved to appendix; selector explicitly *not* presented as a deployable method; fragility under leave-source-out stated. |
| Generalization limited, mostly percent evidence | Scope narrowed in abstract, intro, conclusion, limitations. |
| Cross-backbone/external results unconvincing | Claims reduced to two narrow facts; no cross-model prevalence claim. |
| Stronger case for importance + impactful solution | Importance: §1 now motivates with the "plausible-looking, factor-100 wrong" risk; Limitations add deployment-risk paragraph. Solution: we do not claim one. The impactful finding is the controlled attribution (directional supervision 12–14/17 vs. ≤1/17 for compute-matched controls) plus the blind pass@20 result showing the bottleneck is selection. |

### Reviewer U7dc

| Concern | Change |
|---|---|
| Oracle partly implied by construction | Blind-generation oracle added (Table `blind_oracle`); coverage decomposed by candidate source; claim softened to "generation and selection both matter; selection is the larger gap." |
| "No-leak" ≠ untouched test set; v8–v17 researcher overfitting; no paired test | Renamed label-free; caveat stated in §6 and Limitations; paired bootstrap CI reported in main table; leave-source-out row added **[VERIFY]**; appendix states ladder was not nested CV. |
| Annotation independence; κ meaning; second reviewer scope | §3 "Annotation pipeline" paragraph separates the three stages and states exactly what each does and does not establish. **[TODO if time]**: instructions text for both reviewers in appendix. |
| Row-level overlap with training rows | Placeholder in §3 and Appendix `app:overlap` **[VERIFY — must be filled with the actual count]**. |
| "Strict" definition not strict | Renamed "strict scale-error candidate"; ratio rule is a filter; blinded verification (17/18) is the evidence; percent parsing stated. **[TODO if time]**: sensitivity analysis with narrower bands. |
| Test gain not attributable to scale objective | Compute-matched controls table added (Table `controls`) **[VERIFY all cells]**. |
| 49.52% over 21 examples too precise | Replaced with counts + Wilson CI; per-seed counts promised in Appendix `app:strict_verified` **[VERIFY — table must be filled]**. |
| Scope broader than evidence | Narrowed. |
| Missing implementation details | Appendix `app:pairs` placeholder lists every item requested **[VERIFY — must be filled]**. |
| Novelty positioning | Related work updated with the three suggested benchmarks. |
| Wording items | "tempting and wrong" removed; ScaleQA-Natural-473 named; prefer-large/small; percent parsing; percentage points; 0.0593 example simplified (6.0333 → "should not be expanded"); "transfers more robustly" phrasing removed and replaced by the two measured properties; "deployable" no longer used for the selector. **[TODO]**: Table 3 per-seed baseline std; Table 12 vs Table 4 pipeline labels; oracle definition table (done in appendix). |

### Reviewer y48B

| Concern | Change |
|---|---|
| Marginal improvement, single model, single dataset | Contribution reframed: small aggregate gain is expected because the failure is rare; the targeted effect with controls is the result. Scope stated. Multi-backbone rows retained as diagnostics with narrowed claims. |
| No practical solution | We state plainly that we do not propose one, and recommend against deploying global correction. |
| Visualizations | **[TODO if time]**: pipeline figure and fix–break figure. Current Figure 1 box retained. |

### Reviewer 9k9v

| Concern | Change |
|---|---|
| Benchmark-like language | Removed; "mined diagnostic set" throughout. |
| Absolute counts next to percentages | Done. |
| Annotation pipeline clarity | Done (§3). |
| Simplify selector experiments | Done (Table `selector_main_comparison`). |
| Broader scale types or limit claims | Claims limited; broader types listed as future work in Limitations. |

## Numbers that must be verified before submission

Source: author response of 8 July 2026. None of these are in the repository's `artifacts/evidence/`. Locate the result files, add them to `repro/paper_claims.csv`, then remove the red colouring.

| Number | Where used | Status |
|---|---|---|
| 18 strict candidates on test; 17/18 true; 16/18 clean; 1/30 missed | §3, Appendix | **Reconcile with the 21 ratio-rule candidates in `chartqa_test_subset_results`.** State why 18 ≠ 21. |
| 0/17, 1/17, 12/17, 14/17 | abstract, §1, Table controls | Fill random-negative and non-directional rows. |
| ChartQA val 86.40 → 86.95 (3 seeds) | Table controls | Confirm this is validation (N=1920), consistent with `bootstrap_chartqa_full`. |
| Blind pass@20 79.3 vs single 52.4 on 473 | abstract, §1, §5, §6 | Confirm sampling temperature and that 52.4 is single stochastic sample (not greedy 50.95). |
| Leave-source-out selector result | Table selector_main | Fill with number. |
| Row-level overlap count | §3, Appendix | Fill. |
| Per-item verified table | Appendix | Fill. |

## Not done (time)

- Narrower ratio-band sensitivity analysis.
- Independent from-scratch annotation of a ~200-item sample with human–human κ.
- Multi-model evaluation under one protocol.
- Figures (pipeline, fix–break).
These remain the strongest remaining weaknesses and are the first things to add if the paper goes to another cycle.
