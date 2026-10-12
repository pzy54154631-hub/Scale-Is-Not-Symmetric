# Research Plan, October 2026

Status of this document: working plan, not results. Every number below that is
not already in `repro/paper_claims.csv` is a target or a placeholder, never a
claim. Nothing goes into a paper until it has a result file and a row in
`paper_claims.csv`.

Hardware assumption: one RTX 5090 (32 GB). Everything here is sized for that.
Needs CUDA 12.8+ and a Blackwell-compatible PyTorch/vLLM build; run the smoke
test in `make smoke` before any long job.

---

## Part A. Scale paper (this repository) — ARR Oct 2026 → NAACL/COLING 2027

### A0. Blocking items before submission (tonight)

See `docs/ARR_OCT2026_REVISION_NOTES.md`, last section. In priority order:

1. Reconcile 18 blind-verified candidates vs. 21 ratio-rule candidates on test.
2. Attach result files for every `\verify{}` number; remove red.
3. Fill or delete: leave-source-out row, overlap count, per-item table,
   pair-construction appendix.
4. OpenReview: mark as resubmission, link Submission 3292, paste revision notes.
5. Reviewer registration for all authors.

### A1. Experiments the reviewers asked for that are cheap (run during review, for the camera-ready or next cycle)

| # | Experiment | Why | Cost on 5090 | Script to write |
|---|---|---|---|---|
| A1.1 | Narrower ratio bands: [90,111], exact powers of ten, sign-preserving, percent-aware parse | U7dc "strict is not strict" | CPU, minutes | `scripts/analyze_strict_sensitivity.py` |
| A1.2 | Per-seed recovery counts on verified-17; items recovered by k/5 seeds | U7dc, 9k9v | CPU, minutes (predictions exist) | extend `analyze_seed_consistency_full.py` |
| A1.3 | Fixed dev/test split + nested CV for selector; paired test vs self-check | U7dc selector overfitting | CPU, <1 h | `scripts/scaleqa_selector_nested_cv.py` |
| A1.4 | Row-disjoint recompute if overlap > 0 | U7dc | CPU | filter + rerun eval tables |
| A1.5 | Independent from-scratch annotation of ~200 ScaleQA items by a second person; human–human κ | U7dc, 9k9v | human time, no GPU | annotation sheet export |
| A1.6 | Table-3 per-seed baseline std; Table-12 vs Table-4 label fix | U7dc typos | none | tex |

### A2. Experiments that change the paper's weight (next cycle if NAACL rejects)

| # | Experiment | Target claim | Cost on 5090 |
|---|---|---|---|
| A2.1 | Same protocol on 5–6 open models ≤8B (Qwen2.5-VL-7B, Qwen3-VL small, InternVL2.5/3-8B, LLaVA-OneVision-7B, Idefics3-8B, MiniCPM-V) + 1–2 API models: strict-candidate prevalence, blind-verified rate, subtype asymmetry | "scale-regime errors are a cross-model phenomenon" | inference only, ~2–4 h/model with vLLM |
| A2.2 | Broader scale types: mine thousands/millions/billions, currency, log-axis cases from PlotQA, ChartQAPro, CharXiv; verify blind | removes "only percent" limitation | mining CPU; verification human; inference ~2 h |
| A2.3 | **Unit-first decoding**: model emits (value, unit-as-read) and expected answer convention; deterministic conversion | the "impactful solution" the AC asked for | inference only, ~1 h per model per set |
| A2.4 | Fix–break for A2.3 on full ChartQA test, ScaleQA-473, PlotQA numeric | safety | inference |

Go/no-go for A2.3 as a headline: it must beat self-check (54.55 on ScaleQA-473)
by ≥3 points AND not reduce full ChartQA test accuracy AND hold on ≥2 backbones.
Otherwise it is one more probe in the ladder.

---

## Part B. New paper: test-time selection for chart numeric QA (ICML 2027 Jan 29 / ICCV 2027 Mar)

Motivation already measured: single sample 52.4 vs blind pass@20 79.3 on
ScaleQA-473. The model generates the right number; it does not pick it.

### B1. One-week kill test (run first; abandon the direction if it fails)

- Data: ~500 numeric questions each from ChartQAPro and CharXiv (reasoning,
  numeric answers only). Hold out; do not tune on them.
- Model: Qwen2.5-VL-7B-Instruct, vLLM, N=20 samples, T≈0.7.
- Compare: (1) greedy; (2) majority vote on normalized number; (3) unit-aware
  vote: parse each sample into (value, unit, scale), bucket by unit, vote within
  bucket, choose bucket by question/axis evidence; (4) oracle pass@20.
- Kill rule: if (3) − (2) < 2 points on both sets, stop. If ≥3–5 points, continue.

### B2. If it survives

- Method: structured candidate parsing + chart-consistency checks (axis range,
  tick spacing, unit tokens) + optional counterfactual re-render check for
  charts with source tables (PlotQA, ChartQA with tables): perturb one value,
  re-render, re-ask; a candidate that tracks the perturbation is grounded.
- Baselines: self-consistency, program-of-thought / code execution, chart-to-
  table-then-compute (DePlot-style), self-check prompt, our prefer-large adapter.
- Backbones: 3–4 open models ≤8B; report "best among ≤8B open models" only if
  true on the public leaderboards checked at submission time.
- Benchmarks: ChartQA, ChartQAPro, CharXiv numeric, PlotQA numeric.
- Report fix–break on every set.
- Cost: inference only; ~1 GPU-day per model across sets.

---

## Part C. CV interpretability paper (ICML Jan / ICCV Mar; needs B1-style kill test too)

Question: where inside an MLLM does the scale information get lost?

1. Controlled synthetic charts (matplotlib): same data, vary only unit label
   ("%" vs "millions") or only magnitude. Borrowed design principle from
   MechFlow (`../MECHFLOW_AAAI27`): counterfactual source replacement, held-out
   combinations, causal interventions rather than attention maps.
2. Linear probes per layer for (a) magnitude, (b) unit/scale token: is the unit
   decodable from vision tokens? from the answer position?
3. Activation patching: patch "correct-unit" run into "wrong-unit" run; locate
   layers/heads carrying unit information to the answer.
4. Steering at the located site; report fix–break.

Kill rule: if unit is not linearly decodable at any layer, or patching finds no
localized effect, stop.

Cost: 7B in bf16 ≈ 16 GB; patching at batch 1–4 fits in 32 GB.

---

## Part D. Fast, low-risk outputs for the transfer timeline

AAAI-27 workshops: papers due Nov 20, 2026; notification Dec 2, 2026. Check
each workshop's archival policy (non-archival ones can coexist with a main-
conference submission).

| Material | Candidate workshop topic | Status of evidence |
|---|---|---|
| PW-OFPC (`pw-ofpc` repo) | reasoning / neuro-symbolic | RAVEN 0.537→0.857 with intervention ladder; needs I-RAVEN main table and a specialized-solver baseline |
| Neural Dynamics (`../NEURAL_DYNAMICS_AAAI27`) | NeuroAI / AI4Science | synthetic edge AUC 0.862 vs 0.728 best baseline, 3 seeds consistent; synthetic only |
| MechFlow (`../MECHFLOW_AAAI27`) | interpretability | 2 seeds; warmup effect inside seed noise; needs the planned 10-seed sweep first |

Other venues on the calendar: CVPR 2027 (Nov 16; ambiguity-routing paper is the
candidate, needs the n=22 and "ties scalar MLP" weaknesses addressed), ICME 2027
(Dec, IEEE, CCF-B; confirm date on the official site), IJCAI (Jan 20), ICML
(Jan 29), ICCV (Mar).

---

## Part E. Working rules (so the writing stays honest)

- A number enters a paper only with a result file and a `paper_claims.csv` row.
- Every citation is checked against arXiv/ACL Anthology before it enters
  `references.bib`.
- Counts with Wilson CIs for anything with n < 100.
- Report fix and break for every intervention.
- Do not call a method "deployable" without latency/cost numbers.
- Kill tests come before full experiments; abandon directions that fail them.
