# Project context for Claude Code

Paper: "Scale Is Not Symmetric" (ChartQA scale-error diagnostics). Rejected at ARR/EMNLP 2026
(all reviewers 2.5, AC: "resubmit next cycle"). Resubmitting to ARR October 2026 cycle
(deadline Oct 12 2026 AoE) → NAACL 2027 / COLING 2027 commitment. Working branch:
`arr-oct2026-resubmission`.

## Read these first
- `docs/ARR_OCT2026_REVISION_NOTES.md` — point-by-point response to the reviews; what changed; what still needs verification.
- `docs/RESEARCH_PLAN_OCT2026.md` — experiments planned, kill tests, venues, working rules.
- `docs/UNIT_FIRST_RUNBOOK.md` — the one GPU experiment that could change the paper's weight.
- `repro/paper_claims.csv` — every number in the paper maps to a result file (C1–C16).

## State of the manuscript (paper/main.tex, compile with `latexmk -pdf main.tex`)
- Main text fits 8 pages (Limitations/refs start on p.9).
- `\verify{}` macro prints numbers in RED. These come from the July 2026 author response and
  currently have NO result files in the repo (server data was lost): blinded verification 17/18,
  compute-matched controls 0/1/12/14 of 17, ChartQA val 86.40→86.95, blind pass@20 79.3 vs 52.4.
  Decision pending: remove them from the main text, or keep with the user's explicit OK.
- Numbers added on Oct 12 2026 from files already in the repo (verified, CPU-only):
  row-overlap analysis (363/473 items from ChartQA training rows; adapter +7.3 on the 110
  row-disjoint items, −4.1 on training-row items), leave-source-out selector (below self-check on
  all 4 sources), paired selector statistics. Scripts: `scripts/experiment_policy_selector_v10b_leave_source_out.py`,
  `results/scaleqa_natural/row_overlap_split_analysis.csv`.
- Known inconsistency to resolve: 21 ratio-rule strict test candidates (paper tables) vs 18 in
  the July blinded verification.

## Data situation
Server data is gone. Reconstructible: ChartQA/PlotQA (public); ScaleQA-Natural-473 (join the
candidate pool CSV in `artifacts/evidence/expanded/20260521_scaleqa_routing_reranker_taxonomy_plus_fast/.../candidate_pool.csv`
with ChartQA rows by `source_split`/`source_row_idx`); adapters (retrain: answer-only 3000 steps
r16, scale adapters 300 steps, configs in appendix/`configs/`). Not reconstructible without
re-running: the July rebuttal results; the blinded verification also needs a human annotator.

## Rules the user set
- No number enters the paper without a result file and a `paper_claims.csv` row.
- Every citation checked against arXiv/ACL Anthology before entering `references.bib`.
- Counts + Wilson CIs for n < 100; fix AND break for every intervention.
- Never call the selector "deployable". Scope claims to percentage–decimal, factor-of-100 errors.
- Commit messages: plain; user pushes or has the Claude GitHub App installed.

## Other repos of the user (separate, do not merge into this paper)
pw-ofpc (Raven), ambiguity-is-not-one-scalar (VQA routing, CVPR 2027 candidate),
MECHFLOW_AAAI27 and NEURAL_DYNAMICS_AAAI27 (local folders; workshop candidates).
