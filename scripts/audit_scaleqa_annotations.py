#!/usr/bin/env python3
"""Second-pass audit for ScaleQA-Natural annotations.

This is not a replacement for human IAA. It creates an independent, deterministic
second-pass label from candidate metadata so the paper can report a transparent
two-pass audit and expose conflicts for final human review.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUND2 = Path('/Users/pengziyu/writing/scale_annotation_round2_full_codex_annotated/scaleqa_natural_candidates_round2_after_internvl_train_CODEX_ANNOTATED.csv')
FINAL520 = ROOT / 'results/scaleqa_natural/image_assisted_v3/scaleqa_natural_expanded_520_codex_image_assisted.csv'
OUTDIR = ROOT / 'results/scaleqa_natural/annotation_audit'
OUTDIR.mkdir(parents=True, exist_ok=True)

ACCEPT_SUBTYPES = {
    'decimal_to_percent',
    'percent_to_decimal',
    'value_scaled_up',
    'value_scaled_down',
    'untyped_100x',
}
REVIEW_SUBTYPES = {
    'minor_factor_or_unit_candidate',
    'minor_factor_percentage_candidate',
    'mixed_factor_candidate',
    'count_or_unit_ambiguous_100x',
}
REJECT_SUBTYPES = {
    'multi_field_or_explanation',
    'nonnumeric_or_multi_gold',
}
VALID_FACTORS = {'100x', '1/100x', '10x', '1/10x', '1000x', '1/1000x'}
CORE_FACTORS = {'100x', '1/100x'}

NUM_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?%?')

def read_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def norm(s) -> str:
    return str(s or '').strip()

def nums(s: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(str(s or '')):
        is_pct = m.endswith('%')
        t = m.rstrip('%').replace(',', '')
        try:
            v = float(t)
        except Exception:
            continue
        # Keep the surface number. Whether percent means /100 is exactly the
        # ambiguity under audit, so do not normalize the percent sign away.
        out.append(v)
    return out

def ratio_factor(gold: str, pred: str) -> str:
    gn = nums(gold)
    pn = nums(pred)
    if len(gn) != 1 or len(pn) != 1 or abs(gn[0]) < 1e-12:
        return 'none'
    r = abs(pn[0] / gn[0])
    checks = [
        ('100x', 100.0), ('1/100x', 0.01),
        ('10x', 10.0), ('1/10x', 0.1),
        ('1000x', 1000.0), ('1/1000x', 0.001),
    ]
    for name, target in checks:
        lo, hi = target / 1.25, target * 1.25
        if lo <= r <= hi:
            return name
    return 'other'

def first_pass_label(row: dict) -> str:
    lab = norm(row.get('review_label')).lower()
    action = norm(row.get('benchmark_action')).lower()
    if lab == 'yes' or action == 'include':
        return 'include'
    if lab == 'not_scale' or action == 'not_benchmark':
        return 'reject'
    return 'review'

def second_pass_label(row: dict) -> tuple[str, str]:
    subtype = norm(row.get('scale_error_subtype')).lower()
    factors = {x.strip() for x in norm(row.get('scale_factors')).split('|') if x.strip()}
    directions = {x.strip() for x in norm(row.get('scale_directions')).split('|') if x.strip()}
    conf = 0.0
    try:
        conf = float(row.get('codex_confidence') or 0)
    except Exception:
        pass
    g = norm(row.get('gold_answer'))
    p = norm(row.get('representative_predictions'))
    inferred = ratio_factor(g, p)
    gnums, pnums = nums(g), nums(p)

    reasons = []
    if subtype in REJECT_SUBTYPES:
        return 'reject', f'reject subtype={subtype}'
    if len(gnums) != 1:
        return 'reject', f'gold numeric count={len(gnums)}'
    if len(pnums) != 1:
        if subtype in ACCEPT_SUBTYPES and factors & CORE_FACTORS and conf >= 0.90:
            return 'review', f'prediction numeric count={len(pnums)} but high-confidence core subtype'
        return 'reject', f'prediction numeric count={len(pnums)}'

    if inferred in VALID_FACTORS:
        reasons.append(f'inferred_factor={inferred}')
    else:
        reasons.append(f'inferred_factor={inferred}')

    if subtype in ACCEPT_SUBTYPES and (factors & CORE_FACTORS or inferred in CORE_FACTORS):
        if conf >= 0.85:
            return 'include', '; '.join(reasons + [f'core subtype={subtype}', f'conf={conf:.2f}'])
        return 'review', '; '.join(reasons + [f'core subtype low conf={conf:.2f}'])

    if subtype in ACCEPT_SUBTYPES and (factors & VALID_FACTORS or inferred in VALID_FACTORS):
        if conf >= 0.90 and len(directions) == 1:
            return 'include', '; '.join(reasons + [f'valid non-core factor subtype={subtype}', f'conf={conf:.2f}'])
        return 'review', '; '.join(reasons + [f'valid non-core factor needs review conf={conf:.2f}'])

    if subtype in REVIEW_SUBTYPES:
        if conf >= 0.64 and (factors & VALID_FACTORS or inferred in VALID_FACTORS) and len(directions) == 1:
            return 'review', '; '.join(reasons + [f'review subtype={subtype}', f'conf={conf:.2f}'])
        return 'reject', '; '.join(reasons + [f'weak review subtype={subtype}', f'conf={conf:.2f}'])

    if inferred in CORE_FACTORS and conf >= 0.90:
        return 'review', '; '.join(reasons + [f'unknown subtype={subtype}', f'conf={conf:.2f}'])
    return 'reject', '; '.join(reasons + [f'fallback subtype={subtype}', f'conf={conf:.2f}'])

def cohen_kappa(labels1, labels2, order=('include','review','reject')):
    n = len(labels1)
    if n == 0:
        return float('nan')
    po = sum(a == b for a, b in zip(labels1, labels2)) / n
    c1, c2 = Counter(labels1), Counter(labels2)
    pe = sum((c1[o]/n) * (c2[o]/n) for o in order)
    k = (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else float('nan')
    return po, pe, k

def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

round2 = read_csv(ROUND2)
final = read_csv(FINAL520)
final_ids = {r['candidate_id'] for r in final}

# Candidate-pool audit over the original 739 labeled candidates.
audited = []
for r in round2:
    a1 = first_pass_label(r)
    a2, reason = second_pass_label(r)
    rr = dict(r)
    rr['first_pass_label_3way'] = a1
    rr['second_pass_label_3way'] = a2
    rr['second_pass_reason'] = reason
    rr['in_final_520'] = str(r.get('candidate_id') in final_ids)
    rr['agreement'] = str(a1 == a2)
    audited.append(rr)

labels1 = [r['first_pass_label_3way'] for r in audited]
labels2 = [r['second_pass_label_3way'] for r in audited]
po, pe, kappa = cohen_kappa(labels1, labels2)

conflicts = [r for r in audited if r['first_pass_label_3way'] != r['second_pass_label_3way']]
# Prioritize conflicts that affect included/final candidates.
def conflict_rank(r):
    final_bad = 0 if r['in_final_520'] == 'True' else 1
    include_disagree = 0 if 'include' in (r['first_pass_label_3way'], r['second_pass_label_3way']) else 1
    conf = -float(r.get('codex_confidence') or 0)
    return (final_bad, include_disagree, conf, r.get('candidate_id',''))
conflicts = sorted(conflicts, key=conflict_rank)

# Final-positive audit: among included final positives, how many second pass accepts/reviews/rejects.
final_lookup = {r['candidate_id']: r for r in final}
final_audit = []
for fid, fr in final_lookup.items():
    # Use final row because image-assisted additions have updated fields.
    a2, reason = second_pass_label(fr)
    rr = dict(fr)
    rr['final_label'] = 'include'
    rr['second_pass_label_3way'] = a2
    rr['second_pass_reason'] = reason
    rr['agreement_with_final_include'] = str(a2 == 'include')
    final_audit.append(rr)

summary = {
    'candidate_pool_n': len(audited),
    'candidate_pool_first_pass_counts': dict(Counter(labels1)),
    'candidate_pool_second_pass_counts': dict(Counter(labels2)),
    'candidate_pool_agreement': po,
    'candidate_pool_expected_agreement': pe,
    'candidate_pool_cohen_kappa': kappa,
    'candidate_pool_conflicts': len(conflicts),
    'final_520_n': len(final_audit),
    'final_520_second_pass_counts': dict(Counter(r['second_pass_label_3way'] for r in final_audit)),
    'final_520_second_pass_include_rate': sum(r['second_pass_label_3way'] == 'include' for r in final_audit) / len(final_audit),
    'final_520_review_or_reject_rate': sum(r['second_pass_label_3way'] != 'include' for r in final_audit) / len(final_audit),
    'final_520_annotation_sources': dict(Counter(r.get('annotation_source','') for r in final_audit)),
    'final_520_subtype_counts': dict(Counter(r.get('scale_error_subtype','') for r in final_audit)),
    'notes': [
        'Second-pass labels are deterministic metadata audit labels, not a second human annotator.',
        'Disagreement rows are intended for author review before claiming human IAA.',
    ],
}

fields = list(audited[0].keys())
write_csv(OUTDIR / 'scaleqa_annotation_second_pass_audit_739.csv', audited, fields)
write_csv(OUTDIR / 'scaleqa_annotation_second_pass_conflicts.csv', conflicts, fields)
fields_final = list(final_audit[0].keys())
write_csv(OUTDIR / 'scaleqa_final520_second_pass_audit.csv', final_audit, fields_final)
(OUTDIR / 'scaleqa_annotation_second_pass_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

# LaTeX table for the paper.
table = ROOT / 'paper/tables/scaleqa_annotation_audit.tex'
with table.open('w', encoding='utf-8') as f:
    f.write('\\begin{tabular}{lrr}\n')
    f.write('\\toprule\n')
    f.write('Audit item & Count/Score & Notes \\\\\n')
    f.write('\\midrule\n')
    f.write(f'Candidate pool & {len(audited)} & First-pass reviewed candidates \\\\\n')
    f.write(f'Observed agreement & {po:.3f} & Three-way labels \\\\\n')
    f.write(f'Cohen\\\'s $\\kappa$ & {kappa:.3f} & Include/review/reject \\\\\n')
    f.write(f'Conflicts for review & {len(conflicts)} & Exported separately \\\\\n')
    f.write(f'Final positives & {len(final_audit)} & ScaleQA-Natural-520 \\\\\n')
    f.write(f'Second-pass include & {sum(r["second_pass_label_3way"] == "include" for r in final_audit)} & Deterministic audit \\\\\n')
    f.write(f'Second-pass review/reject & {sum(r["second_pass_label_3way"] != "include" for r in final_audit)} & Requires author audit \\\\\n')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}\n')

print(json.dumps(summary, indent=2))
print('wrote', OUTDIR)
print('table', table)
