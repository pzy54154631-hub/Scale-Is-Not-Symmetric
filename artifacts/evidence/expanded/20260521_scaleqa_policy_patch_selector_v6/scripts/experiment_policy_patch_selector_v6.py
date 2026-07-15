#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

OUT = Path("results/scaleqa_natural/policy_patch_selector_v6")
OUT.mkdir(parents=True, exist_ok=True)

POOL = Path("results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_num(value):
    if pd.isna(value):
        return np.nan
    m = NUM_RE.search(str(value).replace(",", "").replace("−", "-"))
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except ValueError:
        return np.nan


def norm_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def safe_ratio(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12 or abs(b) < 1e-12:
        return 0.0
    return math.log10(abs(a / b))


def build_features(pool: pd.DataFrame):
    score = (
        pool.pivot_table(index="candidate_id", columns="adapter", values="score", aggfunc="first")
        .fillna(0)
        .astype(int)
    )
    pred = pool.pivot_table(index="candidate_id", columns="adapter", values="prediction_norm", aggfunc="first")
    meta = pool.drop_duplicates("candidate_id").set_index("candidate_id")

    preferred = [
        "tta:self_check",
        "tta:axis",
        "tta:tta_majority",
        "tta:direct",
        "tta:unit_guard",
        "verifier:candidate_majority",
        "tta:math",
        "main:answer_tuned",
        "main:large_seed4",
        "main:large_seed5",
        "main:large_seed1",
        "main:large_seed2",
        "main:large_seed3",
        "tta:scale",
        "main:base",
        "verifier:letter_axis",
        "verifier:letter_scale_guard",
    ]
    policies = [p for p in preferred if p in score.columns]
    ids = np.array(sorted(score.index))
    id_to_i = {cid: i for i, cid in enumerate(ids)}

    pred_cols = [p for p in policies if p in pred.columns]
    base_texts = []
    base_nums = []
    pred_values = {}
    pred_strings = {}
    for cid in ids:
        q = norm_text(meta.loc[cid].get("question", ""))
        ctx = norm_text(meta.loc[cid].get("chart_context", ""))
        subtype = norm_text(meta.loc[cid].get("scale_error_subtype_manifest", meta.loc[cid].get("scale_error_subtype", "")))
        direction = norm_text(meta.loc[cid].get("scale_directions_manifest", meta.loc[cid].get("scale_directions", "")))
        preds = [norm_text(pred.loc[cid].get(p, "")) for p in pred_cols]
        pred_strings[cid] = {p: norm_text(pred.loc[cid].get(p, "")) for p in pred_cols}
        pred_values[cid] = {p: parse_num(pred.loc[cid].get(p, "")) for p in pred_cols}
        vals = list(pred_values[cid].values())
        s = f"{q} {ctx}"
        flags = [
            "%" in s or "percent" in s or "percentage" in s,
            "million" in s or "billion" in s or "thousand" in s,
            "average" in s or "mean" in s,
            "difference" in s or "greater" in s or "less" in s or "more" in s or "fewer" in s,
            "highest" in s or "largest" in s or "maximum" in s,
            "lowest" in s or "smallest" in s or "minimum" in s,
            "axis" in s or "ticks=" in s,
            "legend=" in s or "units=" in s,
        ]
        ratios = []
        for i, a in enumerate(vals):
            for b in vals[i + 1 :]:
                ratios.append(safe_ratio(a, b))
        base_texts.append(
            f"[Q] {q} [CTX] {ctx} [SUBTYPE_HINT] {subtype} [DIR_HINT] {direction} [PREDS] "
            + " || ".join(preds)
        )
        base_nums.append(
            [float(x) for x in flags]
            + [
                len(q),
                len(ctx),
                len(preds),
                sum(np.isfinite(v) for v in vals),
                float(any(np.isfinite(v) and abs(v) < 1 for v in vals)),
                float(any(np.isfinite(v) and abs(v) > 100 for v in vals)),
            ]
            + ratios[:32]
        )

    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_base_text = vec.fit_transform(base_texts)
    X_base_num = csr_matrix(np.array(base_nums, dtype=float))

    rows = []
    labels = []
    row_meta = []
    policy_to_j = {p: j for j, p in enumerate(policies)}
    for cid in ids:
        strings = pred_strings[cid]
        values = pred_values[cid]
        counts = pd.Series(list(strings.values())).value_counts().to_dict()
        self_val = values.get("tta:self_check", np.nan)
        base_val = values.get("main:answer_tuned", values.get("main:base", np.nan))
        axis_val = values.get("tta:axis", np.nan)
        for p in policies:
            v = values.get(p, np.nan)
            st = strings.get(p, "")
            onehot = [0.0] * len(policies)
            onehot[policy_to_j[p]] = 1.0
            nums = [
                float(np.isfinite(v)),
                float(np.isfinite(v) and abs(v) < 1),
                float(np.isfinite(v) and abs(v) >= 1),
                float(np.isfinite(v) and abs(v) > 100),
                float("." in st),
                float("%" in st),
                len(st),
                counts.get(st, 0),
                safe_ratio(v, self_val),
                safe_ratio(v, base_val),
                safe_ratio(v, axis_val),
                abs(safe_ratio(v, self_val)),
                abs(safe_ratio(v, base_val)),
                float(p.startswith("tta:")),
                float(p.startswith("main:large")),
                float(p.startswith("verifier:")),
            ] + onehot
            rows.append((id_to_i[cid], nums))
            labels.append(int(score.loc[cid, p]))
            row_meta.append((cid, p))

    base_indices = [r[0] for r in rows]
    X = hstack(
        [
            X_base_text[base_indices],
            X_base_num[base_indices],
            csr_matrix(np.array([r[1] for r in rows], dtype=float)),
        ]
    ).tocsr()
    return ids, policies, score, X, np.array(labels), row_meta


def choose_by_prob(probs, row_meta, candidate_ids, policies):
    out = {}
    for prob, (cid, p) in zip(probs, row_meta):
        if cid not in out or prob > out[cid][0]:
            out[cid] = (float(prob), p)
    return {cid: out[cid][1] for cid in candidate_ids}


def conservative_choose(probs, row_meta, candidate_ids, default_policy, margin):
    grouped = {}
    for prob, (cid, p) in zip(probs, row_meta):
        grouped.setdefault(cid, {})[p] = float(prob)
    chosen = {}
    for cid in candidate_ids:
        dprob = grouped[cid].get(default_policy, 0.0)
        best_p, best_prob = max(grouped[cid].items(), key=lambda x: x[1])
        chosen[cid] = best_p if best_prob >= dprob + margin else default_policy
    return chosen


def score_choices(score, choices):
    return float(np.mean([score.loc[cid, policy] for cid, policy in choices.items()]))


def main():
    print("loading", POOL, flush=True)
    pool = pd.read_csv(POOL)
    ids, policies, score, X, y, row_meta = build_features(pool)
    row_cids = np.array([cid for cid, _ in row_meta])

    models = {
        "logreg": LogisticRegression(max_iter=2000, C=0.7, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", n_jobs=-1),
        "et": ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", n_jobs=-1),
    }
    margins = np.linspace(-0.05, 0.35, 17)
    default_policy = "tta:self_check"
    records = []
    detail_rows = []

    for seed in range(50):
        rng = np.random.default_rng(seed)
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        train = shuffled[: int(0.6 * n)]
        dev = shuffled[int(0.6 * n) : int(0.8 * n)]
        test = shuffled[int(0.8 * n) :]

        train_mask = np.isin(row_cids, train)
        dev_mask = np.isin(row_cids, dev)
        test_mask = np.isin(row_cids, test)

        best = None
        fitted = {}
        for name, model in models.items():
            import copy

            clf = copy.deepcopy(model)
            try:
                clf.set_params(random_state=seed)
            except ValueError:
                pass
            clf.fit(X[train_mask], y[train_mask])
            fitted[name] = clf
            dev_prob = clf.predict_proba(X[dev_mask])[:, 1]
            dev_meta = [m for m, keep in zip(row_meta, dev_mask) if keep]

            stack_choices = choose_by_prob(dev_prob, dev_meta, dev, policies)
            stack_score = score_choices(score, stack_choices)
            key = (stack_score, name, "stack", None)
            if best is None or key[0] > best[0]:
                best = key

            for margin in margins:
                choices = conservative_choose(dev_prob, dev_meta, dev, default_policy, margin)
                val = score_choices(score, choices)
                key = (val, name, "patch", float(margin))
                if key[0] > best[0]:
                    best = key

        _, name, mode, margin = best
        clf = fitted[name]
        test_prob = clf.predict_proba(X[test_mask])[:, 1]
        test_meta = [m for m, keep in zip(row_meta, test_mask) if keep]
        if mode == "stack":
            choices = choose_by_prob(test_prob, test_meta, test, policies)
        else:
            choices = conservative_choose(test_prob, test_meta, test, default_policy, margin)

        rec = {
            "seed": seed,
            "selected_model": name,
            "selected_mode": mode,
            "selected_margin": margin,
            "selector": score_choices(score, choices),
            "self_check": float(score.loc[test, default_policy].mean()),
            "axis": float(score.loc[test, "tta:axis"].mean()),
            "best_fixed_split": float(score.loc[test, policies].mean().max()),
            "policy_oracle": float(score.loc[test, policies].max(axis=1).mean()),
            "switch_rate": float(np.mean([choices[cid] != default_policy for cid in test])),
        }
        records.append(rec)
        for cid, policy in choices.items():
            detail_rows.append({"seed": seed, "candidate_id": cid, "chosen_policy": policy, "score": int(score.loc[cid, policy])})
        if seed % 5 == 0:
            print("seed", seed, rec, flush=True)

    result = pd.DataFrame(records)
    detail = pd.DataFrame(detail_rows)
    summary = {
        "n": int(len(ids)),
        "selector_mean": float(result.selector.mean()),
        "selector_std": float(result.selector.std(ddof=1)),
        "selfcheck_mean": float(result.self_check.mean()),
        "axis_mean": float(result.axis.mean()),
        "best_fixed_split_mean": float(result.best_fixed_split.mean()),
        "policy_oracle_mean": float(result.policy_oracle.mean()),
        "switch_rate_mean": float(result.switch_rate.mean()),
        "selected_model_counts": result.selected_model.value_counts().to_dict(),
        "selected_mode_counts": result.selected_mode.value_counts().to_dict(),
        "selected_margin_counts": result.selected_margin.astype(str).value_counts().to_dict(),
    }
    result.to_csv(OUT / "scaleqa520_policy_patch_selector_v6_results.csv", index=False)
    detail.to_csv(OUT / "scaleqa520_policy_patch_selector_v6_choices.csv", index=False)
    (OUT / "scaleqa520_policy_patch_selector_v6_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
