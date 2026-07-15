#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUT = Path("results/scaleqa_natural/policy_selector_v17_pairwise_top8_noleak")
OUT.mkdir(parents=True, exist_ok=True)
POOL = Path("results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
OPTIONAL = [
    Path("results/eval/20260521_scaleqa520_structured_verifier_top4.csv"),
    Path("results/eval/20260521_scaleqa520_structured_verifier_ordered.csv"),
    Path("results/eval/20260521_scaleqa520_pairwise_verifier.csv"),
    Path("results/eval/20260521_scaleqa520_tta_v2_scale_regime.csv"),
    Path("results/eval/20260522_scaleqa520_rationale_verifier_v4_answer.csv"),
    Path("results/eval/20260522_scaleqa520_rationale_verifier_v4_large1.csv"),
    Path("results/eval/20260522_scaleqa520_rationale_verifier_v4_base.csv"),
    Path("results/eval/20260522_scaleqa520_candidate_likelihood_v5_answer.csv"),
    Path("results/eval/20260522_scaleqa520_candidate_likelihood_v5_large1.csv"),
    Path("results/eval/20260522_scaleqa520_candidate_likelihood_v5_base.csv"),
    Path("results/eval/20260522_scaleqa520_candidate_likelihood_top8_answer.csv"),
    Path("results/eval/20260522_scaleqa520_candidate_likelihood_top8_large1.csv"),
    Path("results/eval/20260522_scaleqa520_candidate_likelihood_top8_base.csv"),
    Path("results/eval/20260522_scaleqa520_pairwise_top8_answer.csv"),
]
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_num(x):
    if pd.isna(x):
        return np.nan
    m = NUM_RE.search(str(x).replace(",", "").replace("−", "-"))
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except ValueError:
        return np.nan


def txt(x):
    return "" if pd.isna(x) else str(x).strip().lower()


def safe_log_ratio(a, b):
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12 or abs(b) < 1e-12:
        return 0.0
    return float(math.log10(abs(a / b)))


def numeric_bucket(v):
    if not np.isfinite(v):
        return [0.0] * 7
    av = abs(v)
    return [
        float(av == 0),
        float(0 < av < 0.01),
        float(0.01 <= av < 0.1),
        float(0.1 <= av < 1),
        float(1 <= av < 100),
        float(100 <= av < 10000),
        float(av >= 10000),
    ]


def build():
    pool = pd.read_csv(POOL)
    base_meta = pool.drop_duplicates("candidate_id").set_index("candidate_id")
    extra_rows = []
    for path in OPTIONAL:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "adapter" not in df.columns or "candidate_id" not in df.columns:
            continue
        for _, r in df.iterrows():
            adapter = txt(r.get("adapter", ""))
            if "oracle" in adapter:
                continue
            cid = str(r["candidate_id"])
            if cid not in base_meta.index:
                continue
            pred_val = r.get("prediction", "")
            if pd.isna(pred_val) or str(pred_val).strip() == "":
                continue
            m = base_meta.loc[cid]
            extra_rows.append(
                {
                    "candidate_id": cid,
                    "adapter": adapter,
                    "question": r.get("question", m.get("question", "")),
                    "answers": r.get("answers", m.get("answers", "")),
                    "prediction": pred_val,
                    "prediction_norm": pred_val,
                    "score": int(r.get("score", 0)),
                    "chart_context": m.get("chart_context", ""),
                }
            )
    if extra_rows:
        pool = pd.concat([pool, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)
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
        "decimal_guard",
        "axis_unit_audit",
        "percent_display",
        "regime_check",
        "tta_majority",
        "pairwise_default_vs_alt",
        "structured_context_letter",
        "structured_scale_audit",
        "structured_order_source_base_last",
        "structured_order_numeric_asc",
        "structured_order_numeric_desc",
    ]
    policies = [p for p in preferred if p in score.columns]
    ids = np.array(sorted(score.index))
    id_to_i = {cid: i for i, cid in enumerate(ids)}
    pred_cols = [p for p in policies if p in pred.columns]

    base_texts, base_num, values, strings = [], [], {}, {}
    for cid in ids:
        q = txt(meta.loc[cid].get("question", ""))
        ctx = txt(meta.loc[cid].get("chart_context", ""))
        strings[cid] = {p: txt(pred.loc[cid].get(p, "")) for p in pred_cols}
        values[cid] = {p: parse_num(pred.loc[cid].get(p, "")) for p in pred_cols}
        vals = list(values[cid].values())
        s = q + " " + ctx
        flags = [
            "%" in s or "percent" in s or "percentage" in s,
            "million" in s or "billion" in s or "thousand" in s,
            "average" in s or "mean" in s,
            "difference" in s or "greater" in s or "less" in s or "more" in s or "fewer" in s,
            "highest" in s or "largest" in s or "maximum" in s,
            "lowest" in s or "smallest" in s or "minimum" in s,
            "axis" in s or "ticks=" in s,
            "legend=" in s or "units=" in s,
            "bar" in s,
            "line" in s,
            "pie" in s,
        ]
        ratios = []
        for i, a in enumerate(vals):
            for b in vals[i + 1 :]:
                ratios.append(safe_log_ratio(a, b))
        finite = [v for v in vals if np.isfinite(v)]
        base_texts.append("[Q] " + q + " [CTX] " + ctx + " [PREDS] " + " || ".join(strings[cid].values()))
        base_num.append(
            [float(x) for x in flags]
            + [
                len(q),
                len(ctx),
                len(strings[cid]),
                sum(np.isfinite(v) for v in vals),
                float(any(np.isfinite(v) and abs(v) < 1 for v in vals)),
                float(any(np.isfinite(v) and abs(v) > 100 for v in vals)),
                float(np.nanmean(finite) if finite else 0.0),
                float(np.nanstd(finite) if len(finite) > 1 else 0.0),
            ]
            + ratios[:36]
        )

    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X_text = vec.fit_transform(base_texts)
    X_base = hstack([X_text, csr_matrix(np.asarray(base_num, dtype=float))]).tocsr()

    policy_to_j = {p: j for j, p in enumerate(policies)}
    row_base, row_extra, row_y, row_meta = [], [], [], []
    for cid in ids:
        counts = pd.Series(list(strings[cid].values())).value_counts().to_dict()
        self_v = values[cid].get("tta:self_check", np.nan)
        axis_v = values[cid].get("tta:axis", np.nan)
        ans_v = values[cid].get("main:answer_tuned", values[cid].get("main:base", np.nan))
        maj_v = values[cid].get("verifier:candidate_majority", np.nan)
        for p in policies:
            v = values[cid].get(p, np.nan)
            st = strings[cid].get(p, "")
            onehot = [0.0] * len(policies)
            onehot[policy_to_j[p]] = 1.0
            extra = (
                [
                    float(np.isfinite(v)),
                    float("." in st),
                    float("%" in st),
                    len(st),
                    counts.get(st, 0),
                    safe_log_ratio(v, self_v),
                    safe_log_ratio(v, axis_v),
                    safe_log_ratio(v, ans_v),
                    safe_log_ratio(v, maj_v),
                    abs(safe_log_ratio(v, self_v)),
                    abs(safe_log_ratio(v, axis_v)),
                    abs(safe_log_ratio(v, ans_v)),
                    float(p.startswith("tta:")),
                    float(p.startswith("main:large")),
                    float(p.startswith("verifier:")),
                    float(p == "tta:self_check"),
                    float(p == "tta:axis"),
                    float(p == "tta:unit_guard"),
                ]
                + numeric_bucket(v)
                + onehot
            )
            row_base.append(id_to_i[cid])
            row_extra.append(extra)
            row_y.append(int(score.loc[cid, p]))
            row_meta.append((cid, p))
    X_row = hstack([X_base[row_base], csr_matrix(np.asarray(row_extra, dtype=float))]).tocsr()
    return ids, policies, score, X_base, X_row, np.asarray(row_y), np.asarray([m[0] for m in row_meta]), row_meta


def score_policy(score, ids, policy):
    return float(score.loc[ids, policy].mean())


def score_choices(score, choices):
    return float(np.mean([score.loc[cid, p] for cid, p in choices.items()]))


def choose_stack(probs, meta, ids):
    best = {}
    for prob, (cid, p) in zip(probs, meta):
        if cid not in best or float(prob) > best[cid][0]:
            best[cid] = (float(prob), p)
    return {cid: best[cid][1] for cid in ids}


def choose_patch(probs, meta, ids, default, margin):
    grouped = {}
    for prob, (cid, p) in zip(probs, meta):
        grouped.setdefault(cid, {})[p] = float(prob)
    choices = {}
    for cid in ids:
        base = grouped[cid].get(default, 0.0)
        best_p, best_prob = max(grouped[cid].items(), key=lambda x: x[1])
        choices[cid] = best_p if best_prob >= base + margin else default
    return choices


def best_policy_labels(score, ids, policies):
    labels = []
    for cid in ids:
        correct = [p for p in policies if int(score.loc[cid, p]) == 1]
        labels.append(correct[0] if correct else policies[0])
    return np.asarray(labels)


def main():
    print("building features", flush=True)
    ids, policies, score, X_base, X_row, y_row, row_cids, row_meta = build()
    default = "tta:self_check"
    margins = np.linspace(-0.05, 0.30, 15)
    row_models = {
        "logreg": LogisticRegression(max_iter=2000, C=0.7, class_weight="balanced"),
        "et_small": ExtraTreesClassifier(n_estimators=180, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=4),
        "rf_small": RandomForestClassifier(n_estimators=160, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=4),
    }
    base_models = {
        "base_logreg": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "base_histgb": make_pipeline(StandardScaler(with_mean=False), TruncatedSVD(n_components=96, random_state=0), HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=15)),
    }
    records = []
    choices_out = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        train = shuffled[: int(0.6 * n)]
        dev = shuffled[int(0.6 * n) : int(0.8 * n)]
        test = shuffled[int(0.8 * n) :]
        train_dev = np.concatenate([train, dev])
        train_mask = np.isin(row_cids, train)
        dev_mask = np.isin(row_cids, dev)
        td_mask = np.isin(row_cids, train_dev)
        test_mask = np.isin(row_cids, test)

        candidates = []
        # Fixed policy selected on dev.
        dev_scores = {p: score_policy(score, dev, p) for p in policies}
        fixed_p = max(dev_scores, key=dev_scores.get)
        candidates.append((dev_scores[fixed_p], "fixed", fixed_p, None, None))

        # Row-level policy correctness models.
        fitted_row = {}
        for name, clf in row_models.items():
            import copy

            model = copy.deepcopy(clf)
            try:
                model.set_params(random_state=seed)
            except ValueError:
                pass
            model.fit(X_row[train_mask], y_row[train_mask])
            fitted_row[name] = model
            probs = model.predict_proba(X_row[dev_mask])[:, 1]
            dev_meta = [m for m, keep in zip(row_meta, dev_mask) if keep]
            stack = choose_stack(probs, dev_meta, dev)
            candidates.append((score_choices(score, stack), name, "stack", None, None))
            for margin in margins:
                patch = choose_patch(probs, dev_meta, dev, default, float(margin))
                candidates.append((score_choices(score, patch), name, "patch", float(margin), None))

        # Multiclass best-policy classifier from candidate-level features.
        id_to_i = {cid: i for i, cid in enumerate(ids)}
        train_i = [id_to_i[cid] for cid in train]
        dev_i = [id_to_i[cid] for cid in dev]
        train_dev_i = [id_to_i[cid] for cid in train_dev]
        test_i = [id_to_i[cid] for cid in test]
        y_best = best_policy_labels(score, ids, policies)
        fitted_base = {}
        for name, clf in base_models.items():
            import copy

            model = copy.deepcopy(clf)
            try:
                model.set_params(random_state=seed)
            except ValueError:
                pass
            model.fit(X_base[train_i], y_best[train_i])
            fitted_base[name] = model
            pred = model.predict(X_base[dev_i])
            choices = {cid: p for cid, p in zip(dev, pred)}
            candidates.append((score_choices(score, choices), name, "multiclass", None, None))

        best = max(candidates, key=lambda x: x[0])
        _, model_name, mode, margin, _ = best
        if model_name == "fixed":
            chosen_policy = mode
            choices = {cid: chosen_policy for cid in test}
        elif mode in {"stack", "patch"}:
            import copy

            model = copy.deepcopy(row_models[model_name])
            try:
                model.set_params(random_state=seed)
            except ValueError:
                pass
            model.fit(X_row[td_mask], y_row[td_mask])
            probs = model.predict_proba(X_row[test_mask])[:, 1]
            test_meta = [m for m, keep in zip(row_meta, test_mask) if keep]
            choices = choose_stack(probs, test_meta, test) if mode == "stack" else choose_patch(probs, test_meta, test, default, margin)
        else:
            import copy

            model = copy.deepcopy(base_models[model_name])
            try:
                model.set_params(random_state=seed)
            except ValueError:
                pass
            model.fit(X_base[train_dev_i], y_best[train_dev_i])
            pred = model.predict(X_base[test_i])
            choices = {cid: p for cid, p in zip(test, pred)}

        rec = {
            "seed": seed,
            "selected_model": model_name,
            "selected_mode": mode,
            "selected_margin": margin,
            "selector": score_choices(score, choices),
            "self_check": score_policy(score, test, default),
            "axis": score_policy(score, test, "tta:axis"),
            "dev_selected_fixed": score_policy(score, test, fixed_p),
            "dev_selected_fixed_policy": fixed_p,
            "best_fixed_split": float(score.loc[test, policies].mean().max()),
            "policy_oracle": float(score.loc[test, policies].max(axis=1).mean()),
            "switch_rate": float(np.mean([choices[cid] != default for cid in test])),
        }
        records.append(rec)
        for cid, p in choices.items():
            choices_out.append({"seed": seed, "candidate_id": cid, "chosen_policy": p, "score": int(score.loc[cid, p])})
        if seed % 5 == 0:
            print("seed", seed, rec, flush=True)

    df = pd.DataFrame(records)
    summary = {
        "n": int(len(ids)),
        "selector_mean": float(df.selector.mean()),
        "selector_std": float(df.selector.std(ddof=1)),
        "selfcheck_mean": float(df.self_check.mean()),
        "axis_mean": float(df.axis.mean()),
        "dev_selected_fixed_mean": float(df.dev_selected_fixed.mean()),
        "best_fixed_split_mean": float(df.best_fixed_split.mean()),
        "policy_oracle_mean": float(df.policy_oracle.mean()),
        "switch_rate_mean": float(df.switch_rate.mean()),
        "selected_model_counts": df.selected_model.value_counts().to_dict(),
        "selected_mode_counts": df.selected_mode.value_counts().to_dict(),
        "dev_selected_fixed_policy_counts": df.dev_selected_fixed_policy.value_counts().to_dict(),
    }
    df.to_csv(OUT / "scaleqa520_policy_selector_v17_pairwise_top8_noleak_results.csv", index=False)
    pd.DataFrame(choices_out).to_csv(OUT / "scaleqa520_policy_selector_v17_pairwise_top8_noleak_choices.csv", index=False)
    (OUT / "scaleqa520_policy_selector_v17_pairwise_top8_noleak_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
