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
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

OUT = Path("results/scaleqa_natural/regime_router_v5_fast")
OUT.mkdir(parents=True, exist_ok=True)

POOL = "results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv"
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_num(value):
    if pd.isna(value):
        return np.nan
    match = NUM_RE.search(str(value).replace(",", "").replace("−", "-"))
    if not match:
        return np.nan
    try:
        return float(match.group(0))
    except ValueError:
        return np.nan


def main() -> None:
    print("loading pool", flush=True)
    pool = pd.read_csv(POOL)
    ids = np.array(sorted(pool.candidate_id.unique()))
    score = (
        pool.pivot_table(index="candidate_id", columns="adapter", values="score", aggfunc="first")
        .fillna(0)
        .astype(int)
    )
    pred = pool.pivot_table(index="candidate_id", columns="adapter", values="prediction_norm", aggfunc="first")
    meta = pool.drop_duplicates("candidate_id").set_index("candidate_id")
    preferred = [
        "main:answer_tuned",
        "main:base",
        "main:large_seed4",
        "main:large_seed5",
        "tta:self_check",
        "tta:axis",
        "tta:direct",
        "tta:math",
        "tta:unit_guard",
        "tta:tta_majority",
        "verifier:candidate_majority",
    ]
    policies = [p for p in preferred if p in score.columns]

    texts, nums, y_sub, y_dir = [], [], [], []
    pred_cols = [
        p
        for p in [
            "main:answer_tuned",
            "main:base",
            "tta:self_check",
            "tta:axis",
            "tta:direct",
            "tta:unit_guard",
            "verifier:candidate_majority",
        ]
        if p in pred.columns
    ]
    for cid in ids:
        q = str(meta.loc[cid].get("question", ""))
        ctx = str(meta.loc[cid].get("chart_context", ""))
        subtype = str(meta.loc[cid].get("scale_error_subtype_manifest", meta.loc[cid].get("scale_error_subtype", "")))
        direction = str(meta.loc[cid].get("scale_directions_manifest", meta.loc[cid].get("scale_directions", "")))
        preds = [str(pred.loc[cid].get(p, "")) for p in pred_cols]
        texts.append((" [Q] " + q + " [CTX] " + ctx + " [PREDS] " + " || ".join(preds)).lower())
        vals = [parse_num(x) for x in preds]
        ratios = []
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, b = vals[i], vals[j]
                if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-12 and abs(b) > 1e-12:
                    ratios.append(math.log10(abs(a / b)))
                else:
                    ratios.append(0.0)
        s = (q + " " + ctx).lower()
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
        nums.append(
            [float(x) for x in flags]
            + [
                len(q),
                len(ctx),
                sum(np.isfinite(v) for v in vals),
                float(any(np.isfinite(v) and abs(v) < 1 for v in vals)),
                float(any(np.isfinite(v) and abs(v) > 100 for v in vals)),
            ]
            + ratios[:15]
        )
        y_sub.append(subtype)
        y_dir.append(direction)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = hstack([vec.fit_transform(texts), csr_matrix(np.array(nums, dtype=float))]).tocsr()
    y_sub = np.array(y_sub)
    y_dir = np.array(y_dir)
    idx = {cid: i for i, cid in enumerate(ids)}

    classifiers = {
        "logreg_sub": (LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"), y_sub),
        "logreg_dir": (LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"), y_dir),
        "et_sub": (
            ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1),
            y_sub,
        ),
        "rf_sub": (
            RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", n_jobs=-1),
            y_sub,
        ),
    }

    def best_map(train_ids, labels):
        labels = np.array(labels)
        train_ids = np.array(train_ids)
        default = score.loc[train_ids, policies].mean().idxmax()
        mapping = {}
        for lab in sorted(set(labels)):
            group = train_ids[labels == lab]
            mapping[lab] = score.loc[group, policies].mean().idxmax() if len(group) >= 3 else default
        return mapping, default

    def eval_labels(test_ids, labels, mapping, default):
        return float(np.mean([score.loc[cid, mapping.get(label, default)] for cid, label in zip(test_ids, labels)]))

    rows = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        train = shuffled[: int(0.6 * n)]
        dev = shuffled[int(0.6 * n) : int(0.8 * n)]
        test = shuffled[int(0.8 * n) :]
        train_i = [idx[x] for x in train]
        dev_i = [idx[x] for x in dev]
        test_i = [idx[x] for x in test]

        best = None
        for name, (clf, labels) in classifiers.items():
            import copy

            model = copy.deepcopy(clf)
            try:
                model.set_params(random_state=seed)
            except ValueError:
                pass
            model.fit(X[train_i], labels[train_i])
            pred_dev = model.predict(X[dev_i])
            mapping, default = best_map(train, labels[train_i])
            policy_acc = eval_labels(dev, pred_dev, mapping, default)
            label_acc = accuracy_score(labels[dev_i], pred_dev)
            key = (policy_acc, label_acc)
            if best is None or key > best[0]:
                best = (key, name, labels)

        _, selected, labels = best
        import copy

        model = copy.deepcopy(classifiers[selected][0])
        try:
            model.set_params(random_state=seed)
        except ValueError:
            pass
        train_dev = np.concatenate([train, dev])
        train_dev_i = [idx[x] for x in train_dev]
        model.fit(X[train_dev_i], labels[train_dev_i])
        pred_test = model.predict(X[test_i])
        mapping, default = best_map(train_dev, labels[train_dev_i])
        rows.append(
            {
                "seed": seed,
                "selected": selected,
                "router": eval_labels(test, pred_test, mapping, default),
                "self_check": float(score.loc[test, "tta:self_check"].mean()),
                "axis": float(score.loc[test, "tta:axis"].mean()),
                "best_fixed_split": float(score.loc[test, policies].mean().max()),
                "true_regime_oracle": eval_labels(test, labels[test_i], mapping, default),
                "label_acc": float(accuracy_score(labels[test_i], pred_test)),
                "default_policy": default,
            }
        )
        if seed % 5 == 0:
            print("seed", seed, "router", rows[-1]["router"], "selected", selected, flush=True)

    result = pd.DataFrame(rows)
    summary = {
        "n": int(len(ids)),
        "router_mean": float(result.router.mean()),
        "router_std": float(result.router.std(ddof=1)),
        "selfcheck_mean": float(result.self_check.mean()),
        "axis_mean": float(result.axis.mean()),
        "best_fixed_split_mean": float(result.best_fixed_split.mean()),
        "true_regime_oracle_mean": float(result.true_regime_oracle.mean()),
        "label_acc_mean": float(result.label_acc.mean()),
        "selected_counts": result.selected.value_counts().to_dict(),
        "default_policy_counts": result.default_policy.value_counts().to_dict(),
    }
    result.to_csv(OUT / "scaleqa520_regime_router_v5_fast_results.csv", index=False)
    (OUT / "scaleqa520_regime_router_v5_fast_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
