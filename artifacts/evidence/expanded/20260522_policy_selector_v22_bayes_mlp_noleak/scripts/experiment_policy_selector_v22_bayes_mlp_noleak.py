#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUT = Path("results/scaleqa_natural/policy_selector_v22_bayes_mlp_noleak")
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
    Path("results/eval/20260522_scaleqa520_pairwise_top8_large1.csv"),
    Path("results/eval/20260522_scaleqa520_pairwise_top8_base.csv"),
    Path("results/eval/20260522_scaleqa520_candidate_letter_top8_answer.csv"),
]
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def txt(x) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def parse_num(x) -> float:
    if pd.isna(x):
        return np.nan
    m = NUM_RE.search(str(x).replace(",", "").replace("−", "-"))
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except ValueError:
        return np.nan


def norm_pred(x) -> str:
    s = txt(x).replace(",", "").replace("−", "-")
    if not s or s in {"nan", "none"}:
        return ""
    m = NUM_RE.search(s)
    if not m:
        return s[:64]
    try:
        v = float(m.group(0))
    except ValueError:
        return s[:64]
    if abs(v) >= 100:
        body = f"{v:.3f}"
    elif abs(v) >= 1:
        body = f"{v:.4f}"
    else:
        body = f"{v:.6f}"
    body = body.rstrip("0").rstrip(".")
    return body + ("%" if "%" in s else "")


def add_rows(pool: pd.DataFrame) -> pd.DataFrame:
    base_meta = pool.drop_duplicates("candidate_id").set_index("candidate_id")
    rows = []
    for path in OPTIONAL:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "adapter" not in df.columns or "candidate_id" not in df.columns:
            continue
        for _, r in df.iterrows():
            adapter = txt(r.get("adapter", ""))
            if not adapter or "oracle" in adapter:
                continue
            cid = str(r["candidate_id"])
            if cid not in base_meta.index:
                continue
            pred_val = r.get("prediction", "")
            if pd.isna(pred_val) or str(pred_val).strip() == "":
                continue
            m = base_meta.loc[cid]
            rows.append(
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
    if rows:
        pool = pd.concat([pool, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return pool


def rank_policy(p: str) -> tuple[int, str]:
    preferred = [
        "tta:self_check",
        "tta:axis",
        "tta:direct",
        "tta:unit_guard",
        "tta:tta_majority",
        "main:answer_tuned",
        "main:large_seed4",
        "main:large_seed5",
        "main:large_seed1",
        "main:large_seed2",
        "main:large_seed3",
        "main:base",
        "decimal_guard",
        "axis_unit_audit",
        "percent_display",
        "regime_check",
        "verifier:candidate_majority",
        "candidate_majority",
    ]
    if p in preferred:
        return preferred.index(p), p
    if "candlik" in p:
        return 200, p
    if "pairwise" in p:
        return 250, p
    if "letter" in p:
        return 300, p
    if "structured" in p:
        return 350, p
    return 500, p


def text_tags(meta: pd.DataFrame, cid: str) -> list[str]:
    s = txt(meta.loc[cid].get("question", "")) + " " + txt(meta.loc[cid].get("chart_context", ""))
    tags = []
    if "%" in s or "percent" in s or "percentage" in s:
        tags.append("percent")
    if any(w in s for w in ["million", "billion", "thousand", "units=", "bn", "usd", "$"]):
        tags.append("unit")
    if any(w in s for w in ["difference", "greater", "less", "more", "fewer", "increase", "decrease"]):
        tags.append("compare")
    if any(w in s for w in ["highest", "lowest", "largest", "smallest", "maximum", "minimum"]):
        tags.append("extreme")
    if any(w in s for w in ["average", "mean"]):
        tags.append("average")
    if not tags:
        tags.append("plain")
    return tags


def bucket(meta: pd.DataFrame, cid: str, fine: bool) -> str:
    tags = text_tags(meta, cid)
    if not fine:
        return tags[0]
    s = txt(meta.loc[cid].get("question", "")) + " " + txt(meta.loc[cid].get("chart_context", ""))
    chart = "bar" if "bar" in s else "line" if "line" in s else "pie" if "pie" in s else "chart"
    return "+".join(tags[:2] + [chart])


def build():
    pool = add_rows(pd.read_csv(POOL))
    pool["adapter"] = pool["adapter"].map(txt)
    pool["candidate_id"] = pool["candidate_id"].astype(str)
    pool = pool[~pool["adapter"].str.contains("oracle", na=False)].copy()
    pool["score"] = pool["score"].fillna(0).astype(int)
    score = pool.pivot_table(index="candidate_id", columns="adapter", values="score", aggfunc="first").fillna(0).astype(int)
    pred = pool.pivot_table(index="candidate_id", columns="adapter", values="prediction_norm", aggfunc="first")
    meta = pool.drop_duplicates("candidate_id").set_index("candidate_id")
    policies = sorted(list(score.columns), key=rank_policy)
    ids = np.array(sorted(score.index))
    norm = {
        cid: {p: norm_pred(pred.loc[cid].get(p, "")) if p in pred.columns else "" for p in policies}
        for cid in ids
    }
    values = {
        cid: {p: parse_num(pred.loc[cid].get(p, "")) if p in pred.columns else np.nan for p in policies}
        for cid in ids
    }
    return ids, policies, score, meta, norm, values


def mean_policy(score, ids, p):
    return float(score.loc[ids, p].mean()) if p in score.columns else 0.0


def mean_choices(score, choices):
    return float(np.mean([score.loc[cid, p] for cid, p in choices.items()]))


def smoothed_priors(score, train, policies, meta, alpha, fine):
    global_mean = {p: mean_policy(score, train, p) for p in policies}
    b_to_ids = defaultdict(list)
    for cid in train:
        b_to_ids[bucket(meta, cid, fine)].append(cid)
    b_prior = {}
    for b, cids in b_to_ids.items():
        n = len(cids)
        b_prior[b] = {
            p: float((score.loc[cids, p].sum() + alpha * global_mean[p]) / (n + alpha))
            for p in policies
        }
    return global_mean, b_prior


def policy_flags(p: str) -> list[float]:
    return [
        float(p.startswith("tta:")),
        float(p.startswith("main:")),
        float("candlik" in p),
        float("pairwise" in p),
        float("letter" in p),
        float("structured" in p),
        float(p == "decimal_guard"),
        float(p == "axis_unit_audit" or p == "tta:axis"),
        float("percent" in p),
        float("regime" in p),
    ]


def make_features(cids, policies, score, meta, norm, values, global_mean, bucket_prior, fine):
    rows = []
    labels = []
    refs = []
    for cid in cids:
        b = bucket(meta, cid, fine)
        pri = bucket_prior.get(b, global_mean)
        counts = Counter(norm[cid].get(p, "") for p in policies if norm[cid].get(p, ""))
        max_count = max(counts.values()) if counts else 1
        tags = set(text_tags(meta, cid))
        for p_i, p in enumerate(policies):
            ns = norm[cid].get(p, "")
            v = values[cid].get(p, np.nan)
            is_num = float(not np.isnan(v))
            log_abs = math.log1p(abs(v)) if not np.isnan(v) else 0.0
            has_percent = float("%" in ns)
            consensus = counts.get(ns, 0)
            global_p = global_mean.get(p, 0.5)
            bucket_p = pri.get(p, global_p)
            self_p = pri.get("tta:self_check", global_mean.get("tta:self_check", 0.5))
            axis_p = pri.get("tta:axis", global_mean.get("tta:axis", 0.5))
            feats = [
                global_p,
                bucket_p,
                bucket_p - global_p,
                bucket_p - self_p,
                bucket_p - axis_p,
                math.log1p(consensus),
                consensus / max(1, max_count),
                p_i / max(1, len(policies) - 1),
                is_num,
                log_abs,
                has_percent,
                float("percent" in tags),
                float("unit" in tags),
                float("compare" in tags),
                float("extreme" in tags),
                float("average" in tags),
                float("plain" in tags),
            ] + policy_flags(p)
            rows.append(feats)
            labels.append(int(score.loc[cid, p]))
            refs.append((cid, p, ns))
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int32), refs


def train_model(X, y, seed, hidden, alpha):
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=0.003,
            max_iter=80,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=seed,
        ),
    )
    clf.fit(X, y)
    return clf


def choose_with_model(clf, cids, policies, score, meta, norm, values, global_mean, bucket_prior, cfg):
    fine, hidden, alpha, group_mode, default, margin, _switch_cap = cfg
    X, _, refs = make_features(cids, policies, score, meta, norm, values, global_mean, bucket_prior, fine)
    prob = clf.predict_proba(X)[:, 1]
    by_cid = defaultdict(list)
    for pr, (cid, p, ns) in zip(prob, refs):
        by_cid[cid].append((float(pr), p, ns))
    choices = {}
    for cid, items in by_cid.items():
        default_score = max((s for s, p, _ in items if p == default), default=-1.0)
        if group_mode:
            group_score = defaultdict(float)
            group_best = {}
            for s, p, ns in items:
                key = ns or f"__{p}"
                group_score[key] += math.exp(4.0 * s)
                if key not in group_best or s > group_best[key][0]:
                    group_best[key] = (s, p)
            best_key, best_group_score = max(group_score.items(), key=lambda x: x[1])
            best_p = group_best[best_key][1]
            best_score = math.log(best_group_score + 1e-12) / 4.0
        else:
            best_score, best_p, _ = max(items, key=lambda x: x[0])
        choices[cid] = best_p if best_score >= default_score + margin else default
    return choices


def main():
    ids, policies, score, meta, norm, values = build()
    print(f"n={len(ids)} policies={len(policies)}", flush=True)
    defaults = [p for p in ["tta:self_check", "tta:axis"] if p in policies]
    if not defaults:
        defaults = [policies[0]]
    cfgs = []
    for fine in [False, True]:
        for hidden in [(16,), (32, 16)]:
            for alpha in [0.001, 0.01]:
                for group_mode in [False, True]:
                    for default in defaults:
                        for margin in [0.08, 0.18]:
                            for switch_cap in [0.45, 0.60]:
                                cfgs.append((fine, hidden, alpha, group_mode, default, margin, switch_cap))
    print(f"cfgs={len(cfgs)} defaults={defaults}", flush=True)

    records = []
    choices_rows = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        train = shuffled[: int(0.6 * n)]
        dev = shuffled[int(0.6 * n): int(0.8 * n)]
        test = shuffled[int(0.8 * n):]
        train_dev = np.concatenate([train, dev])

        candidates = []
        cache = {}
        for cfg in cfgs:
            fine, hidden, alpha, group_mode, default, margin, switch_cap = cfg
            key = (fine, hidden, alpha)
            if key not in cache:
                g, b = smoothed_priors(score, train, policies, meta, alpha=5.0, fine=fine)
                X, y, _ = make_features(train, policies, score, meta, norm, values, g, b, fine)
                cache[key] = (train_model(X, y, seed, hidden, alpha), g, b)
            clf, g, b = cache[key]
            ch = choose_with_model(clf, dev, policies, score, meta, norm, values, g, b, cfg)
            dev_score = mean_choices(score, ch)
            dev_switch = float(np.mean([ch[cid] != "tta:self_check" for cid in dev])) if "tta:self_check" in policies else 0.0
            # Avoid dev-overfit selectors that win by switching almost everything.
            objective = dev_score - max(0.0, dev_switch - switch_cap) * 0.5
            candidates.append((objective, dev_score, cfg))
        fixed_scores = {p: mean_policy(score, dev, p) for p in policies}
        fixed_p = max(fixed_scores, key=fixed_scores.get)
        candidates.append((fixed_scores[fixed_p], fixed_scores[fixed_p], ("fixed", fixed_p)))
        _, _, best_cfg = max(candidates, key=lambda x: x[0])

        if best_cfg[0] == "fixed":
            choices = {cid: best_cfg[1] for cid in test}
            mode = "fixed"
            cfg_json = json.dumps({"mode": "fixed", "policy": best_cfg[1]})
            default = best_cfg[1]
        else:
            fine, hidden, alpha, group_mode, default, margin, switch_cap = best_cfg
            g, b = smoothed_priors(score, train_dev, policies, meta, alpha=5.0, fine=fine)
            X, y, _ = make_features(train_dev, policies, score, meta, norm, values, g, b, fine)
            clf = train_model(X, y, seed, hidden, alpha)
            choices = choose_with_model(clf, test, policies, score, meta, norm, values, g, b, best_cfg)
            mode = "group_mlp" if group_mode else "policy_mlp"
            cfg_json = json.dumps(
                {
                    "fine": fine,
                    "hidden": hidden,
                    "alpha": alpha,
                    "group": group_mode,
                    "default": default,
                    "margin": margin,
                    "switch_cap": switch_cap,
                }
            )
        rec = {
            "seed": seed,
            "selector": mean_choices(score, choices),
            "self_check": mean_policy(score, test, "tta:self_check"),
            "axis": mean_policy(score, test, "tta:axis"),
            "dev_selected_fixed": mean_policy(score, test, fixed_p),
            "dev_selected_fixed_policy": fixed_p,
            "best_fixed_split": float(score.loc[test, policies].mean().max()),
            "policy_oracle": float(score.loc[test, policies].max(axis=1).mean()),
            "switch_rate": float(np.mean([choices[cid] != "tta:self_check" for cid in test])) if "tta:self_check" in policies else np.nan,
            "mode": mode,
            "default": default,
            "cfg": cfg_json,
        }
        records.append(rec)
        for cid, p in choices.items():
            choices_rows.append({"seed": seed, "candidate_id": cid, "chosen_policy": p, "score": int(score.loc[cid, p])})
        print("seed", seed, rec, flush=True)

    df = pd.DataFrame(records)
    summary = {
        "n": int(len(ids)),
        "n_policies": int(len(policies)),
        "selector_mean": float(df.selector.mean()),
        "selector_std": float(df.selector.std(ddof=1)),
        "selfcheck_mean": float(df.self_check.mean()),
        "axis_mean": float(df.axis.mean()),
        "dev_selected_fixed_mean": float(df.dev_selected_fixed.mean()),
        "best_fixed_split_mean": float(df.best_fixed_split.mean()),
        "policy_oracle_mean": float(df.policy_oracle.mean()),
        "switch_rate_mean": float(df.switch_rate.mean()),
        "mode_counts": df["mode"].value_counts().to_dict(),
        "default_counts": df["default"].value_counts().to_dict(),
        "dev_selected_fixed_policy_counts": df.dev_selected_fixed_policy.value_counts().to_dict(),
    }
    df.to_csv(OUT / "scaleqa520_policy_selector_v22_bayes_mlp_noleak_results.csv", index=False)
    pd.DataFrame(choices_rows).to_csv(OUT / "scaleqa520_policy_selector_v22_bayes_mlp_noleak_choices.csv", index=False)
    (OUT / "scaleqa520_policy_selector_v22_bayes_mlp_noleak_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
