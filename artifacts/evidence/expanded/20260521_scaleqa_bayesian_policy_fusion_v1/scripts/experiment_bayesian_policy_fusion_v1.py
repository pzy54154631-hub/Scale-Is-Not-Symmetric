#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(os.environ.get("MIRAGE_BAYES_OUT", "results/scaleqa_natural/bayesian_policy_fusion_v1"))
OUT.mkdir(parents=True, exist_ok=True)

POOL = Path("results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
OPTIONAL = [
    Path("results/eval/20260521_scaleqa520_structured_verifier_top4.csv"),
    Path("results/eval/20260521_scaleqa520_structured_verifier_ordered.csv"),
    Path("results/eval/20260521_scaleqa520_pairwise_verifier.csv"),
    Path("results/eval/20260521_scaleqa520_tta_v2_scale_regime.csv"),
]

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def norm_answer(x: object) -> str:
    s = "" if pd.isna(x) else str(x).strip().lower()
    s = s.replace(",", "").replace("−", "-")
    m = NUM_RE.search(s)
    if m:
        try:
            v = float(m.group(0))
            if abs(v - round(v)) < 1e-8:
                return str(int(round(v)))
            return f"{v:.8g}"
        except Exception:
            pass
    return re.sub(r"\s+", " ", s)


def parse_num(x: object) -> float:
    m = NUM_RE.search(str(x).replace(",", "").replace("−", "-"))
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except Exception:
        return float("nan")


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def context_bucket(row: pd.Series) -> str:
    q = str(row.get("question", "")).lower()
    ctx = str(row.get("chart_context", "")).lower()
    text = q + " " + ctx
    bits = []
    bits.append("pct" if ("%" in text or "percent" in text or "percentage" in text) else "nopct")
    bits.append("unit" if any(w in text for w in ["million", "billion", "thousand", "units=", "legend=", "axis", "ticks="]) else "nounit")
    if any(w in q for w in ["difference", "greater", "less", "more", "fewer"]):
        bits.append("diff")
    elif any(w in q for w in ["average", "mean"]):
        bits.append("avg")
    elif any(w in q for w in ["highest", "largest", "maximum", "lowest", "smallest", "minimum"]):
        bits.append("extreme")
    else:
        bits.append("lookup")
    return "|".join(bits)


def magnitude_bucket(ans: str) -> str:
    v = parse_num(ans)
    if not np.isfinite(v):
        return "nonnum"
    av = abs(v)
    if av == 0:
        return "zero"
    if av < 0.01:
        return "lt001"
    if av < 0.1:
        return "lt01"
    if av < 1:
        return "lt1"
    if av < 100:
        return "1_100"
    if av < 10000:
        return "100_1e4"
    return "ge1e4"


def load_rows() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    rows = []
    for _, r in pool.iterrows():
        rows.append(
            {
                "candidate_id": str(r["candidate_id"]),
                "adapter": str(r["adapter"]),
                "question": r.get("question", ""),
                "answers": r.get("answers", ""),
                "prediction": r.get("prediction_norm", r.get("prediction", "")),
                "answer_key": norm_answer(r.get("prediction_norm", r.get("prediction", ""))),
                "score": int(r.get("score", 0)),
                "chart_context": r.get("chart_context", ""),
            }
        )
    for path in OPTIONAL:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "candidate_id" not in df.columns or "prediction" not in df.columns:
            continue
        for _, r in df.iterrows():
            adapter = str(r.get("adapter", path.stem))
            if "oracle" in adapter:
                continue
            pred = r.get("prediction", "")
            if pd.isna(pred) or str(pred).strip() == "":
                continue
            rows.append(
                {
                    "candidate_id": str(r["candidate_id"]),
                    "adapter": adapter,
                    "question": r.get("question", ""),
                    "answers": r.get("answers", ""),
                    "prediction": pred,
                    "answer_key": norm_answer(pred),
                    "score": int(r.get("score", 0)),
                    "chart_context": "",
                }
            )
    data = pd.DataFrame(rows)
    # Fill context/question from pool metadata when optional files omit it.
    meta = pool.drop_duplicates("candidate_id").set_index("candidate_id")
    for col in ["question", "answers", "chart_context"]:
        miss = data[col].isna() | (data[col].astype(str).str.len() == 0)
        if miss.any():
            data.loc[miss, col] = data.loc[miss, "candidate_id"].map(meta[col] if col in meta.columns else {})
    data["ctx_bucket"] = data.apply(context_bucket, axis=1)
    data["mag_bucket"] = data["answer_key"].map(magnitude_bucket)
    return data


def beta_mean(pos: int, total: int, alpha: float) -> float:
    return (pos + alpha) / (total + 2 * alpha)


def fit_reliability(train: pd.DataFrame, alpha: float):
    global_rel = {}
    adapter_rel = {}
    adapter_ctx_rel = {}
    adapter_mag_rel = {}
    for name, g in train.groupby("adapter"):
        global_rel[name] = beta_mean(int(g.score.sum()), len(g), alpha)
    for key, g in train.groupby(["adapter", "ctx_bucket"]):
        adapter_ctx_rel[key] = beta_mean(int(g.score.sum()), len(g), alpha)
    for key, g in train.groupby(["adapter", "mag_bucket"]):
        adapter_mag_rel[key] = beta_mean(int(g.score.sum()), len(g), alpha)
    # Backoff prior over answer magnitude.
    for key, g in train.groupby("mag_bucket"):
        adapter_rel[key] = beta_mean(int(g.score.sum()), len(g), alpha)
    return global_rel, adapter_ctx_rel, adapter_mag_rel, adapter_rel


def choose_for_ids(df: pd.DataFrame, ids: np.ndarray, rel, weights):
    global_rel, ctx_rel, mag_rel, mag_prior = rel
    w_global, w_ctx, w_mag, w_vote, w_prior = weights
    choices = {}
    for cid in ids:
        g = df[df.candidate_id == cid]
        cand_scores = defaultdict(float)
        cand_votes = Counter()
        # Candidate prior: agreement count and magnitude plausibility.
        for _, r in g.iterrows():
            key = r.answer_key
            cand_votes[key] += 1
            a = r.adapter
            p0 = global_rel.get(a, 0.5)
            pc = ctx_rel.get((a, r.ctx_bucket), p0)
            pm = mag_rel.get((a, r.mag_bucket), p0)
            prior = mag_prior.get(r.mag_bucket, 0.5)
            cand_scores[key] += w_global * logit(p0) + w_ctx * logit(pc) + w_mag * logit(pm) + w_prior * logit(prior)
        max_vote = max(cand_votes.values()) if cand_votes else 1
        for key, v in cand_votes.items():
            cand_scores[key] += w_vote * math.log1p(v) / math.log1p(max_vote)
        best = max(cand_scores, key=cand_scores.get)
        choices[cid] = best
    return choices


def score_choices(df: pd.DataFrame, choices: dict[str, str]) -> float:
    vals = []
    for cid, key in choices.items():
        g = df[(df.candidate_id == cid) & (df.answer_key == key)]
        vals.append(int(g.score.max()) if len(g) else 0)
    return float(np.mean(vals))


def fixed_score(df: pd.DataFrame, ids: np.ndarray, adapter: str) -> float:
    vals = []
    for cid in ids:
        g = df[(df.candidate_id == cid) & (df.adapter == adapter)]
        vals.append(int(g.score.max()) if len(g) else 0)
    return float(np.mean(vals))


def oracle_score(df: pd.DataFrame, ids: np.ndarray) -> float:
    vals = []
    for cid in ids:
        vals.append(int(df[df.candidate_id == cid].score.max()))
    return float(np.mean(vals))


def main():
    df = load_rows()
    ids = np.array(sorted(df.candidate_id.unique()))
    adapters = sorted(df.adapter.unique())
    print(f"n_ids={len(ids)} n_rows={len(df)} n_adapters={len(adapters)} adapters={adapters}", flush=True)
    grid = []
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        for w_ctx in [0.0, 0.5, 1.0, 1.5]:
            for w_mag in [0.0, 0.5, 1.0]:
                for w_vote in [0.0, 0.5, 1.0, 1.5, 2.0]:
                    grid.append((alpha, 1.0, w_ctx, w_mag, w_vote, 0.25))
    records = []
    choices_rows = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        train = shuffled[: int(0.6 * n)]
        dev = shuffled[int(0.6 * n) : int(0.8 * n)]
        test = shuffled[int(0.8 * n) :]
        train_dev = np.concatenate([train, dev])
        train_df = df[df.candidate_id.isin(train)]
        dev_df = df[df.candidate_id.isin(dev)]
        # Tune smoothing/weights on dev.
        best = None
        for params in grid:
            alpha, *weights = params
            rel = fit_reliability(train_df, alpha)
            dev_choices = choose_for_ids(dev_df, dev, rel, weights)
            acc = score_choices(dev_df, dev_choices)
            if best is None or acc > best[0]:
                best = (acc, params)
        alpha, *weights = best[1]
        rel = fit_reliability(df[df.candidate_id.isin(train_dev)], alpha)
        test_df = df[df.candidate_id.isin(test)]
        choices = choose_for_ids(test_df, test, rel, weights)
        acc = score_choices(test_df, choices)
        rec = {
            "seed": seed,
            "bayes_fusion": acc,
            "self_check": fixed_score(df, test, "tta:self_check"),
            "axis": fixed_score(df, test, "tta:axis"),
            "direct": fixed_score(df, test, "tta:direct"),
            "pairwise": fixed_score(df, test, "pairwise_default_vs_alt"),
            "top4_context": fixed_score(df, test, "structured_context_letter"),
            "oracle": oracle_score(df, test),
            "alpha": alpha,
            "w_global": weights[0],
            "w_ctx": weights[1],
            "w_mag": weights[2],
            "w_vote": weights[3],
            "w_prior": weights[4],
            "dev_score": best[0],
        }
        records.append(rec)
        for cid, ans in choices.items():
            choices_rows.append({"seed": seed, "candidate_id": cid, "answer_key": ans})
        if seed % 5 == 0:
            print("seed", seed, rec, flush=True)

    res = pd.DataFrame(records)
    summary = {
        "n": int(len(ids)),
        "n_rows": int(len(df)),
        "bayes_mean": float(res.bayes_fusion.mean()),
        "bayes_std": float(res.bayes_fusion.std(ddof=1)),
        "selfcheck_mean": float(res.self_check.mean()),
        "axis_mean": float(res.axis.mean()),
        "direct_mean": float(res.direct.mean()),
        "pairwise_mean": float(res.pairwise.mean()),
        "top4_context_mean": float(res.top4_context.mean()),
        "oracle_mean": float(res.oracle.mean()),
        "best_params_counts": res[["alpha", "w_ctx", "w_mag", "w_vote"]].astype(str).agg("|".join, axis=1).value_counts().to_dict(),
    }
    res.to_csv(OUT / "scaleqa520_bayesian_policy_fusion_v1_results.csv", index=False)
    pd.DataFrame(choices_rows).to_csv(OUT / "scaleqa520_bayesian_policy_fusion_v1_choices.csv", index=False)
    (OUT / "scaleqa520_bayesian_policy_fusion_v1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
