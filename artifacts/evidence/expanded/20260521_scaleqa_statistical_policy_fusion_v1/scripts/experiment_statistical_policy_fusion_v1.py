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

OUT = Path(os.environ.get("MIRAGE_STAT_OUT", "results/scaleqa_natural/statistical_policy_fusion_v1"))
OUT.mkdir(parents=True, exist_ok=True)

POOL = Path("results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
OPTIONAL = [
    Path("results/eval/20260521_scaleqa520_structured_verifier_top4.csv"),
    Path("results/eval/20260521_scaleqa520_structured_verifier_ordered.csv"),
    Path("results/eval/20260521_scaleqa520_pairwise_verifier.csv"),
    Path("results/eval/20260521_scaleqa520_tta_v2_scale_regime.csv"),
]

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def txt(x: object) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def parse_num(x: object) -> float:
    m = NUM_RE.search(str(x).replace(",", "").replace("−", "-"))
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except Exception:
        return float("nan")


def norm_answer(x: object) -> str:
    s = txt(x).replace(",", "").replace("−", "-")
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


def logit(p: float) -> float:
    p = min(max(float(p), 1e-5), 1 - 1e-5)
    return math.log(p / (1 - p))


def beta_stats(pos: int, total: int, alpha: float, prior: float = 0.5) -> tuple[float, float]:
    # Beta posterior with symmetric strength alpha centered at prior.
    a = pos + alpha * prior * 2
    b = (total - pos) + alpha * (1 - prior) * 2
    mean = a / (a + b)
    var = a * b / (((a + b) ** 2) * (a + b + 1))
    return float(mean), float(math.sqrt(max(var, 0.0)))


def wilson_lcb(pos: int, total: int, z: float = 1.0, prior: float = 0.5, strength: float = 2.0) -> float:
    # Smoothed Wilson lower confidence bound; useful for conservative policy switching.
    n = total + strength
    phat = (pos + prior * strength) / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    rad = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return float((center - rad) / denom)


def bucket_num(v: float) -> str:
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
    if av < 10:
        return "1_10"
    if av < 100:
        return "10_100"
    if av < 10000:
        return "100_1e4"
    return "ge1e4"


def ratio_bucket(a: float, b: float) -> str:
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12 or abs(b) < 1e-12:
        return "na"
    r = math.log10(abs(a / b))
    if r < -2.5:
        return "lt_m2"
    if r < -1.5:
        return "m2"
    if r < -0.5:
        return "m1"
    if r < 0.5:
        return "same"
    if r < 1.5:
        return "p1"
    if r < 2.5:
        return "p2"
    return "gt_p2"


def context_bucket(question: object, context: object) -> str:
    q = txt(question)
    c = txt(context)
    s = q + " " + c
    pct = "pct" if ("%" in s or "percent" in s or "percentage" in s) else "nopct"
    unit = "unit" if any(w in s for w in ["million", "billion", "thousand", "axis", "ticks=", "legend=", "units="]) else "nounit"
    if any(w in q for w in ["difference", "greater", "less", "more", "fewer"]):
        op = "compare"
    elif any(w in q for w in ["average", "mean"]):
        op = "avg"
    elif any(w in q for w in ["highest", "largest", "maximum", "lowest", "smallest", "minimum"]):
        op = "extreme"
    else:
        op = "lookup"
    return f"{pct}|{unit}|{op}"


def load_rows() -> pd.DataFrame:
    pool = pd.read_csv(POOL)
    base_meta = pool.drop_duplicates("candidate_id").set_index("candidate_id")
    rows = []
    for _, r in pool.iterrows():
        rows.append(
            {
                "candidate_id": str(r["candidate_id"]),
                "adapter": str(r["adapter"]),
                "question": r.get("question", ""),
                "answers": r.get("answers", ""),
                "prediction": r.get("prediction_norm", r.get("prediction", "")),
                "score": int(r.get("score", 0)),
                "chart_context": r.get("chart_context", ""),
            }
        )
    for path in OPTIONAL:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "candidate_id" not in df.columns or "adapter" not in df.columns or "prediction" not in df.columns:
            continue
        for _, r in df.iterrows():
            adapter = str(r.get("adapter", path.stem))
            if "oracle" in adapter:
                continue
            cid = str(r["candidate_id"])
            if cid not in base_meta.index:
                continue
            pred = r.get("prediction", "")
            if pd.isna(pred) or str(pred).strip() == "":
                continue
            m = base_meta.loc[cid]
            rows.append(
                {
                    "candidate_id": cid,
                    "adapter": adapter,
                    "question": r.get("question", m.get("question", "")),
                    "answers": r.get("answers", m.get("answers", "")),
                    "prediction": pred,
                    "score": int(r.get("score", 0)),
                    "chart_context": m.get("chart_context", ""),
                }
            )
    data = pd.DataFrame(rows)
    data["answer_key"] = data["prediction"].map(norm_answer)
    data["num"] = data["prediction"].map(parse_num)
    data["mag"] = data["num"].map(bucket_num)
    data["ctx"] = [context_bucket(q, c) for q, c in zip(data["question"], data["chart_context"])]
    return data


def make_item_features(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in data.groupby("candidate_id"):
        num_by = dict(zip(g.adapter, g.num))
        key_votes = g.answer_key.value_counts().to_dict()
        for _, r in g.iterrows():
            n = r.num
            rows.append(
                {
                    "candidate_id": cid,
                    "adapter": r.adapter,
                    "score": int(r.score),
                    "answer_key": r.answer_key,
                    "ctx": r.ctx,
                    "mag": r.mag,
                    "vote": str(min(int(key_votes.get(r.answer_key, 0)), 5)),
                    "ratio_self": ratio_bucket(n, num_by.get("tta:self_check", float("nan"))),
                    "ratio_axis": ratio_bucket(n, num_by.get("tta:axis", float("nan"))),
                    "ratio_direct": ratio_bucket(n, num_by.get("tta:direct", float("nan"))),
                    "ratio_answer": ratio_bucket(n, num_by.get("main:answer_tuned", float("nan"))),
                    "src": (
                        "tta"
                        if str(r.adapter).startswith("tta:")
                        else "main"
                        if str(r.adapter).startswith("main:")
                        else "verifier"
                        if str(r.adapter).startswith("verifier:")
                        else "structured"
                    ),
                }
            )
    return pd.DataFrame(rows)


def fit_tables(train: pd.DataFrame, alpha: float, z: float) -> dict:
    prior = float(train.score.mean())
    tables = {}
    specs = {
        "adapter": ["adapter"],
        "adapter_ctx": ["adapter", "ctx"],
        "adapter_mag": ["adapter", "mag"],
        "adapter_vote": ["adapter", "vote"],
        "adapter_ratio_self": ["adapter", "ratio_self"],
        "adapter_ratio_axis": ["adapter", "ratio_axis"],
        "src_ctx": ["src", "ctx"],
    }
    for name, cols in specs.items():
        tab = {}
        for key, g in train.groupby(cols):
            if not isinstance(key, tuple):
                key = (key,)
            pos, total = int(g.score.sum()), len(g)
            mean, sd = beta_stats(pos, total, alpha, prior)
            tab[key] = {
                "mean": mean,
                "sd": sd,
                "lcb": wilson_lcb(pos, total, z=z, prior=prior, strength=2 * alpha),
                "n": total,
            }
        tables[name] = tab
    tables["prior"] = {"mean": prior, "sd": math.sqrt(max(prior * (1 - prior) / max(len(train), 1), 0.0)), "lcb": prior, "n": len(train)}
    return tables


def lookup(tables: dict, name: str, key: tuple, field: str) -> float:
    prior = tables["prior"][field]
    return tables.get(name, {}).get(tuple(key), {}).get(field, prior)


def row_score(row: pd.Series, tables: dict, params: tuple) -> float:
    metric, w_ctx, w_mag, w_vote, w_rself, w_raxis, w_src, w_agree = params
    field = metric
    # Product-of-experts in logit space. LCB metric is conservative; mean is empirical Bayes.
    parts = [
        (1.0, lookup(tables, "adapter", (row.adapter,), field)),
        (w_ctx, lookup(tables, "adapter_ctx", (row.adapter, row.ctx), field)),
        (w_mag, lookup(tables, "adapter_mag", (row.adapter, row.mag), field)),
        (w_vote, lookup(tables, "adapter_vote", (row.adapter, row.vote), field)),
        (w_rself, lookup(tables, "adapter_ratio_self", (row.adapter, row.ratio_self), field)),
        (w_raxis, lookup(tables, "adapter_ratio_axis", (row.adapter, row.ratio_axis), field)),
        (w_src, lookup(tables, "src_ctx", (row.src, row.ctx), field)),
    ]
    score = 0.0
    for w, p in parts:
        if w:
            score += float(w) * logit(p)
    score += float(w_agree) * math.log1p(int(row.vote))
    return score


def choose_policy(feat: pd.DataFrame, ids: np.ndarray, tables: dict, params: tuple) -> dict[str, str]:
    sub = feat[feat.candidate_id.isin(ids)].copy()
    sub["_s"] = [row_score(r, tables, params) for _, r in sub.iterrows()]
    choices = {}
    for cid, g in sub.groupby("candidate_id"):
        choices[cid] = str(g.sort_values("_s", ascending=False).iloc[0].adapter)
    return choices


def choose_answer(feat: pd.DataFrame, ids: np.ndarray, tables: dict, params: tuple) -> dict[str, str]:
    sub = feat[feat.candidate_id.isin(ids)].copy()
    sub["_s"] = [row_score(r, tables, params) for _, r in sub.iterrows()]
    choices = {}
    for cid, g in sub.groupby("candidate_id"):
        agg = g.groupby("answer_key")["_s"].sum()
        choices[cid] = str(agg.sort_values(ascending=False).index[0])
    return choices


def score_policy_choices(score: pd.DataFrame, choices: dict[str, str]) -> float:
    vals = []
    for cid, pol in choices.items():
        vals.append(int(score.loc[cid, pol]) if pol in score.columns else 0)
    return float(np.mean(vals))


def score_answer_choices(feat: pd.DataFrame, choices: dict[str, str]) -> float:
    vals = []
    by = feat.groupby(["candidate_id", "answer_key"]).score.max()
    for cid, key in choices.items():
        vals.append(int(by.get((cid, key), 0)))
    return float(np.mean(vals))


def fixed_score(score: pd.DataFrame, ids: np.ndarray, policy: str) -> float:
    if policy not in score.columns:
        return float("nan")
    return float(score.loc[ids, policy].fillna(0).mean())


def oracle_score(score: pd.DataFrame, ids: np.ndarray) -> float:
    return float(score.loc[ids].fillna(0).max(axis=1).mean())


def main() -> None:
    data = load_rows()
    feat = make_item_features(data)
    score = feat.pivot_table(index="candidate_id", columns="adapter", values="score", aggfunc="max").fillna(0).astype(int)
    ids = np.array(sorted(score.index))
    grid = []
    # Keep the search deliberately small: this is a no-leak statistical selector,
    # not a high-capacity model search. The grid covers empirical-Bayes posterior
    # means and conservative Wilson lower bounds with a few interpretable factors.
    for metric in ["mean", "lcb"]:
        for alpha in [0.5, 2.0]:
            for z in [1.0]:
                for w_ctx in [0.0, 1.0]:
                    for w_mag in [0.0, 0.5]:
                        for w_vote in [0.0, 1.0]:
                            for w_rself in [0.0, 0.5]:
                                for w_raxis in [0.0, 0.5]:
                                    for w_src in [0.0, 0.5]:
                                        for w_agree in [0.0, 0.25, 0.5]:
                                            grid.append((metric, alpha, z, (metric, w_ctx, w_mag, w_vote, w_rself, w_raxis, w_src, w_agree)))
    records = []
    choice_rows = []
    for seed in range(50):
        rng = np.random.default_rng(seed)
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_ids = shuffled[: int(0.6 * n)]
        dev_ids = shuffled[int(0.6 * n) : int(0.8 * n)]
        test_ids = shuffled[int(0.8 * n) :]
        train = feat[feat.candidate_id.isin(train_ids)]
        best = None
        for metric, alpha, z, params in grid:
            tables = fit_tables(train, alpha, z)
            pol = choose_policy(feat, dev_ids, tables, params)
            pol_score = score_policy_choices(score, pol)
            ans = choose_answer(feat, dev_ids, tables, params)
            ans_score = score_answer_choices(feat, ans)
            mode, val = ("policy", pol_score) if pol_score >= ans_score else ("answer", ans_score)
            if best is None or val > best[0]:
                best = (val, mode, metric, alpha, z, params)
        assert best is not None
        _, mode, metric, alpha, z, params = best
        fit_ids = np.concatenate([train_ids, dev_ids])
        tables = fit_tables(feat[feat.candidate_id.isin(fit_ids)], alpha, z)
        if mode == "policy":
            choices = choose_policy(feat, test_ids, tables, params)
            stat_score = score_policy_choices(score, choices)
            for cid, pol in choices.items():
                choice_rows.append({"seed": seed, "candidate_id": cid, "choice_type": "policy", "choice": pol})
        else:
            choices = choose_answer(feat, test_ids, tables, params)
            stat_score = score_answer_choices(feat, choices)
            for cid, key in choices.items():
                choice_rows.append({"seed": seed, "candidate_id": cid, "choice_type": "answer", "choice": key})
        rec = {
            "seed": seed,
            "stat_fusion": stat_score,
            "selected_mode": mode,
            "selected_metric": metric,
            "alpha": alpha,
            "z": z,
            "dev_score": best[0],
            "self_check": fixed_score(score, test_ids, "tta:self_check"),
            "axis": fixed_score(score, test_ids, "tta:axis"),
            "direct": fixed_score(score, test_ids, "tta:direct"),
            "pairwise": fixed_score(score, test_ids, "pairwise_default_vs_alt"),
            "decimal_guard": fixed_score(score, test_ids, "decimal_guard"),
            "oracle": oracle_score(score, test_ids),
        }
        records.append(rec)
        print("seed", seed, rec, flush=True)
    res = pd.DataFrame(records)
    res.to_csv(OUT / "scaleqa520_statistical_policy_fusion_v1_splits.csv", index=False)
    pd.DataFrame(choice_rows).to_csv(OUT / "scaleqa520_statistical_policy_fusion_v1_choices.csv", index=False)
    summary = {
        "n": int(len(ids)),
        "stat_mean": float(res.stat_fusion.mean()),
        "stat_std": float(res.stat_fusion.std(ddof=0)),
        "selfcheck_mean": float(res.self_check.mean()),
        "axis_mean": float(res.axis.mean()),
        "direct_mean": float(res.direct.mean()),
        "pairwise_mean": float(res.pairwise.mean()),
        "decimal_guard_mean": float(res.decimal_guard.mean()),
        "oracle_mean": float(res.oracle.mean()),
        "selected_modes": {str(k): int(v) for k, v in Counter(res.selected_mode).items()},
        "selected_metrics": {str(k): int(v) for k, v in Counter(res.selected_metric).items()},
        "alpha_counts": {str(k): int(v) for k, v in Counter(res.alpha).items()},
        "z_counts": {str(k): int(v) for k, v in Counter(res.z).items()},
    }
    (OUT / "scaleqa520_statistical_policy_fusion_v1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
