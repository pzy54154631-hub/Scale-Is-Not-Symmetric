#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/scaleqa_natural/bayesian_regime_table_selector_v12")
OUT.mkdir(parents=True, exist_ok=True)

POOL = Path("results/scaleqa_natural/reranker_v2/scaleqa520_policy_candidate_reranker_v2_candidate_pool.csv")
OPTIONAL = [
    Path("results/eval/20260521_scaleqa520_structured_verifier_top4.csv"),
    Path("results/eval/20260521_scaleqa520_structured_verifier_ordered.csv"),
    Path("results/eval/20260521_scaleqa520_pairwise_verifier.csv"),
    Path("results/eval/20260521_scaleqa520_tta_v2_scale_regime.csv"),
    Path("results/eval/20260521_scaleqa520_regime_verifier_v3.csv"),
    Path("results/eval/20260521_scaleqa520_logprob_verifier_answer_tuned.csv"),
    Path("results/eval/20260521_scaleqa520_logprob_verifier_large_seed1.csv"),
    Path("results/eval/20260521_scaleqa520_logprob_verifier_base.csv"),
]

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def txt(x) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def parse_num(x) -> float:
    if pd.isna(x):
        return float("nan")
    m = NUM_RE.search(str(x).replace(",", "").replace("−", "-"))
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except ValueError:
        return float("nan")


def bucket(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    av = abs(v)
    if av == 0:
        return "zero"
    if av < 0.01:
        return "lt001"
    if av < 0.1:
        return "001_01"
    if av < 1:
        return "01_1"
    if av < 100:
        return "1_100"
    if av < 10000:
        return "100_1e4"
    return "ge1e4"


def safe_lratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12 or abs(b) < 1e-12:
        return 0.0
    return float(math.log10(abs(a / b)))


def approx_ratio_tag(a: float, b: float) -> str:
    r = safe_lratio(a, b)
    if abs(r - 2) < 0.18:
        return "x100"
    if abs(r + 2) < 0.18:
        return "div100"
    if abs(r - 1) < 0.18:
        return "x10"
    if abs(r + 1) < 0.18:
        return "div10"
    if abs(r) < 0.08:
        return "same"
    return "other"


def load_pool() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    score = pool.pivot_table(index="candidate_id", columns="adapter", values="score", aggfunc="first").fillna(0).astype(int)
    pred = pool.pivot_table(index="candidate_id", columns="adapter", values="prediction_norm", aggfunc="first")
    meta = pool.drop_duplicates("candidate_id").set_index("candidate_id")
    return meta, score, pred


def policy_list(score: pd.DataFrame) -> list[str]:
    preferred = [
        "tta:self_check", "tta:axis", "decimal_guard", "tta:direct", "pairwise_default_vs_alt",
        "main:answer_tuned", "main:large_seed1", "main:large_seed2", "main:large_seed3",
        "main:large_seed4", "main:large_seed5", "main:base", "tta:unit_guard", "tta:tta_majority",
        "percent_display", "regime_check", "axis_unit_audit",
        "regime_unit_first", "regime_default_safe", "regime_eliminate",
        "logprob_unit_first", "logprob_balanced_fewshot", "logprob_axis_ticks",
        "structured_context_letter", "structured_scale_audit",
        "structured_order_source_base_last", "structured_order_numeric_asc", "structured_order_numeric_desc",
    ]
    return [p for p in preferred if p in score.columns]


def observable_bins(meta: pd.DataFrame, pred: pd.DataFrame, policies: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for cid in score_ids(meta, pred):
        q = txt(meta.loc[cid].get("question", ""))
        ctx = txt(meta.loc[cid].get("chart_context", ""))
        s = q + " " + ctx
        vals = {p: parse_num(pred.loc[cid].get(p, "")) if p in pred.columns else float("nan") for p in policies}
        self_v = vals.get("tta:self_check", vals.get("main:answer_tuned", float("nan")))
        axis_v = vals.get("tta:axis", float("nan"))
        ans_v = vals.get("main:answer_tuned", vals.get("main:base", float("nan")))
        ratio_sa = approx_ratio_tag(self_v, axis_v)
        ratio_sans = approx_ratio_tag(self_v, ans_v)
        flags = {
            "pct": str(int("%" in s or "percent" in s or "percentage" in s)),
            "unit": str(int(any(w in s for w in ["million", "billion", "thousand", "kg", "km"]))),
            "op": "diff" if any(w in s for w in ["difference", "greater", "less", "more", "fewer"]) else ("avg" if any(w in s for w in ["average", "mean"]) else "read"),
            "ext": "max" if any(w in s for w in ["highest", "largest", "maximum"]) else ("min" if any(w in s for w in ["lowest", "smallest", "minimum"]) else "none"),
        }
        out[cid] = {
            "global": "all",
            "scale_basic": f"pct={flags['pct']}|unit={flags['unit']}|self={bucket(self_v)}",
            "op_scale": f"pct={flags['pct']}|op={flags['op']}|self={bucket(self_v)}",
            "axis_disagree": f"pct={flags['pct']}|sa={ratio_sa}|self={bucket(self_v)}|axis={bucket(axis_v)}",
            "answer_disagree": f"pct={flags['pct']}|sans={ratio_sans}|self={bucket(self_v)}|ans={bucket(ans_v)}",
            "full_coarse": f"pct={flags['pct']}|unit={flags['unit']}|op={flags['op']}|ext={flags['ext']}|self={bucket(self_v)}|sa={ratio_sa}",
        }
    return out


def score_ids(meta: pd.DataFrame, pred: pd.DataFrame) -> list[str]:
    return sorted(set(meta.index).intersection(set(pred.index)))


def fixed_score(score: pd.DataFrame, ids: np.ndarray, policy: str) -> float:
    return float(score.loc[ids, policy].mean())


def choice_score(score: pd.DataFrame, choices: dict[str, str]) -> float:
    return float(np.mean([score.loc[cid, p] for cid, p in choices.items()]))


def fit_tables(score: pd.DataFrame, train_ids: np.ndarray, policies: list[str], bins: dict[str, dict[str, str]], bin_name: str, alpha: float, min_count: int):
    global_mean = {p: float(score.loc[train_ids, p].mean()) for p in policies}
    counts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for cid in train_ids:
        b = bins[cid][bin_name]
        for p in policies:
            counts[b][p][0] += int(score.loc[cid, p])
            counts[b][p][1] += 1
    table = {}
    for b, d in counts.items():
        table[b] = {}
        for p in policies:
            succ, n = d[p]
            if n < min_count:
                table[b][p] = global_mean[p]
            else:
                table[b][p] = (succ + alpha * global_mean[p]) / (n + alpha)
    return global_mean, table


def choose_ids(ids: np.ndarray, policies: list[str], bins: dict[str, dict[str, str]], bin_name: str, global_mean, table, default: str, margin: float):
    choices = {}
    for cid in ids:
        b = bins[cid][bin_name]
        reliab = table.get(b, global_mean)
        best_p = max(policies, key=lambda p: reliab.get(p, global_mean[p]))
        best_v = reliab.get(best_p, global_mean[best_p])
        def_v = reliab.get(default, global_mean[default])
        choices[cid] = best_p if best_v >= def_v + margin else default
    return choices


def main():
    meta, score, pred = load_pool()
    policies = policy_list(score)
    ids = np.array(score_ids(meta, pred))
    default = "tta:self_check"
    bins = observable_bins(meta, pred, policies)
    bin_names = ["global", "scale_basic", "op_scale", "axis_disagree", "answer_disagree", "full_coarse"]
    alphas = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0]
    min_counts = [3, 5, 8, 12, 20]
    margins = [-0.02, 0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]

    records, choices_out = [], []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        shuffled = ids.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        train = shuffled[: int(0.6 * n)]
        dev = shuffled[int(0.6 * n): int(0.8 * n)]
        test = shuffled[int(0.8 * n):]
        train_dev = np.concatenate([train, dev])

        candidates = []
        for bin_name in bin_names:
            for alpha in alphas:
                for min_count in min_counts:
                    gm, tb = fit_tables(score, train, policies, bins, bin_name, alpha, min_count)
                    for margin in margins:
                        ch = choose_ids(dev, policies, bins, bin_name, gm, tb, default, margin)
                        candidates.append((choice_score(score, ch), bin_name, alpha, min_count, margin))
        best = max(candidates, key=lambda x: x[0])
        _, bin_name, alpha, min_count, margin = best
        gm, tb = fit_tables(score, train_dev, policies, bins, bin_name, alpha, min_count)
        choices = choose_ids(test, policies, bins, bin_name, gm, tb, default, margin)
        rec = {
            "seed": seed,
            "selector": choice_score(score, choices),
            "self_check": fixed_score(score, test, default),
            "axis": fixed_score(score, test, "tta:axis"),
            "best_fixed_split": float(score.loc[test, policies].mean().max()),
            "policy_oracle": float(score.loc[test, policies].max(axis=1).mean()),
            "selected_bin": bin_name,
            "alpha": alpha,
            "min_count": min_count,
            "margin": margin,
            "switch_rate": float(np.mean([choices[cid] != default for cid in test])),
        }
        records.append(rec)
        choices_out.extend({"seed": seed, "candidate_id": cid, "chosen_policy": p, "score": int(score.loc[cid, p])} for cid, p in choices.items())
        if seed % 10 == 0:
            print("seed", seed, rec, flush=True)

    df = pd.DataFrame(records)
    summary = {
        "n": int(len(ids)),
        "policies": policies,
        "selector_mean": float(df.selector.mean()),
        "selector_std": float(df.selector.std(ddof=1)),
        "selfcheck_mean": float(df.self_check.mean()),
        "axis_mean": float(df.axis.mean()),
        "best_fixed_split_mean": float(df.best_fixed_split.mean()),
        "policy_oracle_mean": float(df.policy_oracle.mean()),
        "switch_rate_mean": float(df.switch_rate.mean()),
        "selected_bin_counts": df.selected_bin.value_counts().to_dict(),
    }
    df.to_csv(OUT / "scaleqa520_bayesian_regime_table_selector_v12_results.csv", index=False)
    pd.DataFrame(choices_out).to_csv(OUT / "scaleqa520_bayesian_regime_table_selector_v12_choices.csv", index=False)
    (OUT / "scaleqa520_bayesian_regime_table_selector_v12_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
