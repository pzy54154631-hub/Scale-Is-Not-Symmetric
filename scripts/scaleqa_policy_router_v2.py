#!/usr/bin/env python
"""Multiclass deployable policy router for ScaleQA-Natural."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import MaxAbsScaler

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", re.I)


def norm(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).lower()


def nums(x):
    out = []
    for m in NUM_RE.findall(str(x).replace(",", "").replace("−", "-")):
        try:
            out.append(float(m))
        except Exception:
            pass
    return out


def num_bucket(x):
    xs = nums(x)
    if not xs:
        return "no_num"
    v = abs(xs[0])
    if v < 0.01:
        return "lt_0.01"
    if v < 0.1:
        return "0.01_0.1"
    if v < 1:
        return "0.1_1"
    if v < 10:
        return "1_10"
    if v < 100:
        return "10_100"
    if v < 1000:
        return "100_1000"
    return "gte_1000"


def qtype(q):
    s = norm(q)
    if any(w in s for w in ["average", "mean"]):
        return "average"
    if any(w in s for w in ["difference", "more than", "less than"]):
        return "difference"
    if any(w in s for w in ["ratio", "times", "a:b"]):
        return "ratio"
    if any(w in s for w in ["percent", "percentage", "%"]):
        return "percent"
    if any(w in s for w in ["highest", "largest", "maximum", "greatest"]):
        return "max"
    if any(w in s for w in ["lowest", "smallest", "minimum", "least"]):
        return "min"
    if any(w in s for w in ["sum", "total"]):
        return "sum"
    return "lookup"


def unit_class(ctx, q):
    s = norm(ctx) + " " + norm(q)
    if "%" in s or "percent" in s or "percentage" in s:
        return "percent"
    if any(w in s for w in ["million", "billion", "thousand"]):
        return "large_unit"
    if any(w in s for w in ["ratio", "rate", "per "]):
        return "ratio_rate"
    if any(w in s for w in ["$", "usd", "dollar", "euro"]):
        return "currency"
    return "none"


def read_policy_csv(spec):
    if "::" in spec:
        prefix, path = spec.split("::", 1)
    else:
        prefix, path = "", spec
    df = pd.read_csv(path)
    df = df[~df["adapter"].astype(str).str.contains("oracle", case=False, na=False)].copy()
    df["policy"] = (prefix + ":" + df["adapter"].astype(str)) if prefix else df["adapter"].astype(str)
    for c in ["candidate_id", "question", "answers", "prediction", "score", "scale_error_subtype", "scale_directions", "source_split", "source_row_idx"]:
        if c not in df.columns:
            df[c] = ""
    return df[["candidate_id", "policy", "question", "answers", "prediction", "score", "scale_error_subtype", "scale_directions", "source_split", "source_row_idx"]]


def build_table(policy_files, chart_context_csv, manifest_csv):
    df = pd.concat([read_policy_csv(x) for x in policy_files], ignore_index=True)
    # One score per policy/example; if duplicate predictions under same policy, keep last.
    df = df.drop_duplicates(["candidate_id", "policy"], keep="last")
    ctx = pd.read_csv(chart_context_csv)
    df = df.merge(ctx[["candidate_id", "chart_context"]], on="candidate_id", how="left")
    if manifest_csv:
        man = pd.read_csv(manifest_csv)
        keep = [c for c in ["candidate_id", "source_split", "source_row_idx", "scale_error_subtype", "scale_directions"] if c in man.columns]
        df = df.merge(man[keep], on="candidate_id", how="left", suffixes=("", "_m"))
        for c in ["source_split", "source_row_idx", "scale_error_subtype", "scale_directions"]:
            cm = c + "_m"
            if cm in df.columns:
                df[c] = df[c].where(df[c].astype(str).str.len() > 0, df[cm])
    df["chart_group"] = df["source_split"].fillna("").astype(str) + "::" + df["source_row_idx"].fillna("").astype(str)
    df.loc[df["chart_group"].isin(["::", "nan::nan"]), "chart_group"] = df["candidate_id"]
    return df


def instance_features(df):
    inst = df.sort_values("policy").drop_duplicates("candidate_id").copy()
    rows = []
    for r in inst.itertuples(index=False):
        q = r.question
        ctx = r.chart_context
        # Use answer-tuned/base predictions as available context without gold subtype.
        group = df[df["candidate_id"].eq(r.candidate_id)]
        pred_map = dict(zip(group["policy"], group["prediction"]))
        answer_pred = pred_map.get("main:answer_tuned", pred_map.get("tta:direct", ""))
        feat_text = (
            f"question={q} chart_context={ctx} answer_prediction={answer_pred} "
            f"qtype={qtype(q)} unit={unit_class(ctx, q)} pred_bucket={num_bucket(answer_pred)}"
        )
        d = {
            "qtype=" + qtype(q): 1.0,
            "unit=" + unit_class(ctx, q): 1.0,
            "pred_bucket=" + num_bucket(answer_pred): 1.0,
            "question_has_percent": float("%" in norm(q) or "percent" in norm(q)),
            "ctx_has_percent": float("%" in norm(ctx) or "percent" in norm(ctx)),
            "pred_is_decimal": float("." in str(answer_pred)),
            "pred_lt_one": float((nums(answer_pred) or [999])[0] < 1),
        }
        rows.append({
            "candidate_id": r.candidate_id,
            "chart_group": r.chart_group,
            "scale_error_subtype": r.scale_error_subtype,
            "text": feat_text,
            "dict": d,
        })
    return pd.DataFrame(rows)


def split(inst, seed, test_size, dev_size):
    ids = inst["candidate_id"].to_numpy()
    groups = inst["chart_group"].to_numpy()
    strata = inst["scale_error_subtype"].fillna("unknown").to_numpy()
    try:
        g = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        td_idx, te_idx = next(g.split(ids, strata, groups))
    except Exception:
        s = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        td_idx, te_idx = next(s.split(ids, strata))
    td = inst.iloc[td_idx].reset_index(drop=True)
    te_ids = set(inst.iloc[te_idx]["candidate_id"])
    try:
        g2 = GroupShuffleSplit(n_splits=1, test_size=dev_size / (1 - test_size), random_state=seed + 111)
        tr_rel, dv_rel = next(g2.split(td["candidate_id"], td["scale_error_subtype"], td["chart_group"]))
    except Exception:
        s2 = StratifiedShuffleSplit(n_splits=1, test_size=dev_size / (1 - test_size), random_state=seed + 111)
        tr_rel, dv_rel = next(s2.split(td["candidate_id"], td["scale_error_subtype"].fillna("unknown")))
    return set(td.iloc[tr_rel]["candidate_id"]), set(td.iloc[dv_rel]["candidate_id"]), te_ids


def policy_scores(df, ids):
    return df[df["candidate_id"].isin(ids)].groupby("policy")["score"].mean().to_dict()


def choose_score(df, choices):
    vals = []
    for cid, pol in choices.items():
        sub = df[(df["candidate_id"].eq(cid)) & (df["policy"].eq(pol))]
        if len(sub):
            vals.append(float(sub["score"].iloc[0]))
    return float(np.mean(vals)) if vals else float("nan")


def labels_for(df, ids, priors):
    sub = df[df["candidate_id"].isin(ids)].copy()
    labels = {}
    for cid, g in sub.groupby("candidate_id"):
        best = g[g["score"].eq(g["score"].max())].copy()
        best["prior"] = best["policy"].map(priors).fillna(0)
        labels[cid] = best.sort_values(["score", "prior"], ascending=[False, False])["policy"].iloc[0]
    return labels


def fit_model(train, dev, y_train, y_dev, model_name):
    if model_name == "logreg_text":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=16000)
        xtr = vec.fit_transform(train["text"])
        xdv = vec.transform(dev["text"])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="liblinear")
    else:
        dv = DictVectorizer()
        xtr = dv.fit_transform(train["dict"])
        xdv = dv.transform(dev["dict"])
        scaler = MaxAbsScaler()
        xtr = scaler.fit_transform(xtr)
        xdv = scaler.transform(xdv)
        if model_name == "extra_trees":
            clf = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=7, n_jobs=-1)
        elif model_name == "random_forest":
            clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=7, n_jobs=-1)
        else:
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="liblinear")
    clf.fit(xtr, y_train)
    return vec if model_name == "logreg_text" else (dv, scaler), clf


def transform(model_name, prep, frame):
    if model_name == "logreg_text":
        return prep.transform(frame["text"])
    dv, scaler = prep
    return scaler.transform(dv.transform(frame["dict"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-file", action="append", required=True)
    ap.add_argument("--chart-context-csv", required=True)
    ap.add_argument("--manifest-csv", default="")
    ap.add_argument("--out-dir", default="results/scaleqa_natural/policy_router_v2")
    ap.add_argument("--name", default="scaleqa520_policy_router_v2")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--dev-size", type=float, default=0.2)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_table(args.policy_file, args.chart_context_csv, args.manifest_csv)
    inst = instance_features(df)
    records = []
    choice_frames = []
    model_names = ["logreg_text", "logreg_num", "extra_trees", "random_forest"]
    for seed in range(args.seeds):
        tr_ids, dv_ids, te_ids = split(inst, seed, args.test_size, args.dev_size)
        train_scores = policy_scores(df, tr_ids)
        priors = train_scores
        y_tr_map = labels_for(df, tr_ids, priors)
        y_dv_map = labels_for(df, dv_ids, priors)
        tr = inst[inst["candidate_id"].isin(tr_ids)].copy()
        dv = inst[inst["candidate_id"].isin(dv_ids)].copy()
        te = inst[inst["candidate_id"].isin(te_ids)].copy()
        y_tr = tr["candidate_id"].map(y_tr_map)
        y_dv = dv["candidate_id"].map(y_dv_map)
        best_dev = -1
        best = None
        for mn in model_names:
            try:
                prep, clf = fit_model(tr, dv, y_tr, y_dv, mn)
                pred_dv = clf.predict(transform(mn, prep, dv))
                dev_choices = dict(zip(dv["candidate_id"], pred_dv))
                dev_acc = choose_score(df, dev_choices)
                if dev_acc > best_dev:
                    best_dev = dev_acc
                    best = (mn, prep, clf)
            except Exception:
                continue
        mn, prep, clf = best
        pred_te = clf.predict(transform(mn, prep, te))
        learned_choices = dict(zip(te["candidate_id"], pred_te))
        learned_acc = choose_score(df, learned_choices)
        fixed = policy_scores(df, te_ids)
        global_best = max(train_scores, key=train_scores.get)
        global_acc = fixed.get(global_best, float("nan"))
        oracle = float(df[df["candidate_id"].isin(te_ids)].groupby("candidate_id")["score"].max().mean())
        row = {
            "seed": seed,
            "best_model": mn,
            "global_policy": global_best,
            "global_train_selected": global_acc,
            "learned_policy_router": learned_acc,
            "oracle_policy": oracle,
            "dev_selected_acc": best_dev,
            "n_test": len(te_ids),
        }
        row.update({f"fixed_{k}": v for k, v in fixed.items()})
        records.append(row)
        ch = pd.DataFrame({"seed": seed, "candidate_id": list(learned_choices.keys()), "chosen_policy": list(learned_choices.values())})
        choice_frames.append(ch)
    res = pd.DataFrame(records)
    choices = pd.concat(choice_frames, ignore_index=True)
    res_path = out / f"{args.name}_results.csv"
    ch_path = out / f"{args.name}_choices.csv"
    res.to_csv(res_path, index=False)
    choices.to_csv(ch_path, index=False)
    metric_cols = [c for c in res.columns if c.startswith("fixed_")] + ["global_train_selected", "learned_policy_router", "oracle_policy"]
    summary = {
        "name": args.name,
        "n": int(inst["candidate_id"].nunique()),
        "n_policies": int(df["policy"].nunique()),
        "policies": sorted(df["policy"].unique().tolist()),
        "metrics": {c: {"mean": float(res[c].mean()), "std": float(res[c].std(ddof=1)), "min": float(res[c].min()), "max": float(res[c].max())} for c in metric_cols if c in res},
        "chosen_model_counts": res["best_model"].value_counts().to_dict(),
        "files": {"results": str(res_path), "choices": str(ch_path)},
    }
    (out / f"{args.name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Policy router: {args.name}", "", f"N={summary['n']}, policies={summary['n_policies']}", "", "| Metric | Mean | Std | Min | Max |", "|---|---:|---:|---:|---:|"]
    for k, m in summary["metrics"].items():
        lines.append(f"| {k} | {m['mean']:.4f} | {m['std']:.4f} | {m['min']:.4f} | {m['max']:.4f} |")
    (out / f"{args.name}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
