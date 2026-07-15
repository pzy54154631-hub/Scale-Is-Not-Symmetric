#!/usr/bin/env python
"""Deployable-style candidate reranker for ScaleQA-Natural.

The reranker is intentionally lightweight: it receives candidate answers from
existing prompting/adaptation systems plus chart-side context strings extracted
from the image.  It never uses gold subtype labels as input.  We evaluate with
grouped held-out splits so candidates from the same source chart stay together.
"""

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", re.I)


def norm_text(x: object) -> str:
    s = "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
    s = s.lower().replace("−", "-").replace(",", "")
    s = s.replace("%", " percent ")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;\"'[](){}")


def nums(x: object) -> list[float]:
    out = []
    for m in NUM_RE.findall("" if x is None else str(x).replace(",", "").replace("−", "-")):
        try:
            out.append(float(m))
        except Exception:
            pass
    return out


def first_num(x: object) -> float | None:
    xs = nums(x)
    return xs[0] if xs else None


def to_list(x: object) -> list[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    s = str(x).strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            y = fn(s)
            if isinstance(y, list):
                return [str(v) for v in y]
            return [str(y)]
        except Exception:
            pass
    return [s]


def num_dict(prefix: str, value: object) -> dict[str, float]:
    text = "" if value is None else str(value)
    n = first_num(text)
    if n is None:
        return {
            f"{prefix}_has_num": 0.0,
            f"{prefix}_is_decimal": 0.0,
            f"{prefix}_lt_one": 0.0,
            f"{prefix}_log_abs": 0.0,
            f"{prefix}_digits": 0.0,
            f"{prefix}_dec_places": 0.0,
            f"{prefix}_has_percent": float("%" in text),
        }
    raw = NUM_RE.findall(text.replace(",", "").replace("−", "-"))[0]
    dec_places = 0
    if "." in raw:
        dec_places = len(raw.split(".", 1)[1].split("e", 1)[0].split("E", 1)[0])
    return {
        f"{prefix}_has_num": 1.0,
        f"{prefix}_is_decimal": float("." in raw),
        f"{prefix}_lt_one": float(abs(n) < 1.0),
        f"{prefix}_log_abs": math.log10(abs(n) + 1e-12),
        f"{prefix}_digits": float(sum(ch.isdigit() for ch in raw)),
        f"{prefix}_dec_places": float(dec_places),
        f"{prefix}_has_percent": float("%" in text),
    }


def appears_in_context(value: object, context: object) -> dict[str, float]:
    text = "" if value is None else str(value)
    ctx = "" if context is None else str(context)
    v = first_num(text)
    ctx_norm = ctx.replace(",", "")
    feats = {
        "cand_text_in_context": float(norm_text(text) in norm_text(ctx) and bool(norm_text(text))),
        "context_has_percent": float("%" in ctx or "percent" in ctx.lower() or "percentage" in ctx.lower()),
        "context_has_million": float(any(w in ctx.lower() for w in ["million", "millions", "m "])),
        "context_has_billion": float(any(w in ctx.lower() for w in ["billion", "billions", "bn"])),
        "context_has_thousand": float(any(w in ctx.lower() for w in ["thousand", "thousands", "k "])),
    }
    if v is None:
        feats.update({
            "cand_num_in_context": 0.0,
            "cand_x100_in_context": 0.0,
            "cand_div100_in_context": 0.0,
        })
        return feats
    candidates = [v, v * 100.0, v / 100.0]
    names = ["cand_num_in_context", "cand_x100_in_context", "cand_div100_in_context"]
    ctx_nums = nums(ctx_norm)
    for name, c in zip(names, candidates):
        ok = any(abs(c - z) <= max(1e-4, 0.01 * max(abs(c), 1.0)) for z in ctx_nums)
        feats[name] = float(ok)
    return feats


def question_type(q: object) -> str:
    s = norm_text(q)
    if any(w in s for w in ["average", "mean"]):
        return "average"
    if any(w in s for w in ["difference", "more than", "less than"]):
        return "difference"
    if any(w in s for w in ["ratio", "times", "divide"]):
        return "ratio"
    if any(w in s for w in ["percent", "percentage", "%"]):
        return "percent"
    if any(w in s for w in ["highest", "largest", "maximum", "max"]):
        return "max"
    if any(w in s for w in ["lowest", "smallest", "minimum", "min"]):
        return "min"
    if any(w in s for w in ["sum", "total"]):
        return "sum"
    return "lookup"


def read_policy_csv(path: str, name_prefix: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"candidate_id", "adapter", "question", "answers", "prediction", "score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {missing}")
    out = df.copy()
    if name_prefix:
        out["adapter"] = name_prefix + ":" + out["adapter"].astype(str)
    keep = [
        "candidate_id", "adapter", "question", "answers", "prediction", "score",
        "idx", "scale_error_subtype", "scale_directions", "source_split", "source_row_idx",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = ""
    return out[keep]


def build_candidates(policy_files: list[str], chart_context_csv: str, manifest_csv: str | None) -> pd.DataFrame:
    frames = []
    for spec in policy_files:
        if "::" in spec:
            prefix, path = spec.split("::", 1)
        else:
            prefix, path = None, spec
        frames.append(read_policy_csv(path, prefix))
    df = pd.concat(frames, ignore_index=True)
    # Remove oracle rows and unusable empty predictions.
    df = df[~df["adapter"].str.contains("oracle", case=False, na=False)].copy()
    df["prediction_norm"] = df["prediction"].map(norm_text)
    df = df[df["prediction_norm"].astype(bool)].copy()
    df = df.drop_duplicates(["candidate_id", "adapter", "prediction_norm"], keep="last")

    ctx = pd.read_csv(chart_context_csv)
    df = df.merge(ctx[["candidate_id", "chart_context"]], on="candidate_id", how="left")
    if manifest_csv:
        man = pd.read_csv(manifest_csv)
        man_keep = [c for c in ["candidate_id", "source_split", "source_row_idx", "scale_error_subtype", "scale_directions"] if c in man.columns]
        df = df.merge(man[man_keep], on="candidate_id", how="left", suffixes=("", "_manifest"))
        for c in ["source_split", "source_row_idx", "scale_error_subtype", "scale_directions"]:
            alt = c + "_manifest"
            if alt in df.columns:
                df[c] = df[c].where(df[c].astype(str).str.len() > 0, df[alt])
    # Stable chart group for leakage-aware splits.
    df["chart_group"] = (
        df["source_split"].fillna("").astype(str)
        + "::"
        + df["source_row_idx"].fillna("").astype(str)
    )
    bad = df["chart_group"].isin(["::", "nan::nan"])
    df.loc[bad, "chart_group"] = df.loc[bad, "candidate_id"]
    return df


def instance_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate_id", "question", "answers", "scale_error_subtype", "scale_directions", "chart_group", "chart_context"]
    inst = df.sort_values("adapter").drop_duplicates("candidate_id")[cols].copy()
    return inst


def split_instances(inst: pd.DataFrame, seed: int, test_size: float, dev_size: float):
    groups = inst["chart_group"].to_numpy()
    ids = inst["candidate_id"].to_numpy()
    strata = inst["scale_error_subtype"].fillna("unknown").to_numpy()
    # Group split first; fall back to stratified ID split if groups are too coarse.
    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_dev_idx, test_idx = next(gss.split(ids, strata, groups))
    except Exception:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_dev_idx, test_idx = next(sss.split(ids, strata))
    train_dev = inst.iloc[train_dev_idx].reset_index(drop=True)
    test_ids = set(inst.iloc[test_idx]["candidate_id"])
    dev_frac = dev_size / max(1e-6, 1.0 - test_size)
    try:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=dev_frac, random_state=seed + 1000)
        tr_rel, dev_rel = next(gss2.split(train_dev["candidate_id"], train_dev["scale_error_subtype"], train_dev["chart_group"]))
    except Exception:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=dev_frac, random_state=seed + 1000)
        tr_rel, dev_rel = next(sss2.split(train_dev["candidate_id"], train_dev["scale_error_subtype"].fillna("unknown")))
    return set(train_dev.iloc[tr_rel]["candidate_id"]), set(train_dev.iloc[dev_rel]["candidate_id"]), test_ids


def make_text(row) -> str:
    q = "" if pd.isna(row.question) else str(row.question)
    pred = "" if pd.isna(row.prediction) else str(row.prediction)
    ctx = "" if pd.isna(row.chart_context) else str(row.chart_context)
    return (
        f"source={row.adapter} qtype={question_type(q)} question={q} "
        f"candidate={pred} chart_context={ctx}"
    )


def make_dict(row) -> dict[str, float | str]:
    q = "" if pd.isna(row.question) else str(row.question)
    pred = "" if pd.isna(row.prediction) else str(row.prediction)
    ctx = "" if pd.isna(row.chart_context) else str(row.chart_context)
    d: dict[str, float | str] = {
        "adapter=" + str(row.adapter): 1.0,
        "qtype=" + question_type(q): 1.0,
        "question_has_percent": float("%" in q or "percent" in q.lower() or "percentage" in q.lower()),
        "question_has_decimal_word": float("decimal" in q.lower()),
    }
    d.update(num_dict("cand", pred))
    d.update(appears_in_context(pred, ctx))
    return d


def fit_predict(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, model_name: str):
    if model_name == "logreg_text":
        vec = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=10000, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=10000, min_df=1)),
        ])
        xtr = vec.fit_transform(train["text"])
        xdv = vec.transform(dev["text"])
        xte = vec.transform(test["text"])
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="liblinear")
    else:
        dv = DictVectorizer()
        xtr = dv.fit_transform(train["dict"])
        xdv = dv.transform(dev["dict"])
        xte = dv.transform(test["dict"])
        scaler = MaxAbsScaler()
        xtr = scaler.fit_transform(xtr)
        xdv = scaler.transform(xdv)
        xte = scaler.transform(xte)
        if model_name == "extra_trees":
            clf = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=13, n_jobs=-1)
        elif model_name == "random_forest":
            clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=13, n_jobs=-1)
        elif model_name == "logreg_num":
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", solver="liblinear")
        else:
            raise ValueError(model_name)
    clf.fit(xtr, train["score"].astype(int))
    dev_prob = clf.predict_proba(xdv)[:, 1]
    test_prob = clf.predict_proba(xte)[:, 1]
    return dev_prob, test_prob


def choose_accuracy(cands: pd.DataFrame, ids: set[str], prob_col: str) -> tuple[float, pd.DataFrame]:
    sub = cands[cands["candidate_id"].isin(ids)].copy()
    choice = sub.loc[sub.groupby("candidate_id")[prob_col].idxmax()].copy()
    return float(choice["score"].mean()), choice


def fixed_scores(cands: pd.DataFrame, ids: set[str]) -> dict[str, float]:
    sub = cands[cands["candidate_id"].isin(ids)].copy()
    return sub.groupby("adapter")["score"].mean().to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-file", action="append", required=True, help="Optional prefix::path CSV.")
    ap.add_argument("--chart-context-csv", required=True)
    ap.add_argument("--manifest-csv", default="")
    ap.add_argument("--out-dir", default="results/scaleqa_natural/reranker_v2")
    ap.add_argument("--name", default="scaleqa520_candidate_reranker_v2")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--dev-size", type=float, default=0.2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = build_candidates(args.policy_file, args.chart_context_csv, args.manifest_csv or None)
    cands["text"] = [make_text(r) for r in cands.itertuples(index=False)]
    cands["dict"] = [make_dict(r) for r in cands.itertuples(index=False)]
    inst = instance_table(cands)

    records = []
    choices = []
    model_names = ["logreg_text", "logreg_num", "extra_trees", "random_forest"]
    for seed in range(args.seeds):
        train_ids, dev_ids, test_ids = split_instances(inst, seed, args.test_size, args.dev_size)
        train = cands[cands["candidate_id"].isin(train_ids)].copy()
        dev = cands[cands["candidate_id"].isin(dev_ids)].copy()
        test = cands[cands["candidate_id"].isin(test_ids)].copy()

        dev_scores = {}
        test_probs = {}
        for model_name in model_names:
            dev_prob, test_prob = fit_predict(train, dev, test, model_name)
            dev[f"prob_{model_name}"] = dev_prob
            test[f"prob_{model_name}"] = test_prob
            dev_acc, _ = choose_accuracy(dev, dev_ids, f"prob_{model_name}")
            dev_scores[model_name] = dev_acc
            test_probs[model_name] = test_prob
        best_model = max(dev_scores, key=dev_scores.get)
        test[f"prob_best"] = test_probs[best_model]
        learned_acc, choice = choose_accuracy(test, test_ids, "prob_best")
        fixed = fixed_scores(cands, test_ids)
        train_fixed = fixed_scores(cands, train_ids)
        best_fixed_name = max(train_fixed, key=train_fixed.get)
        best_fixed_acc = fixed.get(best_fixed_name, float("nan"))
        oracle = float(test.groupby("candidate_id")["score"].max().mean())
        majority_acc = fixed.get("tta:candidate_majority", fixed.get("candidate_majority", float("nan")))
        row = {
            "seed": seed,
            "n_train": len(train_ids),
            "n_dev": len(dev_ids),
            "n_test": len(test_ids),
            "best_model": best_model,
            "learned_reranker": learned_acc,
            "train_selected_fixed": best_fixed_name,
            "train_selected_fixed_acc": best_fixed_acc,
            "oracle_candidate": oracle,
            "candidate_majority": majority_acc,
        }
        for k, v in fixed.items():
            row[f"fixed_{k}"] = v
        records.append(row)
        choice["seed"] = seed
        choice["chosen_model"] = best_model
        choices.append(choice[["seed", "candidate_id", "adapter", "prediction", "score", "chosen_model", "scale_error_subtype", "scale_directions"]])

    results = pd.DataFrame(records)
    choices_df = pd.concat(choices, ignore_index=True)
    results_csv = out_dir / f"{args.name}_results.csv"
    choices_csv = out_dir / f"{args.name}_choices.csv"
    candidates_csv = out_dir / f"{args.name}_candidate_pool.csv"
    results.to_csv(results_csv, index=False)
    choices_df.to_csv(choices_csv, index=False)
    cands.drop(columns=["dict", "text"]).to_csv(candidates_csv, index=False)

    metric_cols = [c for c in results.columns if c.startswith("fixed_")] + [
        "learned_reranker", "train_selected_fixed_acc", "candidate_majority", "oracle_candidate"
    ]
    summary = {
        "name": args.name,
        "n_instances": int(inst["candidate_id"].nunique()),
        "n_candidate_rows": int(len(cands)),
        "n_unique_policy_names": int(cands["adapter"].nunique()),
        "policy_names": sorted(cands["adapter"].unique().tolist()),
        "metrics": {
            c: {
                "mean": float(results[c].mean()),
                "std": float(results[c].std(ddof=1)),
                "min": float(results[c].min()),
                "max": float(results[c].max()),
            }
            for c in metric_cols if c in results.columns
        },
        "chosen_model_counts": results["best_model"].value_counts().to_dict(),
        "files": {
            "results_csv": str(results_csv),
            "choices_csv": str(choices_csv),
            "candidate_pool_csv": str(candidates_csv),
        },
    }
    summary_json = out_dir / f"{args.name}_summary.json"
    summary_md = out_dir / f"{args.name}_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Candidate reranker: {args.name}", "", f"N={summary['n_instances']}, candidate rows={summary['n_candidate_rows']}"]
    lines += ["", "| Metric | Mean | Std | Min | Max |", "|---|---:|---:|---:|---:|"]
    for c, m in summary["metrics"].items():
        lines.append(f"| {c} | {m['mean']:.4f} | {m['std']:.4f} | {m['min']:.4f} | {m['max']:.4f} |")
    lines += ["", "## Chosen model counts"]
    for k, v in summary["chosen_model_counts"].items():
        lines.append(f"- {k}: {v}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
