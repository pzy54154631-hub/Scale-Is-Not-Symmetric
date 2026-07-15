#!/usr/bin/env python
"""Train lightweight scale-regime routers on ScaleQA-Natural diagnostics.

The goal is not to claim a production-ready router.  It is to replace an
oracle routed upper bound with a deployable-style diagnostic: given the
question and the base prediction, can a small classifier select among fixed
scale adapters better than any single global direction?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion


ADAPTER_ORDER = [
    "base",
    "large_pos",
    "small_pos",
    "balanced",
    "real_only",
    "all_pairs",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-csv",
        action="append",
        required=True,
        help="Long-format eval CSV with adapter/candidate_id/score columns.",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", default="results/scaleqa_natural/router")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dev-size", type=float, default=0.2)
    parser.add_argument(
        "--adapters",
        default=",".join(ADAPTER_ORDER),
        help="Comma-separated adapter names to route among.",
    )
    return parser.parse_args()


def read_long_eval(paths: list[str], adapters: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["adapter"].isin(adapters)].copy()
    data["score"] = data["score"].astype(float)
    # Avoid duplicated adapter rows if an eval file is passed twice.
    data = data.drop_duplicates(["candidate_id", "adapter"], keep="last")
    missing = set(adapters) - set(data["adapter"].unique())
    if missing:
        raise ValueError(f"Missing adapters in eval data: {sorted(missing)}")
    counts = data.groupby("candidate_id")["adapter"].nunique()
    bad = counts[counts != len(adapters)]
    if len(bad):
        raise ValueError(f"{len(bad)} candidates do not have all adapters")
    return data


def num_features(value: object) -> dict[str, float]:
    text = "" if pd.isna(value) else str(value)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not nums:
        return {
            "has_num": 0.0,
            "abs_log10": 0.0,
            "is_decimal": 0.0,
            "lt_one": 0.0,
            "num_len": 0.0,
            "decimal_places": 0.0,
        }
    raw = nums[0]
    try:
        val = abs(float(raw))
    except ValueError:
        val = 0.0
    decimal_places = 0
    if "." in raw:
        decimal_places = len(raw.split(".", 1)[1].split("e", 1)[0].split("E", 1)[0])
    return {
        "has_num": 1.0,
        "abs_log10": math.log10(val + 1e-12),
        "is_decimal": float("." in raw),
        "lt_one": float(val < 1.0),
        "num_len": float(len(raw)),
        "decimal_places": float(decimal_places),
    }


def build_instance_table(data: pd.DataFrame) -> pd.DataFrame:
    base = data[data["adapter"] == "base"].copy()
    keep_cols = [
        "candidate_id",
        "idx",
        "question",
        "answers",
        "prediction",
        "source_split",
        "source_row_idx",
        "scale_error_subtype",
        "scale_factors",
        "scale_directions",
    ]
    base = base[keep_cols].rename(columns={"prediction": "base_prediction"})
    return base.reset_index(drop=True)


def make_router_rows(data: pd.DataFrame, instances: pd.DataFrame) -> pd.DataFrame:
    merged = data.merge(
        instances[["candidate_id", "base_prediction"]],
        on="candidate_id",
        how="left",
    )
    rows = []
    for row in merged.itertuples(index=False):
        nf = num_features(row.base_prediction)
        pf = num_features(row.prediction)
        question = "" if pd.isna(row.question) else str(row.question)
        base_pred = "" if pd.isna(row.base_prediction) else str(row.base_prediction)
        adapter_pred = "" if pd.isna(row.prediction) else str(row.prediction)
        adapter = str(row.adapter)
        text = (
            f"adapter={adapter} "
            f"question={question} "
            f"base_prediction={base_pred} "
            f"adapter_prediction={adapter_pred} "
            f"has_percent={int('%' in question or 'percent' in question.lower() or 'percentage' in question.lower())} "
            f"base_decimal={int(nf['is_decimal'])} base_lt_one={int(nf['lt_one'])} "
            f"base_dec_places={int(nf['decimal_places'])}"
            f" cand_decimal={int(pf['is_decimal'])} cand_lt_one={int(pf['lt_one'])} "
            f"cand_dec_places={int(pf['decimal_places'])}"
        )
        rows.append(
            {
                "candidate_id": row.candidate_id,
                "adapter": adapter,
                "score": float(row.score),
                "router_text": text,
            }
        )
    return pd.DataFrame(rows)


def split_ids(instances: pd.DataFrame, seed: int, test_size: float, dev_size: float):
    ids = instances["candidate_id"].to_numpy()
    strata = instances["scale_error_subtype"].fillna("unknown").to_numpy()
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_dev_idx, test_idx = next(split1.split(ids, strata))
    train_dev_ids = ids[train_dev_idx]
    test_ids = ids[test_idx]
    train_dev_strata = strata[train_dev_idx]
    dev_fraction_of_train_dev = dev_size / (1.0 - test_size)
    split2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=dev_fraction_of_train_dev,
        random_state=seed + 1000,
    )
    train_idx_rel, dev_idx_rel = next(split2.split(train_dev_ids, train_dev_strata))
    return set(train_dev_ids[train_idx_rel]), set(train_dev_ids[dev_idx_rel]), set(test_ids)


def eval_fixed(data: pd.DataFrame, ids: set[str]) -> dict[str, float]:
    subset = data[data["candidate_id"].isin(ids)]
    return {
        adapter: float(g["score"].mean())
        for adapter, g in subset.groupby("adapter")
    }


def choose_by_adapter(data: pd.DataFrame, ids: set[str], choices: dict[str, str]) -> float:
    subset = data[data["candidate_id"].isin(ids)]
    rows = []
    for cid, group in subset.groupby("candidate_id"):
        choice = choices[cid]
        rows.append(float(group[group["adapter"] == choice]["score"].iloc[0]))
    return float(np.mean(rows)) if rows else float("nan")


def oracle_accuracy(data: pd.DataFrame, ids: set[str]) -> float:
    subset = data[data["candidate_id"].isin(ids)]
    return float(subset.groupby("candidate_id")["score"].max().mean())


def subtype_router(data: pd.DataFrame, instances: pd.DataFrame, train_ids: set[str], test_ids: set[str]):
    train = data[data["candidate_id"].isin(train_ids)].copy()
    if "scale_error_subtype" not in train.columns:
        train = train.merge(
            instances[["candidate_id", "scale_error_subtype"]],
            on="candidate_id",
            how="left",
        )
    subtype_best = {}
    for subtype, group in train.groupby("scale_error_subtype"):
        means = group.groupby("adapter")["score"].mean()
        subtype_best[subtype] = str(means.sort_values(ascending=False).index[0])
    global_best = str(train.groupby("adapter")["score"].mean().sort_values(ascending=False).index[0])
    test_meta = instances[instances["candidate_id"].isin(test_ids)]
    choices = {
        row.candidate_id: subtype_best.get(row.scale_error_subtype, global_best)
        for row in test_meta.itertuples(index=False)
    }
    return choose_by_adapter(data, test_ids, choices), subtype_best


def learned_router(
    router_rows: pd.DataFrame,
    data: pd.DataFrame,
    train_ids: set[str],
    dev_ids: set[str],
    test_ids: set[str],
):
    train = router_rows[router_rows["candidate_id"].isin(train_ids)]
    dev = router_rows[router_rows["candidate_id"].isin(dev_ids)]
    test = router_rows[router_rows["candidate_id"].isin(test_ids)]

    best_model = None
    best_dev = -1.0
    best_c = None
    for c in [0.1, 0.3, 1.0, 3.0, 10.0]:
        model = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=6000,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        max_features=6000,
                    ),
                ),
            ]
        )
        x_train = model.fit_transform(train["router_text"])
        clf = LogisticRegression(
            C=c,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
        clf.fit(x_train, train["score"].astype(int))
        x_dev = model.transform(dev["router_text"])
        probs = clf.predict_proba(x_dev)[:, 1]
        dev_eval = dev[["candidate_id", "adapter", "score"]].copy()
        dev_eval["prob"] = probs
        choices = dev_eval.loc[dev_eval.groupby("candidate_id")["prob"].idxmax()]
        dev_acc = float(choices["score"].mean())
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_model = (model, clf)
            best_c = c

    vectorizer, clf = best_model
    x_test = vectorizer.transform(test["router_text"])
    probs = clf.predict_proba(x_test)[:, 1]
    test_eval = test[["candidate_id", "adapter", "score"]].copy()
    test_eval["prob"] = probs
    choices_df = test_eval.loc[test_eval.groupby("candidate_id")["prob"].idxmax()].copy()
    choices = dict(zip(choices_df["candidate_id"], choices_df["adapter"]))
    acc = choose_by_adapter(data, test_ids, choices)
    return acc, best_dev, best_c, choices_df


def instance_router_text(row) -> str:
    nf = num_features(row.base_prediction)
    question = "" if pd.isna(row.question) else str(row.question)
    base_pred = "" if pd.isna(row.base_prediction) else str(row.base_prediction)
    return (
        f"question={question} "
        f"base_prediction={base_pred} "
        f"has_percent={int('%' in question or 'percent' in question.lower() or 'percentage' in question.lower())} "
        f"base_decimal={int(nf['is_decimal'])} base_lt_one={int(nf['lt_one'])} "
        f"base_dec_places={int(nf['decimal_places'])} "
        f"base_log10_bucket={int(np.clip(np.floor(nf['abs_log10']), -6, 6))}"
    )


def learned_subtype_router(
    data: pd.DataFrame,
    instances: pd.DataFrame,
    train_ids: set[str],
    dev_ids: set[str],
    test_ids: set[str],
):
    train = instances[instances["candidate_id"].isin(train_ids)].copy()
    dev = instances[instances["candidate_id"].isin(dev_ids)].copy()
    test = instances[instances["candidate_id"].isin(test_ids)].copy()
    for frame in [train, dev, test]:
        frame["router_text"] = [instance_router_text(r) for r in frame.itertuples(index=False)]

    # Learn which adapter is best for each gold subtype on the training split.
    train_long = data[data["candidate_id"].isin(train_ids)].copy()
    subtype_best = {}
    train_long = train_long.merge(
        instances[["candidate_id", "scale_error_subtype"]],
        on="candidate_id",
        how="left",
        suffixes=("", "_inst"),
    )
    subtype_col = "scale_error_subtype"
    if "scale_error_subtype_inst" in train_long.columns:
        subtype_col = "scale_error_subtype_inst"
    for subtype, group in train_long.groupby(subtype_col):
        means = group.groupby("adapter")["score"].mean()
        subtype_best[subtype] = str(means.sort_values(ascending=False).index[0])
    global_best = str(train_long.groupby("adapter")["score"].mean().sort_values(ascending=False).index[0])

    best_model = None
    best_dev = -1.0
    best_c = None
    for c in [0.1, 0.3, 1.0, 3.0, 10.0]:
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=6000,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=1,
                        max_features=6000,
                    ),
                ),
            ]
        )
        x_train = vectorizer.fit_transform(train["router_text"])
        clf = LogisticRegression(
            C=c,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            multi_class="ovr",
        )
        clf.fit(x_train, train["scale_error_subtype"])
        pred_dev = clf.predict(vectorizer.transform(dev["router_text"]))
        dev_choices = {
            cid: subtype_best.get(pred, global_best)
            for cid, pred in zip(dev["candidate_id"], pred_dev)
        }
        dev_acc = choose_by_adapter(data, dev_ids, dev_choices)
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_c = c
            best_model = (vectorizer, clf)

    vectorizer, clf = best_model
    pred_test = clf.predict(vectorizer.transform(test["router_text"]))
    choices = {
        cid: subtype_best.get(pred, global_best)
        for cid, pred in zip(test["candidate_id"], pred_test)
    }
    acc = choose_by_adapter(data, test_ids, choices)
    subtype_acc = accuracy_score(test["scale_error_subtype"], pred_test)
    choice_df = pd.DataFrame(
        {
            "candidate_id": test["candidate_id"].to_numpy(),
            "gold_subtype": test["scale_error_subtype"].to_numpy(),
            "pred_subtype": pred_test,
            "chosen_adapter": [choices[cid] for cid in test["candidate_id"]],
        }
    )
    return acc, subtype_acc, best_dev, best_c, choice_df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapters = [x.strip() for x in args.adapters.split(",") if x.strip()]
    data = read_long_eval(args.eval_csv, adapters)
    instances = build_instance_table(data)
    router_rows = make_router_rows(data, instances)

    records = []
    choice_frames = []
    subtype_maps = []
    for seed in range(args.seeds):
        train_ids, dev_ids, test_ids = split_ids(
            instances,
            seed=seed,
            test_size=args.test_size,
            dev_size=args.dev_size,
        )
        fixed = eval_fixed(data, test_ids)
        train_fixed = eval_fixed(data, train_ids)
        best_global = max(train_fixed, key=train_fixed.get)
        global_choices = {cid: best_global for cid in test_ids}
        global_acc = choose_by_adapter(data, test_ids, global_choices)
        subtype_acc, subtype_best = subtype_router(data, instances, train_ids, test_ids)
        learned_acc, learned_dev, best_c, choices = learned_router(
            router_rows, data, train_ids, dev_ids, test_ids
        )
        learned_subtype_acc, subtype_pred_acc, subtype_dev, subtype_c, subtype_choices = (
            learned_subtype_router(data, instances, train_ids, dev_ids, test_ids)
        )
        oracle = oracle_accuracy(data, test_ids)
        row = {
            "seed": seed,
            "n_train": len(train_ids),
            "n_dev": len(dev_ids),
            "n_test": len(test_ids),
            "global_best_adapter": best_global,
            "global_best": global_acc,
            "subtype_router": subtype_acc,
            "learned_router": learned_acc,
            "learned_dev": learned_dev,
            "learned_C": best_c,
            "learned_subtype_router": learned_subtype_acc,
            "learned_subtype_pred_acc": subtype_pred_acc,
            "learned_subtype_dev": subtype_dev,
            "learned_subtype_C": subtype_c,
            "oracle_instance": oracle,
        }
        row.update({f"fixed_{k}": v for k, v in fixed.items()})
        records.append(row)
        choices["seed"] = seed
        choice_frames.append(choices)
        subtype_choices["seed"] = seed
        subtype_choices.to_csv(
            out_dir / f"{args.name}_learned_subtype_router_choices_seed{seed}.csv",
            index=False,
        )
        subtype_maps.append({"seed": seed, **subtype_best})

    results = pd.DataFrame(records)
    choices_all = pd.concat(choice_frames, ignore_index=True)
    subtype_map_df = pd.DataFrame(subtype_maps)

    result_csv = out_dir / f"{args.name}_router_results.csv"
    choices_csv = out_dir / f"{args.name}_learned_router_choices.csv"
    subtype_csv = out_dir / f"{args.name}_subtype_router_maps.csv"
    results.to_csv(result_csv, index=False)
    choices_all.to_csv(choices_csv, index=False)
    subtype_map_df.to_csv(subtype_csv, index=False)

    metric_cols = [
        c for c in results.columns
        if c.startswith("fixed_")
        or c in ["global_best", "subtype_router", "learned_router", "oracle_instance"]
    ]
    summary = {
        "name": args.name,
        "n_instances": int(instances["candidate_id"].nunique()),
        "adapters": adapters,
        "splits": {
            "source_split_counts": instances["source_split"].value_counts().to_dict(),
            "subtype_counts": instances["scale_error_subtype"].value_counts().to_dict(),
        },
        "metrics": {
            col: {
                "mean": float(results[col].mean()),
                "std": float(results[col].std(ddof=1)),
                "min": float(results[col].min()),
                "max": float(results[col].max()),
            }
            for col in metric_cols
        },
        "files": {
            "results_csv": str(result_csv),
            "choices_csv": str(choices_csv),
            "subtype_maps_csv": str(subtype_csv),
        },
    }

    summary_json = out_dir / f"{args.name}_router_summary.json"
    summary_md = out_dir / f"{args.name}_router_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"# Router summary: {args.name}", ""]
    lines.append(f"N instances: {summary['n_instances']}")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|")
    for col in metric_cols:
        m = summary["metrics"][col]
        lines.append(
            f"| {col} | {m['mean']:.4f} | {m['std']:.4f} | {m['min']:.4f} | {m['max']:.4f} |"
        )
    lines.append("")
    lines.append("## Subtype counts")
    for k, v in summary["splits"]["subtype_counts"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Source split counts")
    for k, v in summary["splits"]["source_split_counts"].items():
        lines.append(f"- {k}: {v}")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
