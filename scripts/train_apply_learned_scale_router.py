import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract_numbers(x):
    s = str(x).lower().replace(",", "").replace("−", "-")
    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", s)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def one_number(x):
    nums = extract_numbers(x)
    if len(nums) != 1:
        return None
    return nums[0]


def has_decimal_string(x):
    return int(bool(re.search(r"\d+\.\d+", str(x))))


def is_integer_like_value(v):
    if v is None:
        return 0
    return int(abs(v - round(v)) < 1e-8)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def qfeat(q):
    q = str(q).lower()

    keys = {
        "has_percent": ["percentage", "percent", "%"],
        "has_rate": ["rate", "ratio", "proportion"],
        "has_value": ["value", "values"],
        "has_diff": ["difference", "gap"],
        "has_max": ["highest", "largest", "maximum", "max"],
        "has_min": ["lowest", "smallest", "minimum", "min"],
        "has_avg": ["average", "mean", "median"],
        "has_sum": ["sum", "total", "add up"],
        "has_count": ["how many", "number of", "count"],
        "has_year": ["which year", "what year", "year was", "year did"],
        "has_bar": ["bar", "bars"],
        "has_line": ["line", "lines"],
    }

    out = {}
    for name, words in keys.items():
        out[name] = int(any(w in q for w in words))

    out["q_len"] = len(q.split())
    return out


def is_factor100_conflict(base_pred, comp_pred):
    b = one_number(base_pred)
    c = one_number(comp_pred)
    if b is None or c is None or abs(c) < 1e-12:
        return False
    ratio = abs(b / c)
    return 80 <= ratio <= 125 and abs(c) < abs(b)


def load_margin0(path):
    df = pd.read_csv(path)

    if "margin" in df.columns:
        df["_margin_float"] = df["margin"].apply(lambda x: safe_float(x, default=999))
        df = df[df["_margin_float"].abs() < 1e-9].copy()

    df = df.drop_duplicates("idx").copy()
    return df


def make_features(df):
    rows = []

    for _, r in df.iterrows():
        base_pred = r["base_pred"]
        comp_pred = r["comp_pred"]
        q = r["question"]

        b = one_number(base_pred)
        c = one_number(comp_pred)

        if b is None or c is None or abs(c) < 1e-12:
            continue

        ratio = abs(b / c)

        if not (80 <= ratio <= 125 and abs(c) < abs(b)):
            continue

        item = {
            "idx": int(r["idx"]),
            "base_value": b,
            "comp_value": c,
            "abs_base": abs(b),
            "abs_comp": abs(c),
            "ratio": ratio,
            "log_ratio": math.log(max(ratio, 1e-8)),
            "delta_b_minus_a": safe_float(r.get("delta_b_minus_a", 0.0), default=0.0),
            "logp_a": safe_float(r.get("logp_a", 0.0), default=0.0),
            "logp_b": safe_float(r.get("logp_b", 0.0), default=0.0),
            "base_has_decimal": has_decimal_string(base_pred),
            "comp_has_decimal": has_decimal_string(comp_pred),
            "base_integer_like": is_integer_like_value(b),
            "comp_integer_like": is_integer_like_value(c),
            "comp_small_decimal": int(abs(c) <= 1.5 and has_decimal_string(comp_pred)),
            "base_large": int(abs(b) >= 2.0),
            "base_score": int(r["base_score"]),
            "comp_score": int(r["comp_score"]),
            "question": q,
            "answers": r["answers"],
            "base_pred": base_pred,
            "comp_pred": comp_pred,
        }

        item.update(qfeat(q))
        rows.append(item)

    return pd.DataFrame(rows)


def get_feature_cols():
    return [
        "abs_base",
        "abs_comp",
        "ratio",
        "log_ratio",
        "delta_b_minus_a",
        "logp_a",
        "logp_b",
        "base_has_decimal",
        "comp_has_decimal",
        "base_integer_like",
        "comp_integer_like",
        "comp_small_decimal",
        "base_large",
        "has_percent",
        "has_rate",
        "has_value",
        "has_diff",
        "has_max",
        "has_min",
        "has_avg",
        "has_sum",
        "has_count",
        "has_year",
        "has_bar",
        "has_line",
        "q_len",
    ]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_model(train_df, feature_cols):
    X = train_df[feature_cols].fillna(0).values.astype(float)
    y = train_df["label"].values.astype(int)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                C=1.0,
            ),
        )
        clf.fit(X, y)

        def predict_prob(df):
            X2 = df[feature_cols].fillna(0).values.astype(float)
            return clf.predict_proba(X2)[:, 1]

        return predict_prob, "logistic_regression"

    except Exception as e:
        print("sklearn unavailable or failed, fallback to delta-only router:", repr(e))

        delta = train_df["delta_b_minus_a"].fillna(0).values.astype(float)
        pos_mean = delta[y == 1].mean() if (y == 1).sum() else 0.0
        neg_mean = delta[y == 0].mean() if (y == 0).sum() else 0.0
        scale = 5.0 if abs(pos_mean - neg_mean) < 1e-6 else 5.0 / abs(pos_mean - neg_mean)
        center = (pos_mean + neg_mean) / 2.0

        def predict_prob(df):
            d = df["delta_b_minus_a"].fillna(0).values.astype(float)
            return sigmoid((d - center) * scale)

        return predict_prob, "delta_fallback"


def evaluate_threshold(full_df, conflict_df, prob, tau, base_name="base", comp_name="comp"):
    selected = full_df.copy()
    selected["selected_from"] = base_name
    selected["score"] = selected["base_score"]

    conflict_map = dict(zip(conflict_df["idx"], prob))
    selected["router_prob_comp"] = selected["idx"].map(conflict_map)

    use_comp = selected["router_prob_comp"].notna() & (selected["router_prob_comp"] >= tau)

    selected.loc[use_comp, "selected_from"] = comp_name
    selected.loc[use_comp, "score"] = selected.loc[use_comp, "comp_score"]

    wins = selected[(selected["score"] == 1) & (selected["base_score"] == 0)]
    losses = selected[(selected["score"] == 0) & (selected["base_score"] == 1)]

    return {
        "score": float(selected["score"].mean()),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "net": int(len(wins) - len(losses)),
        "selected": int(use_comp.sum()),
        "df": selected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-logprob", required=True)
    parser.add_argument("--val-logprob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--loss-penalty", type=float, default=2.0)
    parser.add_argument("--prefer-zero-loss", action="store_true")
    args = parser.parse_args()

    train_raw = load_margin0(args.train_logprob)
    val_raw = load_margin0(args.val_logprob)

    train_conf = make_features(train_raw)
    val_conf = make_features(val_raw)

    # Train only on cases where base and comp disagree in correctness.
    train_sup = train_conf[train_conf["base_score"] != train_conf["comp_score"]].copy()
    train_sup["label"] = (train_sup["comp_score"] > train_sup["base_score"]).astype(int)

    print("train total rows:", len(train_raw))
    print("train factor100 conflicts:", len(train_conf))
    print("train supervised conflicts:", len(train_sup))
    print("train comp-positive:", int(train_sup["label"].sum()))
    print("train base-positive:", int((1 - train_sup["label"]).sum()))

    print("val total rows:", len(val_raw))
    print("val factor100 conflicts:", len(val_conf))

    if len(train_sup) < 5 or train_sup["label"].nunique() < 2:
        raise RuntimeError("Not enough supervised conflict examples. Increase train max-samples.")

    feature_cols = get_feature_cols()
    predict_prob, model_type = train_model(train_sup, feature_cols)

    train_prob = predict_prob(train_conf)
    val_prob = predict_prob(val_conf)

    # Tune threshold on train.
    candidates = np.linspace(0.05, 0.95, 91)

    train_results = []
    for tau in candidates:
        res = evaluate_threshold(train_raw, train_conf, train_prob, tau)
        objective = res["score"] - args.loss_penalty * (res["losses"] / max(len(train_raw), 1))
        train_results.append((objective, tau, res))

    if args.prefer_zero_loss:
        zero_loss = [x for x in train_results if x[2]["losses"] == 0]
        if zero_loss:
            train_results = zero_loss

    train_results.sort(key=lambda x: (x[0], x[2]["score"], -x[2]["losses"]), reverse=True)
    best_obj, best_tau, best_train = train_results[0]

    print("\nChosen model:", model_type)
    print("Chosen tau:", round(float(best_tau), 4))
    print("Train score:", round(best_train["score"], 4))
    print("Train wins:", best_train["wins"])
    print("Train losses:", best_train["losses"])
    print("Train selected:", best_train["selected"])

    val_res = evaluate_threshold(val_raw, val_conf, val_prob, best_tau)

    out_df = val_res["df"].copy()

    # Add predictions.
    out_df["selected_pred"] = np.where(
        out_df["selected_from"].eq("comp"),
        out_df["comp_pred"],
        out_df["base_pred"],
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    # Print selected comp cases.
    print("\nVAL SUMMARY")
    print("val base:", round(float(val_raw["base_score"].mean()), 4))
    print("val comp:", round(float(val_raw["comp_score"].mean()), 4))
    print("val oracle:", round(float(val_raw[["base_score", "comp_score"]].max(axis=1).mean()), 4))
    print("val learned router:", round(val_res["score"], 4))
    print("val wins:", val_res["wins"])
    print("val losses:", val_res["losses"])
    print("val selected:", val_res["selected"])

    selected_comp = out_df[out_df["selected_from"] == "comp"]
    print("\nSelected comp cases:")
    for _, r in selected_comp.iterrows():
        print("\n" + "=" * 80)
        print("idx:", r["idx"])
        print("Q:", r["question"])
        print("Ans:", r["answers"])
        print("base:", r["base_pred"], "score", r["base_score"])
        print("comp:", r["comp_pred"], "score", r["comp_score"])
        print("prob_comp:", r["router_prob_comp"])

    summary = {
        "model_type": model_type,
        "tau": round(float(best_tau), 4),
        "train_score": round(float(best_train["score"]), 4),
        "train_wins": int(best_train["wins"]),
        "train_losses": int(best_train["losses"]),
        "train_selected": int(best_train["selected"]),
        "val_base": round(float(val_raw["base_score"].mean()), 4),
        "val_comp": round(float(val_raw["comp_score"].mean()), 4),
        "val_oracle": round(float(val_raw[["base_score", "comp_score"]].max(axis=1).mean()), 4),
        "val_router": round(float(val_res["score"]), 4),
        "val_wins": int(val_res["wins"]),
        "val_losses": int(val_res["losses"]),
        "val_net": int(val_res["net"]),
        "val_selected": int(val_res["selected"]),
        "train_conflicts": int(len(train_conf)),
        "val_conflicts": int(len(val_conf)),
    }

    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSaved:", out)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
