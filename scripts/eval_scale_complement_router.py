import argparse
import json
import re
from pathlib import Path

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


def is_factor_100(base_pred, comp_pred, lo=80.0, hi=125.0):
    b = one_number(base_pred)
    c = one_number(comp_pred)

    if b is None or c is None:
        return False

    if abs(c) < 1e-12:
        return False

    ratio = abs(b / c)

    return lo <= ratio <= hi


def question_excluded(q):
    q = str(q).lower()

    # count/year questions are dangerous: 4 vs 0.04 is usually not a scale issue.
    bad_words = [
        "how many",
        "number of",
        "count",
        "which year",
        "what year",
        "year was",
        "year did",
    ]

    return any(w in q for w in bad_words)


def question_scale_related(q):
    q = str(q).lower()

    good_words = [
        "percentage",
        "percent",
        "%",
        "rate",
        "value",
        "difference",
        "highest",
        "largest",
        "maximum",
        "smallest",
        "lowest",
        "average",
        "mean",
        "median",
        "sum",
        "total",
        "bar",
        "line",
    ]

    return any(w in q for w in good_words)


def choose_scale_router(question, base_pred, comp_pred, mode):
    b = one_number(base_pred)
    c = one_number(comp_pred)

    if b is None or c is None:
        return base_pred, "base_no_single_num"

    if not is_factor_100(base_pred, comp_pred):
        return base_pred, "base_no_factor100"

    # We only want cases like base=92, comp=0.92.
    # If comp is not smaller than base, keep base.
    if abs(c) >= abs(b):
        return base_pred, "base_comp_not_smaller"

    if mode == "factor100_any":
        return comp_pred, "comp_factor100_any"

    if mode == "factor100_no_count_year":
        if question_excluded(question):
            return base_pred, "base_excluded_count_year"
        return comp_pred, "comp_factor100_no_count_year"

    if mode == "factor100_scale_keywords":
        if question_excluded(question):
            return base_pred, "base_excluded_count_year"
        if question_scale_related(question):
            return comp_pred, "comp_factor100_scale_keywords"
        return base_pred, "base_no_scale_keyword"

    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/eval/chartqa_comp_200.csv")
    parser.add_argument("--base", default="sfg_cem_best")
    parser.add_argument("--comp", default="comp_final")
    parser.add_argument("--output", default="results/eval/chartqa_scale_router_200.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    score = df.pivot_table(
        index="idx",
        columns="adapter",
        values="score",
        aggfunc="first",
    )

    pred = df.pivot_table(
        index="idx",
        columns="adapter",
        values="prediction",
        aggfunc="first",
    )

    meta = df.drop_duplicates("idx").set_index("idx")

    modes = [
        "factor100_any",
        "factor100_no_count_year",
        "factor100_scale_keywords",
    ]

    rows = []

    print("Base:", args.base, float(score[args.base].mean()))
    print("Comp:", args.comp, float(score[args.comp].mean()))
    print("Oracle:", float(score[[args.base, args.comp]].max(axis=1).mean()))

    for mode in modes:
        chosen_scores = []
        chosen_rows = []

        for idx in score.index:
            q = meta.loc[idx, "question"]
            answers = meta.loc[idx, "answers"]

            base_pred = pred.loc[idx, args.base]
            comp_pred = pred.loc[idx, args.comp]

            selected_pred, reason = choose_scale_router(
                q,
                base_pred,
                comp_pred,
                mode,
            )

            # Since this script is evaluating existing predictions,
            # we infer score by selecting the corresponding existing adapter score.
            if str(selected_pred) == str(comp_pred):
                selected_from = args.comp
                s = score.loc[idx, args.comp]
            else:
                selected_from = args.base
                s = score.loc[idx, args.base]

            chosen_scores.append(s)

            chosen_rows.append({
                "idx": idx,
                "mode": mode,
                "question": q,
                "answers": answers,
                "base_pred": base_pred,
                "comp_pred": comp_pred,
                "selected_from": selected_from,
                "selected_pred": selected_pred,
                "reason": reason,
                "score": s,
                "base_score": score.loc[idx, args.base],
                "comp_score": score.loc[idx, args.comp],
            })

        out_df = pd.DataFrame(chosen_rows)

        wins = out_df.index[
            (out_df["score"] == 1) & (out_df["base_score"] == 0)
        ].tolist()

        losses = out_df.index[
            (out_df["score"] == 0) & (out_df["base_score"] == 1)
        ].tolist()

        print("\n" + "=" * 80)
        print("MODE:", mode)
        print("router score:", round(float(out_df["score"].mean()), 4))
        print("wins over base:", len(wins))
        print("losses to base:", len(losses))
        print("net:", len(wins) - len(losses))
        print(out_df["reason"].value_counts().to_string())

        rows.extend(chosen_rows)

    final_df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out, index=False)

    summary = (
        final_df.groupby("mode")["score"]
        .mean()
        .round(4)
        .to_dict()
    )

    summary["_base"] = round(float(score[args.base].mean()), 4)
    summary["_comp"] = round(float(score[args.comp].mean()), 4)
    summary["_oracle"] = round(float(score[[args.base, args.comp]].max(axis=1).mean()), 4)

    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSaved:", out)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
