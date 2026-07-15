import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


def to_answer_list(x):
    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    s = str(x).strip()

    # Try JSON first
    try:
        y = json.loads(s)
        if isinstance(y, list):
            return y
        return [y]
    except Exception:
        pass

    # Try Python literal: "['59']"
    try:
        y = ast.literal_eval(s)
        if isinstance(y, list):
            return y
        return [y]
    except Exception:
        return [s]


def norm_text(x):
    s = str(x).strip().lower()
    s = s.replace("−", "-")
    s = s.replace(",", "")
    s = s.replace("%", " percent ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;\"'[](){}")
    return s


def extract_numbers(x):
    s = str(x).lower()
    s = s.replace(",", "")
    s = s.replace("−", "-")
    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", s)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def close_num(pred, gold, tol=0.05):
    # ChartQA-style relaxed numeric accuracy.
    if abs(gold) < 1e-12:
        return abs(pred - gold) <= 1e-4
    return abs(pred - gold) <= tol * abs(gold)


def numeric_match(pred, gold, allow_scale=False):
    ps = extract_numbers(pred)
    gs = extract_numbers(gold)

    if not ps or not gs:
        return False

    for p in ps:
        for g in gs:
            if close_num(p, g):
                return True

            # Diagnostic only: 0.59 <-> 59 scale ambiguity.
            if allow_scale:
                if close_num(p * 100.0, g):
                    return True
                if close_num(p, g * 100.0):
                    return True

    return False


def score_one(pred, answers, allow_scale=False):
    pred_n = norm_text(pred)
    ans_list = [norm_text(a) for a in to_answer_list(answers) if str(a).strip()]

    if not ans_list:
        return 0

    # Exact normalized string.
    if any(pred_n == a for a in ans_list):
        return 1

    # Numeric relaxed score.
    if any(numeric_match(pred, a, allow_scale=allow_scale) for a in ans_list):
        return 1

    # Mild substring fallback for non-numeric short answers.
    # Keep this conservative to avoid false positives.
    for a in ans_list:
        if len(a) >= 3 and len(pred_n) >= 3:
            if a in pred_n or pred_n in a:
                return 1

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "prediction" not in df.columns:
        raise ValueError("CSV must contain column: prediction")

    answer_col = "answers" if "answers" in df.columns else "answer"
    if answer_col not in df.columns:
        raise ValueError("CSV must contain column: answers or answer")

    df["score_official_v2"] = [
        score_one(p, a, allow_scale=False)
        for p, a in zip(df["prediction"], df[answer_col])
    ]

    df["score_scaled_v2"] = [
        score_one(p, a, allow_scale=True)
        for p, a in zip(df["prediction"], df[answer_col])
    ]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Saved:", out)

    print("\n=== Original score ===")
    if "score" in df.columns:
        print(df.groupby("adapter")["score"].mean().round(4).to_string())

    print("\n=== official_v2 ===")
    print(df.groupby("adapter")["score_official_v2"].mean().round(4).to_string())

    print("\n=== scaled_v2 diagnostic ===")
    print(df.groupby("adapter")["score_scaled_v2"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
