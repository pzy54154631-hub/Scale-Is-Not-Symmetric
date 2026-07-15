import argparse
import ast
import re
import pandas as pd


def nums(x):
    s = str(x).lower().replace(",", "").replace("−", "-")
    out = []
    for n in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", s):
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def first_answer(ans):
    s = str(ans)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and obj:
            return str(obj[0])
    except Exception:
        pass
    return s


def one_number(x):
    ns = nums(x)
    return ns[0] if len(ns) == 1 else None


def is_gold_scale_error(pred, gold):
    p = one_number(pred)
    g = one_number(gold)

    if p is None or g is None:
        return False
    if abs(p) < 1e-12 or abs(g) < 1e-12:
        return False

    ratio = abs(p / g)

    return (0.008 <= ratio <= 0.0125) or (80 <= ratio <= 125)


def get_pred_col(df):
    if "prediction" in df.columns:
        return "prediction"
    if "pred" in df.columns:
        return "pred"
    cols = [c for c in df.columns if "pred" in c.lower()]
    if not cols:
        raise ValueError("No prediction column found.")
    return cols[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--base-name", default="base")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.eval_csv).drop_duplicates(["adapter", "idx"]).copy()
    pred_col = get_pred_col(df)

    base = df[df["adapter"] == args.base_name].copy()
    if len(base) == 0:
        raise ValueError(f"No base adapter named {args.base_name}")

    base["gold"] = base["answers"].apply(first_answer)
    base["gold_scale_error"] = [
        is_gold_scale_error(pred, gold) and int(score) == 0
        for pred, gold, score in zip(base[pred_col], base["gold"], base["score"])
    ]

    scale_idx = set(base.loc[base["gold_scale_error"], "idx"].astype(int))

    print("gold-scale-error subset size:", len(scale_idx))
    print("idx:", sorted(scale_idx))

    rows = []

    for adapter, g in df.groupby("adapter"):
        g = g.drop_duplicates("idx").copy()
        g["is_gold_scale_error"] = g["idx"].astype(int).isin(scale_idx)

        rows.append({
            "method": adapter,
            "overall": g["score"].mean(),
            "gold_scale_error_subset": g[g["is_gold_scale_error"]]["score"].mean(),
            "non_gold_scale_error": g[~g["is_gold_scale_error"]]["score"].mean(),
            "gold_scale_n": int(g["is_gold_scale_error"].sum()),
            "non_gold_scale_n": int((~g["is_gold_scale_error"]).sum()),
        })

    out = pd.DataFrame(rows)

    print("\n=== Summary ===")
    print(out.round(4).to_string(index=False))

    seed = out[out["method"].str.contains("seed", na=False)].copy()
    base_row = out[out["method"] == args.base_name].iloc[0]

    if len(seed):
        print("\n=== Seed mean ===")
        for col in ["overall", "gold_scale_error_subset", "non_gold_scale_error"]:
            print(col, "mean =", round(seed[col].mean(), 4), "std =", round(seed[col].std(), 4))

        print("\n=== Mean delta over base ===")
        print("overall_delta =", round(seed["overall"].mean() - base_row["overall"], 4))
        print("gold_scale_delta =", round(seed["gold_scale_error_subset"].mean() - base_row["gold_scale_error_subset"], 4))
        print("non_scale_delta =", round(seed["non_gold_scale_error"].mean() - base_row["non_gold_scale_error"], 4))

    out.to_csv(args.output, index=False)
    print("\nSaved:", args.output)


if __name__ == "__main__":
    main()
