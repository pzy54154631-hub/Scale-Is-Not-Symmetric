import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PARAMS = [
    # name, init, low, high
    ("base_lambda",     0.08, 0.04, 0.14),
    ("lambda_max",      0.10, 0.07, 0.16),
    ("constant_weight", 0.45, 0.05, 0.80),
    ("evidence_weight", 0.25, 0.00, 0.70),
    ("private_weight",  0.20, 0.00, 0.70),
    ("shared_weight",   0.10, 0.00, 0.50),
    ("tau_share",       0.15, -0.20, 0.50),
    ("tau_private",     0.45, 0.15, 0.85),
    ("tau_conflict",   -0.10, -0.40, 0.25),
    ("temp_energy",     0.12, 0.04, 0.35),
    ("temp_share",      0.20, 0.05, 0.45),
    ("temp_private",    0.20, 0.05, 0.45),
    ("temp_conflict",   0.15, 0.05, 0.40),
]


def clip_vec(x):
    y = x.copy()
    for i, (_, _, lo, hi) in enumerate(PARAMS):
        y[i] = float(np.clip(y[i], lo, hi))
    return y


def vec_to_dict(x):
    return {name: float(x[i]) for i, (name, _, _, _) in enumerate(PARAMS)}


def run(cmd, env=None):
    print("\n$", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env)


def read_mean_w_chart(out_dir):
    p = Path(out_dir) / "mirage_spf_stats.json"
    if not p.exists():
        return None
    xs = json.loads(p.read_text())
    if not xs:
        return None
    return sum(x["w_chart"] for x in xs) / len(xs)


def make_adapter(args, name, params):
    out_dir = f"{args.prefix}_{name}"

    cmd = [
        sys.executable,
        "scripts/merge_mirage_spf_lite.py",
        "--text", args.text,
        "--chart", args.chart,
        "--average", args.average,
        "--output", out_dir,

        "--base-lambda", str(params["base_lambda"]),
        "--lambda-min", str(args.lambda_min),
        "--lambda-max", str(params["lambda_max"]),

        "--constant-weight", str(params["constant_weight"]),
        "--evidence-weight", str(params["evidence_weight"]),
        "--private-weight", str(params["private_weight"]),
        "--shared-weight", str(params["shared_weight"]),

        "--tau-share", str(params["tau_share"]),
        "--tau-private", str(params["tau_private"]),
        "--tau-conflict", str(params["tau_conflict"]),

        "--temp-energy", str(params["temp_energy"]),
        "--temp-share", str(params["temp_share"]),
        "--temp-private", str(params["temp_private"]),
        "--temp-conflict", str(params["temp_conflict"]),

        "--prior-visual-mlp", "1.0",
        "--prior-visual-other", "0.0",
        "--prior-lang-mlp", "0.0",
        "--prior-lang-other", "0.0",
        "--prior-other", "0.0",
    ]

    run(cmd)
    return out_dir


def eval_generation(args, gen, names_to_dirs):
    out_csv = f"results/eval/chartqa_sfg_cem_gen{gen}_200.csv"

    adapters = [
        f"average={args.average}",
        f"vmlp_chart060={args.prefix}_vmlp_chartw_060",
        f"pas_group_stable={args.prefix}_pas_group_stable",
        f"spf_safe={args.prefix}_spf_safe",
    ]

    for name, path in names_to_dirs.items():
        adapters.append(f"{name}={path}")

    cmd = [
        sys.executable,
        "scripts/eval_tiny_adapters.py",
        "--task", "chartqa",
        "--split", "val",
        "--max-samples", str(args.max_samples),
        "--output", out_csv,
        "--adapters",
        *adapters,
    ]

    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"

    run(cmd, env=env)

    df = pd.read_csv(out_csv)
    scores = df.groupby("adapter")["score"].mean().to_dict()
    return scores, out_csv


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", default="outputs/adapters/textvqa_answer_only_3000_r16")
    parser.add_argument("--chart", default="outputs/adapters/chartqa_answer_only_3000_r16")
    parser.add_argument("--average", default="outputs/merged/textvqa_chartqa_answer_only_3000_r16_average")
    parser.add_argument("--prefix", default="outputs/merged/textvqa_chartqa_answer_only_3000_r16")

    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=200)

    parser.add_argument("--lambda-min", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)

    # 避免公式学得太激进，偏离安全区太远
    parser.add_argument("--target-w-chart", type=float, default=0.55)
    parser.add_argument("--wchart-penalty", type=float, default=0.03)

    args = parser.parse_args()

    np.random.seed(args.seed)
    Path("results/eval").mkdir(parents=True, exist_ok=True)

    mean = np.array([init for _, init, _, _ in PARAMS], dtype=np.float64)
    std = np.array([(hi - lo) * 0.18 for _, _, lo, hi in PARAMS], dtype=np.float64)

    history = []
    best = None

    for gen in range(args.generations):
        print("\n" + "=" * 80)
        print(f"GENERATION {gen}")
        print("=" * 80)

        candidates = []

        # 第一代强制包含 safe-like center
        if gen == 0:
            samples = [mean.copy()]
            while len(samples) < args.population:
                samples.append(clip_vec(np.random.normal(mean, std)))
        else:
            samples = [clip_vec(np.random.normal(mean, std)) for _ in range(args.population)]

        names_to_dirs = {}

        for i, vec in enumerate(samples):
            params = vec_to_dict(vec)
            name = f"sfg_cem_g{gen}_i{i}"
            out_dir = make_adapter(args, name, params)
            names_to_dirs[name] = out_dir

            w_chart = read_mean_w_chart(out_dir)
            candidates.append({
                "name": name,
                "path": out_dir,
                "vec": vec.tolist(),
                "params": params,
                "mean_w_chart": w_chart,
            })

        scores, csv_path = eval_generation(args, gen, names_to_dirs)

        for cand in candidates:
            score = float(scores.get(cand["name"], -1.0))
            w_chart = cand["mean_w_chart"]

            penalty = 0.0
            if w_chart is not None:
                penalty = args.wchart_penalty * abs(w_chart - args.target_w_chart)

            objective = score - penalty

            cand["score"] = score
            cand["objective"] = objective
            cand["eval_csv"] = csv_path

            print(
                cand["name"],
                "score=", round(score, 4),
                "w_chart=", None if w_chart is None else round(w_chart, 4),
                "objective=", round(objective, 4),
            )

        candidates = sorted(candidates, key=lambda x: x["objective"], reverse=True)

        if best is None or candidates[0]["objective"] > best["objective"]:
            best = candidates[0]

        elite = candidates[:args.elite]
        elite_vecs = np.array([e["vec"] for e in elite], dtype=np.float64)

        mean = elite_vecs.mean(axis=0)
        std = elite_vecs.std(axis=0) + 1e-3
        std = np.maximum(std, np.array([(hi - lo) * 0.04 for _, _, lo, hi in PARAMS]))
        std = np.minimum(std, np.array([(hi - lo) * 0.25 for _, _, lo, hi in PARAMS]))

        history.extend(candidates)

        print("\nCurrent best:")
        print(json.dumps({
            "name": best["name"],
            "path": best["path"],
            "score": best["score"],
            "objective": best["objective"],
            "mean_w_chart": best["mean_w_chart"],
            "params": best["params"],
        }, indent=2))

        Path("results/eval/sfg_cem_history.json").write_text(json.dumps(history, indent=2))
        Path("results/eval/sfg_cem_best.json").write_text(json.dumps(best, indent=2))

    print("\n" + "=" * 80)
    print("FINAL BEST")
    print("=" * 80)
    print(json.dumps({
        "name": best["name"],
        "path": best["path"],
        "score": best["score"],
        "objective": best["objective"],
        "mean_w_chart": best["mean_w_chart"],
        "params": best["params"],
    }, indent=2))

    print("\nSaved:")
    print("results/eval/sfg_cem_history.json")
    print("results/eval/sfg_cem_best.json")


if __name__ == "__main__":
    main()
