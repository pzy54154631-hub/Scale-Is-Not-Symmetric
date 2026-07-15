import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sigmoid(x):
    x = max(min(float(x), 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def module_type(key: str):
    if ".visual.blocks." in key and ".mlp." in key:
        return "visual_mlp"
    if ".visual.blocks." in key:
        return "visual_other"
    if ".language_model." in key and ".mlp." in key:
        return "lang_mlp"
    if ".language_model." in key:
        return "lang_other"
    return "other"


def get_block_id(key: str):
    parts = key.split(".")
    if "blocks" not in parts:
        return None
    i = parts.index("blocks")
    try:
        return int(parts[i + 1])
    except Exception:
        return None


def depth_factor(key: str, shallow=0.95, middle=1.0, deep=1.05):
    bid = get_block_id(key)
    if bid is None:
        return 1.0
    if bid < 10:
        return shallow
    if bid < 22:
        return middle
    return deep


def module_prior(key: str, args):
    t = module_type(key)
    if t == "visual_mlp":
        return args.prior_visual_mlp
    if t == "visual_other":
        return args.prior_visual_other
    if t == "lang_mlp":
        return args.prior_lang_mlp
    if t == "lang_other":
        return args.prior_lang_other
    return args.prior_other


def lora_norm(B, A, scale):
    B = B.float().cpu()
    A = A.float().cpu()
    val = torch.sum((B.T @ B) * (A @ A.T).T) * scale * scale
    return torch.sqrt(torch.clamp(val, min=1e-12)).item()


def lora_cross(B1, A1, s1, B2, A2, s2):
    B1 = B1.float().cpu()
    A1 = A1.float().cpu()
    B2 = B2.float().cpu()
    A2 = A2.float().cpu()
    return (torch.sum((B1.T @ B2) * (A2 @ A1.T).T) * s1 * s2).item()


def compress_raw_product(B, A, target_r):
    dtype, device = B.dtype, B.device

    B = B.float().cpu()
    A = A.float().cpu()

    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")

    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)

    r = min(target_r, S.numel())
    S = torch.clamp(S[:r], min=0)
    root = torch.sqrt(S)

    B_new = Qb @ (U[:, :r] * root.unsqueeze(0))
    A_new = (root.unsqueeze(1) * Vh[:r, :]) @ Qa.T

    return B_new.to(device=device, dtype=dtype), A_new.to(device=device, dtype=dtype)


def compute_lambda(key, E_text, E_chart, cos, delta, args):
    prior = module_prior(key, args)

    if prior <= 0:
        return 0.0, {
            "prior": prior,
            "share_gate": 0.0,
            "private_gate": 0.0,
            "chart_evidence": 0.0,
            "conflict_gate": 0.0,
            "depth_factor": 1.0,
        }

    # shared component: high cosine + small disagreement
    share_gate = sigmoid((cos - args.tau_share) / args.temp_share)

    # private component: large difference between adapters
    private_gate = sigmoid((delta - args.tau_private) / args.temp_private)

    # chart evidence: chart update has larger module energy
    chart_ratio = E_chart / (E_text + E_chart + 1e-12)
    chart_evidence = sigmoid((chart_ratio - 0.5) / args.temp_energy)

    # conflict gate: negative cosine should reduce correction
    conflict_gate = sigmoid((cos - args.tau_conflict) / args.temp_conflict)

    d_factor = depth_factor(
        key,
        shallow=args.shallow_scale,
        middle=args.middle_scale,
        deep=args.deep_scale,
    )

    # SPF idea:
    # - strong shared gate means average is already good, so private correction should be moderate
    # - private_gate + chart_evidence increases ChartQA private residual
    # - conflict_gate prevents destructive correction
    raw = (
        args.base_lambda
        * prior
        * d_factor
        * conflict_gate
        * (
            args.private_weight * private_gate
            + args.evidence_weight * chart_evidence
            + args.shared_weight * (1.0 - share_gate)
            + args.constant_weight
        )
    )

    lam = clamp(raw, args.lambda_min, args.lambda_max)

    return lam, {
        "prior": prior,
        "share_gate": share_gate,
        "private_gate": private_gate,
        "chart_evidence": chart_evidence,
        "conflict_gate": conflict_gate,
        "depth_factor": d_factor,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", required=True)
    parser.add_argument("--chart", required=True)
    parser.add_argument("--average", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--target-r", type=int, default=16)
    parser.add_argument("--target-alpha", type=int, default=32)

    # module priors
    parser.add_argument("--prior-visual-mlp", type=float, default=1.0)
    parser.add_argument("--prior-visual-other", type=float, default=0.0)
    parser.add_argument("--prior-lang-mlp", type=float, default=0.0)
    parser.add_argument("--prior-lang-other", type=float, default=0.0)
    parser.add_argument("--prior-other", type=float, default=0.0)

    # lambda range
    parser.add_argument("--base-lambda", type=float, default=0.10)
    parser.add_argument("--lambda-min", type=float, default=0.00)
    parser.add_argument("--lambda-max", type=float, default=0.12)

    # gates
    parser.add_argument("--tau-share", type=float, default=0.15)
    parser.add_argument("--temp-share", type=float, default=0.20)

    parser.add_argument("--tau-private", type=float, default=0.45)
    parser.add_argument("--temp-private", type=float, default=0.20)

    parser.add_argument("--tau-conflict", type=float, default=-0.10)
    parser.add_argument("--temp-conflict", type=float, default=0.15)

    parser.add_argument("--temp-energy", type=float, default=0.12)

    # mixture coefficients
    parser.add_argument("--private-weight", type=float, default=0.35)
    parser.add_argument("--evidence-weight", type=float, default=0.35)
    parser.add_argument("--shared-weight", type=float, default=0.20)
    parser.add_argument("--constant-weight", type=float, default=0.35)

    # depth prior
    parser.add_argument("--shallow-scale", type=float, default=0.95)
    parser.add_argument("--middle-scale", type=float, default=1.00)
    parser.add_argument("--deep-scale", type=float, default=1.05)

    args = parser.parse_args()

    text_dir = Path(args.text)
    chart_dir = Path(args.chart)
    avg_dir = Path(args.average)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_t = load_file(str(text_dir / "adapter_model.safetensors"))
    state_c = load_file(str(chart_dir / "adapter_model.safetensors"))
    state_avg = load_file(str(avg_dir / "adapter_model.safetensors"))

    cfg_t = json.loads((text_dir / "adapter_config.json").read_text())
    cfg_c = json.loads((chart_dir / "adapter_config.json").read_text())
    cfg_out = json.loads((avg_dir / "adapter_config.json").read_text())

    r_t = int(cfg_t.get("r", args.target_r))
    r_c = int(cfg_c.get("r", args.target_r))
    alpha_t = float(cfg_t.get("lora_alpha", r_t))
    alpha_c = float(cfg_c.get("lora_alpha", r_c))

    scale_t = alpha_t / r_t
    scale_c = alpha_c / r_c
    out_scale = float(args.target_alpha) / float(args.target_r)

    new_state = dict(state_avg)
    stats = []

    for a_key in state_t:
        if ".lora_A." not in a_key:
            continue

        b_key = a_key.replace(".lora_A.", ".lora_B.")

        if a_key not in state_c or b_key not in state_t or b_key not in state_c:
            continue

        prior = module_prior(a_key, args)
        if prior <= 0:
            # keep average exactly for inactive modules
            continue

        A_t, B_t = state_t[a_key], state_t[b_key]
        A_c, B_c = state_c[a_key], state_c[b_key]

        E_t = lora_norm(B_t, A_t, scale_t)
        E_c = lora_norm(B_c, A_c, scale_c)
        cross = lora_cross(B_t, A_t, scale_t, B_c, A_c, scale_c)

        cos = cross / (E_t * E_c + 1e-12)
        cos = clamp(cos, -1.0, 1.0)

        diff_sq = max(E_t * E_t + E_c * E_c - 2.0 * cross, 0.0)
        delta = math.sqrt(diff_sq) / (E_t + E_c + 1e-12)

        lam, meta = compute_lambda(a_key, E_t, E_c, cos, delta, args)

        # SPF:
        # avg + lambda(chart - text)
        # = (0.5 - lambda) text + (0.5 + lambda) chart
        w_text = clamp(0.5 - lam, 0.0, 1.0)
        w_chart = clamp(0.5 + lam, 0.0, 1.0)

        c_text = max(w_text * scale_t / out_scale, 0.0)
        c_chart = max(w_chart * scale_c / out_scale, 0.0)

        B_cat = torch.cat(
            [
                math.sqrt(c_text) * B_t,
                math.sqrt(c_chart) * B_c,
            ],
            dim=1,
        )

        A_cat = torch.cat(
            [
                math.sqrt(c_text) * A_t,
                math.sqrt(c_chart) * A_c,
            ],
            dim=0,
        )

        B_new, A_new = compress_raw_product(B_cat, A_cat, args.target_r)

        new_state[a_key] = A_new
        new_state[b_key] = B_new

        stats.append(
            {
                "module": a_key.replace(".lora_A.", "."),
                "module_type": module_type(a_key),
                "block": get_block_id(a_key),
                "E_text": E_t,
                "E_chart": E_c,
                "chart_ratio": E_c / (E_t + E_c + 1e-12),
                "cos": cos,
                "delta": delta,
                "lambda": lam,
                "w_text": w_text,
                "w_chart": w_chart,
                **meta,
            }
        )

    cfg_out["r"] = args.target_r
    cfg_out["lora_alpha"] = args.target_alpha
    cfg_out["rank_pattern"] = {}
    cfg_out["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg_out, indent=2))
    (out_dir / "mirage_spf_stats.json").write_text(json.dumps(stats, indent=2))

    for f in avg_dir.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved:", out_dir)
    print("active module pairs:", len(stats))

    if stats:
        print("mean lambda:", sum(x["lambda"] for x in stats) / len(stats))
        print("mean w_chart:", sum(x["w_chart"] for x in stats) / len(stats))
        print("min lambda:", min(x["lambda"] for x in stats))
        print("max lambda:", max(x["lambda"] for x in stats))

        by_type = {}
        for x in stats:
            by_type.setdefault(x["module_type"], []).append(x)

        for t, xs in sorted(by_type.items()):
            print(
                t,
                "n=", len(xs),
                "mean_lambda=", sum(x["lambda"] for x in xs) / len(xs),
                "mean_w_chart=", sum(x["w_chart"] for x in xs) / len(xs),
            )


if __name__ == "__main__":
    main()
