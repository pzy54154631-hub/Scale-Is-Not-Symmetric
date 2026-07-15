import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def is_visual_mlp(key: str) -> bool:
    return ".visual.blocks." in key and ".mlp." in key


def get_block_id(key: str):
    parts = key.split(".")
    if "blocks" not in parts:
        return None
    i = parts.index("blocks")
    try:
        return int(parts[i + 1])
    except Exception:
        return None


def depth_scale(key: str, shallow: float, middle: float, deep: float) -> float:
    bid = get_block_id(key)
    if bid is None:
        return 1.0
    if bid < 10:
        return shallow
    if bid < 22:
        return middle
    return deep


def lora_norm(B, A, scale: float) -> float:
    B = B.float().cpu()
    A = A.float().cpu()
    # ||scale * B A||_F^2 = scale^2 * trace(B^T B A A^T)
    val = torch.sum((B.T @ B) * (A @ A.T).T) * scale * scale
    return torch.sqrt(torch.clamp(val, min=1e-12)).item()


def lora_cross(B1, A1, s1: float, B2, A2, s2: float) -> float:
    B1 = B1.float().cpu()
    A1 = A1.float().cpu()
    B2 = B2.float().cpu()
    A2 = A2.float().cpu()
    # <s1 B1A1, s2 B2A2> = s1 s2 trace(B1A1 A2^T B2^T)
    return (torch.sum((B1.T @ B2) * (A2 @ A1.T).T) * s1 * s2).item()


def compress_raw_product(B, A, target_r: int):
    """
    Compress raw product B @ A back to LoRA rank target_r.
    """
    dtype, device = B.dtype, B.device

    B = B.float().cpu()
    A = A.float().cpu()

    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")

    small = Rb @ Ra.T
    U, S, Vh = torch.linalg.svd(small, full_matrices=False)

    r = min(target_r, S.numel())
    S = torch.clamp(S[:r], min=0)
    root = torch.sqrt(S)

    B_new = Qb @ (U[:, :r] * root.unsqueeze(0))
    A_new = (root.unsqueeze(1) * Vh[:r, :]) @ Qa.T

    return B_new.to(device=device, dtype=dtype), A_new.to(device=device, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", required=True)
    parser.add_argument("--chart", required=True)
    parser.add_argument("--average", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--target-r", type=int, default=16)
    parser.add_argument("--target-alpha", type=int, default=32)

    # Core PAS correction.
    # base_lambda=0.10 means:
    # avg + 0.10(chart-text) = 0.40 text + 0.60 chart
    parser.add_argument("--base-lambda", type=float, default=0.10)
    parser.add_argument("--lambda-min", type=float, default=0.04)
    parser.add_argument("--lambda-max", type=float, default=0.14)

    # Structure prior.
    parser.add_argument("--shallow-scale", type=float, default=0.92)
    parser.add_argument("--middle-scale", type=float, default=1.00)
    parser.add_argument("--deep-scale", type=float, default=1.08)

    # Energy and conflict controls.
    parser.add_argument("--energy-gain", type=float, default=0.35)
    parser.add_argument("--conflict-damping", type=float, default=0.50)
    parser.add_argument("--delta-threshold", type=float, default=0.75)
    parser.add_argument("--delta-damping", type=float, default=0.30)

    # Optional global scale for visual MLP correction.
    parser.add_argument("--pas-scale", type=float, default=1.00)

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

        if not is_visual_mlp(a_key):
            continue

        b_key = a_key.replace(".lora_A.", ".lora_B.")

        if a_key not in state_c or b_key not in state_t or b_key not in state_c:
            continue

        A_t, B_t = state_t[a_key], state_t[b_key]
        A_c, B_c = state_c[a_key], state_c[b_key]

        E_t = lora_norm(B_t, A_t, scale_t)
        E_c = lora_norm(B_c, A_c, scale_c)
        cross = lora_cross(B_t, A_t, scale_t, B_c, A_c, scale_c)

        cos = cross / (E_t * E_c + 1e-12)
        cos = max(min(cos, 1.0), -1.0)

        diff_sq = max(E_t * E_t + E_c * E_c - 2.0 * cross, 0.0)
        delta = math.sqrt(diff_sq) / (E_t + E_c + 1e-12)

        # Chart energy ratio. If ChartQA update is stronger, correction slightly increases.
        chart_ratio = E_c / (E_t + E_c + 1e-12)
        energy_gate = 1.0 + args.energy_gain * (chart_ratio - 0.5) * 2.0

        # If two updates are negatively aligned, reduce correction.
        conflict_gate = 1.0 - args.conflict_damping * max(0.0, -cos)
        conflict_gate = max(0.0, conflict_gate)

        # If disagreement is too large, reduce correction.
        delta_gate = 1.0 - args.delta_damping * max(0.0, delta - args.delta_threshold)
        delta_gate = max(0.0, delta_gate)

        dscale = depth_scale(
            a_key,
            shallow=args.shallow_scale,
            middle=args.middle_scale,
            deep=args.deep_scale,
        )

        lam = (
            args.base_lambda
            * args.pas_scale
            * dscale
            * energy_gate
            * conflict_gate
            * delta_gate
        )

        lam = max(args.lambda_min, min(args.lambda_max, lam))

        # PAS correction:
        # avg + lambda(chart - text)
        # = (0.5 - lambda) text + (0.5 + lambda) chart
        w_text = 0.5 - lam
        w_chart = 0.5 + lam

        # Keep positive and stable.
        w_text = max(0.0, min(1.0, w_text))
        w_chart = max(0.0, min(1.0, w_chart))

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
                "block": get_block_id(a_key),
                "E_text": E_t,
                "E_chart": E_c,
                "chart_ratio": chart_ratio,
                "cos": cos,
                "delta": delta,
                "depth_scale": dscale,
                "lambda": lam,
                "w_text": w_text,
                "w_chart": w_chart,
            }
        )

    cfg_out["r"] = args.target_r
    cfg_out["lora_alpha"] = args.target_alpha
    cfg_out["rank_pattern"] = {}
    cfg_out["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg_out, indent=2))
    (out_dir / "mirage_pas_stats.json").write_text(json.dumps(stats, indent=2))

    for f in avg_dir.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved:", out_dir)
    print("visual_mlp_pairs:", len(stats))

    if stats:
        for key in ["lambda", "w_text", "w_chart", "chart_ratio", "cos", "delta"]:
            print("mean_" + key + ":", sum(x[key] for x in stats) / len(stats))

        print("min_lambda:", min(x["lambda"] for x in stats))
        print("max_lambda:", max(x["lambda"] for x in stats))


if __name__ == "__main__":
    main()
