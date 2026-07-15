import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def fro_norm_lora(B, A, scale):
    B = B.float().cpu()
    A = A.float().cpu()
    gb = B.T @ B
    ga = A @ A.T
    n2 = torch.sum(gb * ga.T) * (scale ** 2)
    return torch.sqrt(torch.clamp(n2, min=1e-12))


def cross_lora(B1, A1, s1, B2, A2, s2):
    B1 = B1.float().cpu()
    A1 = A1.float().cpu()
    B2 = B2.float().cpu()
    A2 = A2.float().cpu()
    cross = torch.sum((B1.T @ B2) * (A2 @ A1.T).T) * s1 * s2
    return cross


def compress_raw_lora(B, A, target_r):
    """
    Compress raw low-rank product B @ A to target rank.
    B: [out, r]
    A: [r, in]
    """
    dtype = B.dtype
    device = B.device

    Bf = B.float().cpu()
    Af = A.float().cpu()

    Qb, Rb = torch.linalg.qr(Bf, mode="reduced")
    Qa, Ra = torch.linalg.qr(Af.T, mode="reduced")

    small = Rb @ Ra.T
    U, S, Vh = torch.linalg.svd(small, full_matrices=False)

    r = min(target_r, S.numel())
    S = torch.clamp(S[:r], min=0)
    sqrtS = torch.sqrt(S)

    B_new = Qb @ (U[:, :r] * sqrtS.unsqueeze(0))
    A_new = (sqrtS.unsqueeze(1) * Vh[:r, :]) @ Qa.T

    return B_new.to(device=device, dtype=dtype), A_new.to(device=device, dtype=dtype)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--target-r", type=int, default=16)
    parser.add_argument("--target-alpha", type=int, default=32)

    parser.add_argument("--energy-power", type=float, default=0.5)
    parser.add_argument("--balance-power", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--align-threshold", type=float, default=0.10)
    parser.add_argument("--conflict-damping", type=float, default=0.50)
    parser.add_argument("--synergy-gain", type=float, default=0.10)
    parser.add_argument("--scale-min", type=float, default=0.60)
    parser.add_argument("--scale-max", type=float, default=1.20)

    args = parser.parse_args()

    dir1 = Path(args.adapter_a)
    dir2 = Path(args.adapter_b)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    state1 = load_file(str(dir1 / "adapter_model.safetensors"))
    state2 = load_file(str(dir2 / "adapter_model.safetensors"))

    cfg1 = json.loads((dir1 / "adapter_config.json").read_text())
    cfg2 = json.loads((dir2 / "adapter_config.json").read_text())

    r1 = int(cfg1.get("r", 16))
    r2 = int(cfg2.get("r", 16))
    alpha1 = float(cfg1.get("lora_alpha", r1))
    alpha2 = float(cfg2.get("lora_alpha", r2))

    scale1 = alpha1 / r1
    scale2 = alpha2 / r2
    out_scale = float(args.target_alpha) / float(args.target_r)

    new_state = {}
    stats = []

    pairs = 0

    for a_key in state1:
        if ".lora_A." not in a_key:
            continue

        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if a_key not in state2 or b_key not in state1 or b_key not in state2:
            continue

        A1 = state1[a_key]
        B1 = state1[b_key]
        A2 = state2[a_key]
        B2 = state2[b_key]

        E1 = fro_norm_lora(B1, A1, scale1).item()
        E2 = fro_norm_lora(B2, A2, scale2).item()
        cross = cross_lora(B1, A1, scale1, B2, A2, scale2).item()
        cos = cross / (E1 * E2 + 1e-12)
        cos = max(min(cos, 1.0), -1.0)

        # Energy-based task preference.
        e1p = (E1 + 1e-12) ** args.energy_power
        e2p = (E2 + 1e-12) ** args.energy_power
        p1 = e1p / (e1p + e2p + 1e-12)
        p2 = e2p / (e1p + e2p + 1e-12)

        # Balance gate: if one task dominates a module, reduce shared averaging.
        balance = 2.0 * min(E1, E2) / (E1 + E2 + 1e-12)
        balance_gate = balance ** args.balance_power

        # Compatibility gate.
        temp = max(args.temperature, 1e-6)
        g = sigmoid((cos - args.align_threshold) / temp) * balance_gate

        # Conflict damping and positive synergy.
        conflict = max(0.0, -cos)
        positive = max(0.0, cos)
        scale_gate = (1.0 - args.conflict_damping * conflict) * (1.0 + args.synergy_gain * positive)
        scale_gate = max(args.scale_min, min(args.scale_max, scale_gate))

        w1 = scale_gate * (g * 0.5 + (1.0 - g) * p1)
        w2 = scale_gate * (g * 0.5 + (1.0 - g) * p2)

        # We want:
        # out_scale * Bnew Anew ~= w1*scale1*B1A1 + w2*scale2*B2A2
        raw_c1 = max(w1 * scale1 / out_scale, 0.0)
        raw_c2 = max(w2 * scale2 / out_scale, 0.0)

        Bcat = torch.cat([
            math.sqrt(raw_c1) * B1,
            math.sqrt(raw_c2) * B2,
        ], dim=1)

        Acat = torch.cat([
            math.sqrt(raw_c1) * A1,
            math.sqrt(raw_c2) * A2,
        ], dim=0)

        Bnew, Anew = compress_raw_lora(Bcat, Acat, args.target_r)

        new_state[a_key] = Anew
        new_state[b_key] = Bnew

        stats.append({
            "module": a_key.replace(".lora_A.", "."),
            "cos": cos,
            "E1": E1,
            "E2": E2,
            "balance": balance,
            "gate": g,
            "scale_gate": scale_gate,
            "w1": w1,
            "w2": w2,
        })

        pairs += 1

    # Copy possible non-LoRA tensors.
    for k, v in state1.items():
        if k not in new_state and ".lora_A." not in k and ".lora_B." not in k:
            new_state[k] = v

    out_cfg = dict(cfg1)
    out_cfg["r"] = args.target_r
    out_cfg["lora_alpha"] = args.target_alpha
    out_cfg["rank_pattern"] = {}
    out_cfg["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(out_cfg, indent=2))
    (out_dir / "mirage_v2_stats.json").write_text(json.dumps(stats, indent=2))

    # Copy processor/tokenizer files.
    for p in dir1.iterdir():
        if p.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / p.name
        if p.is_file() and not dst.exists():
            shutil.copy2(p, dst)

    print("Saved:", out_dir)
    print("processed_pairs:", pairs)

    if stats:
        print("mean_cos:", sum(x["cos"] for x in stats) / len(stats))
        print("mean_gate:", sum(x["gate"] for x in stats) / len(stats))
        print("mean_scale_gate:", sum(x["scale_gate"] for x in stats) / len(stats))
        print("mean_w1:", sum(x["w1"] for x in stats) / len(stats))
        print("mean_w2:", sum(x["w2"] for x in stats) / len(stats))


if __name__ == "__main__":
    main()
