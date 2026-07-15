import argparse, json, math, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

def sigmoid(x):
    x = max(min(float(x), 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-x))

def norm_lora(B, A, scale):
    B = B.float().cpu()
    A = A.float().cpu()
    return torch.sqrt(torch.clamp(torch.sum((B.T @ B) * (A @ A.T).T) * scale * scale, min=1e-12)).item()

def cross_lora(B1, A1, s1, B2, A2, s2):
    B1 = B1.float().cpu()
    A1 = A1.float().cpu()
    B2 = B2.float().cpu()
    A2 = A2.float().cpu()
    return (torch.sum((B1.T @ B2) * (A2 @ A1.T).T) * s1 * s2).item()

def compress(B, A, target_r):
    dtype, device = B.dtype, B.device
    B = B.float().cpu()
    A = A.float().cpu()
    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")
    U, S, Vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    r = min(target_r, S.numel())
    S = torch.clamp(S[:r], min=0)
    root = torch.sqrt(S)
    Bn = Qb @ (U[:, :r] * root.unsqueeze(0))
    An = (root.unsqueeze(1) * Vh[:r, :]) @ Qa.T
    return Bn.to(device=device, dtype=dtype), An.to(device=device, dtype=dtype)

def module_type(k):
    if ".visual.blocks." in k and ".mlp." in k:
        return "visual_mlp"
    if ".visual.blocks." in k:
        return "visual_attn"
    if ".language_model.layers." in k:
        return "language"
    return "other"

def prior(k, args):
    t = module_type(k)
    if t == "visual_mlp":
        return args.prior_visual_mlp
    if t == "visual_attn":
        return args.prior_visual_attn
    if t == "language":
        return args.prior_language
    return args.prior_other

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-a", required=True)
    p.add_argument("--adapter-b", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-r", type=int, default=16)
    p.add_argument("--target-alpha", type=int, default=32)

    p.add_argument("--prior-visual-mlp", type=float, default=1.0)
    p.add_argument("--prior-visual-attn", type=float, default=0.0)
    p.add_argument("--prior-language", type=float, default=0.0)
    p.add_argument("--prior-other", type=float, default=0.0)

    p.add_argument("--energy-power", type=float, default=1.0)
    p.add_argument("--tau-cos", type=float, default=0.0)
    p.add_argument("--temp-cos", type=float, default=0.25)
    p.add_argument("--tau-delta", type=float, default=0.45)
    p.add_argument("--temp-delta", type=float, default=0.20)
    p.add_argument("--delta-weight", type=float, default=1.0)
    p.add_argument("--risk-weight", type=float, default=0.7)
    p.add_argument("--dominance-weight", type=float, default=0.3)
    p.add_argument("--gate-sharpness", type=float, default=1.0)
    p.add_argument("--conflict-damping", type=float, default=0.3)
    p.add_argument("--synergy-gain", type=float, default=0.05)
    p.add_argument("--scale-min", type=float, default=0.70)
    p.add_argument("--scale-max", type=float, default=1.10)
    args = p.parse_args()

    da = Path(args.adapter_a)
    db = Path(args.adapter_b)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    sa = load_file(str(da / "adapter_model.safetensors"))
    sb = load_file(str(db / "adapter_model.safetensors"))
    ca = json.loads((da / "adapter_config.json").read_text())
    cb = json.loads((db / "adapter_config.json").read_text())

    ra = int(ca.get("r", args.target_r))
    rb = int(cb.get("r", args.target_r))
    aa = float(ca.get("lora_alpha", ra))
    ab = float(cb.get("lora_alpha", rb))
    scale_a = aa / ra
    scale_b = ab / rb
    out_scale = float(args.target_alpha) / float(args.target_r)

    new_state = {}
    stats = []

    for a_key in sa:
        if ".lora_A." not in a_key:
            continue

        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if a_key not in sb or b_key not in sa or b_key not in sb:
            continue

        A1, B1 = sa[a_key], sa[b_key]
        A2, B2 = sb[a_key], sb[b_key]

        E1 = norm_lora(B1, A1, scale_a)
        E2 = norm_lora(B2, A2, scale_b)
        cross = cross_lora(B1, A1, scale_a, B2, A2, scale_b)

        cos = cross / (E1 * E2 + 1e-12)
        cos = max(min(cos, 1.0), -1.0)

        diff_sq = max(E1 * E1 + E2 * E2 - 2 * cross, 0.0)
        delta = math.sqrt(diff_sq) / (E1 + E2 + 1e-12)

        e1p = (E1 + 1e-12) ** args.energy_power
        e2p = (E2 + 1e-12) ** args.energy_power
        P1 = e1p / (e1p + e2p + 1e-12)
        P2 = e2p / (e1p + e2p + 1e-12)

        dominance = abs(P1 - P2) * 2.0

        risk_logit = (
            (args.tau_cos - cos) / max(args.temp_cos, 1e-6)
            + args.delta_weight * (delta - args.tau_delta) / max(args.temp_delta, 1e-6)
        )
        risk = sigmoid(risk_logit)

        q_prior = prior(a_key, args)
        raw_q = args.risk_weight * risk + args.dominance_weight * dominance
        raw_q = max(0.0, min(1.0, raw_q))
        Q = q_prior * (raw_q ** args.gate_sharpness)
        Q = max(0.0, min(1.0, Q))

        W1 = (1.0 - Q) * 0.5 + Q * P1
        W2 = (1.0 - Q) * 0.5 + Q * P2

        gamma = 1.0 - args.conflict_damping * max(0.0, -cos) + args.synergy_gain * max(0.0, cos)
        gamma = max(args.scale_min, min(args.scale_max, gamma))

        W1 *= gamma
        W2 *= gamma

        c1 = max(W1 * scale_a / out_scale, 0.0)
        c2 = max(W2 * scale_b / out_scale, 0.0)

        Bcat = torch.cat([math.sqrt(c1) * B1, math.sqrt(c2) * B2], dim=1)
        Acat = torch.cat([math.sqrt(c1) * A1, math.sqrt(c2) * A2], dim=0)

        Bn, An = compress(Bcat, Acat, args.target_r)
        new_state[a_key] = An
        new_state[b_key] = Bn

        stats.append({
            "module": a_key.replace(".lora_A.", "."),
            "type": module_type(a_key),
            "cos": cos,
            "delta": delta,
            "risk": risk,
            "dominance": dominance,
            "Q": Q,
            "gamma": gamma,
            "W_textvqa": W1,
            "W_chartqa": W2,
        })

    for k, v in sa.items():
        if k not in new_state and ".lora_A." not in k and ".lora_B." not in k:
            new_state[k] = v

    cfg = dict(ca)
    cfg["r"] = args.target_r
    cfg["lora_alpha"] = args.target_alpha
    cfg["rank_pattern"] = {}
    cfg["alpha_pattern"] = {}

    save_file(new_state, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    (out / "mirage_v4_stats.json").write_text(json.dumps(stats, indent=2))

    for f in da.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved:", out)
    print("processed_pairs:", len(stats))
    if stats:
        for key in ["cos", "delta", "risk", "dominance", "Q", "gamma", "W_textvqa", "W_chartqa"]:
            print("mean_" + key + ":", sum(x[key] for x in stats) / len(stats))

if __name__ == "__main__":
    main()
