import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def get_block_id(key):
    parts = key.split(".")
    if "blocks" not in parts:
        return None
    i = parts.index("blocks")
    try:
        return int(parts[i + 1])
    except Exception:
        return None


def get_depth_group(key):
    bid = get_block_id(key)
    if bid is None:
        return "other"
    if bid < 10:
        return "shallow"
    if bid < 22:
        return "middle"
    return "deep"


def get_proj_group(key):
    if ".gate_proj." in key:
        return "gate"
    if ".up_proj." in key:
        return "up"
    if ".down_proj." in key:
        return "down"
    return "other"


def is_visual_mlp(key):
    return ".visual.blocks." in key and ".mlp." in key


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--chart", required=True)
    p.add_argument("--average", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--target-r", type=int, default=16)
    p.add_argument("--target-alpha", type=int, default=32)

    # default lambda
    p.add_argument("--default-lambda", type=float, default=0.10)

    # group lambdas
    p.add_argument("--shallow-gate", type=float, default=None)
    p.add_argument("--shallow-up", type=float, default=None)
    p.add_argument("--shallow-down", type=float, default=None)

    p.add_argument("--middle-gate", type=float, default=None)
    p.add_argument("--middle-up", type=float, default=None)
    p.add_argument("--middle-down", type=float, default=None)

    p.add_argument("--deep-gate", type=float, default=None)
    p.add_argument("--deep-up", type=float, default=None)
    p.add_argument("--deep-down", type=float, default=None)

    args = p.parse_args()

    text_dir = Path(args.text)
    chart_dir = Path(args.chart)
    avg_dir = Path(args.average)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    st = load_file(str(text_dir / "adapter_model.safetensors"))
    sc = load_file(str(chart_dir / "adapter_model.safetensors"))
    sa = load_file(str(avg_dir / "adapter_model.safetensors"))

    ct = json.loads((text_dir / "adapter_config.json").read_text())
    cc = json.loads((chart_dir / "adapter_config.json").read_text())
    cfg = json.loads((avg_dir / "adapter_config.json").read_text())

    rt = int(ct.get("r", args.target_r))
    rc = int(cc.get("r", args.target_r))
    at = float(ct.get("lora_alpha", rt))
    ac = float(cc.get("lora_alpha", rc))

    scale_t = at / rt
    scale_c = ac / rc
    out_scale = args.target_alpha / args.target_r

    group_lambda = {
        "shallow_gate": args.shallow_gate,
        "shallow_up": args.shallow_up,
        "shallow_down": args.shallow_down,
        "middle_gate": args.middle_gate,
        "middle_up": args.middle_up,
        "middle_down": args.middle_down,
        "deep_gate": args.deep_gate,
        "deep_up": args.deep_up,
        "deep_down": args.deep_down,
    }

    new_state = dict(sa)
    stats = []

    for a_key in st:
        if ".lora_A." not in a_key:
            continue
        if not is_visual_mlp(a_key):
            continue

        b_key = a_key.replace(".lora_A.", ".lora_B.")
        if a_key not in sc or b_key not in st or b_key not in sc:
            continue

        depth = get_depth_group(a_key)
        proj = get_proj_group(a_key)
        g = f"{depth}_{proj}"

        lam = group_lambda.get(g)
        if lam is None:
            lam = args.default_lambda

        lam = max(0.0, min(0.20, lam))

        w_text = 0.5 - lam
        w_chart = 0.5 + lam

        A_t, B_t = st[a_key], st[b_key]
        A_c, B_c = sc[a_key], sc[b_key]

        c_t = max(w_text * scale_t / out_scale, 0.0)
        c_c = max(w_chart * scale_c / out_scale, 0.0)

        Bcat = torch.cat(
            [
                math.sqrt(c_t) * B_t,
                math.sqrt(c_c) * B_c,
            ],
            dim=1,
        )

        Acat = torch.cat(
            [
                math.sqrt(c_t) * A_t,
                math.sqrt(c_c) * A_c,
            ],
            dim=0,
        )

        Bn, An = compress(Bcat, Acat, args.target_r)
        new_state[a_key] = An
        new_state[b_key] = Bn

        stats.append({
            "module": a_key.replace(".lora_A.", "."),
            "block": get_block_id(a_key),
            "depth": depth,
            "proj": proj,
            "group": g,
            "lambda": lam,
            "w_text": w_text,
            "w_chart": w_chart,
        })

    cfg["r"] = args.target_r
    cfg["lora_alpha"] = args.target_alpha
    cfg["rank_pattern"] = {}
    cfg["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "mirage_pas_group_stats.json").write_text(json.dumps(stats, indent=2))

    for f in avg_dir.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved:", out_dir)
    print("visual_mlp_pairs:", len(stats))

    from collections import defaultdict
    bucket = defaultdict(list)
    for x in stats:
        bucket[x["group"]].append(x["lambda"])

    for g, vals in sorted(bucket.items()):
        print(g, "n=", len(vals), "lambda=", sum(vals) / len(vals))


if __name__ == "__main__":
    main()
