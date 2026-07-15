import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def find_adapter_file(d):
    d = Path(d)
    p = d / "adapter_model.safetensors"
    if p.exists():
        return p
    raise FileNotFoundError(f"adapter_model.safetensors not found in {d}")


def compress_raw_product(B, A, target_r):
    dtype = B.dtype
    device = B.device

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


def lora_norm(B, A, scale):
    B = B.float().cpu()
    A = A.float().cpu()
    val = torch.sum((B.T @ B) * (A @ A.T).T) * scale * scale
    return torch.sqrt(torch.clamp(val, min=1e-12)).item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--residual-adapter", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--target-r", type=int, default=16)
    parser.add_argument("--target-alpha", type=int, default=32)

    args = parser.parse_args()

    base_dir = Path(args.base_adapter)
    res_dir = Path(args.residual_adapter)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base adapter:", base_dir)
    base_state = load_file(str(find_adapter_file(base_dir)))

    print("Loading residual adapter:", res_dir)
    res_state = load_file(str(find_adapter_file(res_dir)))

    base_cfg = json.loads((base_dir / "adapter_config.json").read_text())
    res_cfg = json.loads((res_dir / "adapter_config.json").read_text())

    base_r = int(base_cfg.get("r", args.target_r))
    base_alpha = float(base_cfg.get("lora_alpha", base_r))
    base_scale = base_alpha / base_r

    res_r = int(res_cfg.get("r", 4))
    res_alpha = float(res_cfg.get("lora_alpha", res_r))
    res_scale = res_alpha / res_r

    out_scale = args.target_alpha / args.target_r

    new_state = dict(base_state)
    stats = []

    merged = 0
    skipped = 0

    for a_key in base_state.keys():
        if ".lora_A." not in a_key:
            continue

        b_key = a_key.replace(".lora_A.", ".lora_B.")

        if b_key not in base_state:
            continue

        if a_key not in res_state or b_key not in res_state:
            skipped += 1
            continue

        A_base = base_state[a_key]
        B_base = base_state[b_key]
        A_res = res_state[a_key]
        B_res = res_state[b_key]

        # We want:
        # out_scale * B_final A_final
        # ~= base_scale * B_base A_base + eta * res_scale * B_res A_res
        c_base = max(base_scale / out_scale, 0.0)
        c_res = max(args.eta * res_scale / out_scale, 0.0)

        B_cat = torch.cat(
            [
                math.sqrt(c_base) * B_base,
                math.sqrt(c_res) * B_res,
            ],
            dim=1,
        )

        A_cat = torch.cat(
            [
                math.sqrt(c_base) * A_base,
                math.sqrt(c_res) * A_res,
            ],
            dim=0,
        )

        B_new, A_new = compress_raw_product(B_cat, A_cat, args.target_r)

        new_state[a_key] = A_new
        new_state[b_key] = B_new

        base_norm = lora_norm(B_base, A_base, base_scale)
        res_norm = lora_norm(B_res, A_res, res_scale)

        stats.append({
            "module": a_key,
            "base_norm": base_norm,
            "residual_norm": res_norm,
            "eta": args.eta,
            "residual_over_base": res_norm / (base_norm + 1e-12),
        })

        merged += 1

    out_cfg = dict(base_cfg)
    out_cfg["r"] = args.target_r
    out_cfg["lora_alpha"] = args.target_alpha
    out_cfg["rank_pattern"] = {}
    out_cfg["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(out_cfg, indent=2))
    (out_dir / "mirage_rf_merge_stats.json").write_text(json.dumps(stats, indent=2))

    for f in base_dir.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved:", out_dir)
    print("merged modules:", merged)
    print("skipped modules:", skipped)

    if stats:
        print("mean residual/base norm:",
              sum(x["residual_over_base"] for x in stats) / len(stats))
        print("max residual/base norm:",
              max(x["residual_over_base"] for x in stats))


if __name__ == "__main__":
    main()
