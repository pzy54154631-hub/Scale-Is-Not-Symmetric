import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def compress_lora_pair(B, A, target_r=16, scale_ratio=1.0):
    """
    Compress raw LoRA product B @ A to target rank.

    B: [out_dim, r_in]
    A: [r_in, in_dim]

    We avoid materializing full B @ A.
    Use:
      B = Qb Rb
      A.T = Qa Ra
      B A = Qb (Rb Ra.T) Qa.T
    Then SVD only on the small r_in x r_in matrix.
    """
    dtype = B.dtype
    device = B.device

    Bf = B.float().cpu()
    Af = A.float().cpu()

    r_in = A.shape[0]
    if r_in <= target_r and abs(scale_ratio - 1.0) < 1e-12:
        return B.clone(), A.clone()

    Qb, Rb = torch.linalg.qr(Bf, mode="reduced")
    Qa, Ra = torch.linalg.qr(Af.T, mode="reduced")

    small = (Rb @ Ra.T) * float(scale_ratio)

    U, S, Vh = torch.linalg.svd(small, full_matrices=False)

    r = min(target_r, S.numel())
    S = torch.clamp(S[:r], min=0)
    sqrtS = torch.sqrt(S)

    B_new = Qb @ (U[:, :r] * sqrtS.unsqueeze(0))
    A_new = (sqrtS.unsqueeze(1) * Vh[:r, :]) @ Qa.T

    return B_new.to(device=device, dtype=dtype), A_new.to(device=device, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input adapter directory")
    parser.add_argument("--output", required=True, help="Output adapter directory")
    parser.add_argument("--target-r", type=int, default=16)
    parser.add_argument("--target-alpha", type=int, default=32)
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_model = in_dir / "adapter_model.safetensors"
    in_config = in_dir / "adapter_config.json"

    if not in_model.exists():
        raise FileNotFoundError(in_model)
    if not in_config.exists():
        raise FileNotFoundError(in_config)

    cfg = json.loads(in_config.read_text())

    state = load_file(str(in_model))
    new_state = {}

    old_cfg_r = int(cfg.get("r", args.target_r))
    old_alpha = float(cfg.get("lora_alpha", old_cfg_r))

    processed_pairs = 0

    for k, v in state.items():
        if ".lora_A." not in k:
            continue

        a_key = k
        b_key = k.replace(".lora_A.", ".lora_B.")

        if b_key not in state:
            continue

        A = state[a_key]
        B = state[b_key]

        r_in = A.shape[0]

        old_scale = old_alpha / float(r_in)
        new_scale = float(args.target_alpha) / float(args.target_r)

        # We want:
        # new_scale * (B_new @ A_new) ~= old_scale * (B @ A)
        # so raw target product is (old_scale / new_scale) * B @ A
        scale_ratio = old_scale / new_scale

        B_new, A_new = compress_lora_pair(
            B,
            A,
            target_r=args.target_r,
            scale_ratio=scale_ratio,
        )

        new_state[a_key] = A_new
        new_state[b_key] = B_new
        processed_pairs += 1

    # Copy non-LoRA tensors if any.
    for k, v in state.items():
        if k not in new_state:
            if ".lora_A." in k or ".lora_B." in k:
                continue
            new_state[k] = v

    cfg["r"] = args.target_r
    cfg["lora_alpha"] = args.target_alpha

    # Avoid inconsistent per-module rank patterns.
    if "rank_pattern" in cfg:
        cfg["rank_pattern"] = {}
    if "alpha_pattern" in cfg:
        cfg["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    # Copy tokenizer / processor files if they exist.
    for p in in_dir.iterdir():
        if p.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / p.name
        if p.is_file() and not dst.exists():
            shutil.copy2(p, dst)

    print("Input:", in_dir)
    print("Output:", out_dir)
    print("processed_lora_pairs:", processed_pairs)
    print("target_r:", args.target_r)
    print("target_alpha:", args.target_alpha)
    print("Done.")


if __name__ == "__main__":
    main()
