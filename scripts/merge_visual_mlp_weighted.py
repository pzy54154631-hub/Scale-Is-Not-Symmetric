import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


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


def is_visual_mlp(key):
    return ".visual.blocks." in key and ".mlp." in key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--chart", required=True)
    p.add_argument("--average", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--w-chart", type=float, required=True)
    p.add_argument("--target-r", type=int, default=16)
    p.add_argument("--target-alpha", type=int, default=32)
    args = p.parse_args()

    w_chart = args.w_chart
    w_text = 1.0 - w_chart

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

    rt = int(ct.get("r", 16))
    rc = int(cc.get("r", 16))
    at = float(ct.get("lora_alpha", rt))
    ac = float(cc.get("lora_alpha", rc))

    scale_t = at / rt
    scale_c = ac / rc
    out_scale = args.target_alpha / args.target_r

    new_state = dict(sa)
    replaced_pairs = 0

    for a_key in st:
        if ".lora_A." not in a_key:
            continue

        if not is_visual_mlp(a_key):
            continue

        b_key = a_key.replace(".lora_A.", ".lora_B.")

        if a_key not in sc or b_key not in st or b_key not in sc:
            continue

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
        replaced_pairs += 1

    cfg["r"] = args.target_r
    cfg["lora_alpha"] = args.target_alpha
    cfg["rank_pattern"] = {}
    cfg["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    for f in avg_dir.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved:", out_dir)
    print("w_text:", w_text)
    print("w_chart:", w_chart)
    print("replaced_visual_mlp_pairs:", replaced_pairs)


if __name__ == "__main__":
    main()
