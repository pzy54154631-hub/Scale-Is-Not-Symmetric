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
    # example: base_model.model.model.visual.blocks.12.mlp.up_proj.lora_A.weight
    parts = key.split(".")
    if "blocks" not in parts:
        return None
    i = parts.index("blocks")
    try:
        return int(parts[i + 1])
    except Exception:
        return None


def depth_scale(key: str, shallow=0.95, middle=1.00, deep=1.05):
    bid = get_block_id(key)
    if bid is None:
        return 1.0
    # Qwen2.5-VL visual blocks usually around 32 layers
    if bid < 10:
        return shallow
    if bid < 22:
        return middle
    return deep


def compress_raw_product(B, A, target_r):
    dtype, device = B.dtype, B.device
    Bf = B.float().cpu()
    Af = A.float().cpu()

    Qb, Rb = torch.linalg.qr(Bf, mode="reduced")
    Qa, Ra = torch.linalg.qr(Af.T, mode="reduced")

    small = Rb @ Ra.T
    U, S, Vh = torch.linalg.svd(small, full_matrices=False)

    r = min(target_r, S.numel())
    S = torch.clamp(S[:r], min=0)
    root = torch.sqrt(S)

    Bn = Qb @ (U[:, :r] * root.unsqueeze(0))
    An = (root.unsqueeze(1) * Vh[:r, :]) @ Qa.T

    return Bn.to(device=device, dtype=dtype), An.to(device=device, dtype=dtype)


def lora_to_small_svd(B, A):
    """
    Computes thin SVD of raw LoRA product B@A without materializing full matrix.
    B: [out, r], A: [r, in]
    Returns U_full [out,k], S [k], Vh_full [k,in].
    """
    Bf = B.float().cpu()
    Af = A.float().cpu()

    Qb, Rb = torch.linalg.qr(Bf, mode="reduced")
    Qa, Ra = torch.linalg.qr(Af.T, mode="reduced")

    small = Rb @ Ra.T
    U, S, Vh = torch.linalg.svd(small, full_matrices=False)

    U_full = Qb @ U
    Vh_full = Vh @ Qa.T

    return U_full, S, Vh_full


def make_direction_matrix(U, Vh, j):
    # Only used for small r direction comparison through factor dot identities.
    # Direction = u_j v_j^T, unit Frobenius if u and v are unit.
    return U[:, j], Vh[j, :]


def direction_cos(u1, v1, u2, v2):
    # <u1 v1^T, u2 v2^T> = <u1,u2> * <v1,v2>
    return torch.dot(u1, u2).item() * torch.dot(v1, v2).item()


def orient_to_reference(u, v, u_ref, v_ref):
    # Since u v^T == (-u)(-v)^T, choose sign closer to reference.
    c = direction_cos(u, v, u_ref, v_ref)
    if c < 0:
        return -u, v, -c
    return u, v, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--chart", required=True)
    p.add_argument("--average", required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--target-r", type=int, default=16)
    p.add_argument("--target-alpha", type=int, default=32)

    # KAS hyperparams
    p.add_argument("--shared-threshold", type=float, default=0.35)
    p.add_argument("--conflict-threshold", type=float, default=-0.20)
    p.add_argument("--dominance-ratio", type=float, default=1.15)

    p.add_argument("--shared-scale", type=float, default=1.00)
    p.add_argument("--dominant-scale", type=float, default=1.00)
    p.add_argument("--conflict-scale", type=float, default=0.00)

    p.add_argument("--chart-bias", type=float, default=1.00)
    p.add_argument("--text-bias", type=float, default=1.00)

    p.add_argument("--shallow-scale", type=float, default=0.95)
    p.add_argument("--middle-scale", type=float, default=1.00)
    p.add_argument("--deep-scale", type=float, default=1.05)

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

        A_t, B_t = st[a_key], st[b_key]
        A_c, B_c = sc[a_key], sc[b_key]

        # Raw LoRA product is B@A. Effective product has scale alpha/r.
        U_t, S_t, Vh_t = lora_to_small_svd(B_t, A_t)
        U_c, S_c, Vh_c = lora_to_small_svd(B_c, A_c)

        k = min(S_t.numel(), S_c.numel(), args.target_r)

        B_parts = []
        A_parts = []

        n_shared = 0
        n_conflict = 0
        n_text_dom = 0
        n_chart_dom = 0

        ds = depth_scale(a_key, args.shallow_scale, args.middle_scale, args.deep_scale)

        for j in range(k):
            ut, vt = make_direction_matrix(U_t, Vh_t, j)
            uc, vc = make_direction_matrix(U_c, Vh_c, j)

            uc, vc, cos = orient_to_reference(uc, vc, ut, vt)

            sig_t = S_t[j].item() * scale_t * args.text_bias
            sig_c = S_c[j].item() * scale_c * args.chart_bias

            # Decide direction behavior.
            if cos >= args.shared_threshold:
                # Aligned: merge both singular directions.
                weight = args.shared_scale * ds
                sig = 0.5 * (sig_t + sig_c) * weight
                u = (ut + uc)
                v = (vt + vc)

                un = torch.norm(u) + 1e-12
                vn = torch.norm(v) + 1e-12
                u = u / un
                v = v / vn

                n_shared += 1

            elif cos <= args.conflict_threshold:
                # Conflict: either prune or keep a very damped dominant direction.
                weight = args.conflict_scale * ds

                if sig_c >= sig_t:
                    sig = sig_c * weight
                    u, v = uc, vc
                    n_chart_dom += 1
                else:
                    sig = sig_t * weight
                    u, v = ut, vt
                    n_text_dom += 1

                n_conflict += 1

            else:
                # Ambiguous: keep dominant task if sufficiently stronger.
                weight = args.dominant_scale * ds

                if sig_c >= args.dominance_ratio * sig_t:
                    sig = sig_c * weight
                    u, v = uc, vc
                    n_chart_dom += 1
                elif sig_t >= args.dominance_ratio * sig_c:
                    sig = sig_t * weight
                    u, v = ut, vt
                    n_text_dom += 1
                else:
                    # If neither dominates, fall back to average-like direction.
                    sig = 0.5 * (sig_t + sig_c) * weight
                    u = (ut + uc)
                    v = (vt + vc)
                    u = u / (torch.norm(u) + 1e-12)
                    v = v / (torch.norm(v) + 1e-12)
                    n_shared += 1

            # We need raw product so that out_scale * Bn@An approximates effective delta.
            raw_sig = max(sig / out_scale, 0.0)
            if raw_sig <= 0:
                continue

            root = math.sqrt(raw_sig)
            B_parts.append(root * u.reshape(-1, 1))
            A_parts.append(root * v.reshape(1, -1))

        if not B_parts:
            continue

        Bcat = torch.cat(B_parts, dim=1)
        Acat = torch.cat(A_parts, dim=0)

        Bn, An = compress_raw_product(Bcat.to(B_t.device, B_t.dtype), Acat.to(A_t.device, A_t.dtype), args.target_r)

        new_state[a_key] = An
        new_state[b_key] = Bn

        stats.append({
            "module": a_key.replace(".lora_A.", "."),
            "block": get_block_id(a_key),
            "depth_scale": ds,
            "shared": n_shared,
            "conflict": n_conflict,
            "text_dom": n_text_dom,
            "chart_dom": n_chart_dom,
        })

    cfg["r"] = args.target_r
    cfg["lora_alpha"] = args.target_alpha
    cfg["rank_pattern"] = {}
    cfg["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "mirage_kas_stats.json").write_text(json.dumps(stats, indent=2))

    for f in avg_dir.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved:", out_dir)
    print("visual_mlp_pairs:", len(stats))
    if stats:
        for key in ["shared", "conflict", "text_dom", "chart_dom"]:
            print("mean_" + key + ":", sum(x[key] for x in stats) / len(stats))


if __name__ == "__main__":
    main()
