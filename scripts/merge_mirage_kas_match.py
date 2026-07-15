import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def is_visual_mlp(key):
    return ".visual.blocks." in key and ".mlp." in key


def get_block_id(key):
    parts = key.split(".")
    if "blocks" not in parts:
        return None
    i = parts.index("blocks")
    try:
        return int(parts[i + 1])
    except Exception:
        return None


def depth_scale(key, shallow, middle, deep):
    bid = get_block_id(key)
    if bid is None:
        return 1.0
    if bid < 10:
        return shallow
    if bid < 22:
        return middle
    return deep


def lora_svd(B, A):
    B = B.float().cpu()
    A = A.float().cpu()

    Qb, Rb = torch.linalg.qr(B, mode="reduced")
    Qa, Ra = torch.linalg.qr(A.T, mode="reduced")

    small = Rb @ Ra.T
    U, S, Vh = torch.linalg.svd(small, full_matrices=False)

    U = Qb @ U
    Vh = Vh @ Qa.T
    return U, S, Vh


def dir_cos(U1, V1, j, U2, V2, k):
    return torch.dot(U1[:, j], U2[:, k]).item() * torch.dot(V1[j, :], V2[k, :]).item()


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

    p.add_argument("--shared-threshold", type=float, default=0.30)
    p.add_argument("--conflict-threshold", type=float, default=-0.15)
    p.add_argument("--dominance-ratio", type=float, default=1.08)

    p.add_argument("--shared-scale", type=float, default=1.00)
    p.add_argument("--dominant-scale", type=float, default=1.03)
    p.add_argument("--conflict-scale", type=float, default=0.00)

    p.add_argument("--text-bias", type=float, default=1.00)
    p.add_argument("--chart-bias", type=float, default=1.03)

    p.add_argument("--shallow-scale", type=float, default=0.95)
    p.add_argument("--middle-scale", type=float, default=1.00)
    p.add_argument("--deep-scale", type=float, default=1.08)

    p.add_argument("--unmatched-chart-scale", type=float, default=0.00)
    p.add_argument("--unmatched-text-scale", type=float, default=0.00)

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

        Ut, St, Vt = lora_svd(B_t, A_t)
        Uc, Sc, Vc = lora_svd(B_c, A_c)

        rt0 = min(args.target_r, St.numel())
        rc0 = min(args.target_r, Sc.numel())

        candidates = []
        for j in range(rt0):
            for k in range(rc0):
                c = dir_cos(Ut, Vt, j, Uc, Vc, k)
                candidates.append((abs(c), c, j, k))

        candidates.sort(reverse=True)

        used_t = set()
        used_c = set()
        matches = []

        for _, c, j, k in candidates:
            if j in used_t or k in used_c:
                continue
            used_t.add(j)
            used_c.add(k)
            matches.append((j, k, c))
            if len(matches) >= args.target_r:
                break

        ds = depth_scale(a_key, args.shallow_scale, args.middle_scale, args.deep_scale)

        B_parts = []
        A_parts = []

        n_shared = 0
        n_conflict = 0
        n_text_dom = 0
        n_chart_dom = 0

        for j, k, cos in matches:
            ut, vt = Ut[:, j], Vt[j, :]
            uc, vc = Uc[:, k], Vc[k, :]

            sig_t = St[j].item() * scale_t * args.text_bias
            sig_c = Sc[k].item() * scale_c * args.chart_bias

            if cos >= args.shared_threshold:
                sig = 0.5 * (sig_t + sig_c) * args.shared_scale * ds
                u = ut + uc
                v = vt + vc
                u = u / (torch.norm(u) + 1e-12)
                v = v / (torch.norm(v) + 1e-12)
                n_shared += 1

            elif cos <= args.conflict_threshold:
                n_conflict += 1
                if args.conflict_scale <= 0:
                    continue

                if sig_c >= sig_t:
                    sig = sig_c * args.conflict_scale * ds
                    u, v = uc, vc
                    n_chart_dom += 1
                else:
                    sig = sig_t * args.conflict_scale * ds
                    u, v = ut, vt
                    n_text_dom += 1

            else:
                if sig_c >= args.dominance_ratio * sig_t:
                    sig = sig_c * args.dominant_scale * ds
                    u, v = uc, vc
                    n_chart_dom += 1
                elif sig_t >= args.dominance_ratio * sig_c:
                    sig = sig_t * args.dominant_scale * ds
                    u, v = ut, vt
                    n_text_dom += 1
                else:
                    sig = 0.5 * (sig_t + sig_c) * args.shared_scale * ds
                    u = ut + uc
                    v = vt + vc
                    u = u / (torch.norm(u) + 1e-12)
                    v = v / (torch.norm(v) + 1e-12)
                    n_shared += 1

            raw_sig = max(sig / out_scale, 0.0)
            if raw_sig <= 0:
                continue

            root = math.sqrt(raw_sig)
            B_parts.append(root * u.reshape(-1, 1))
            A_parts.append(root * v.reshape(1, -1))

        # Optional unmatched directions. Usually keep 0.00 first.
        for j in range(rt0):
            if j in used_t or args.unmatched_text_scale <= 0:
                continue
            sig = St[j].item() * scale_t * args.unmatched_text_scale * ds / out_scale
            root = math.sqrt(max(sig, 0.0))
            if root > 0:
                B_parts.append(root * Ut[:, j].reshape(-1, 1))
                A_parts.append(root * Vt[j, :].reshape(1, -1))

        for k in range(rc0):
            if k in used_c or args.unmatched_chart_scale <= 0:
                continue
            sig = Sc[k].item() * scale_c * args.unmatched_chart_scale * ds / out_scale
            root = math.sqrt(max(sig, 0.0))
            if root > 0:
                B_parts.append(root * Uc[:, k].reshape(-1, 1))
                A_parts.append(root * Vc[k, :].reshape(1, -1))

        if not B_parts:
            continue

        Bcat = torch.cat(B_parts, dim=1).to(B_t.device, B_t.dtype)
        Acat = torch.cat(A_parts, dim=0).to(A_t.device, A_t.dtype)

        Bn, An = compress(Bcat, Acat, args.target_r)

        new_state[a_key] = An
        new_state[b_key] = Bn

        stats.append({
            "module": a_key.replace(".lora_A.", "."),
            "block": get_block_id(a_key),
            "num_matches": len(matches),
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
    (out_dir / "mirage_kas_match_stats.json").write_text(json.dumps(stats, indent=2))

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
