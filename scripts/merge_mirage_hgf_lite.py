import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def module_type(k):
    if ".visual.blocks." in k and ".mlp." in k:
        return "visual_mlp"
    if ".visual.blocks." in k and (".attn." in k or ".self_attn." in k):
        return "visual_attn"
    if ".language_model." in k and ".mlp." in k:
        return "lang_mlp"
    if ".language_model." in k and ".self_attn." in k:
        return "lang_attn"
    return "other"


def module_prior(k):
    t = module_type(k)
    if t == "visual_mlp":
        return 1.00
    if t == "visual_attn":
        return 0.70
    if t == "lang_mlp":
        return 0.30
    if t == "lang_attn":
        return 0.20
    return 0.10


def lora_norm_sq(A, B, scale):
    # || scale * B A ||_F^2 = scale^2 * tr((B^T B)(A A^T))
    A = A.float().cpu()
    B = B.float().cpu()
    BtB = B.T @ B
    AAt = A @ A.T
    val = torch.sum(BtB * AAt.T).item()
    return max(scale * scale * val, 0.0)


def lora_inner(A1, B1, s1, A2, B2, s2):
    # <s1 B1A1, s2 B2A2>
    A1 = A1.float().cpu()
    B1 = B1.float().cpu()
    A2 = A2.float().cpu()
    B2 = B2.float().cpu()
    BtB = B1.T @ B2
    A2A1t = A2 @ A1.T
    val = torch.trace(BtB @ A2A1t).item()
    return s1 * s2 * val


def compress(B, A, target_r):
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

    Bn = Qb @ (U[:, :r] * root.unsqueeze(0))
    An = (root.unsqueeze(1) * Vh[:r, :]) @ Qa.T

    return Bn.to(device=device, dtype=dtype), An.to(device=device, dtype=dtype)


def choose_weight(k, cos, delta, norm_ratio, mode):
    prior = module_prior(k)

    if mode == "safe":
        base_lam = 0.055
        max_w = 0.62
    elif mode == "balanced":
        base_lam = 0.075
        max_w = 0.65
    elif mode == "chart":
        base_lam = 0.095
        max_w = 0.68
    else:
        base_lam = 0.070
        max_w = 0.64

    # conflict 越高，越回退 average
    conflict = sigmoid((-cos - 0.05) / 0.12) + sigmoid((delta - 0.80) / 0.15)
    conflict = clamp(conflict / 2.0, 0.0, 1.0)

    # 共享方向越强，越允许非 average 融合
    agree_gate = sigmoid((cos + 0.05) / 0.15) * sigmoid((0.85 - delta) / 0.20)

    # Expert 1: average
    w_avg = 0.50

    # Expert 2: probabilistic norm fusion
    # chart adapter norm 明显更强时略偏 chart，但强度受限
    w_pf = 0.50 + 0.06 * math.tanh(norm_ratio) + 0.04 * prior * agree_gate
    w_pf = clamp(w_pf, 0.42, max_w)

    # Expert 3: PAS mild correction
    lam = base_lam * prior * agree_gate
    w_pas = clamp(0.50 + lam, 0.50, max_w)

    # Expert 4: conservative conflict fallback
    w_cons = 0.50 if conflict > 0.45 else w_pas

    # hidden-style routing weights
    r_avg = 1.00 + 1.50 * conflict
    r_pf = 0.40 + max(cos, 0.0)
    r_pas = 0.25 + prior * agree_gate
    r_cons = 0.40 + conflict

    total = r_avg + r_pf + r_pas + r_cons

    w_chart = (
        r_avg * w_avg +
        r_pf * w_pf +
        r_pas * w_pas +
        r_cons * w_cons
    ) / total

    # 安全边界：不能过度偏 ChartQA
    w_chart = clamp(w_chart, 0.45, max_w)
    return w_chart, {
        "prior": prior,
        "conflict": conflict,
        "agree_gate": agree_gate,
        "w_avg": w_avg,
        "w_pf": w_pf,
        "w_pas": w_pas,
        "w_cons": w_cons,
        "r_avg": r_avg,
        "r_pf": r_pf,
        "r_pas": r_pas,
        "r_cons": r_cons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--chart", required=True)
    ap.add_argument("--average", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", default="safe", choices=["safe", "balanced", "chart"])
    ap.add_argument("--target-r", type=int, default=16)
    ap.add_argument("--target-alpha", type=int, default=32)
    args = ap.parse_args()

    text_dir = Path(args.text)
    chart_dir = Path(args.chart)
    avg_dir = Path(args.average)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    st = load_file(str(text_dir / "adapter_model.safetensors"))
    sc = load_file(str(chart_dir / "adapter_model.safetensors"))
    sa = load_file(str(avg_dir / "adapter_model.safetensors"))

    cfg_t = json.loads((text_dir / "adapter_config.json").read_text())
    cfg_c = json.loads((chart_dir / "adapter_config.json").read_text())
    cfg = json.loads((avg_dir / "adapter_config.json").read_text())

    rt = int(cfg_t.get("r", args.target_r))
    rc = int(cfg_c.get("r", args.target_r))
    at = float(cfg_t.get("lora_alpha", rt))
    ac = float(cfg_c.get("lora_alpha", rc))

    scale_t = at / rt
    scale_c = ac / rc
    out_scale = args.target_alpha / args.target_r

    new_state = dict(sa)
    stats = []

    for a_key in st.keys():
        if ".lora_A." not in a_key:
            continue

        b_key = a_key.replace(".lora_A.", ".lora_B.")

        if a_key not in sc or b_key not in st or b_key not in sc:
            continue

        A_t, B_t = st[a_key], st[b_key]
        A_c, B_c = sc[a_key], sc[b_key]

        nt2 = lora_norm_sq(A_t, B_t, scale_t)
        nc2 = lora_norm_sq(A_c, B_c, scale_c)
        inner = lora_inner(A_t, B_t, scale_t, A_c, B_c, scale_c)

        nt = math.sqrt(max(nt2, 1e-12))
        nc = math.sqrt(max(nc2, 1e-12))

        cos = inner / (nt * nc + 1e-12)
        cos = clamp(cos, -1.0, 1.0)

        diff2 = max(nt2 + nc2 - 2.0 * inner, 0.0)
        delta = math.sqrt(diff2) / (nt + nc + 1e-12)

        norm_ratio = math.log((nc + 1e-8) / (nt + 1e-8))

        w_chart, meta = choose_weight(a_key, cos, delta, norm_ratio, args.mode)
        w_text = 1.0 - w_chart

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
            "module_type": module_type(a_key),
            "cos": cos,
            "delta": delta,
            "norm_ratio_chart_over_text": norm_ratio,
            "w_text": w_text,
            "w_chart": w_chart,
            **meta,
        })

    cfg["r"] = args.target_r
    cfg["lora_alpha"] = args.target_alpha
    cfg["rank_pattern"] = {}
    cfg["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "mirage_hgf_lite_stats.json").write_text(json.dumps(stats, indent=2))

    for f in avg_dir.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved:", out_dir)
    print("mode:", args.mode)
    print("merged pairs:", len(stats))

    if stats:
        print("mean w_chart:", sum(x["w_chart"] for x in stats) / len(stats))
        for mt in ["visual_mlp", "visual_attn", "lang_mlp", "lang_attn", "other"]:
            xs = [x for x in stats if x["module_type"] == mt]
            if xs:
                print(mt, "n=", len(xs), "mean_w_chart=", sum(x["w_chart"] for x in xs) / len(xs))


if __name__ == "__main__":
    main()
