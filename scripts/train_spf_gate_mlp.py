import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# -----------------------------
# Basic utilities
# -----------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def module_type(key: str):
    if ".visual.blocks." in key and ".mlp." in key:
        return "visual_mlp"
    if ".visual.blocks." in key:
        return "visual_other"
    if ".language_model." in key and ".mlp." in key:
        return "lang_mlp"
    if ".language_model." in key:
        return "lang_other"
    return "other"


def type_onehot(key: str):
    t = module_type(key)
    return [
        1.0 if t == "visual_mlp" else 0.0,
        1.0 if t == "visual_other" else 0.0,
        1.0 if t == "lang_mlp" else 0.0,
        1.0 if t == "lang_other" else 0.0,
        1.0 if t == "other" else 0.0,
    ]


def get_block_id(key: str):
    parts = key.split(".")
    if "blocks" in parts:
        i = parts.index("blocks")
        try:
            return int(parts[i + 1])
        except Exception:
            return -1
    if "layers" in parts:
        i = parts.index("layers")
        try:
            return int(parts[i + 1])
        except Exception:
            return -1
    return -1


def normalize_depth(key: str):
    bid = get_block_id(key)
    if bid < 0:
        return 0.0
    return bid / 32.0


def lora_norm(B, A, scale):
    B = B.float().cpu()
    A = A.float().cpu()
    val = torch.sum((B.T @ B) * (A @ A.T).T) * scale * scale
    return torch.sqrt(torch.clamp(val, min=1e-12)).item()


def lora_cross(B1, A1, s1, B2, A2, s2):
    B1 = B1.float().cpu()
    A1 = A1.float().cpu()
    B2 = B2.float().cpu()
    A2 = A2.float().cpu()
    return (torch.sum((B1.T @ B2) * (A2 @ A1.T).T) * s1 * s2).item()


def compress_raw_product(B, A, target_r):
    dtype, device = B.dtype, B.device

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


# -----------------------------
# MLP gate
# -----------------------------

class SPFGateMLP(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=16, lambda_max=0.12):
        super().__init__()
        self.lambda_max = lambda_max

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

        # 初始略微偏向 ChartQA，但很保守
        with torch.no_grad():
            self.net[-1].bias.fill_(0.25)

    def forward(self, z):
        raw = self.net(z).squeeze(-1)
        return self.lambda_max * torch.tanh(raw)


# -----------------------------
# Mixed LoRA wrapper
# -----------------------------

class MixedLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear,
        A_text,
        B_text,
        A_chart,
        B_chart,
        scale_text,
        scale_chart,
        feature,
        gate,
        module_name,
    ):
        super().__init__()

        self.base = base_linear
        self.gate = gate
        self.module_name = module_name

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.register_buffer("A_text", A_text.detach().clone())
        self.register_buffer("B_text", B_text.detach().clone())
        self.register_buffer("A_chart", A_chart.detach().clone())
        self.register_buffer("B_chart", B_chart.detach().clone())

        self.scale_text = float(scale_text)
        self.scale_chart = float(scale_chart)

        self.register_buffer("feature", feature.detach().clone())

    def forward(self, x):
        y = self.base(x)

        # lambda_m is learned by the hidden gate
        lam = self.gate(self.feature.unsqueeze(0))[0]

        w_text = 0.5 - lam
        w_chart = 0.5 + lam

        # LoRA update: x @ A.T @ B.T
        dtype = x.dtype
        A_t = self.A_text.to(dtype=dtype)
        B_t = self.B_text.to(dtype=dtype)
        A_c = self.A_chart.to(dtype=dtype)
        B_c = self.B_chart.to(dtype=dtype)

        upd_t = F.linear(F.linear(x, A_t), B_t) * self.scale_text
        upd_c = F.linear(F.linear(x, A_c), B_c) * self.scale_chart

        return y + w_text * upd_t + w_chart * upd_c


def resolve_module_name(lora_a_key):
    name = lora_a_key
    name = name.replace(".lora_A.weight", "")
    name = name.replace(".lora_A.default.weight", "")

    # PEFT keys often start with base_model.model.
    prefixes = [
        "base_model.model.",
        "model.",
    ]

    cands = [name]
    for p in prefixes:
        if name.startswith(p):
            cands.append(name[len(p):])

    # 去重但保序
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def get_module_by_name(model, name):
    cur = model
    for part in name.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            if not hasattr(cur, part):
                raise AttributeError(part)
            cur = getattr(cur, part)
    return cur


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    cur = model

    for part in parts[:-1]:
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)

    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = new_module
    else:
        setattr(cur, last, new_module)


# -----------------------------
# Dataset and loss
# -----------------------------

def load_examples(task, split, max_samples):
    rows = []

    if task == "textvqa":
        ds = load_dataset("lmms-lab/textvqa", split=f"{split}[:{max_samples}]")
        for ex in ds:
            rows.append({
                "image": ex["image"],
                "question": ex["question"],
                "answers": ex["answers"],
            })
        return rows

    if task == "chartqa":
        ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{split}[:{max_samples}]")
        for ex in ds:
            rows.append({
                "image": ex["image"],
                "question": ex["query"],
                "answers": ex["label"],
            })
        return rows

    raise ValueError(task)


def first_answer(answers):
    if isinstance(answers, list):
        if len(answers) == 0:
            return ""
        return str(answers[0])
    return str(answers)


def make_messages(image, question, answer=None):
    prompt = (
        "Read the image carefully and answer the question. "
        "Return only the final short answer.\n"
        f"Question: {question}"
    )

    content = [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]

    messages = [{"role": "user", "content": content}]

    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    return messages


def compute_answer_nll(model, processor, image, question, answer, device):
    # prompt only
    prompt_messages = make_messages(image, question, answer=None)
    full_messages = make_messages(image, question, answer=answer)

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(full_messages)

    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    labels = inputs["input_ids"].clone()

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    outputs = model(**inputs, labels=labels)
    return outputs.loss


# -----------------------------
# Feature construction
# -----------------------------

def build_features(state_t, state_c, cfg_t, cfg_c, target_r):
    r_t = int(cfg_t.get("r", target_r))
    r_c = int(cfg_c.get("r", target_r))
    alpha_t = float(cfg_t.get("lora_alpha", r_t))
    alpha_c = float(cfg_c.get("lora_alpha", r_c))

    scale_t = alpha_t / r_t
    scale_c = alpha_c / r_c

    features = {}

    for a_key in state_t:
        if ".lora_A" not in a_key:
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
        cos = clamp(cos, -1.0, 1.0)

        diff_sq = max(E_t * E_t + E_c * E_c - 2.0 * cross, 0.0)
        delta = math.sqrt(diff_sq) / (E_t + E_c + 1e-12)

        chart_ratio = E_c / (E_t + E_c + 1e-12)
        log_norm_text = math.log(E_t + 1e-8)
        log_norm_chart = math.log(E_c + 1e-8)

        feat = [
            cos,
            delta,
            chart_ratio,
            log_norm_text,
            log_norm_chart,
            normalize_depth(a_key),
            *type_onehot(a_key),
        ]

        features[a_key] = torch.tensor(feat, dtype=torch.float32)

    return features, scale_t, scale_c


def inject_mixed_lora(model, state_t, state_c, features, gate, scale_t, scale_c):
    injected = 0
    missing = []

    for a_key, feat in features.items():
        b_key = a_key.replace(".lora_A.", ".lora_B.")

        A_t = state_t[a_key]
        B_t = state_t[b_key]
        A_c = state_c[a_key]
        B_c = state_c[b_key]

        found = False
        for cand in resolve_module_name(a_key):
            try:
                mod = get_module_by_name(model, cand)
                if isinstance(mod, nn.Linear):
                    wrapper = MixedLoRALinear(
                        mod,
                        A_t.to(mod.weight.device),
                        B_t.to(mod.weight.device),
                        A_c.to(mod.weight.device),
                        B_c.to(mod.weight.device),
                        scale_t,
                        scale_c,
                        feat.to(mod.weight.device),
                        gate,
                        cand,
                    )
                    set_module_by_name(model, cand, wrapper)
                    injected += 1
                    found = True
                    break
            except Exception:
                continue

        if not found:
            missing.append(a_key)

    print("Injected modules:", injected)
    print("Missing modules:", len(missing))
    if missing[:5]:
        print("Missing examples:", missing[:5])


# -----------------------------
# Save learned merged adapter
# -----------------------------

def save_learned_adapter(
    gate,
    features,
    state_t,
    state_c,
    avg_dir,
    text_dir,
    chart_dir,
    out_dir,
    target_r,
    target_alpha,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_t = json.loads((text_dir / "adapter_config.json").read_text())
    cfg_c = json.loads((chart_dir / "adapter_config.json").read_text())
    cfg_out = json.loads((avg_dir / "adapter_config.json").read_text())

    r_t = int(cfg_t.get("r", target_r))
    r_c = int(cfg_c.get("r", target_r))
    alpha_t = float(cfg_t.get("lora_alpha", r_t))
    alpha_c = float(cfg_c.get("lora_alpha", r_c))

    scale_t = alpha_t / r_t
    scale_c = alpha_c / r_c
    out_scale = target_alpha / target_r

    avg_state = load_file(str(avg_dir / "adapter_model.safetensors"))
    new_state = dict(avg_state)

    stats = []

    gate.eval()
    with torch.no_grad():
        for a_key, feat in features.items():
            b_key = a_key.replace(".lora_A.", ".lora_B.")

            A_t, B_t = state_t[a_key], state_t[b_key]
            A_c, B_c = state_c[a_key], state_c[b_key]

            lam = gate(feat.unsqueeze(0).to(next(gate.parameters()).device))[0].item()
            lam = clamp(lam, -0.18, 0.18)

            w_text = 0.5 - lam
            w_chart = 0.5 + lam

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

            B_new, A_new = compress_raw_product(B_cat, A_cat, target_r)

            new_state[a_key] = A_new
            new_state[b_key] = B_new

            stats.append({
                "module": a_key,
                "module_type": module_type(a_key),
                "lambda": lam,
                "w_text": w_text,
                "w_chart": w_chart,
            })

    cfg_out["r"] = target_r
    cfg_out["lora_alpha"] = target_alpha
    cfg_out["rank_pattern"] = {}
    cfg_out["alpha_pattern"] = {}

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg_out, indent=2))
    (out_dir / "spf_gate_mlp_stats.json").write_text(json.dumps(stats, indent=2))

    for f in avg_dir.iterdir():
        if f.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)

    print("Saved learned adapter:", out_dir)
    print("mean lambda:", sum(x["lambda"] for x in stats) / len(stats))
    print("mean w_chart:", sum(x["w_chart"] for x in stats) / len(stats))


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--text", required=True)
    parser.add_argument("--chart", required=True)
    parser.add_argument("--average", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--chart-samples", type=int, default=32)
    parser.add_argument("--text-samples", type=int, default=32)
    parser.add_argument("--split-chart", default="val")
    parser.add_argument("--split-text", default="validation")

    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lambda-max", type=float, default=0.12)
    parser.add_argument("--lambda-prior", type=float, default=0.04)

    parser.add_argument("--target-r", type=int, default=16)
    parser.add_argument("--target-alpha", type=int, default=32)

    parser.add_argument("--text-weight", type=float, default=1.0)
    parser.add_argument("--chart-weight", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=0.05)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_dir = Path(args.text)
    chart_dir = Path(args.chart)
    avg_dir = Path(args.average)
    out_dir = Path(args.output)

    print("Loading adapters...")
    state_t = load_file(str(text_dir / "adapter_model.safetensors"))
    state_c = load_file(str(chart_dir / "adapter_model.safetensors"))

    cfg_t = json.loads((text_dir / "adapter_config.json").read_text())
    cfg_c = json.loads((chart_dir / "adapter_config.json").read_text())

    features, scale_t, scale_c = build_features(state_t, state_c, cfg_t, cfg_c, args.target_r)
    print("Feature modules:", len(features))

    gate = SPFGateMLP(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        lambda_max=args.lambda_max,
    ).to(device)

    # Anchor the gate around SPF-safe instead of pure average.
    with torch.no_grad():
        ratio = max(min(args.lambda_prior / args.lambda_max, 0.95), -0.95)
        init_bias = torch.atanh(torch.tensor(ratio)).item()
        gate.net[-1].bias.fill_(init_bias)
    print(f"Initialized gate around lambda_prior={args.lambda_prior:.4f}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    # Freeze all model params
    for p in model.parameters():
        p.requires_grad_(False)

    print("Injecting mixed LoRA wrappers...")
    inject_mixed_lora(model, state_t, state_c, features, gate, scale_t, scale_c)

    print("Loading calibration examples...")
    chart_rows = load_examples("chartqa", args.split_chart, args.chart_samples)
    text_rows = load_examples("textvqa", args.split_text, args.text_samples)

    calib = []
    for r in chart_rows:
        calib.append(("chartqa", r))
    for r in text_rows:
        calib.append(("textvqa", r))

    import random
    random.seed(42)
    random.shuffle(calib)

    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr)

    print("Training SPF gate...")
    for step in range(args.steps):
        task, ex = calib[step % len(calib)]

        image = ex["image"]
        question = ex["question"]
        answer = first_answer(ex["answers"])

        loss = compute_answer_nll(
            model=model,
            processor=processor,
            image=image,
            question=question,
            answer=answer,
            device=device,
        )

        weight = args.chart_weight if task == "chartqa" else args.text_weight

        # regularize lambda toward safe small correction
        feats = torch.stack([f.to(device) for f in features.values()], dim=0)
        lambdas = gate(feats)
        reg = ((lambdas - args.lambda_prior) ** 2).mean()

        total = weight * loss + args.reg_weight * reg

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0 or step == args.steps - 1:
            with torch.no_grad():
                mean_lam = gate(feats).mean().item()
                mean_w_chart = 0.5 + mean_lam
            print(
                f"step={step:03d} task={task} "
                f"loss={loss.item():.4f} reg={reg.item():.6f} "
                f"mean_lambda={mean_lam:.4f} mean_w_chart={mean_w_chart:.4f}"
            )

    print("Saving learned merged adapter...")
    save_learned_adapter(
        gate=gate,
        features=features,
        state_t=state_t,
        state_c=state_c,
        avg_dir=avg_dir,
        text_dir=text_dir,
        chart_dir=chart_dir,
        out_dir=out_dir,
        target_r=args.target_r,
        target_alpha=args.target_alpha,
    )


if __name__ == "__main__":
    main()
