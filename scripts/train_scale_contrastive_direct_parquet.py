#!/usr/bin/env python3
"""Train a direct scale-contrastive LoRA adapter from local ChartQA parquet.

This is a lightweight cross-backbone variant used when a backbone-specific
answer-only base adapter is unavailable.  It keeps the backbone frozen and
optimizes only newly inserted LoRA weights with the same pairwise scale
objective used in the main experiments.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(str(Path(__file__).parent))
from train_conflict_guided_scale_final import answer_nll, load_margin0, normal_prompt, one_number
from train_scale_contrastive_final import build_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-logprob", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--output-adapter", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument(
        "--pair-mode",
        default="large_pos_only",
        choices=["all", "large_pos_only", "small_pos_only", "balanced_scale", "real_only"],
    )
    parser.add_argument("--max-pairs-per-kind", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def load_image(obj: Any) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, dict):
        data = obj.get("bytes")
        if data is not None:
            return Image.open(io.BytesIO(data)).convert("RGB")
        path = obj.get("path")
        if path:
            return Image.open(path).convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")
    raise TypeError(f"Unsupported image object: {type(obj)}")


def pair_kind(item: dict[str, Any]) -> str:
    pv = one_number(item["pos"])
    nv = one_number(item["neg"])
    if pv is None or nv is None:
        return "other"
    if abs(pv) >= 2 and abs(nv) <= 2:
        return "large_pos"
    if abs(pv) <= 2 and abs(nv) >= 2:
        return "small_pos"
    return "other"


def filter_pairs(pairs: list[dict[str, Any]], mode: str, max_pairs: int) -> list[dict[str, Any]]:
    for item in pairs:
        item["pair_kind"] = pair_kind(item)
    if mode == "real_only":
        pairs = [x for x in pairs if x["source"] == "real_base_comp_conflict"]
    elif mode == "large_pos_only":
        pairs = [x for x in pairs if x["pair_kind"] == "large_pos"]
    elif mode == "small_pos_only":
        pairs = [x for x in pairs if x["pair_kind"] == "small_pos"]
    elif mode == "balanced_scale":
        large = [x for x in pairs if x["pair_kind"] == "large_pos"]
        small = [x for x in pairs if x["pair_kind"] == "small_pos"]
        real = [x for x in pairs if x["source"] == "real_base_comp_conflict"]
        k = min(len(large), len(small), max_pairs)
        random.shuffle(large)
        random.shuffle(small)
        pairs = large[:k] + small[:k] + real
    if mode in {"large_pos_only", "small_pos_only"}:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs] if max_pairs > 0 else pairs
    random.shuffle(pairs)
    return pairs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_raw = load_margin0(args.train_logprob)
    pairs = filter_pairs(build_pairs(train_raw), args.pair_mode, args.max_pairs_per_kind)
    if not pairs:
        raise RuntimeError(f"No pairs left after pair_mode={args.pair_mode}")
    print("contrastive pairs:", len(pairs))
    print("source counts:")
    print(pd.Series([p["source"] for p in pairs]).value_counts().to_string())
    print("kind counts:")
    print(pd.Series([p["pair_kind"] for p in pairs]).value_counts().to_string())

    train_df = pd.read_parquet(args.train_parquet)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    for step in range(args.steps):
        row = random.choice(pairs)
        image = load_image(train_df.iloc[int(row["idx"])]["image"])
        prompt = normal_prompt(row["question"])
        pos_nll = answer_nll(model, processor, image, prompt, row["pos"])
        neg_nll = answer_nll(model, processor, image, prompt, row["neg"])
        diff = neg_nll - pos_nll
        loss = F.softplus(-args.beta * (diff - args.margin))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(
                f"step={step:04d} loss={loss.item():.4f} diff={diff.item():.4f} "
                f"pos={row['pos']} neg={row['neg']}"
            )

    out = Path(args.output_adapter)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    summary = {
        "model_dir": args.model_dir,
        "output_adapter": str(out),
        "steps": args.steps,
        "lr": args.lr,
        "beta": args.beta,
        "margin": args.margin,
        "pairs": len(pairs),
        "pair_mode": args.pair_mode,
        "max_pairs_per_kind": args.max_pairs_per_kind,
        "direct_from_backbone": True,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
