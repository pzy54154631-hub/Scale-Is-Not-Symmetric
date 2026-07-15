import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(str(Path(__file__).parent))
from train_conflict_guided_scale_final import (
    answer_nll,
    first_answer,
    one_number,
    is_factor100_conflict,
    load_margin0,
    normal_prompt,
)


def fmt_num(x):
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def make_scaled_wrong(v):
    if 1500 <= abs(v) <= 2200:
        return None
    if abs(v) >= 2:
        return v / 100.0
    if 0 < abs(v) <= 1.5:
        return v * 100.0
    return None


def build_pairs(train_raw):
    pairs = []

    # 1. Gold answer vs artificial scale-corrupted answer
    for _, r in train_raw.iterrows():
        gold = first_answer(r["answers"])
        v = one_number(gold)
        if v is None:
            continue

        wrong = make_scaled_wrong(v)
        if wrong is None:
            continue

        pos = fmt_num(v)
        neg = fmt_num(wrong)

        if pos == neg:
            continue

        pairs.append({
            "idx": int(r["idx"]),
            "question": r["question"],
            "pos": pos,
            "neg": neg,
            "source": "gold_vs_scaled_wrong",
        })

    # 2. Real base/comp conflict pairs
    for _, r in train_raw.iterrows():
        if not is_factor100_conflict(r["base_pred"], r["comp_pred"]):
            continue

        base_s = int(r["base_score"])
        comp_s = int(r["comp_score"])

        if base_s == comp_s:
            continue

        if base_s == 1 and comp_s == 0:
            pos, neg = str(r["base_pred"]), str(r["comp_pred"])
        elif comp_s == 1 and base_s == 0:
            pos, neg = str(r["comp_pred"]), str(r["base_pred"])
        else:
            continue

        pairs.append({
            "idx": int(r["idx"]),
            "question": r["question"],
            "pos": pos,
            "neg": neg,
            "source": "real_base_comp_conflict",
        })

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--train-logprob", required=True)
    parser.add_argument("--output-adapter", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-max-samples", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--pair-mode", default="all",
        choices=["all", "real_only", "large_pos_only", "small_pos_only", "balanced_scale"])
    parser.add_argument("--max-pairs-per-kind", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_raw = load_margin0(args.train_logprob)
    pairs = build_pairs(train_raw)

    def pair_kind(x):
        pv = one_number(x["pos"])
        nv = one_number(x["neg"])
        if pv is None or nv is None:
            return "other"
        if abs(pv) >= 2 and abs(nv) <= 2:
            return "large_pos"
        if abs(pv) <= 2 and abs(nv) >= 2:
            return "small_pos"
        return "other"

    for x in pairs:
        x["pair_kind"] = pair_kind(x)

    if args.pair_mode == "real_only":
        pairs = [x for x in pairs if x["source"] == "real_base_comp_conflict"]

    elif args.pair_mode == "large_pos_only":
        pairs = [x for x in pairs if x["pair_kind"] == "large_pos"]

    elif args.pair_mode == "small_pos_only":
        pairs = [x for x in pairs if x["pair_kind"] == "small_pos"]

    elif args.pair_mode == "balanced_scale":
        large = [x for x in pairs if x["pair_kind"] == "large_pos"]
        small = [x for x in pairs if x["pair_kind"] == "small_pos"]
        real = [x for x in pairs if x["source"] == "real_base_comp_conflict"]
        random.shuffle(large)
        random.shuffle(small)
        k = min(len(large), len(small), args.max_pairs_per_kind)
        pairs = large[:k] + small[:k] + real
        random.shuffle(pairs)

    if not pairs:
        raise RuntimeError(f"No pairs left after pair_mode={args.pair_mode}")

    print("contrastive pairs:", len(pairs))
    print("pair mode:", args.pair_mode)
    print("source counts:")
    print(pd.Series([p["source"] for p in pairs]).value_counts().to_string())
    print("kind counts:")
    print(pd.Series([p["pair_kind"] for p in pairs]).value_counts().to_string())

    train_ds = load_dataset(
        "HuggingFaceM4/ChartQA",
        split=f"{args.train_split}[:{args.train_max_samples}]",
        cache_dir="/root/autodl-tmp/datasets",
    )

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False

    model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    for step in range(args.steps):
        row = random.choice(pairs)
        image = train_ds[int(row["idx"])]["image"]
        prompt = normal_prompt(row["question"])

        pos_nll = answer_nll(model, processor, image, prompt, row["pos"])
        neg_nll = answer_nll(model, processor, image, prompt, row["neg"])

        # logp_pos - logp_neg = -pos_nll - (-neg_nll) = neg_nll - pos_nll
        diff = neg_nll - pos_nll
        loss = F.softplus(-args.beta * (diff - args.margin))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 50 == 0 or step == args.steps - 1:
            print(
                f"step={step:04d} "
                f"src={row['source']} "
                f"loss={loss.item():.4f} "
                f"diff={diff.item():.4f} "
                f"pos={row['pos']} neg={row['neg']}"
            )

    out = Path(args.output_adapter)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)

    summary = {
        "base_adapter": args.base_adapter,
        "output_adapter": str(out),
        "steps": args.steps,
        "lr": args.lr,
        "beta": args.beta,
        "margin": args.margin,
        "pairs": len(pairs),
        "pair_mode": args.pair_mode,
        "max_pairs_per_kind": args.max_pairs_per_kind,
    }

    (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved:", out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
