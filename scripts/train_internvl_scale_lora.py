#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", re.I)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/InternVL2_5-1B")
    ap.add_argument("--output-adapter", required=True)
    ap.add_argument("--pair-mode", choices=["large_pos", "small_pos"], default="large_pos")
    ap.add_argument("--train-max-samples", type=int, default=1500)
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1)
    return ap.parse_args()


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=4, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def image_to_pixels(image: Image.Image, max_num: int = 4) -> torch.Tensor:
    transform = build_transform(input_size=448)
    images = dynamic_preprocess(image.convert("RGB"), image_size=448, max_num=max_num, use_thumbnail=True)
    return torch.stack([transform(img) for img in images])


def first_answer(ans: Any) -> str:
    if hasattr(ans, "tolist"):
        ans = ans.tolist()
    if isinstance(ans, (list, tuple)) and ans:
        return str(ans[0])
    return str(ans)


def one_number(text: Any) -> float | None:
    nums = NUM_RE.findall(str(text).replace(",", "").replace("−", "-"))
    if len(nums) != 1:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


def fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-8:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def build_pairs(train_ds, pair_mode: str, max_pairs: int, seed: int) -> list[dict[str, Any]]:
    pairs = []
    for idx, ex in enumerate(train_ds):
        v = one_number(first_answer(ex["label"]))
        if v is None or v == 0 or 1500 <= abs(v) <= 2200:
            continue
        if pair_mode == "large_pos" and abs(v) >= 2:
            pos, neg = fmt_num(v), fmt_num(v / 100.0)
        elif pair_mode == "small_pos" and 0 < abs(v) <= 1.5:
            pos, neg = fmt_num(v), fmt_num(v * 100.0)
        else:
            continue
        if pos == neg:
            continue
        pairs.append({"idx": idx, "question": str(ex["query"]), "pos": pos, "neg": neg})
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def unwrap_chat_model(model):
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "model"):
        return base.model
    return model


def make_text(tokenizer, chat_model, question: str, answer: str | None, num_patches: int) -> str:
    model_dir = Path(chat_model.config._name_or_path)
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from conversation import get_conv_template

    template = get_conv_template(chat_model.template)
    template.system_message = chat_model.system_message
    user_q = "<image>\n" + question + " Answer briefly with the final value only."
    template.append_message(template.roles[0], user_q)
    template.append_message(template.roles[1], answer)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * chat_model.num_image_token * num_patches + IMG_END_TOKEN
    return query.replace("<image>", image_tokens, 1)


def answer_nll(model, tokenizer, image: Image.Image, question: str, answer: str) -> torch.Tensor:
    chat_model = unwrap_chat_model(model)
    chat_model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    pixel_values = image_to_pixels(image).to(torch.bfloat16).cuda()
    full_text = make_text(tokenizer, chat_model, question, answer, pixel_values.shape[0])
    prefix_text = make_text(tokenizer, chat_model, question, None, pixel_values.shape[0])
    full = tokenizer(full_text, return_tensors="pt")
    prefix = tokenizer(prefix_text, return_tensors="pt")
    input_ids = full["input_ids"].cuda()
    attention_mask = full["attention_mask"].cuda()
    labels = input_ids.clone()
    labels[:, : prefix["input_ids"].shape[1]] = -100
    image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device="cuda")
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        labels=labels,
        use_cache=False,
    )
    return outputs.loss


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds = load_dataset(
        "HuggingFaceM4/ChartQA",
        split=f"train[:{args.train_max_samples}]",
        cache_dir="/root/autodl-tmp/datasets",
    )
    pairs = build_pairs(train_ds, args.pair_mode, args.max_pairs, args.seed)
    if not pairs:
        raise RuntimeError(f"No pairs for {args.pair_mode}")
    print("pairs", len(pairs), "mode", args.pair_mode)

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=False,
        tokenizer_type="qwen2",
    )
    model = AutoModel.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).cuda()
    model.config.use_cache = False
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, config)
    model.train()
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    for step in range(args.steps):
        row = random.choice(pairs)
        image = train_ds[int(row["idx"])]["image"]
        pos_nll = answer_nll(model, tokenizer, image, row["question"], row["pos"])
        neg_nll = answer_nll(model, tokenizer, image, row["question"], row["neg"])
        diff = neg_nll - pos_nll
        loss = F.softplus(-args.beta * (diff - args.margin))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step={step:04d} loss={loss.item():.4f} diff={diff.item():.4f} pos={row['pos']} neg={row['neg']}", flush=True)

    out = Path(args.output_adapter)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    summary = vars(args) | {"pairs": len(pairs)}
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved", out)


if __name__ == "__main__":
    main()
