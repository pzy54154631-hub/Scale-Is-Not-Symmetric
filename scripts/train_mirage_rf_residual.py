import argparse
import json
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def first_answer(x):
    if isinstance(x, list):
        return str(x[0]) if len(x) else ""
    return str(x)


def load_examples(task, split, max_samples):
    rows = []

    if task == "chartqa":
        ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{split}[:{max_samples}]")
        for ex in ds:
            rows.append({
                "task": "chartqa",
                "image": ex["image"],
                "question": ex["query"],
                "answer": first_answer(ex["label"]),
            })
        return rows

    if task == "textvqa":
        ds = load_dataset("lmms-lab/textvqa", split=f"{split}[:{max_samples}]")
        for ex in ds:
            rows.append({
                "task": "textvqa",
                "image": ex["image"],
                "question": ex["question"],
                "answer": first_answer(ex["answers"]),
            })
        return rows

    raise ValueError(task)


def make_messages(image, question, answer=None):
    prompt = (
        "Read the image carefully and answer the question. "
        "Return only the final short answer.\n"
        f"Question: {question}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if answer is not None:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        })

    return messages


def compute_answer_nll(model, processor, ex, device):
    answer = ex["answer"]
    if answer is None or len(str(answer).strip()) == 0:
        return None

    prompt_messages = make_messages(ex["image"], ex["question"], answer=None)
    full_messages = make_messages(ex["image"], ex["question"], answer=answer)

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

    out = model(**inputs, labels=labels)
    return out.loss


def keep_lora_param(name, scope):
    if "lora_" not in name:
        return False

    if scope == "all":
        return True

    if scope == "visual_mlp":
        return ".visual.blocks." in name and ".mlp." in name

    if scope == "visual":
        return ".visual.blocks." in name

    if scope == "lang_mlp":
        return ".language_model." in name and ".mlp." in name

    return True


def freeze_by_scope(model, scope):
    trainable = 0
    frozen = 0

    for name, p in model.named_parameters():
        if "lora_" in name:
            keep = keep_lora_param(name, scope)
            p.requires_grad_(keep)
            if keep:
                trainable += p.numel()
            else:
                frozen += p.numel()
        else:
            p.requires_grad_(False)

    print(f"LoRA trainable params under scope={scope}: {trainable}")
    print(f"LoRA frozen params: {frozen}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"),
    )

    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--chart-samples", type=int, default=64)
    parser.add_argument("--text-samples", type=int, default=32)
    parser.add_argument("--chart-split", default="val")
    parser.add_argument("--text-split", default="validation")

    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--residual-r", type=int, default=4)
    parser.add_argument("--residual-alpha", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--scope", default="visual_mlp", choices=["visual_mlp", "visual", "lang_mlp", "all"])

    parser.add_argument("--chart-weight", type=float, default=1.2)
    parser.add_argument("--text-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_adapter_dir = Path(args.base_adapter)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False

    print("Loading base merged adapter:", base_adapter_dir)
    model = PeftModel.from_pretrained(
        model,
        str(base_adapter_dir),
        is_trainable=False,
    )

    print("Merging base adapter into model weights...")
    model = model.merge_and_unload()

    # Read target modules from the base adapter config.
    cfg = json.loads((base_adapter_dir / "adapter_config.json").read_text())
    target_modules = cfg.get("target_modules", None)
    if target_modules is None:
        raise ValueError("target_modules not found in base adapter_config.json")

    print("Target modules:", target_modules)

    print("Adding residual LoRA...")
    residual_cfg = LoraConfig(
        r=args.residual_r,
        lora_alpha=args.residual_alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, residual_cfg)
    freeze_by_scope(model, args.scope)

    model.train()

    print("Loading calibration examples...")
    chart_rows = load_examples("chartqa", args.chart_split, args.chart_samples)
    text_rows = load_examples("textvqa", args.text_split, args.text_samples)

    calib = []
    calib.extend(chart_rows)
    calib.extend(text_rows)
    random.shuffle(calib)

    print("Calibration size:", len(calib))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("Training residual LoRA...")
    for step in range(args.steps):
        ex = calib[step % len(calib)]

        loss = compute_answer_nll(model, processor, ex, device)
        if loss is None:
            continue

        weight = args.chart_weight if ex["task"] == "chartqa" else args.text_weight
        total = weight * loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 10 == 0 or step == args.steps - 1:
            print(
                f"step={step:03d} "
                f"task={ex['task']} "
                f"loss={loss.item():.4f} "
                f"weighted={total.item():.4f}"
            )

    print("Saving residual adapter:", output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    meta = {
        "base_adapter": str(base_adapter_dir),
        "scope": args.scope,
        "residual_r": args.residual_r,
        "residual_alpha": args.residual_alpha,
        "chart_samples": args.chart_samples,
        "text_samples": args.text_samples,
        "steps": args.steps,
        "lr": args.lr,
        "chart_weight": args.chart_weight,
        "text_weight": args.text_weight,
    }

    (output_dir / "mirage_rf_train_meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
