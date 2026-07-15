import argparse
import ast
import json
import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def to_answer_list(x):
    if isinstance(x, list):
        return x
    if x is None:
        return []
    s = str(x).strip()
    try:
        y = json.loads(s)
        return y if isinstance(y, list) else [y]
    except Exception:
        pass
    try:
        y = ast.literal_eval(s)
        return y if isinstance(y, list) else [y]
    except Exception:
        return [s]


def first_answer(x):
    xs = to_answer_list(x)
    return str(xs[0]) if xs else ""


def make_prompt(question):
    return (
        "Read the image carefully and answer the question.\n"
        "Return only the final short answer.\n"
        f"Question: {question}"
    )


def get_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def completion_logp_and_nll(model, processor, image, question, answer):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": make_prompt(question)},
            ],
        }
    ]

    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(answer)}
            ],
        }
    ]

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

    device = get_device(model)

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

    out = model(**inputs)
    logits = out.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)

    logp = F.log_softmax(shift_logits, dim=-1)
    tok_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    denom = mask.sum().clamp(min=1)
    mean_logp = (tok_logp * mask).sum() / denom
    nll = -mean_logp

    return mean_logp, nll


def load_chartqa(split, max_samples):
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{split}[:{max_samples}]")
    return list(ds)


def load_textvqa(split, max_samples):
    ds = load_dataset("lmms-lab/textvqa", split=f"{split}[:{max_samples}]")
    return list(ds)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--hard-csv", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--chart-split", default="train")
    parser.add_argument("--chart-max-samples", type=int, default=1500)
    parser.add_argument("--text-split", default="train")
    parser.add_argument("--text-samples", type=int, default=800)

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--hard-prob", type=float, default=0.55)
    parser.add_argument("--chart-anchor-prob", type=float, default=0.25)
    parser.add_argument("--text-prob", type=float, default=0.20)

    parser.add_argument("--margin-weight", type=float, default=0.03)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--chart-anchor-weight", type=float, default=0.35)

    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading hard cases:", args.hard_csv)
    hard_rows = pd.read_csv(args.hard_csv).to_dict("records")
    if len(hard_rows) == 0:
        raise ValueError("No hard cases found.")

    print("hard cases:", len(hard_rows))

    print("Loading ChartQA anchor data...")
    chart_rows = load_chartqa(args.chart_split, args.chart_max_samples)

    print("Loading TextVQA replay data...")
    text_rows = load_textvqa(args.text_split, args.text_samples)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False

    print("Loading trainable adapter:", args.base_adapter)
    model = PeftModel.from_pretrained(
        model,
        args.base_adapter,
        is_trainable=True,
    )

    model.train()
    model.print_trainable_parameters()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    probs = [
        ("hard", args.hard_prob),
        ("chart_anchor", args.chart_anchor_prob),
        ("text", args.text_prob),
    ]

    total_p = sum(p for _, p in probs)
    probs = [(n, p / total_p) for n, p in probs]

    def sample_mode():
        r = random.random()
        acc = 0
        for name, p in probs:
            acc += p
            if r <= acc:
                return name
        return probs[-1][0]

    print("Training complementarity-aware adapter...")

    for step in range(args.steps):
        mode = sample_mode()

        if mode == "hard":
            row = random.choice(hard_rows)
            idx = int(row["idx"])
            ex = chart_rows[idx]

            image = ex["image"]
            question = ex["query"]
            chosen = str(row["chosen"])
            rejected = str(row["rejected"])

            chosen_logp, chosen_nll = completion_logp_and_nll(
                model,
                processor,
                image,
                question,
                chosen,
            )

            loss = chosen_nll

            margin_loss_val = None

            if args.margin_weight > 0 and rejected.strip():
                rejected_logp, _ = completion_logp_and_nll(
                    model,
                    processor,
                    image,
                    question,
                    rejected,
                )

                # Encourage gold completion to be more likely than the old wrong answer.
                margin_loss = F.softplus(rejected_logp - chosen_logp + args.margin)
                loss = loss + args.margin_weight * margin_loss
                margin_loss_val = margin_loss.item()

            desc = f"hard idx={idx} chosen={chosen} rejected={rejected}"

        elif mode == "chart_anchor":
            ex = random.choice(chart_rows)

            image = ex["image"]
            question = ex["query"]
            answer = first_answer(ex["label"])

            _, nll = completion_logp_and_nll(
                model,
                processor,
                image,
                question,
                answer,
            )

            loss = args.chart_anchor_weight * nll
            margin_loss_val = None
            desc = f"chart_anchor ans={answer}"

        else:
            ex = random.choice(text_rows)

            image = ex["image"]
            question = ex["question"]
            answer = first_answer(ex["answers"])

            _, nll = completion_logp_and_nll(
                model,
                processor,
                image,
                question,
                answer,
            )

            loss = args.text_weight * nll
            margin_loss_val = None
            desc = f"text ans={answer}"

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 10 == 0 or step == args.steps - 1:
            msg = f"step={step:04d} mode={mode} loss={loss.item():.4f} {desc}"
            if margin_loss_val is not None:
                msg += f" margin={margin_loss_val:.4f}"
            print(msg)

        if args.save_every > 0 and step > 0 and step % args.save_every == 0:
            ckpt = out_dir.parent / f"{out_dir.name}_step{step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt), safe_serialization=True)
            print("Saved checkpoint:", ckpt)

    print("Saving final adapter:", out_dir)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    meta = vars(args)
    (out_dir / "mirage_comp_meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
