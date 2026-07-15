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


def first_answer(x):
    if isinstance(x, list):
        return str(x[0]) if x else ""
    return str(x)


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


def make_texts(processor, image, question, answer):
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
            "content": [{"type": "text", "text": str(answer)}],
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
    return prompt_text, full_text, image_inputs, video_inputs


def mean_logprob_completion(model, processor, image, question, answer):
    prompt_text, full_text, image_inputs, video_inputs = make_texts(
        processor, image, question, answer
    )

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

    nll = -(tok_logp * mask).sum() / denom

    return mean_logp, nll


def load_chartqa_images(split, max_samples):
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{split}[:{max_samples}]")
    rows = {}
    for i, ex in enumerate(ds):
        rows[i] = {
            "image": ex["image"],
            "question": ex["query"],
            "answers": ex["label"],
        }
    return rows


def load_textvqa_replay(split, max_samples):
    ds = load_dataset("lmms-lab/textvqa", split=f"{split}[:{max_samples}]")
    rows = []
    for ex in ds:
        rows.append({
            "image": ex["image"],
            "question": ex["question"],
            "answer": first_answer(ex["answers"]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--chart-split", default="train")
    parser.add_argument("--chart-max-samples", type=int, default=1000)
    parser.add_argument("--text-split", default="train")
    parser.add_argument("--text-samples", type=int, default=500)

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--sft-weight", type=float, default=0.2)
    parser.add_argument("--text-replay-weight", type=float, default=0.3)
    parser.add_argument("--text-replay-every", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pairs:", args.pairs)
    pairs = pd.read_csv(args.pairs).to_dict("records")
    if len(pairs) == 0:
        raise ValueError("No VPO pairs found.")

    print("num pairs:", len(pairs))

    print("Loading ChartQA images...")
    chart_rows = load_chartqa_images(args.chart_split, args.chart_max_samples)

    print("Loading TextVQA replay...")
    text_rows = load_textvqa_replay(args.text_split, args.text_samples)

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

    print("Training VPO...")
    for step in range(args.steps):
        ex = random.choice(pairs)
        idx = int(ex["idx"])
        chart_ex = chart_rows[idx]

        chosen = str(ex["chosen"])
        rejected = str(ex["rejected"])

        chosen_logp, chosen_nll = mean_logprob_completion(
            model,
            processor,
            chart_ex["image"],
            chart_ex["question"],
            chosen,
        )

        rejected_logp, _ = mean_logprob_completion(
            model,
            processor,
            chart_ex["image"],
            chart_ex["question"],
            rejected,
        )

        pref_loss = -F.logsigmoid(args.beta * (chosen_logp - rejected_logp))
        loss = pref_loss + args.sft_weight * chosen_nll

        # TextVQA replay to protect OCR ability.
        if text_rows and args.text_replay_every > 0 and step % args.text_replay_every == 0:
            tex = random.choice(text_rows)
            _, text_nll = mean_logprob_completion(
                model,
                processor,
                tex["image"],
                tex["question"],
                tex["answer"],
            )
            loss = loss + args.text_replay_weight * text_nll
        else:
            text_nll = None

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 10 == 0 or step == args.steps - 1:
            msg = (
                f"step={step:04d} "
                f"loss={loss.item():.4f} "
                f"pref={pref_loss.item():.4f} "
                f"chosen_logp={chosen_logp.item():.4f} "
                f"rejected_logp={rejected_logp.item():.4f} "
                f"chosen={chosen} rejected={rejected}"
            )
            if text_nll is not None:
                msg += f" text_nll={text_nll.item():.4f}"
            print(msg)

    print("Saving:", out_dir)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    meta = vars(args)
    (out_dir / "mirage_vpo_meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
