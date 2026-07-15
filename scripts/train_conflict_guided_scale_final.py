import argparse
import ast
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def extract_numbers(x):
    s = str(x).lower().replace(",", "").replace("−", "-")
    nums = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", s)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            pass
    return out


def one_number(x):
    nums = extract_numbers(x)
    if len(nums) != 1:
        return None
    return nums[0]


def first_answer(ans):
    s = str(ans)
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and obj:
            return str(obj[0])
    except Exception:
        pass
    return s


def is_factor100_conflict(base_pred, comp_pred):
    b = one_number(base_pred)
    c = one_number(comp_pred)
    if b is None or c is None or abs(c) < 1e-12:
        return False
    ratio = abs(b / c)
    return 80 <= ratio <= 125 and abs(c) < abs(b)


def load_margin0(path):
    df = pd.read_csv(path)
    if "margin" in df.columns:
        df["_m"] = pd.to_numeric(df["margin"], errors="coerce")
        df = df[df["_m"].abs() < 1e-9].copy()
    return df.drop_duplicates("idx").copy()


def normal_prompt(question):
    return (
        "Read the chart carefully and answer the question.\n"
        "Return only the final short answer.\n"
        f"Question: {question}"
    )


def scale_prompt(question):
    return (
        "Read the chart carefully and answer the question.\n"
        "Pay special attention to the original chart scale, axis tick labels, value labels, legend, and units.\n"
        "Preserve the scale shown in the chart.\n"
        "If the chart value is shown as 59, answer 59, not 0.59.\n"
        "If the chart value is shown as 0.59, answer 0.59, not 59.\n"
        "Do not convert percentages to decimals or decimals to percentages unless the question explicitly asks.\n"
        "Return only the final short answer.\n"
        f"Question: {question}"
    )


def get_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def answer_nll(model, processor, image, prompt, answer):
    prompt_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    full_messages = prompt_messages + [{
        "role": "assistant",
        "content": [{"type": "text", "text": str(answer)}],
    }]

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
    logits = out.logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)

    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    denom = mask.sum().clamp(min=1)
    mean_logp = (tok_logp * mask).sum() / denom
    return -mean_logp


def build_training_pools(train_raw):
    general_rows = []
    hard_rows = []
    conflict_rows = []

    for _, r in train_raw.iterrows():
        ans = first_answer(r["answers"])

        item = {
            "idx": int(r["idx"]),
            "question": r["question"],
            "answer": ans,
            "source": "general",
        }

        general_rows.append(item)

        if int(r.get("base_score", 1)) == 0:
            hard_item = dict(item)
            hard_item["source"] = "hard_base_wrong"
            hard_rows.append(hard_item)

        if is_factor100_conflict(r["base_pred"], r["comp_pred"]):
            conflict_item = dict(item)
            conflict_item["source"] = "scale_conflict"

            if int(r["base_score"]) != int(r["comp_score"]):
                if int(r["comp_score"]) > int(r["base_score"]):
                    conflict_item["conflict_type"] = "comp_correct_base_wrong"
                else:
                    conflict_item["conflict_type"] = "base_correct_comp_wrong"
                conflict_rows.append(conflict_item)

    return general_rows, hard_rows, conflict_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--base-adapter", required=True)
    parser.add_argument("--train-logprob", required=True)
    parser.add_argument("--output-adapter", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-max-samples", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--conflict-prob", type=float, default=0.45)
    parser.add_argument("--hard-prob", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_raw = load_margin0(args.train_logprob)
    general_rows, hard_rows, conflict_rows = build_training_pools(train_raw)

    print("general rows:", len(general_rows))
    print("hard rows:", len(hard_rows))
    print("scale conflict rows:", len(conflict_rows))

    if not general_rows:
        raise RuntimeError("No general training rows found.")

    print("Loading ChartQA train images...")
    train_ds = load_dataset(
        "HuggingFaceM4/ChartQA",
        split=f"{args.train_split}[:{args.train_max_samples}]",
        cache_dir="/root/autodl-tmp/datasets",
    )

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

    print("Loading trainable base adapter:", args.base_adapter)
    model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    print("Training conflict-guided scale-aware final adapter...")

    stat = {"general": 0, "hard": 0, "conflict": 0}

    for step in range(args.steps):
        u = random.random()

        if conflict_rows and u < args.conflict_prob:
            row = random.choice(conflict_rows)
            prompt = scale_prompt(row["question"])
            stat["conflict"] += 1

        elif hard_rows and u < args.conflict_prob + args.hard_prob:
            row = random.choice(hard_rows)
            prompt = scale_prompt(row["question"])
            stat["hard"] += 1

        else:
            row = random.choice(general_rows)
            # general replay keeps ordinary ChartQA ability
            prompt = normal_prompt(row["question"])
            stat["general"] += 1

        idx = int(row["idx"])
        image = train_ds[idx]["image"]
        answer = row["answer"]

        loss = answer_nll(model, processor, image, prompt, answer)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if step % 50 == 0 or step == args.steps - 1:
            print(
                f"step={step:04d} "
                f"src={row['source']} "
                f"loss={loss.item():.4f} "
                f"answer={answer}"
            )

    out_adapter = Path(args.output_adapter)
    out_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_adapter), safe_serialization=True)

    summary = {
        "base_adapter": args.base_adapter,
        "output_adapter": str(out_adapter),
        "steps": args.steps,
        "lr": args.lr,
        "conflict_prob": args.conflict_prob,
        "hard_prob": args.hard_prob,
        "general_rows": len(general_rows),
        "hard_rows": len(hard_rows),
        "scale_conflict_rows": len(conflict_rows),
        "sample_counts": stat,
    }

    summary_path = out_adapter / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Saved adapter:", out_adapter)
    print("Saved summary:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
