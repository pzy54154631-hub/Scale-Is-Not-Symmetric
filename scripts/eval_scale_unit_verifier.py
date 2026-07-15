import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from datasets import load_dataset


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


def factor100_conflict(base_pred, comp_pred):
    b = one_number(base_pred)
    c = one_number(comp_pred)
    if b is None or c is None or abs(c) < 1e-12:
        return False
    ratio = abs(b / c)
    return 80 <= ratio <= 125 and abs(c) < abs(b)


def parse_choice(x):
    s = str(x).strip().upper()
    if re.search(r"\bA\b", s):
        return "A"
    if re.search(r"\bB\b", s):
        return "B"
    if s.startswith("A"):
        return "A"
    if s.startswith("B"):
        return "B"
    return "A"


def verifier_prompt(question, base_pred, comp_pred):
    return (
        "You are a chart scale verifier.\n"
        "Look carefully at the chart axis labels, tick values, legend, and units.\n"
        "Two candidate answers differ mainly by scale.\n"
        "Choose the candidate that matches the original chart scale.\n"
        "Do not convert percentages to decimals unless the chart itself uses decimal values.\n"
        "Do not convert decimals to percentages unless the chart itself uses percent values.\n\n"
        f"Question: {question}\n"
        f"Candidate A: {base_pred}\n"
        f"Candidate B: {comp_pred}\n\n"
        "Return only A or B."
    )


def generate_choice(model, processor, image, prompt, max_new_tokens=8):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return parse_choice(raw), raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", default="results/eval/chartqa_comp_200.csv")
    parser.add_argument("--base", default="sfg_cem_best")
    parser.add_argument("--comp", default="comp_final")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output", default="results/eval/chartqa_scale_unit_verifier_200.csv")
    args = parser.parse_args()

    print("Loading eval CSV...")
    df = pd.read_csv(args.input)

    score = df.pivot_table(index="idx", columns="adapter", values="score", aggfunc="first")
    pred = df.pivot_table(index="idx", columns="adapter", values="prediction", aggfunc="first")
    meta = df.drop_duplicates("idx").set_index("idx")

    print("Loading ChartQA images...")
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{args.split}[:{args.max_samples}]")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    print("Loading adapter:", args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    rows = []

    for idx in score.index:
        question = meta.loc[idx, "question"]
        answers = meta.loc[idx, "answers"]
        base_pred = pred.loc[idx, args.base]
        comp_pred = pred.loc[idx, args.comp]

        selected_from = args.base
        selected_pred = base_pred
        reason = "base_default"
        raw = ""

        if factor100_conflict(base_pred, comp_pred):
            image = ds[int(idx)]["image"]
            prompt = verifier_prompt(question, base_pred, comp_pred)
            choice, raw = generate_choice(model, processor, image, prompt)

            if choice == "B":
                selected_from = args.comp
                selected_pred = comp_pred
                reason = "verifier_choose_comp"
            else:
                selected_from = args.base
                selected_pred = base_pred
                reason = "verifier_choose_base"

        if selected_from == args.comp:
            s = score.loc[idx, args.comp]
        else:
            s = score.loc[idx, args.base]

        rows.append({
            "idx": idx,
            "question": question,
            "answers": answers,
            "base_pred": base_pred,
            "comp_pred": comp_pred,
            "selected_from": selected_from,
            "selected_pred": selected_pred,
            "reason": reason,
            "verifier_raw": raw,
            "score": s,
            "base_score": score.loc[idx, args.base],
            "comp_score": score.loc[idx, args.comp],
        })

        print(
            f"[{idx}] reason={reason} score={s} "
            f"base={base_pred} comp={comp_pred} raw={raw}"
        )

    out_df = pd.DataFrame(rows)

    wins = out_df[(out_df["score"] == 1) & (out_df["base_score"] == 0)]
    losses = out_df[(out_df["score"] == 0) & (out_df["base_score"] == 1)]

    print("\nSUMMARY")
    print("base:", round(float(score[args.base].mean()), 4))
    print("comp:", round(float(score[args.comp].mean()), 4))
    print("oracle:", round(float(score[[args.base, args.comp]].max(axis=1).mean()), 4))
    print("scale_unit_verifier:", round(float(out_df["score"].mean()), 4))
    print("wins:", len(wins), wins["idx"].tolist())
    print("losses:", len(losses), losses["idx"].tolist())
    print("net:", len(wins) - len(losses))
    print(out_df["reason"].value_counts().to_string())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    summary = {
        "base": round(float(score[args.base].mean()), 4),
        "comp": round(float(score[args.comp].mean()), 4),
        "oracle": round(float(score[[args.base, args.comp]].max(axis=1).mean()), 4),
        "scale_unit_verifier": round(float(out_df["score"].mean()), 4),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "net": int(len(wins) - len(losses)),
    }

    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print("Saved:", out)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
