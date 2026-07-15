import argparse
import json
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


def factor100_conflict(base_pred, comp_pred):
    b = one_number(base_pred)
    c = one_number(comp_pred)
    if b is None or c is None or abs(c) < 1e-12:
        return False
    ratio = abs(b / c)
    return 80 <= ratio <= 125 and abs(c) < abs(b)


def verifier_prompt(question, base_pred, comp_pred):
    return (
        "You are a chart scale verifier.\n"
        "Look at the chart axis labels, tick values, legend, and units.\n"
        "Two candidate answers differ mainly by scale.\n"
        "Choose the candidate that matches the original chart scale.\n"
        "A chart value shown as 68 should remain 68.\n"
        "A chart value shown as 0.68 should remain 0.68.\n"
        "Do not convert percentages to decimals unless the chart itself uses decimals.\n"
        "Do not convert decimals to percentages unless the chart itself uses percentages.\n\n"
        f"Question: {question}\n"
        f"Candidate A: {base_pred}\n"
        f"Candidate B: {comp_pred}\n\n"
        "Correct choice:"
    )


def get_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def completion_mean_logp(model, processor, image, prompt, answer):
    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
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

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)

    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    denom = mask.sum().clamp(min=1)
    return float((tok_logp * mask).sum() / denom)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", default="results/eval/chartqa_comp_200.csv")
    parser.add_argument("--base", default="sfg_cem_best")
    parser.add_argument("--comp", default="comp_final")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output", default="results/eval/chartqa_scale_unit_verifier_logprob_200.csv")
    args = parser.parse_args()

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

    conflict_rows = {}

    for idx in score.index:
        bp = pred.loc[idx, args.base]
        cp = pred.loc[idx, args.comp]

        if factor100_conflict(bp, cp):
            q = meta.loc[idx, "question"]
            image = ds[int(idx)]["image"]
            prompt = verifier_prompt(q, bp, cp)

            logp_a = completion_mean_logp(model, processor, image, prompt, "A")
            logp_b = completion_mean_logp(model, processor, image, prompt, "B")

            conflict_rows[idx] = {
                "logp_a": logp_a,
                "logp_b": logp_b,
                "delta_b_minus_a": logp_b - logp_a,
            }

            print(
                f"[conflict {idx}] A={logp_a:.4f} B={logp_b:.4f} "
                f"delta={logp_b-logp_a:.4f} base={bp} comp={cp}"
            )

    margins = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]
    all_rows = []

    print("\nBase:", round(float(score[args.base].mean()), 4))
    print("Comp:", round(float(score[args.comp].mean()), 4))
    print("Oracle:", round(float(score[[args.base, args.comp]].max(axis=1).mean()), 4))

    for margin in margins:
        rows = []

        for idx in score.index:
            q = meta.loc[idx, "question"]
            ans = meta.loc[idx, "answers"]
            bp = pred.loc[idx, args.base]
            cp = pred.loc[idx, args.comp]

            selected_from = args.base
            selected_pred = bp
            reason = "base_default"
            logp_a = None
            logp_b = None
            delta = None

            if idx in conflict_rows:
                logp_a = conflict_rows[idx]["logp_a"]
                logp_b = conflict_rows[idx]["logp_b"]
                delta = conflict_rows[idx]["delta_b_minus_a"]

                if delta > margin:
                    selected_from = args.comp
                    selected_pred = cp
                    reason = f"choose_comp_margin_{margin}"
                else:
                    reason = f"choose_base_margin_{margin}"

            if selected_from == args.comp:
                s = score.loc[idx, args.comp]
            else:
                s = score.loc[idx, args.base]

            row = {
                "idx": idx,
                "margin": margin,
                "question": q,
                "answers": ans,
                "base_pred": bp,
                "comp_pred": cp,
                "selected_from": selected_from,
                "selected_pred": selected_pred,
                "reason": reason,
                "logp_a": logp_a,
                "logp_b": logp_b,
                "delta_b_minus_a": delta,
                "score": s,
                "base_score": score.loc[idx, args.base],
                "comp_score": score.loc[idx, args.comp],
            }

            rows.append(row)
            all_rows.append(row)

        out_df = pd.DataFrame(rows)

        wins = out_df[(out_df["score"] == 1) & (out_df["base_score"] == 0)]
        losses = out_df[(out_df["score"] == 0) & (out_df["base_score"] == 1)]

        print("\n" + "=" * 80)
        print("margin:", margin)
        print("score:", round(float(out_df["score"].mean()), 4))
        print("wins:", len(wins), wins["idx"].tolist())
        print("losses:", len(losses), losses["idx"].tolist())
        print("net:", len(wins) - len(losses))
        print(out_df["reason"].value_counts().to_string())

    final_df = pd.DataFrame(all_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out, index=False)

    summary = (
        final_df.groupby("margin")["score"]
        .mean()
        .round(4)
        .to_dict()
    )

    summary["_base"] = round(float(score[args.base].mean()), 4)
    summary["_comp"] = round(float(score[args.comp].mean()), 4)
    summary["_oracle"] = round(float(score[[args.base, args.comp]].max(axis=1).mean()), 4)
    summary["_num_conflicts"] = len(conflict_rows)

    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSaved:", out)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
