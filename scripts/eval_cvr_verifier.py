import argparse
import ast
import json
import os
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def to_answer_list(x):
    if isinstance(x, list):
        return x
    if x is None or (isinstance(x, float) and pd.isna(x)):
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


def norm_text(x):
    s = str(x).strip().lower()
    s = s.replace("−", "-")
    s = s.replace(",", "")
    s = s.replace("%", " percent ")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;\"'[](){}")


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


def close_num(p, g, tol=0.05):
    if abs(g) < 1e-12:
        return abs(p - g) <= 1e-4
    return abs(p - g) <= tol * abs(g)


def relaxed_score(pred, answers):
    pred_n = norm_text(pred)
    ans_list = [a for a in to_answer_list(answers) if str(a).strip()]
    ans_norm = [norm_text(a) for a in ans_list]

    if not ans_norm:
        return 0

    if any(pred_n == a for a in ans_norm):
        return 1

    ps = extract_numbers(pred)
    for a in ans_list:
        gs = extract_numbers(a)
        for p in ps:
            for g in gs:
                if close_num(p, g):
                    return 1

    for a in ans_norm:
        if len(a) >= 3 and len(pred_n) >= 3:
            if a in pred_n or pred_n in a:
                return 1

    return 0


def classify_question(q):
    ql = q.lower()

    if any(w in ql for w in ["sum", "total", "add up", "combined"]):
        return "sum"
    if any(w in ql for w in ["difference", "how much more", "how much less"]):
        return "difference"
    if any(w in ql for w in ["how many", "number of", "count"]):
        return "count"
    if any(w in ql for w in ["highest", "maximum", "max", "largest", "greatest"]):
        return "max"
    if any(w in ql for w in ["lowest", "minimum", "min", "smallest", "least"]):
        return "min"
    if any(w in ql for w in ["greater than", "times", "ratio", "percent", "percentage"]):
        return "ratio_or_scale"
    if any(w in ql for w in ["which year", "what year", "year"]):
        return "year"
    if any(w in ql for w in ["color", "line", "bar", "legend"]):
        return "legend_or_label"

    return "direct_read"


def load_chartqa_examples(split, max_samples):
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{split}[:{max_samples}]")
    rows = {}
    for i, ex in enumerate(ds):
        rows[i] = {
            "image": ex["image"],
            "question": ex["query"],
            "answers": ex["label"],
        }
    return rows


def parse_final_answer(text):
    s = str(text).strip()

    # Prefer FINAL: xxx
    m = re.search(r"FINAL\s*:\s*(.+)", s, flags=re.I | re.S)
    if m:
        s = m.group(1).strip()

    # If model outputs option and answer, keep answer part.
    m = re.search(r"(?:answer|final answer)\s*[:：]\s*(.+)", s, flags=re.I | re.S)
    if m:
        s = m.group(1).strip()

    # Remove common wrappers.
    s = s.split("\n")[0].strip()
    s = s.strip(" .,:;\"'[](){}")

    # Avoid verbose outputs.
    if len(s.split()) > 12:
        nums = extract_numbers(s)
        if nums:
            v = nums[-1]
            if abs(v - round(v)) < 1e-6:
                return str(int(round(v)))
            return f"{v:.6g}"

    return s


def make_candidate_block(candidates):
    lines = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, (adapter, pred) in enumerate(candidates):
        letter = letters[i]
        lines.append(f"{letter}. {pred}    [from {adapter}]")
    return "\n".join(lines)


def verifier_prompt(question, qtype, candidate_block):
    type_hint = {
        "sum": "This is a sum or total question. Verify the values from the chart and add them carefully.",
        "difference": "This is a difference question. Identify both chart values and subtract carefully.",
        "count": "This is a counting question. Count visible items, groups, bars, lines, or data points carefully.",
        "max": "This asks for the maximum or highest value. Locate the largest value in the chart.",
        "min": "This asks for the minimum or lowest value. Locate the smallest value in the chart.",
        "ratio_or_scale": "This question may involve ratio, percentage, or scale. Preserve the chart's original unit and scale.",
        "year": "This asks for a year. Check the x-axis or labels carefully.",
        "legend_or_label": "This asks about legend, color, line, bar, or label. Match the visual element to the legend carefully.",
        "direct_read": "This is a direct chart reading question. Read the relevant chart value carefully.",
    }[qtype]

    return (
        "You are a strict chart-answer verifier.\n"
        "Your job is not to repeat a candidate blindly. You must verify the answer from the chart.\n\n"
        f"Question type: {qtype}\n"
        f"Instruction: {type_hint}\n\n"
        "Candidate answers from different adapters:\n"
        f"{candidate_block}\n\n"
        "Rules:\n"
        "1. Check the chart value, unit, legend, and scale directly from the image.\n"
        "2. Do not convert 59 to 0.59 or 0.59 to 59 unless the question explicitly asks for conversion.\n"
        "3. If one candidate is correct, output that answer.\n"
        "4. If all candidates are wrong, output your corrected answer.\n"
        "5. Return exactly one short final answer.\n\n"
        f"Question: {question}\n"
        "FINAL:"
    )


def run_verifier(model, processor, image, question, candidates, max_new_tokens=32):
    qtype = classify_question(question)
    candidate_block = make_candidate_block(candidates)

    prompt = verifier_prompt(question, qtype, candidate_block)

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
    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    trimmed = [
        o[len(i):]
        for i, o in zip(inputs["input_ids"], out)
    ]

    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    final = parse_final_answer(raw)
    return final, raw, qtype


def dedup_candidates(rows, adapters):
    cand = []
    seen = set()

    for a in adapters:
        sub = rows[rows["adapter"] == a]
        if len(sub) == 0:
            continue
        pred = str(sub.iloc[0]["prediction"]).strip()
        key = norm_text(pred)
        if key and key not in seen:
            cand.append((a, pred))
            seen.add(key)

    return cand


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=200)

    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"),
    )
    parser.add_argument("--verifier-adapter", default=None)

    parser.add_argument(
        "--adapters",
        nargs="+",
        default=[
            "average",
            "vmlp_chart060",
            "pas_group_stable",
            "spf_safe",
            "sfg_cem_best",
        ],
    )

    parser.add_argument("--max-new-tokens", type=int, default=32)

    args = parser.parse_args()

    print("Loading candidate CSV:", args.candidate_csv)
    cand_df = pd.read_csv(args.candidate_csv)

    print("Loading ChartQA examples...")
    examples = load_chartqa_examples(args.split, args.max_samples)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    print("Loading verifier model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )

    if args.verifier_adapter:
        print("Loading verifier adapter:", args.verifier_adapter)
        model = PeftModel.from_pretrained(model, args.verifier_adapter, is_trainable=False)

    model.eval()

    rows = []

    grouped = cand_df.groupby("idx")

    for idx in sorted(grouped.groups.keys()):
        idx_int = int(idx)
        if idx_int not in examples:
            continue

        g = grouped.get_group(idx)
        ex = examples[idx_int]

        candidates = dedup_candidates(g, args.adapters)
        if not candidates:
            continue

        pred, raw, qtype = run_verifier(
            model=model,
            processor=processor,
            image=ex["image"],
            question=ex["question"],
            candidates=candidates,
            max_new_tokens=args.max_new_tokens,
        )

        score = relaxed_score(pred, ex["answers"])

        rows.append({
            "task": "chartqa",
            "adapter": "cvr_verifier",
            "idx": idx_int,
            "question": ex["question"],
            "answers": json.dumps(to_answer_list(ex["answers"]), ensure_ascii=False),
            "prediction": pred,
            "raw_verifier_output": raw,
            "question_type": qtype,
            "candidates": json.dumps(
                [{"adapter": a, "prediction": p} for a, p in candidates],
                ensure_ascii=False,
            ),
            "score": score,
        })

        print(f"[cvr] {idx_int}: score={score} pred={pred}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)

    summary = {
        "cvr_verifier": float(out_df["score"].mean()) if len(out_df) else 0.0,
        "n": int(len(out_df)),
    }

    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print("Saved:", out)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
