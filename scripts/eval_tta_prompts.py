import argparse
import ast
import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import torch
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


def norm_text(x):
    s = str(x).strip().lower()
    s = s.replace("−", "-").replace(",", "")
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


def clean_answer(x):
    s = str(x).strip()

    m = re.search(r"FINAL\s*:\s*(.+)", s, flags=re.I)
    if m:
        s = m.group(1)

    m = re.search(r"answer\s*[:：]\s*(.+)", s, flags=re.I)
    if m:
        s = m.group(1)

    s = s.splitlines()[0].strip()
    s = s.strip(" .,:;\"'[](){}")

    # 如果输出太长，优先抽最后一个数字
    if len(s.split()) > 10:
        nums = extract_numbers(s)
        if nums:
            v = nums[-1]
            if abs(v - round(v)) < 1e-6:
                return str(int(round(v)))
            return f"{v:.6g}"

    return s


PROMPTS = {
    "direct": (
        "Read the image carefully and answer the question.\n"
        "Return only the final short answer.\n"
        "Question: {question}"
    ),

    "scale": (
        "Read the chart carefully and answer the question.\n"
        "Preserve the original unit and scale shown in the chart.\n"
        "If the value is 59, answer 59, not 0.59.\n"
        "If the value is 0.59, answer 0.59, not 59.\n"
        "Return only the final short answer.\n"
        "Question: {question}"
    ),

    "axis": (
        "Read the axis labels, tick marks, legend, and data values carefully.\n"
        "Use the chart's displayed numbers exactly.\n"
        "Return only one short final answer.\n"
        "Question: {question}"
    ),

    "math": (
        "Solve the chart question carefully.\n"
        "If arithmetic is required, identify the relevant values from the chart and compute silently.\n"
        "Do not explain. Return only the final answer.\n"
        "Question: {question}"
    ),

    "count": (
        "This may require counting visible bars, lines, points, categories, or groups.\n"
        "Count carefully from the chart.\n"
        "Return only the final short answer.\n"
        "Question: {question}"
    ),

    "legend": (
        "Pay special attention to colors, legends, labels, and categories in the chart.\n"
        "Return only the final short answer.\n"
        "Question: {question}"
    ),
}


def generate(model, processor, image, prompt, max_new_tokens=32):
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

    return clean_answer(raw), raw


def majority_vote(preds, fallback_name="direct"):
    """
    preds: list of (prompt_name, pred)
    """
    norm_to_preds = defaultdict(list)

    for name, pred in preds:
        key = norm_text(pred)
        if key:
            norm_to_preds[key].append((name, pred))

    if not norm_to_preds:
        for name, pred in preds:
            if name == fallback_name:
                return pred
        return preds[0][1]

    counts = [(len(v), k, v) for k, v in norm_to_preds.items()]
    counts.sort(reverse=True, key=lambda x: x[0])

    # 有多数票，选多数
    if len(counts) == 1 or counts[0][0] > counts[1][0]:
        return counts[0][2][0][1]

    # 平票时回到 direct
    for name, pred in preds:
        if name == fallback_name:
            return pred

    return counts[0][2][0][1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output", default="results/eval/chartqa_tta_prompts_200.csv")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    print("Loading ChartQA...")
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

    for idx, ex in enumerate(ds):
        image = ex["image"]
        question = ex["query"]
        answers = ex["label"]

        preds = []
        prompt_scores = []

        for pname, tmpl in PROMPTS.items():
            prompt = tmpl.format(question=question)
            pred, raw = generate(
                model,
                processor,
                image,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            score = relaxed_score(pred, answers)

            preds.append((pname, pred))
            prompt_scores.append(score)

            rows.append({
                "task": "chartqa",
                "adapter": pname,
                "idx": idx,
                "question": question,
                "answers": json.dumps(to_answer_list(answers), ensure_ascii=False),
                "prediction": pred,
                "raw": raw,
                "score": score,
            })

        maj = majority_vote(preds, fallback_name="direct")
        maj_score = relaxed_score(maj, answers)

        oracle_score = max(prompt_scores)

        rows.append({
            "task": "chartqa",
            "adapter": "tta_majority",
            "idx": idx,
            "question": question,
            "answers": json.dumps(to_answer_list(answers), ensure_ascii=False),
            "prediction": maj,
            "raw": json.dumps(dict(preds), ensure_ascii=False),
            "score": maj_score,
        })

        rows.append({
            "task": "chartqa",
            "adapter": "tta_oracle",
            "idx": idx,
            "question": question,
            "answers": json.dumps(to_answer_list(answers), ensure_ascii=False),
            "prediction": "",
            "raw": json.dumps(dict(preds), ensure_ascii=False),
            "score": oracle_score,
        })

        print(
            f"[{idx}] direct={prompt_scores[0]} maj={maj_score} "
            f"oracle={oracle_score} pred={maj}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    summary = df.groupby("adapter")["score"].mean().round(4).to_dict()
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print("Saved:", out)
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
