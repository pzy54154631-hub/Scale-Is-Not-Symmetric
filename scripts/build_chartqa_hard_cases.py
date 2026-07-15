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


def get_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return s


def make_prompt(question):
    return (
        "Read the image carefully and answer the question.\n"
        "Return only the final short answer.\n"
        f"Question: {question}"
    )


def generate_answer(model, processor, image, question, max_new_tokens=24):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": make_prompt(question)},
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

    device = get_device(model)
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

    return clean_answer(raw), raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--output", default="datasets/comp/chartqa_hard_train1500.csv")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

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

        pred, raw = generate_answer(
            model,
            processor,
            image,
            question,
            max_new_tokens=args.max_new_tokens,
        )

        score = relaxed_score(pred, answers)
        chosen = first_answer(answers)

        if score == 0 and norm_text(pred) != norm_text(chosen):
            rows.append({
                "idx": idx,
                "question": question,
                "answers": json.dumps(to_answer_list(answers), ensure_ascii=False),
                "chosen": chosen,
                "rejected": pred,
                "raw_rejected": raw,
                "score": score,
            })

        if idx % 20 == 0:
            print(f"idx={idx} pred={pred} score={score} hard={len(rows)}")

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    print("Saved:", out)
    print("hard cases:", len(df))
    if len(df):
        print(df.head().to_string())


if __name__ == "__main__":
    main()
