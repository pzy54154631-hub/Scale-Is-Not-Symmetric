import argparse
import ast
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
import torch
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


def generate_answer(model, processor, image, question, max_new_tokens, temperature):
    prompt = (
        "Read the image carefully and answer the question.\n"
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
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out)]
    pred = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return pred.split("\n")[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", default="datasets/vpo/chartqa_vpo_pairs.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        chosen = first_answer(answers)

        seen = set()

        for k in range(args.num_generations):
            pred = generate_answer(
                model,
                processor,
                image,
                question,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            key = norm_text(pred)
            if not key or key in seen:
                continue
            seen.add(key)

            score = relaxed_score(pred, answers)

            if score == 0 and norm_text(chosen) != key:
                rows.append({
                    "idx": idx,
                    "question": question,
                    "answers": json.dumps(to_answer_list(answers), ensure_ascii=False),
                    "chosen": chosen,
                    "rejected": pred,
                    "source": "model_wrong",
                })

        if idx % 20 == 0:
            print(f"idx={idx} pairs={len(rows)}")

    pd.DataFrame(rows).to_csv(out, index=False)

    print("Saved:", out)
    print("num_pairs:", len(rows))


if __name__ == "__main__":
    main()
