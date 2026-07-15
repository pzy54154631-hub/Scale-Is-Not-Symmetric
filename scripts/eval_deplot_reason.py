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
    s = s.splitlines()[0].strip()
    s = s.strip(" .,:;\"'[](){}")
    return s


def generate_with_image(model, processor, image, prompt, max_new_tokens=256):
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
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def generate_text_only(model, processor, prompt, max_new_tokens=64):
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
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
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def derender_prompt():
    return (
        "Convert the chart into a compact data table.\n"
        "Only include information needed for answering chart questions.\n"
        "Preserve original units and scales exactly.\n"
        "Do not convert percentages to decimals or decimals to percentages.\n"
        "Output a markdown table if possible.\n"
        "If there are multiple series, include a column for each series.\n"
        "Return only the table and very short notes about units if needed."
    )


def reason_prompt(table_text, question):
    return (
        "You are answering a chart question using the extracted chart table below.\n"
        "Use the table values exactly. Preserve the original unit and scale.\n"
        "Do not convert 59 to 0.59 or 0.59 to 59 unless explicitly asked.\n"
        "If arithmetic is required, compute it carefully.\n"
        "Return exactly one short final answer.\n\n"
        f"Extracted table:\n{table_text}\n\n"
        f"Question: {question}\n"
        "FINAL:"
    )


def direct_prompt(question):
    return (
        "Read the image carefully and answer the question.\n"
        "Return only the final short answer.\n"
        f"Question: {question}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output", default="results/eval/chartqa_deplot_reason_200.csv")
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

        # direct baseline from the same adapter
        direct_raw = generate_with_image(
            model,
            processor,
            image,
            direct_prompt(question),
            max_new_tokens=24,
        )
        direct_pred = clean_answer(direct_raw)
        direct_score = relaxed_score(direct_pred, answers)

        # DePlot-style table extraction
        table_text = generate_with_image(
            model,
            processor,
            image,
            derender_prompt(),
            max_new_tokens=256,
        )

        # table-only reasoning
        table_raw = generate_text_only(
            model,
            processor,
            reason_prompt(table_text, question),
            max_new_tokens=64,
        )
        table_pred = clean_answer(table_raw)
        table_score = relaxed_score(table_pred, answers)

        rows.append({
            "task": "chartqa",
            "adapter": "direct_sfg",
            "idx": idx,
            "question": question,
            "answers": json.dumps(to_answer_list(answers), ensure_ascii=False),
            "prediction": direct_pred,
            "score": direct_score,
            "table_text": "",
            "raw": direct_raw,
        })

        rows.append({
            "task": "chartqa",
            "adapter": "deplot_reason",
            "idx": idx,
            "question": question,
            "answers": json.dumps(to_answer_list(answers), ensure_ascii=False),
            "prediction": table_pred,
            "score": table_score,
            "table_text": table_text,
            "raw": table_raw,
        })

        print(f"[{idx}] direct={direct_score}:{direct_pred} | deplot={table_score}:{table_pred}")

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
