import argparse
import csv
import json
import os
import re
import torch
from datasets import load_dataset
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

BASE_MODEL_DIR = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"

def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9.]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def relaxed_score(pred, answers):
    import json
    import re

    def norm(x):
        x = str(x).lower().strip()
        x = re.sub(r"[^a-z0-9.\-/% ]+", " ", x)
        x = re.sub(r"\s+", " ", x).strip()
        return x

    def has_number(x):
        return re.search(r"-?\d+(?:\.\d+)?", str(x)) is not None

    def extract_number(x):
        x = str(x).replace(",", "")
        m = re.search(r"-?\d+(?:\.\d+)?", x)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    if isinstance(answers, str):
        try:
            ans_list = json.loads(answers)
        except Exception:
            ans_list = [answers]
    elif isinstance(answers, list):
        ans_list = answers
    else:
        ans_list = [answers]

    pred_n = norm(pred)

    for ans in ans_list:
        ans_n = norm(ans)

        # exact normalized match
        if pred_n == ans_n:
            return 1

        p_num = extract_number(pred_n)
        a_num = extract_number(ans_n)

        # numeric relaxed match: 5% tolerance
        if p_num is not None and a_num is not None:
            tol = max(0.05 * abs(a_num), 1e-3)
            if abs(p_num - a_num) <= tol:
                return 1
            # Important: do NOT allow substring match for numeric answers.
            continue

        # textual relaxed containment only when neither side is numeric
        if not has_number(pred_n) and not has_number(ans_n):
            if len(ans_n) >= 3 and (ans_n in pred_n or pred_n in ans_n):
                return 1

    return 0

def load_examples(task, max_samples, split):
    if task == "textvqa":
        ds = load_dataset("lmms-lab/textvqa", split=f"{split}[:{max_samples}]")
        rows = []
        for ex in ds:
            rows.append({
                "image": ex["image"],
                "question": ex["question"],
                "answers": ex["answers"],
            })
        return rows

    if task == "chartqa":
        ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{split}[:{max_samples}]")
        rows = []
        for ex in ds:
            rows.append({
                "image": ex["image"],
                "question": ex["query"],
                "answers": ex["label"],
            })
        return rows

    raise ValueError(f"Unknown task: {task}")

def generate_answer(model, processor, image, question, max_new_tokens=24):
    if isinstance(image, Image.Image):
        image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + " Answer briefly."},
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
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0].strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["textvqa", "chartqa"])
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapters", nargs="+", required=True)
    args = parser.parse_args()

    if args.split is None:
        args.split = "validation" if args.task == "textvqa" else "val"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading examples...")
    examples = load_examples(args.task, args.max_samples, args.split)
    print(f"Loaded {len(examples)} examples.")

    print("Loading processor...")
    processor_kwargs = {"local_files_only": True}
    min_pixels = os.environ.get("MIN_PIXELS")
    max_pixels = os.environ.get("MAX_PIXELS")

    if min_pixels is not None:
        processor_kwargs["min_pixels"] = int(min_pixels)

    if max_pixels is not None:
        processor_kwargs["max_pixels"] = int(max_pixels)

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_DIR,
        **processor_kwargs,
    )

    print(
        "Processor pixel settings:",
        "min_pixels=", processor_kwargs.get("min_pixels"),
        "max_pixels=", processor_kwargs.get("max_pixels"),
    )
    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    base_model.eval()

    adapter_specs = []
    for item in args.adapters:
        name, path = item.split("=", 1)
        adapter_specs.append((name, path))

    results = []
    peft_model = None
    loaded_adapters = set()

    for adapter_name, adapter_path in adapter_specs:
        print(f"\n=== Evaluating {adapter_name} ===")

        if adapter_path == "NONE":
            model = base_model
        else:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    adapter_name=adapter_name,
                    local_files_only=True,
                )
                loaded_adapters.add(adapter_name)
            elif adapter_name not in loaded_adapters:
                peft_model.load_adapter(
                    adapter_path,
                    adapter_name=adapter_name,
                    is_trainable=False,
                    local_files_only=True,
                )
                loaded_adapters.add(adapter_name)

            peft_model.set_adapter(adapter_name)
            peft_model.eval()
            model = peft_model

        correct = 0

        for i, ex in enumerate(examples):
            pred = generate_answer(model, processor, ex["image"], ex["question"])
            score = relaxed_score(pred, ex["answers"])
            correct += score

            row = {
                "task": args.task,
                "adapter": adapter_name,
                "idx": i,
                "question": ex["question"],
                "answers": json.dumps(ex["answers"], ensure_ascii=False),
                "prediction": pred,
                "score": score,
            }
            results.append(row)

            print(f"[{adapter_name}] {i}: score={score} pred={pred[:80]}")

        acc = correct / max(1, len(examples))
        print(f"SUMMARY {adapter_name}: {correct}/{len(examples)} = {acc:.4f}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "adapter", "idx", "question", "answers", "prediction", "score"],
        )
        writer.writeheader()
        writer.writerows(results)

    summary = {}
    for adapter_name, _ in adapter_specs:
        rows = [r for r in results if r["adapter"] == adapter_name]
        summary[adapter_name] = sum(r["score"] for r in rows) / max(1, len(rows))

    summary_path = args.output.replace(".csv", ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved:", args.output)
    print("Saved:", summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
