import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def first_answer(x):
    if isinstance(x, list):
        return str(x[0]) if x else ""
    return str(x)


def infer_qtype(q):
    ql = q.lower()
    if any(w in ql for w in ["sum", "total", "add up", "combined"]):
        return "sum"
    if any(w in ql for w in ["difference", "how much more", "how much less"]):
        return "difference"
    if any(w in ql for w in ["how many", "number of", "count"]):
        return "count"
    if any(w in ql for w in ["highest", "maximum", "largest", "greatest"]):
        return "max"
    if any(w in ql for w in ["lowest", "minimum", "smallest", "least"]):
        return "min"
    if any(w in ql for w in ["which year", "what year", "year"]):
        return "year"
    if any(w in ql for w in ["color", "line", "bar", "legend"]):
        return "legend_or_label"
    if any(w in ql for w in ["times", "ratio", "percent", "percentage"]):
        return "ratio_or_scale"
    return "direct_read"


def make_pseudo_csr(question, answer):
    qtype = infer_qtype(question)
    return {
        "chart_type": "unknown",
        "question_type": qtype,
        "relevant_entities": [],
        "extracted_values": [],
        "operation": qtype,
        "computed_answer": str(answer),
        "final_answer": str(answer),
    }


def load_synth(jsonl, max_samples):
    rows = []
    if not jsonl:
        return rows

    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            ex = json.loads(line)
            rows.append({
                "source": "synthetic_csr",
                "image": Image.open(ex["image"]).convert("RGB"),
                "question": ex["question"],
                "answer": ex["answer"],
                "csr_target": ex["csr_target"],
                "mode": "csr",
            })
    return rows


def load_chartqa(split, max_samples, mode="csr"):
    ds = load_dataset("HuggingFaceM4/ChartQA", split=f"{split}[:{max_samples}]")
    rows = []
    for ex in ds:
        answer = first_answer(ex["label"])
        rows.append({
            "source": "chartqa",
            "image": ex["image"],
            "question": ex["query"],
            "answer": answer,
            "csr_target": make_pseudo_csr(ex["query"], answer),
            "mode": mode,
        })
    return rows


def load_textvqa(split, max_samples):
    ds = load_dataset("lmms-lab/textvqa", split=f"{split}[:{max_samples}]")
    rows = []
    for ex in ds:
        answer = first_answer(ex["answers"])
        rows.append({
            "source": "textvqa",
            "image": ex["image"],
            "question": ex["question"],
            "answer": answer,
            "csr_target": None,
            "mode": "answer",
        })
    return rows


def csr_prompt(question):
    return (
        "Read the chart carefully and solve the question using structured reasoning.\n"
        "Return a compact JSON object with exactly these keys:\n"
        "chart_type, question_type, relevant_entities, extracted_values, operation, computed_answer, final_answer.\n"
        "Preserve the original unit and scale shown in the chart.\n"
        "Do not convert 59 to 0.59 or 0.59 to 59 unless the question explicitly asks.\n"
        f"Question: {question}"
    )


def answer_prompt(question):
    return (
        "Read the image carefully and answer the question.\n"
        "Return only the final short answer.\n"
        f"Question: {question}"
    )


def make_messages(ex):
    if ex["mode"] == "csr":
        user_text = csr_prompt(ex["question"])
        target = json.dumps(ex["csr_target"], ensure_ascii=False, separators=(",", ":"))
    else:
        user_text = answer_prompt(ex["question"])
        target = str(ex["answer"])

    prompt_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image"]},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    full_messages = prompt_messages + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        }
    ]

    return prompt_messages, full_messages, target


def get_device(model):
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(model, processor, ex):
    prompt_messages, full_messages, _ = make_messages(ex)

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

    out = model(**inputs, labels=labels)
    return out.loss


def read_target_modules(adapter_dir):
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    return cfg.get("target_modules")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--base-adapter", default=None)
    parser.add_argument("--target-config-adapter", default="outputs/adapters/chartqa_answer_only_3000_r16")
    parser.add_argument("--output", required=True)

    parser.add_argument("--synthetic-jsonl", default="datasets/csr_synth/csr_synth.jsonl")
    parser.add_argument("--synthetic-samples", type=int, default=1000)
    parser.add_argument("--chart-samples", type=int, default=500)
    parser.add_argument("--real-csr-samples", type=int, default=0)
    parser.add_argument("--chart-answer-samples", type=int, default=1000)
    parser.add_argument("--text-samples", type=int, default=500)

    parser.add_argument("--chart-split", default="val")
    parser.add_argument("--text-split", default="validation")

    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--chart-weight", type=float, default=1.2)
    parser.add_argument("--synth-weight", type=float, default=1.0)
    parser.add_argument("--text-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)

    print("Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False

    if args.base_adapter:
        print("Continuing from adapter:", args.base_adapter)
        model = PeftModel.from_pretrained(
            model,
            args.base_adapter,
            is_trainable=True,
        )
    else:
        print("Adding new LoRA adapter...")
        target_modules = read_target_modules(args.target_config_adapter)
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        print("target_modules:", target_modules)

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)

    model.train()
    model.print_trainable_parameters()

    print("Loading training data...")
    rows = []

    # Synthetic charts have real structure labels, so they are safe for CSR JSON supervision.
    rows += load_synth(args.synthetic_jsonl, args.synthetic_samples)

    # Real ChartQA does NOT have true extracted_values / operation labels.
    # Therefore, use pseudo-CSR only if explicitly enabled.
    if args.real_csr_samples > 0:
        rows += load_chartqa(args.chart_split, args.real_csr_samples, mode="csr")

    # Real ChartQA should mainly be answer-only supervision.
    if args.chart_answer_samples > 0:
        rows += load_chartqa(args.chart_split, args.chart_answer_samples, mode="answer")

    # TextVQA replay prevents OCR/text ability collapse.
    rows += load_textvqa(args.text_split, args.text_samples)

    random.shuffle(rows)

    print("Total training rows:", len(rows))
    print("Sources:", {s: sum(r["source"] == s for r in rows) for s in sorted(set(r["source"] for r in rows))})

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    print("Training CSR...")
    opt.zero_grad()

    for step in range(args.steps):
        ex = rows[step % len(rows)]

        loss = compute_loss(model, processor, ex)

        if ex["source"] == "synthetic_csr":
            w = args.synth_weight
        elif ex["source"] == "chartqa":
            w = args.chart_weight
        else:
            w = args.text_weight

        total = loss * w / args.grad_accum
        total.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()

        if step % 10 == 0 or step == args.steps - 1:
            print(
                f"step={step:04d} source={ex['source']} mode={ex['mode']} "
                f"loss={loss.item():.4f} weighted={(loss*w).item():.4f}"
            )

    print("Saving adapter:", out_dir)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    meta = vars(args)
    (out_dir / "mirage_csr_train_meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
