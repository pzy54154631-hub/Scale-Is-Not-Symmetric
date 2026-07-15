#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append("scripts")
import eval_scaleqa_structured_verifier as base  # noqa: E402


PROMPTS = {
    "candlik_direct": (
        "Answer the chart question with only the final numeric answer. "
        "Use the chart image and context. The answer may be a decimal, percent-scale number, or raw value.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\n\nFinal answer:"
    ),
    "candlik_scale_guard": (
        "Answer the chart question with the candidate whose numeric scale matches the chart. "
        "Check axis ticks, percent signs, units, title, legend, and whether the question asks for a conversion. "
        "Do not prefer larger or smaller values by default.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\n\nScale-correct answer:"
    ),
    "candlik_unit_first": (
        "First determine whether the chart displays a percent-number, decimal fraction, raw count/value, "
        "or thousand/million/billion-scaled value. Then answer in that displayed unit unless the question explicitly asks to convert.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
}


def prompt_text(processor, image, prompt: str) -> tuple[str, list, list]:
    image = base.normalize_image(image)
    msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msg)
    return text, image_inputs, video_inputs


def avg_logprob_candidate(model, processor, image, prompt: str, candidate: str) -> float:
    ptext, image_inputs, video_inputs = prompt_text(processor, image, prompt)
    cand = " " + str(candidate).strip()
    full = ptext + cand
    pref_inputs = processor(text=[ptext], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    full_inputs = processor(text=[full], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    pref_len = int(pref_inputs["input_ids"].shape[1])
    full_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in full_inputs.items()}
    with torch.no_grad():
        out = model(**full_inputs)
        logits = out.logits[0].float()
    ids = full_inputs["input_ids"][0]
    if pref_len >= len(ids):
        return -1e9
    labels = ids[pref_len:]
    pred_logits = logits[pref_len - 1 : len(ids) - 1]
    logp = torch.log_softmax(pred_logits, dim=-1)
    vals = logp[torch.arange(len(labels), device=labels.device), labels]
    return float(vals.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--candidate-csv", action="append", default=[])
    ap.add_argument("--context-csv", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapters/chartqa_answer_only_3000_r16")
    ap.add_argument("--adapter-name", default="")
    ap.add_argument("--max-candidates", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    pools = base.build_pool(args.candidate_csv, args.max_candidates, True)
    contexts = base.load_context(args.context_csv)
    print(f"[info] examples={len(df)} pools={len(pools)} contexts={len(contexts)} adapter={args.adapter}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    if args.adapter and args.adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    prefix = args.adapter_name.strip() or ("base" if not args.adapter or args.adapter.lower() == "none" else Path(args.adapter).name)
    rows = []
    for n, (idx, ex) in enumerate(df.iterrows(), start=1):
        cid = str(ex.get("candidate_id", idx))
        q = str(ex.get("query", ex.get("question", "")))
        answers = ex.get("label", ex.get("answers", ""))
        ctx = contexts.get(cid, "")
        cands = pools.get(cid, [])
        if not cands:
            cands = [("fallback", str(ex.get("prediction", "")) or "0")]
        cands = cands[: args.max_candidates]
        oracle = max(base.score(pred, answers) for _, pred in cands) if cands else 0
        rows.append(
            {
                "adapter": f"{prefix}:candlik_candidate_oracle",
                "idx": idx,
                "candidate_id": cid,
                "question": q,
                "answers": json.dumps(base.to_list(answers), ensure_ascii=False),
                "prediction": "",
                "raw": json.dumps(cands, ensure_ascii=False),
                "score": oracle,
                "scale_error_subtype": ex.get("scale_error_subtype", ""),
                "scale_directions": ex.get("scale_directions", ""),
                "n_candidates": len(cands),
            }
        )
        for pname, tmpl in PROMPTS.items():
            prompt = tmpl.format(context=ctx or "(not available)", question=q)
            scored = []
            for source, pred in cands:
                lp = avg_logprob_candidate(model, processor, ex["image"], prompt, pred)
                scored.append((lp, source, pred))
            best_lp, best_source, best_pred = max(scored, key=lambda x: x[0])
            rows.append(
                {
                    "adapter": f"{prefix}:{pname}",
                    "idx": idx,
                    "candidate_id": cid,
                    "question": q,
                    "answers": json.dumps(base.to_list(answers), ensure_ascii=False),
                    "prediction": best_pred,
                    "raw": json.dumps([[s, p, lp] for lp, s, p in scored], ensure_ascii=False),
                    "score": base.score(best_pred, answers),
                    "scale_error_subtype": ex.get("scale_error_subtype", ""),
                    "scale_directions": ex.get("scale_directions", ""),
                    "n_candidates": len(cands),
                }
            )
        if n % 25 == 0:
            tmp = pd.DataFrame(rows)
            msg = " ".join(
                f"{prefix}:{p}={tmp[tmp.adapter == f'{prefix}:{p}']['score'].mean():.4f}" for p in PROMPTS
            )
            print(f"[progress] {n}/{len(df)} {msg}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    summary = {
        "overall": res.groupby("adapter")["score"].mean().to_dict(),
        "by_direction": res.groupby(["adapter", "scale_directions"])["score"].mean().reset_index().to_dict("records"),
        "by_subtype": res.groupby(["adapter", "scale_error_subtype"])["score"].mean().reset_index().to_dict("records"),
        "n_examples": len(df),
    }
    out.with_suffix(out.suffix + ".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
