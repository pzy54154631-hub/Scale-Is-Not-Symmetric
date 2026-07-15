#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

sys.path.append("scripts")
import eval_scaleqa_structured_verifier as base  # noqa: E402

LETTERS = "ABCD"

PROMPTS = {
    "logprob_unit_first": (
        "You are selecting the correct answer to a chart question. Candidate answers may differ by scale.\n"
        "Use the chart image and context to determine the displayed numeric unit: percent, decimal fraction, raw value, "
        "thousands/millions, or another unit. Then choose the candidate in that unit.\n"
        "Return one letter only.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nCorrect letter:"
    ),
    "logprob_balanced_fewshot": (
        "Choose the candidate with the correct numeric scale for the chart.\n"
        "Examples of scale choices:\n"
        "- If the chart shows 61 or 61%, choose 61 rather than 0.61.\n"
        "- If the chart shows 0.61 as a fraction/proportion, choose 0.61 rather than 61.\n"
        "- If the question asks for the displayed value, do not convert units unless the chart/question explicitly says so.\n"
        "Now solve the actual chart question. Return one letter only.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nCorrect letter:"
    ),
    "logprob_axis_ticks": (
        "Inspect the chart axis ticks, labels, title, legend, and visible value scale. "
        "Select the candidate whose magnitude could plausibly be read from that scale. "
        "Do not choose only because it is the largest, smallest, or first candidate. Return one letter only.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nCorrect letter:"
    ),
}


def letter_token_ids(tokenizer, letter: str) -> list[int]:
    ids = []
    for s in [letter, " " + letter, letter + ".", " " + letter + "."]:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if toks:
            ids.append(toks[0])
    return sorted(set(ids))


def score_letters(model, processor, image, prompt: str, n: int) -> tuple[str, dict[str, float]]:
    image = base.normalize_image(image)
    msg = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msg)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
    vals: dict[str, float] = {}
    for letter in LETTERS[:n]:
        ids = letter_token_ids(processor.tokenizer, letter)
        vals[letter] = float(torch.logsumexp(logp[ids], dim=0).item()) if ids else -1e9
    best = max(vals, key=vals.get)
    return best, vals


def candidate_block(cands):
    return "\n".join(f"{LETTERS[i]}. {pred}" for i, (_, pred) in enumerate(cands[: len(LETTERS)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--candidate-csv", action="append", default=[])
    ap.add_argument("--context-csv", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapters/chartqa_answer_only_3000_r16")
    ap.add_argument("--max-candidates", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    pools = base.build_pool(args.candidate_csv, args.max_candidates, True)
    contexts = base.load_context(args.context_csv)
    print(f"[info] examples={len(df)} pools={len(pools)} contexts={len(contexts)}", flush=True)

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

    rows = []
    for n, (idx, ex) in enumerate(df.iterrows(), start=1):
        cid = str(ex.get("candidate_id", idx))
        q = str(ex.get("query", ex.get("question", "")))
        answers = ex.get("label", ex.get("answers", ""))
        cands = pools.get(cid, [])
        if not cands:
            cands = [("fallback", str(ex.get("prediction", "")) or "0")]
        cands = cands[: args.max_candidates]
        ctx = contexts.get(cid, "")
        block = candidate_block(cands)
        oracle = max(base.score(pred, answers) for _, pred in cands) if cands else 0
        rows.append(
            {
                "adapter": "logprob_candidate_oracle",
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
            letter, scores = score_letters(
                model,
                processor,
                ex["image"],
                tmpl.format(context=ctx or "(not available)", question=q, candidates=block),
                len(cands),
            )
            pred = cands[LETTERS.index(letter)][1]
            rows.append(
                {
                    "adapter": pname,
                    "idx": idx,
                    "candidate_id": cid,
                    "question": q,
                    "answers": json.dumps(base.to_list(answers), ensure_ascii=False),
                    "prediction": pred,
                    "raw": json.dumps({"letter": letter, "logprobs": scores}, ensure_ascii=False),
                    "score": base.score(pred, answers),
                    "scale_error_subtype": ex.get("scale_error_subtype", ""),
                    "scale_directions": ex.get("scale_directions", ""),
                    "n_candidates": len(cands),
                }
            )
        if n % 25 == 0:
            tmp = pd.DataFrame(rows)
            print(
                f"[progress] {n}/{len(df)} "
                + " ".join(f"{p}={tmp[tmp.adapter == p]['score'].mean():.4f}" for p in PROMPTS),
                flush=True,
            )

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
