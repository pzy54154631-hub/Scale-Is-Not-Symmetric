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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append("scripts")
import eval_scaleqa_structured_verifier as base  # noqa: E402


PROMPT = (
    "You are a careful chart-question verifier. Candidate answers may differ only by numeric scale, "
    "such as 0.61 vs 61.\n"
    "Use BOTH the image and the structured chart context. Pay special attention to axis tick labels, "
    "percent signs, legends, and units.\n"
    "Choose the candidate whose numeric scale matches the chart and question. Do not choose a candidate "
    "because it appears first; choose by chart evidence. Output only LETTER: X.\n\n"
    "Structured chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nLETTER:"
)


def num_key(item):
    tag, pred = item
    v = base.parse_num(pred)
    if v is None or not math.isfinite(v):
        return (1, 0.0, str(pred))
    return (0, abs(v), float(v))


def order_candidates(cands, mode):
    cands = list(cands)
    if mode == "source":
        return cands
    if mode == "numeric_asc":
        return sorted(cands, key=num_key)
    if mode == "numeric_desc":
        return list(reversed(sorted(cands, key=num_key)))
    if mode == "source_base_last":
        base_like = []
        other = []
        for tag, pred in cands:
            if str(tag).startswith("main:") or str(tag).startswith("tta:"):
                base_like.append((tag, pred))
            else:
                other.append((tag, pred))
        return other + base_like
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--candidate-csv", action="append", default=[])
    ap.add_argument("--context-csv", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapters/chartqa_answer_only_3000_r16")
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=32)
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

    modes = ["numeric_asc", "numeric_desc", "source_base_last"]
    rows = []
    for n, (idx, ex) in enumerate(df.iterrows(), start=1):
        cid = str(ex.get("candidate_id", idx))
        q = str(ex.get("query", ex.get("question", "")))
        answers = ex.get("label", ex.get("answers", ""))
        cands = pools.get(cid, [])
        if not cands:
            cands = [("fallback", str(ex.get("prediction", "")) or "0")]
        ctx = contexts.get(cid, "")
        for mode in modes:
            ordered = order_candidates(cands, mode)[: args.max_candidates]
            block = base.candidate_block(ordered)
            raw = base.gen(
                model,
                processor,
                ex["image"],
                PROMPT.format(context=ctx or "(not available)", question=q, candidates=block),
                args.max_new_tokens,
            )
            letter = base.parse_letter(raw, len(ordered))
            pred = ordered[base.LETTERS.index(letter)][1]
            rows.append(
                {
                    "adapter": f"structured_order_{mode}",
                    "idx": idx,
                    "candidate_id": cid,
                    "question": q,
                    "answers": json.dumps(base.to_list(answers), ensure_ascii=False),
                    "prediction": pred,
                    "raw": raw,
                    "score": base.score(pred, answers),
                    "scale_error_subtype": ex.get("scale_error_subtype", ""),
                    "scale_directions": ex.get("scale_directions", ""),
                    "n_candidates": len(ordered),
                }
            )
        if n % 25 == 0:
            tmp = pd.DataFrame(rows)
            print(
                f"[progress] {n}/{len(df)} "
                + " ".join(f"{m}={tmp[tmp.adapter == 'structured_order_' + m]['score'].mean():.4f}" for m in modes),
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
