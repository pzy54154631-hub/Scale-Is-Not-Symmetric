#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append("scripts")
import eval_scaleqa_structured_verifier as base  # noqa: E402


PROMPTS = {
    "regime_unit_first": (
        "You are selecting the scale-correct answer to a chart question.\n"
        "Step 1: from the chart image and context, identify the displayed unit/scale "
        "(percent, decimal fraction, raw value, thousands/millions, etc.).\n"
        "Step 2: decide whether the question asks for the displayed value or a conversion.\n"
        "Step 3: choose the candidate whose numeric scale matches that decision.\n"
        "Do not prefer large or small values by default. Output only LETTER: X.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nLETTER:"
    ),
    "regime_eliminate": (
        "A chart QA model may confuse scales such as 0.61 and 61. Eliminate candidates "
        "that use the wrong chart unit. Check axis ticks, labels, legend, title, and percent signs.\n"
        "If the chart shows percentages, choose the percentage-number candidate; if it shows "
        "fractions or proportions, choose the decimal candidate; otherwise choose the raw displayed value.\n"
        "Return exactly one candidate letter, formatted as LETTER: X.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nLETTER:"
    ),
    "regime_default_safe": (
        "Choose the answer candidate that is numerically correct for this chart. "
        "Use the image first, then the context. A candidate can be wrong even if it has the right digits "
        "but the decimal point or percent scale is wrong.\n"
        "When the evidence is ambiguous, prefer the candidate closest to the chart's visible tick/value scale, "
        "not necessarily the largest or smallest number. Output only LETTER: X.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nLETTER:"
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--candidate-csv", action="append", default=[])
    ap.add_argument("--context-csv", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapters/chartqa_answer_only_3000_r16")
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=20)
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
        block = base.candidate_block(cands)
        oracle = max(base.score(pred, answers) for _, pred in cands) if cands else 0
        rows.append(
            {
                "adapter": "regime_v3_candidate_oracle",
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
            raw = base.gen(
                model,
                processor,
                ex["image"],
                tmpl.format(context=ctx or "(not available)", question=q, candidates=block),
                args.max_new_tokens,
            )
            letter = base.parse_letter(raw, len(cands))
            pred = cands[base.LETTERS.index(letter)][1]
            rows.append(
                {
                    "adapter": pname,
                    "idx": idx,
                    "candidate_id": cid,
                    "question": q,
                    "answers": json.dumps(base.to_list(answers), ensure_ascii=False),
                    "prediction": pred,
                    "raw": raw,
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
