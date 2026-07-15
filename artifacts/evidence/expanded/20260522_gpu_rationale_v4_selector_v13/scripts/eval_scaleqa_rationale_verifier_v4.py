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
    "rationale_unit_table": (
        "You are a strict chart-scale verifier. Candidate answers may contain the same digits but the wrong scale.\n"
        "First identify the chart's visible numeric regime: percent-number (e.g. 61), decimal fraction (e.g. 0.61), "
        "raw count/value, or unit-scaled value (thousand/million/billion). Then compare candidates against that regime.\n"
        "Be especially careful: a percent chart usually displays 61 rather than 0.61; a fraction chart displays 0.61 rather than 61. "
        "Do not prefer larger or smaller numbers by default.\n"
        "Return a short reason and then exactly 'LETTER: X'.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nReason and LETTER:"
    ),
    "rationale_counterfactual": (
        "Solve this as a scale-conflict multiple-choice problem. For each candidate, ask: would this value be plausible if read "
        "directly from the chart axis/ticks/legend/title? Would it require multiplying or dividing by 10, 100, or 1000 without the "
        "question asking for that conversion? Reject such candidates.\n"
        "If evidence is ambiguous, choose the answer consistent with the chart's displayed units, not the numerically transformed variant. "
        "End with exactly 'LETTER: X'.\n\n"
        "Chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nAnalysis and LETTER:"
    ),
    "rationale_default_guard": (
        "Choose the candidate answer for this chart question. Treat scale changes as dangerous: only switch from the model's apparent "
        "default answer if the chart context or image strongly indicates a decimal/percent/unit mismatch. Check percent signs, axis tick "
        "magnitudes, legend units, and whether the question asks for an arithmetic operation.\n"
        "Output one final choice as 'LETTER: X'.\n\n"
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
    ap.add_argument("--adapter-name", default="")
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=80)
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

    rows = []
    prefix = args.adapter_name.strip() or ("base" if not args.adapter or args.adapter.lower() == "none" else Path(args.adapter).name)
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
                "adapter": f"{prefix}:rationale_candidate_oracle",
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
                    "adapter": f"{prefix}:{pname}",
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
            msg = " ".join(f"{prefix}:{p}={tmp[tmp.adapter == f'{prefix}:{p}']['score'].mean():.4f}" for p in PROMPTS)
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
