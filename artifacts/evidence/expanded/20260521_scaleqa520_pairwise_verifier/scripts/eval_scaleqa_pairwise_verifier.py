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

PROMPT = (
    "You are comparing two candidate answers to a chart question. The two answers may differ only by numeric scale "
    "(for example, 0.61 vs 61). Use the chart image and structured context to decide which answer has the correct "
    "scale for the question. Do not prefer the larger number or the first answer by default.\n\n"
    "Structured chart context:\n{context}\n\n"
    "Question: {question}\n"
    "A. {a}\n"
    "B. {b}\n\n"
    "Return only LETTER: A or LETTER: B."
)


def parse_ab(raw: str) -> str:
    s = str(raw).upper()
    if "LETTER" in s:
        if "A" in s.split("LETTER", 1)[1][:8]:
            return "A"
        if "B" in s.split("LETTER", 1)[1][:8]:
            return "B"
    for ch in s:
        if ch in {"A", "B"}:
            return ch
    return "A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--candidate-csv", action="append", default=[])
    ap.add_argument("--context-csv", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--adapter", default="outputs/adapters/chartqa_answer_only_3000_r16")
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=16)
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

        votes = {i: 0 for i in range(len(cands))}
        raw_pairs = []
        # Compare the source/default candidate against each alternative. Alternate A/B
        # placement to reduce first-letter bias without doubling the generation cost.
        for j in range(1, len(cands)):
            if j % 2:
                ia, ib = 0, j
            else:
                ia, ib = j, 0
            raw = base.gen(
                model,
                processor,
                ex["image"],
                PROMPT.format(context=ctx or "(not available)", question=q, a=cands[ia][1], b=cands[ib][1]),
                args.max_new_tokens,
            )
            letter = parse_ab(raw)
            win = ia if letter == "A" else ib
            votes[win] += 1
            raw_pairs.append({"against": j, "ia": ia, "ib": ib, "raw": raw, "letter": letter, "winner": win})
        best_i = max(votes, key=lambda i: (votes[i], -i))
        pred = cands[best_i][1]
        rows.append(
            {
                "adapter": "pairwise_default_vs_alt",
                "idx": idx,
                "candidate_id": cid,
                "question": q,
                "answers": json.dumps(base.to_list(answers), ensure_ascii=False),
                "prediction": pred,
                "raw": json.dumps(raw_pairs, ensure_ascii=False),
                "score": base.score(pred, answers),
                "scale_error_subtype": ex.get("scale_error_subtype", ""),
                "scale_directions": ex.get("scale_directions", ""),
                "n_candidates": len(cands),
            }
        )
        if n % 25 == 0:
            tmp = pd.DataFrame(rows)
            print(f"[progress] {n}/{len(df)} pairwise={tmp.score.mean():.4f}", flush=True)

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
