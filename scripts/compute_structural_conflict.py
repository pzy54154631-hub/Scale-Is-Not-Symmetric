#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mirage.io import load_adapter, get_lora_scale
from mirage.lora_utils import check_same_lora_structure, canonical_lora_svd_from_factors, canonical_component_cosine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute canonical layer-wise/rank-wise structural conflict among LoRA adapters.")
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundles = [load_adapter(p, device="cpu", dtype=torch.float32) for p in args.adapters]
    names = args.names or [f"adapter_{i}" for i in range(len(bundles))]
    states = [b.state for b in bundles]
    scales = [get_lora_scale(b.config) for b in bundles]
    pairs = check_same_lora_structure(states)

    out: Dict[str, Dict[str, float]] = {}
    for pair in pairs:
        module_out: Dict[str, float] = {}
        all_comps = []
        for i, state in enumerate(states):
            comps = canonical_lora_svd_from_factors(
                A=state[pair.key_a].to(args.device),
                B=state[pair.key_b].to(args.device),
                scale=float(scales[i]),
            )
            for ridx, comp in enumerate(comps):
                comp.adapter_idx = i
                comp.rank_idx = ridx
            all_comps.append(comps)

        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                vals = []
                neg_vals = []
                # Components are unordered, so compare each component to its best match.
                for ci in all_comps[i]:
                    best = -1.0
                    for cj in all_comps[j]:
                        cos = float(canonical_component_cosine(ci, cj).clamp(-1, 1).item())
                        best = max(best, cos)
                    vals.append(best)
                    neg_vals.append(max(0.0, -best))
                mean_best_cos = sum(vals) / max(1, len(vals))
                conflict = sum(neg_vals) / max(1, len(neg_vals))
                module_out[f"{names[i]}__vs__{names[j]}__mean_best_cos"] = mean_best_cos
                module_out[f"{names[i]}__vs__{names[j]}__conflict"] = conflict
        out[pair.module] = module_out

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved canonical conflict JSON to {args.output_json}")


if __name__ == "__main__":
    main()
