#!/usr/bin/env python
"""Calibration importance scaffold for MIRAGE.

The full MLLM-specific version depends on the model class and processor. This script creates
an importance JSON template with all component weights initialized to 1.0. You can later replace
these values using activation-difference or gradient-based calibration.

Expected output format:
{
  "module_name": {
    "adapter_0/rank_0": 1.0,
    "adapter_0/rank_1": 1.0,
    ...
  }
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mirage.io import load_adapter
from mirage.lora_utils import check_same_lora_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundles = [load_adapter(p, device="cpu", dtype=torch.float32) for p in args.adapters]
    states = [b.state for b in bundles]
    pairs = check_same_lora_structure(states)

    out = {}
    for pair in pairs:
        module_scores = {}
        for adapter_idx, state in enumerate(states):
            rank = state[pair.key_a].shape[0]
            for rank_idx in range(rank):
                module_scores[f"adapter_{adapter_idx}/rank_{rank_idx}"] = 1.0
        out[pair.module] = module_scores

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved importance template to {args.output_json}")


if __name__ == "__main__":
    main()
