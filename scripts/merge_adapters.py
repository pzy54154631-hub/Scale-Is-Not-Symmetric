#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mirage.io import load_adapter, save_adapter
from mirage.merge import merge_adapters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA adapters with MIRAGE or baselines.")
    parser.add_argument("--method", required=True, choices=["average", "task_arithmetic", "ties", "dare", "breadcrumbs", "mirage", "mirage_canonical", "mirage_v2", "mirage_legacy", "mirage_raw"])
    parser.add_argument("--adapters", nargs="+", required=True, help="Paths to PEFT adapter directories.")
    parser.add_argument("--output", required=True, help="Output merged adapter directory.")
    parser.add_argument("--density", type=float, default=0.5, help="Density for TIES/DARE/Breadcrumbs baselines.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=float, nargs="*", default=None, help="Optional task arithmetic weights.")
    parser.add_argument("--mirage-beta", type=float, default=8.0)
    parser.add_argument("--mirage-margin", type=float, default=0.0, help="Only used by mirage_legacy.")
    parser.add_argument("--mirage-gamma", type=float, default=2.0, help="Sharedness reward for canonical MIRAGE.")
    parser.add_argument("--mirage-temperature", type=float, default=1.0, help="Softmax temperature for canonical MIRAGE.")
    parser.add_argument("--mirage-energy-power", type=float, default=0.5, help="How strongly singular-value energy affects MIRAGE weights.")
    parser.add_argument("--importance-json", type=str, default=None)
    parser.add_argument("--target-rank", type=int, default=None, help="Optional fixed-rank SVD compression. Can be memory-heavy.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda. MIRAGE structural merge usually works on CPU.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--unsafe-bin", action="store_true", help="Save adapter_model.bin instead of safetensors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if args.weights is not None and len(args.weights) != len(args.adapters):
        raise ValueError("--weights length must match --adapters length.")

    bundles = [load_adapter(path, device="cpu", dtype=dtype) for path in args.adapters]
    merged_state, merged_config = merge_adapters(
        bundles=bundles,
        method=args.method,
        density=args.density,
        seed=args.seed,
        weights=args.weights,
        mirage_beta=args.mirage_beta,
        mirage_margin=args.mirage_margin,
        mirage_gamma=args.mirage_gamma,
        mirage_temperature=args.mirage_temperature,
        mirage_energy_power=args.mirage_energy_power,
        importance_json=args.importance_json,
        target_rank=args.target_rank,
        device=args.device,
    )
    save_adapter(
        state=merged_state,
        config=merged_config,
        output_dir=args.output,
        template_dir=args.adapters[0],
        safe_serialization=not args.unsafe_bin,
    )
    print(f"Saved merged adapter to {args.output}")
    print(f"Method: {args.method}")
    print(f"Output rank r={merged_config.get('r')} alpha={merged_config.get('lora_alpha')}")


if __name__ == "__main__":
    main()
