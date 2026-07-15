#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conflict-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-layers", type=int, default=80)
    return parser.parse_args()


def layer_id(module: str) -> int:
    m = re.search(r"layers\.(\d+)", module)
    return int(m.group(1)) if m else 10**9


def main() -> None:
    args = parse_args()
    with Path(args.conflict_json).open("r", encoding="utf-8") as f:
        data = json.load(f)

    modules = sorted(data.keys(), key=layer_id)[: args.max_layers]
    pairs = sorted({p for mod in modules for p in data[mod].keys() if p.endswith("__conflict")})
    # Fall back to all keys for backward compatibility with older conflict JSON files.
    if not pairs:
        pairs = sorted({p for mod in modules for p in data[mod].keys()})
    matrix = np.zeros((len(pairs), len(modules)), dtype=float)
    for x, mod in enumerate(modules):
        for y, p in enumerate(pairs):
            matrix[y, x] = data[mod].get(p, np.nan)

    fig = plt.figure(figsize=(max(8, len(modules) * 0.22), max(4, len(pairs) * 0.35)))
    ax = plt.gca()
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xlabel("LoRA module / layer order")
    ax.set_ylabel("Task pair")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_xticks(range(0, len(modules), max(1, len(modules)//10)))
    ax.set_xticklabels([str(layer_id(modules[i])) if layer_id(modules[i]) < 10**9 else str(i) for i in range(0, len(modules), max(1, len(modules)//10))])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Canonical rank-wise conflict score")
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Saved heatmap to {args.output}")


if __name__ == "__main__":
    main()
