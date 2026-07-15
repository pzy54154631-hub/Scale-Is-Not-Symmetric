#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lmms-eval for a base MLLM + optional PEFT adapter.")
    parser.add_argument("--config", default="configs/tasks.yaml")
    parser.add_argument("--model-name", required=True, help="lmms-eval model name, e.g., qwen2_5_vl. Edit as needed.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tasks", nargs="+", required=True, help="Logical task names defined in configs/tasks.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    task_map: Dict[str, Any] = cfg["tasks"]
    lmms_tasks = []
    for name in args.tasks:
        if name not in task_map:
            raise KeyError(f"Task {name} not found in {cfg_path}. Available: {list(task_map)}")
        lmms_tasks.append(task_map[name]["lmms_eval_name"])

    model_args = [f"pretrained={args.base_model}"]
    if args.adapter:
        # You may need to change this according to your lmms-eval backend.
        # Many backends accept peft=... or adapter=...; verify locally.
        model_args.append(f"peft={args.adapter}")

    cmd = [
        "lmms-eval",
        "--model", args.model_name,
        "--model_args", ",".join(model_args),
        "--tasks", ",".join(lmms_tasks),
        "--num_fewshot", str(args.num_fewshot),
        "--batch_size", str(args.batch_size),
        "--output_path", str(Path(args.output).parent),
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        with Path(args.output).open("w", encoding="utf-8") as f:
            json.dump({"dry_run_command": cmd}, f, indent=2)
        return

    subprocess.run(cmd, check=True)

    # lmms-eval usually writes its own JSON in output_path. This marker file helps table scripts.
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump({"command": cmd, "note": "Find actual lmms-eval result JSON in the same output directory."}, f, indent=2)


if __name__ == "__main__":
    main()
