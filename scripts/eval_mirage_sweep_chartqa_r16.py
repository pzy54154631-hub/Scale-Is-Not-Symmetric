import os
import subprocess
from pathlib import Path

base_cmd = [
    "python", "scripts/eval_tiny_adapters.py",
    "--task", "chartqa",
    "--split", "val",
    "--max-samples", "200",
]

adapters = [
    "base=NONE",
    "average=outputs/merged/textvqa_chartqa_answer_only_3000_r16_average",
    "ties=outputs/merged/textvqa_chartqa_answer_only_3000_r16_ties",
    "dare=outputs/merged/textvqa_chartqa_answer_only_3000_r16_dare",
]

for p in sorted(Path("outputs/merged").glob("textvqa_chartqa_answer_only_3000_r16_mirage_b*")):
    adapters.append(f"{p.name}={p}")

out = "results/eval/chartqa_mirage_sweep_r16_200_eval.csv"

cmd = base_cmd + ["--output", out, "--adapters"] + adapters

env = os.environ.copy()
env["TRANSFORMERS_OFFLINE"] = "1"
env["HF_HUB_OFFLINE"] = "1"
env["MODEL_DIR"] = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"

print("Running ChartQA sweep eval with", len(adapters), "adapters")
subprocess.run(cmd, check=True, env=env)
