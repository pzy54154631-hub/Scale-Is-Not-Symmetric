import os
import json
import subprocess
from pathlib import Path

os.makedirs("results/eval/sweep_chartqa_r16", exist_ok=True)

env = os.environ.copy()
env["TRANSFORMERS_OFFLINE"] = "1"
env["HF_HUB_OFFLINE"] = "1"
env["MODEL_DIR"] = "/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct"
env["HF_ENDPOINT"] = "https://hf-mirror.com"
env["HF_HOME"] = "/root/autodl-tmp/hf_cache"
env["HF_HUB_CACHE"] = "/root/autodl-tmp/hf_cache"
env["HF_DATASETS_CACHE"] = "/root/autodl-tmp/datasets"

items = [
    ("base", "NONE"),
    ("average", "outputs/merged/textvqa_chartqa_answer_only_3000_r16_average"),
    ("ties", "outputs/merged/textvqa_chartqa_answer_only_3000_r16_ties"),
    ("dare", "outputs/merged/textvqa_chartqa_answer_only_3000_r16_dare"),
]

for p in sorted(Path("outputs/merged").glob("textvqa_chartqa_answer_only_3000_r16_mirage_b*")):
    safe_name = p.name.replace(".", "p").replace("-", "_")
    items.append((safe_name, str(p)))

results = {}

for name, adapter_path in items:
    print(f"\n=== Evaluating {name} ===")
    out_csv = f"results/eval/sweep_chartqa_r16/{name}.csv"
    out_summary = f"results/eval/sweep_chartqa_r16/{name}.summary.json"

    cmd = [
        "python", "scripts/eval_tiny_adapters.py",
        "--task", "chartqa",
        "--split", "val",
        "--max-samples", "200",
        "--output", out_csv,
        "--adapters",
        f"{name}={adapter_path}",
    ]

    try:
        subprocess.run(cmd, check=True, env=env)
        data = json.loads(Path(out_summary).read_text())
        score = list(data.values())[0]
        results[name] = score
        print(f"OK {name}: {score}")
    except Exception as e:
        results[name] = None
        print(f"FAILED {name}: {repr(e)}")

Path("results/eval/chartqa_mirage_sweep_r16_200_safe_summary.json").write_text(
    json.dumps(results, indent=2)
)

print("\n=== TOP RESULTS ===")
valid = [(k, v) for k, v in results.items() if v is not None]
valid = sorted(valid, key=lambda x: x[1], reverse=True)
for k, v in valid[:20]:
    print(f"{k:90s} {v:.4f}")
