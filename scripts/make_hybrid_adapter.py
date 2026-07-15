import argparse
import json
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def use_v2_key(key, mode):
    if mode == "language_only":
        return "language_model.layers" in key

    if mode == "no_visual_mlp":
        return not (".visual.blocks." in key and ".mlp." in key)

    if mode == "visual_attn_only":
        return ".visual.blocks." in key and ".mlp." not in key

    if mode == "visual_mlp_only":
        return ".visual.blocks." in key and ".mlp." in key

    if mode == "all_v2":
        return True

    raise ValueError(f"Unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--average", required=True)
    parser.add_argument("--v2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    avg_dir = Path(args.average)
    v2_dir = Path(args.v2)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    avg = load_file(str(avg_dir / "adapter_model.safetensors"))
    v2 = load_file(str(v2_dir / "adapter_model.safetensors"))

    new_state = {}
    replaced = 0
    kept = 0

    for k, v in avg.items():
        if k in v2 and use_v2_key(k, args.mode):
            new_state[k] = v2[k]
            replaced += 1
        else:
            new_state[k] = v
            kept += 1

    save_file(new_state, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})

    cfg = json.loads((avg_dir / "adapter_config.json").read_text())
    cfg["r"] = 16
    cfg["lora_alpha"] = 32
    cfg["rank_pattern"] = {}
    cfg["alpha_pattern"] = {}
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    for p in avg_dir.iterdir():
        if p.name in ["adapter_model.safetensors", "adapter_config.json"]:
            continue
        dst = out_dir / p.name
        if p.is_file() and not dst.exists():
            shutil.copy2(p, dst)

    print("mode:", args.mode)
    print("output:", out_dir)
    print("replaced tensors:", replaced)
    print("kept tensors:", kept)
    print("Done.")


if __name__ == "__main__":
    main()
