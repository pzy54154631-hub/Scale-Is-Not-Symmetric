from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import torch

try:
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
except Exception:  # pragma: no cover - lets .bin workflows run before safetensors is installed
    safe_load_file = None
    safe_save_file = None

TensorDict = Dict[str, torch.Tensor]


@dataclass
class AdapterBundle:
    path: Path
    config: Dict[str, Any]
    state: TensorDict


def _find_state_file(adapter_dir: Path) -> Path:
    st = adapter_dir / "adapter_model.safetensors"
    pt = adapter_dir / "adapter_model.bin"
    if st.exists():
        return st
    if pt.exists():
        return pt
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {adapter_dir}")


def load_adapter(adapter_dir: str | Path, device: str = "cpu", dtype: torch.dtype | None = None) -> AdapterBundle:
    adapter_dir = Path(adapter_dir)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    state_path = _find_state_file(adapter_dir)
    if state_path.suffix == ".safetensors":
        if safe_load_file is None:
            raise ImportError("safetensors is required to load adapter_model.safetensors. Install it or use adapter_model.bin.")
        state = safe_load_file(str(state_path), device=device)
    else:
        state = torch.load(state_path, map_location=device)

    if dtype is not None:
        state = {k: v.to(dtype=dtype) if torch.is_floating_point(v) else v for k, v in state.items()}

    return AdapterBundle(path=adapter_dir, config=config, state=state)


def save_adapter(
    state: TensorDict,
    config: Dict[str, Any],
    output_dir: str | Path,
    template_dir: str | Path | None = None,
    safe_serialization: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer or metadata files if present. The adapter itself only requires config + weights.
    if template_dir is not None:
        template_dir = Path(template_dir)
        for name in ["README.md", "tokenizer_config.json", "special_tokens_map.json"]:
            src = template_dir / name
            if src.exists() and not (output_dir / name).exists():
                shutil.copy2(src, output_dir / name)

    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    cpu_state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("safetensors is required for safe serialization. Use --unsafe-bin or install safetensors.")
        safe_save_file(cpu_state, str(output_dir / "adapter_model.safetensors"))
    else:
        torch.save(cpu_state, output_dir / "adapter_model.bin")


def get_lora_scale(config: Dict[str, Any]) -> float:
    r = config.get("r", None)
    alpha = config.get("lora_alpha", None)
    if r is None or alpha is None:
        # Most PEFT LoRA uses scaling = alpha / r. If unavailable, assume matrices are unscaled.
        return 1.0
    return float(alpha) / float(r)


def clone_config_for_rank(config: Dict[str, Any], new_rank: int) -> Dict[str, Any]:
    out = dict(config)
    # We store merged A/B such that B @ A equals the actual merged update.
    # Setting alpha = r makes PEFT scaling alpha/r = 1.
    out["r"] = int(new_rank)
    out["lora_alpha"] = int(new_rank)
    return out
