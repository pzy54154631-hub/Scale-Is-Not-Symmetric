from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch

from .io import AdapterBundle, get_lora_scale, clone_config_for_rank
from .lora_utils import (
    TensorDict,
    CanonicalComponent,
    check_same_lora_structure,
    rank_component_cosine,
    canonical_component_cosine,
    canonical_lora_svd_from_factors,
    concatenate_rank_components,
    concatenate_canonical_components,
    recompress_lora_factors_via_small_svd,
)


def _mean_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack([t.float() for t in tensors], dim=0).mean(dim=0).to(tensors[0].dtype)


def _task_arithmetic_tensors(tensors: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
    if weights is None:
        weights = [1.0 / len(tensors)] * len(tensors)
    out = torch.zeros_like(tensors[0].float())
    for w, t in zip(weights, tensors):
        out += float(w) * t.float()
    return out.to(tensors[0].dtype)


def _ties_tensor(tensors: List[torch.Tensor], density: float = 0.5, eps: float = 1e-12) -> torch.Tensor:
    """TIES-style tensor merge: trim by magnitude, elect sign, average aligned values.

    This compact implementation captures the trim-elect-merge mechanism for adapter tensors.
    """
    X = torch.stack([t.float() for t in tensors], dim=0)
    if not (0 < density <= 1):
        raise ValueError("density must be in (0, 1].")

    if density < 1.0:
        flat_abs = X.abs().flatten(start_dim=1)
        k = max(1, int(math.ceil(density * flat_abs.shape[1])))
        thresh = torch.topk(flat_abs, k=k, dim=1).values[:, -1]
        mask = X.abs() >= thresh.view(-1, *([1] * (X.ndim - 1)))
        X = X * mask

    sign_votes = torch.sign(X).sum(dim=0)
    elected_sign = torch.sign(sign_votes)
    elected_sign[elected_sign == 0] = 1
    aligned = torch.sign(X) == elected_sign.unsqueeze(0)
    aligned = aligned & (X != 0)
    denom = aligned.float().sum(dim=0).clamp_min(1.0)
    merged = (X * aligned.float()).sum(dim=0) / denom
    return merged.to(tensors[0].dtype)


def _dare_tensor(tensors: List[torch.Tensor], density: float = 0.5, seed: int = 42) -> torch.Tensor:
    """DARE-style random pruning and rescaling before averaging."""
    if not (0 < density <= 1):
        raise ValueError("density must be in (0, 1].")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    outs = []
    for t in tensors:
        x = t.float().cpu()
        if density < 1.0:
            mask = torch.rand(x.shape, generator=gen) < density
            x = x * mask.float() / density
        outs.append(x)
    return torch.stack(outs, dim=0).mean(dim=0).to(tensors[0].dtype)


def _breadcrumb_tensor(tensors: List[torch.Tensor], density: float = 0.8, remove_top: float = 0.01) -> torch.Tensor:
    """Model Breadcrumbs-style sparse masking.

    Keeps medium-magnitude updates and removes very small and extreme outliers before averaging.
    """
    if not (0 < density <= 1):
        raise ValueError("density must be in (0, 1].")
    if not (0 <= remove_top < 1):
        raise ValueError("remove_top must be in [0, 1).")
    outs = []
    for t in tensors:
        x = t.float()
        flat = x.abs().flatten()
        if flat.numel() == 0:
            outs.append(x)
            continue
        low_q = max(0.0, 1.0 - density)
        high_q = max(low_q, 1.0 - remove_top)
        low = torch.quantile(flat, low_q)
        high = torch.quantile(flat, high_q)
        mask = (x.abs() >= low) & (x.abs() <= high)
        outs.append(x * mask.float())
    return torch.stack(outs, dim=0).mean(dim=0).to(tensors[0].dtype)


def merge_parameter_space(
    bundles: List[AdapterBundle],
    method: str,
    density: float = 0.5,
    seed: int = 42,
    weights: Optional[List[float]] = None,
) -> Tuple[TensorDict, Dict[str, Any]]:
    states = [b.state for b in bundles]
    keys = sorted(states[0].keys())
    for s in states[1:]:
        if sorted(s.keys()) != keys:
            raise ValueError("Adapter states have different keys. Use compatible LoRA adapters.")

    merged: TensorDict = {}
    for key in keys:
        tensors = [s[key] for s in states]
        if not torch.is_floating_point(tensors[0]):
            merged[key] = tensors[0].clone()
            continue
        if method == "average":
            merged[key] = _mean_tensors(tensors)
        elif method == "task_arithmetic":
            merged[key] = _task_arithmetic_tensors(tensors, weights=weights)
        elif method == "ties":
            merged[key] = _ties_tensor(tensors, density=density)
        elif method == "dare":
            merged[key] = _dare_tensor(tensors, density=density, seed=seed)
        elif method == "breadcrumbs":
            merged[key] = _breadcrumb_tensor(tensors, density=density)
        else:
            raise ValueError(f"Unsupported parameter-space merge method: {method}")

    return merged, dict(bundles[0].config)


def _load_importance(importance_json: Optional[str | Path]) -> Dict[str, Dict[str, float]]:
    if importance_json is None:
        return {}
    path = Path(importance_json)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Expected format:
    # {
    #   "module_name": {"adapter_0/rank_0": 1.0, "adapter_0/rank_1": 0.8, ...}
    # }
    return data


# -----------------------------------------------------------------------------
# MIRAGE legacy: original raw LoRA-factor rank decomposition.
# Kept only for ablations. The paper method should use merge_mirage_canonical().
# -----------------------------------------------------------------------------

def _component_weight_from_conflict_legacy(
    component_idx: int,
    adapter_idx: int,
    all_components: List[Tuple[int, int, torch.Tensor, torch.Tensor]],
    beta: float = 8.0,
    margin: float = 0.0,
    importance: float = 1.0,
) -> float:
    _, _, b, a = all_components[component_idx]
    best_per_other: Dict[int, float] = {}
    for j, (other_adapter, _, b2, a2) in enumerate(all_components):
        if j == component_idx or other_adapter == adapter_idx:
            continue
        cos = float(rank_component_cosine(b.float(), a.float(), b2.float(), a2.float()).clamp(-1, 1).item())
        if other_adapter not in best_per_other or cos > best_per_other[other_adapter]:
            best_per_other[other_adapter] = cos

    compatibility = 1.0 if not best_per_other else sum(best_per_other.values()) / len(best_per_other)
    gate = 1.0 / (1.0 + math.exp(-float(beta) * (compatibility - float(margin))))
    return float(importance) * gate


def merge_mirage_legacy(
    bundles: List[AdapterBundle],
    beta: float = 8.0,
    margin: float = 0.0,
    importance_json: Optional[str | Path] = None,
    target_rank: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[TensorDict, Dict[str, Any]]:
    """Original MIRAGE prototype using raw LoRA factors.

    This is useful as an ablation, but it is not the recommended paper method because
    raw LoRA factor components are non-identifiable: BA = (BQ)(Q^{-1}A).
    """
    states = [b.state for b in bundles]
    pairs = check_same_lora_structure(states)
    scales = [get_lora_scale(b.config) for b in bundles]
    importance = _load_importance(importance_json)

    merged: TensorDict = {}
    lora_keys = {p.key_a for p in pairs} | {p.key_b for p in pairs}
    for key, tensor in states[0].items():
        if key not in lora_keys:
            merged[key] = tensor.clone()

    out_rank = None
    for pair in pairs:
        all_components: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
        for adapter_idx, state in enumerate(states):
            A = state[pair.key_a].detach().to(device)
            B = state[pair.key_b].detach().to(device)
            r = A.shape[0]
            for rank_idx in range(r):
                a = A[rank_idx, :]
                b = B[:, rank_idx] * float(scales[adapter_idx])
                all_components.append((adapter_idx, rank_idx, b, a))

        weighted_components: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        for comp_idx, (adapter_idx, rank_idx, b, a) in enumerate(all_components):
            imp_key = f"adapter_{adapter_idx}/rank_{rank_idx}"
            imp = float(importance.get(pair.module, {}).get(imp_key, 1.0))
            w = _component_weight_from_conflict_legacy(
                component_idx=comp_idx,
                adapter_idx=adapter_idx,
                all_components=all_components,
                beta=beta,
                margin=margin,
                importance=imp,
            )
            weighted_components.append((b, a, w))

        B_cat, A_cat = concatenate_rank_components(weighted_components, device=device)
        if target_rank is not None:
            A_new, B_new = recompress_lora_factors_via_small_svd(A_cat.float(), B_cat.float(), rank=target_rank)
        else:
            A_new, B_new = A_cat, B_cat

        merged[pair.key_a] = A_new.detach().cpu().to(states[0][pair.key_a].dtype)
        merged[pair.key_b] = B_new.detach().cpu().to(states[0][pair.key_b].dtype)
        out_rank = A_new.shape[0]

    if out_rank is None:
        raise RuntimeError("No LoRA pairs were merged.")
    config = clone_config_for_rank(bundles[0].config, new_rank=out_rank)
    return merged, config


# -----------------------------------------------------------------------------
# MIRAGE v2: canonical SVD + conflict-regularized component weighting.
# This is the version aligned with the paper equations.
# -----------------------------------------------------------------------------

def _canonical_components_for_pair(
    states: List[TensorDict],
    scales: List[float],
    pair_key_a: str,
    pair_key_b: str,
    device: str,
) -> List[CanonicalComponent]:
    comps: List[CanonicalComponent] = []
    for adapter_idx, state in enumerate(states):
        A = state[pair_key_a].detach().to(device)
        B = state[pair_key_b].detach().to(device)
        adapter_comps = canonical_lora_svd_from_factors(A=A, B=B, scale=float(scales[adapter_idx]))
        for rank_idx, comp in enumerate(adapter_comps):
            comp.adapter_idx = adapter_idx
            comp.rank_idx = rank_idx
            comps.append(comp)
    return comps


def _component_statistics(
    comps: List[CanonicalComponent],
    comp_idx: int,
) -> Tuple[float, float]:
    """Return (sharedness, conflict) for one component using best cross-adapter matching.

    Components are unordered across adapters, so for each other adapter we compare against
    the best matching component. This is a lightweight approximation to Hungarian matching.
    """
    comp = comps[comp_idx]
    best_per_other: Dict[int, float] = {}
    for j, other in enumerate(comps):
        if j == comp_idx or other.adapter_idx == comp.adapter_idx:
            continue
        cos = float(canonical_component_cosine(comp, other).clamp(-1, 1).item())
        prev = best_per_other.get(other.adapter_idx)
        if prev is None or cos > prev:
            best_per_other[other.adapter_idx] = cos

    if not best_per_other:
        return 1.0, 0.0

    best_vals = list(best_per_other.values())
    sharedness = sum(max(0.0, v) for v in best_vals) / len(best_vals)
    conflict = sum(max(0.0, -v) for v in best_vals) / len(best_vals)
    return sharedness, conflict


def _mirage_softmax_weights(
    comps: List[CanonicalComponent],
    module: str,
    importance: Dict[str, Dict[str, float]],
    beta: float,
    gamma: float,
    temperature: float,
    energy_power: float,
    eps: float = 1e-12,
) -> List[float]:
    """Conflict-regularized component weights.

    The paper objective is:

      min_alpha -sum I_k alpha_k + lambda sum alpha_k alpha_m C_km + mu||alpha||^2.

    For a scalable implementation, we use its softmax relaxation:

      alpha_k ∝ exp((log I_k + p log sigma_k - beta*C_k + gamma*S_k) / tau),

    where C_k is matched cross-adapter conflict and S_k is sharedness.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    logits: List[float] = []
    for idx, comp in enumerate(comps):
        imp_key = f"adapter_{comp.adapter_idx}/rank_{comp.rank_idx}"
        imp = float(importance.get(module, {}).get(imp_key, 1.0))
        imp = max(imp, eps)
        sigma = max(float(comp.sigma.item()), eps)
        sharedness, conflict = _component_statistics(comps, idx)
        logit = math.log(imp) + float(energy_power) * math.log(sigma) - float(beta) * conflict + float(gamma) * sharedness
        logits.append(logit / float(temperature))

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps)
    return [x / z for x in exps]


def merge_mirage_canonical(
    bundles: List[AdapterBundle],
    beta: float = 8.0,
    gamma: float = 2.0,
    temperature: float = 1.0,
    energy_power: float = 0.5,
    importance_json: Optional[str | Path] = None,
    target_rank: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[TensorDict, Dict[str, Any]]:
    """MIRAGE v2: canonical low-rank, conflict-regularized LoRA merging.

    For each LoRA module and adapter, we compute a canonical SVD of the actual update
    Delta W = scale * B A using a QR-small-SVD trick. This avoids the non-identifiability
    of raw LoRA factors. We then estimate rank-wise sharedness/conflict through best-match
    cross-adapter cosine and compute a softmax approximation to a conflict-regularized
    merging objective. The result is re-factorized as one deployable LoRA adapter.
    """
    states = [b.state for b in bundles]
    pairs = check_same_lora_structure(states)
    scales = [get_lora_scale(b.config) for b in bundles]
    importance = _load_importance(importance_json)

    merged: TensorDict = {}
    lora_keys = {p.key_a for p in pairs} | {p.key_b for p in pairs}
    for key, tensor in states[0].items():
        if key not in lora_keys:
            merged[key] = tensor.clone()

    out_rank = None
    for pair in pairs:
        comps = _canonical_components_for_pair(
            states=states,
            scales=scales,
            pair_key_a=pair.key_a,
            pair_key_b=pair.key_b,
            device=device,
        )
        if not comps:
            raise RuntimeError(f"No canonical components for module {pair.module}")

        weights = _mirage_softmax_weights(
            comps=comps,
            module=pair.module,
            importance=importance,
            beta=beta,
            gamma=gamma,
            temperature=temperature,
            energy_power=energy_power,
        )
        B_cat, A_cat = concatenate_canonical_components(list(zip(comps, weights)), device=device)

        if target_rank is not None:
            A_new, B_new = recompress_lora_factors_via_small_svd(A_cat.float(), B_cat.float(), rank=target_rank)
        else:
            A_new, B_new = A_cat, B_cat

        merged[pair.key_a] = A_new.detach().cpu().to(states[0][pair.key_a].dtype)
        merged[pair.key_b] = B_new.detach().cpu().to(states[0][pair.key_b].dtype)
        out_rank = A_new.shape[0]

    if out_rank is None:
        raise RuntimeError("No LoRA pairs were merged.")
    config = clone_config_for_rank(bundles[0].config, new_rank=out_rank)
    return merged, config


# Backward-compatible alias. In v2, "mirage" means canonical MIRAGE.
def merge_mirage(*args: Any, **kwargs: Any) -> Tuple[TensorDict, Dict[str, Any]]:
    return merge_mirage_canonical(*args, **kwargs)


def merge_adapters(
    bundles: List[AdapterBundle],
    method: str,
    density: float = 0.5,
    seed: int = 42,
    weights: Optional[List[float]] = None,
    mirage_beta: float = 8.0,
    mirage_margin: float = 0.0,  # kept for compatibility with legacy CLI
    mirage_gamma: float = 2.0,
    mirage_temperature: float = 1.0,
    mirage_energy_power: float = 0.5,
    importance_json: Optional[str | Path] = None,
    target_rank: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[TensorDict, Dict[str, Any]]:
    method = method.lower()
    if method in {"average", "task_arithmetic", "ties", "dare", "breadcrumbs"}:
        return merge_parameter_space(
            bundles=bundles,
            method=method,
            density=density,
            seed=seed,
            weights=weights,
        )
    if method in {"mirage", "mirage_canonical", "mirage_v2"}:
        return merge_mirage_canonical(
            bundles=bundles,
            beta=mirage_beta,
            gamma=mirage_gamma,
            temperature=mirage_temperature,
            energy_power=mirage_energy_power,
            importance_json=importance_json,
            target_rank=target_rank,
            device=device,
        )
    if method in {"mirage_legacy", "mirage_raw"}:
        return merge_mirage_legacy(
            bundles=bundles,
            beta=mirage_beta,
            margin=mirage_margin,
            importance_json=importance_json,
            target_rank=target_rank,
            device=device,
        )
    raise ValueError(f"Unknown merge method: {method}")
