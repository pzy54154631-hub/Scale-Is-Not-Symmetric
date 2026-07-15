from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch

TensorDict = Dict[str, torch.Tensor]


@dataclass(frozen=True)
class LoraPair:
    module: str
    key_a: str
    key_b: str


@dataclass
class CanonicalComponent:
    """One canonical rank-1 component of a LoRA update.

    It represents sigma * u v^T, where ||u||_2 = ||v||_2 = 1 and sigma >= 0.
    adapter_idx and rank_idx identify where it came from.
    """

    adapter_idx: int
    rank_idx: int
    sigma: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor


def _strip_lora_suffix(key: str, marker: str) -> str:
    # Handles PEFT key styles such as:
    #   ...q_proj.lora_A.weight
    #   ...q_proj.lora_A.default.weight
    #   ...q_proj.lora_A.adapter_name.weight
    left, _, right = key.partition(marker)
    return left


def find_lora_pairs(state: TensorDict) -> List[LoraPair]:
    a_by_module: Dict[str, str] = {}
    b_by_module: Dict[str, str] = {}

    for key in state.keys():
        if ".lora_A" in key and key.endswith(".weight"):
            module = _strip_lora_suffix(key, ".lora_A")
            a_by_module[module] = key
        elif ".lora_B" in key and key.endswith(".weight"):
            module = _strip_lora_suffix(key, ".lora_B")
            b_by_module[module] = key

    pairs: List[LoraPair] = []
    for module in sorted(set(a_by_module) & set(b_by_module)):
        pairs.append(LoraPair(module=module, key_a=a_by_module[module], key_b=b_by_module[module]))
    return pairs


def check_same_lora_structure(states: List[TensorDict]) -> List[LoraPair]:
    if not states:
        raise ValueError("No adapter states provided.")
    ref_pairs = find_lora_pairs(states[0])
    ref_modules = [p.module for p in ref_pairs]
    if not ref_pairs:
        raise ValueError("No LoRA A/B pairs found in the first adapter.")

    for idx, state in enumerate(states[1:], start=1):
        pairs = find_lora_pairs(state)
        modules = [p.module for p in pairs]
        if modules != ref_modules:
            missing = set(ref_modules) - set(modules)
            extra = set(modules) - set(ref_modules)
            raise ValueError(
                f"Adapter {idx} LoRA structure differs from adapter 0. "
                f"Missing={sorted(missing)[:5]}, Extra={sorted(extra)[:5]}"
            )
    return ref_pairs


def rank_component_cosine(
    b1: torch.Tensor,
    a1: torch.Tensor,
    b2: torch.Tensor,
    a2: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Cosine between two rank-1 matrices b1 a1^T and b2 a2^T.

    Frobenius inner product of two outer products is:
        <b1 a1^T, b2 a2^T> = <b1, b2> * <a1, a2>

    This avoids materializing large dense matrices.
    """
    numerator = torch.dot(b1.flatten(), b2.flatten()) * torch.dot(a1.flatten(), a2.flatten())
    denom = torch.linalg.vector_norm(b1) * torch.linalg.vector_norm(a1) * torch.linalg.vector_norm(b2) * torch.linalg.vector_norm(a2)
    return numerator / torch.clamp(denom, min=eps)


def canonical_component_cosine(c1: CanonicalComponent, c2: CanonicalComponent) -> torch.Tensor:
    """Cosine between canonical components sigma*u*v^T.

    Since u and v are unit vectors and sigma is non-negative, the Frobenius cosine is
    <u1,u2> * <v1,v2>. Singular values are intentionally ignored in the cosine and
    handled separately as component energy/importance.
    """
    return torch.dot(c1.u.flatten(), c2.u.flatten()) * torch.dot(c1.v.flatten(), c2.v.flatten())


def canonical_lora_svd_from_factors(
    A: torch.Tensor,
    B: torch.Tensor,
    scale: float = 1.0,
    max_rank: Optional[int] = None,
    eps: float = 1e-12,
) -> List[CanonicalComponent]:
    """Canonical SVD of a low-rank LoRA update scale * B @ A without materializing B @ A.

    A has shape [r, d_in], B has shape [d_out, r].  Directly computing the SVD of
    B @ A is expensive.  We instead use QR decompositions:

        B = Q_b R_b,     A^T = Q_a R_a,
        B A = Q_b (R_b R_a^T) Q_a^T.

    Then only the small r x r matrix R_b R_a^T needs a dense SVD.
    This gives a canonical low-rank basis for the actual update matrix and avoids
    the non-identifiability of raw LoRA factors.
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("LoRA A and B must be 2D tensors.")
    if A.shape[0] != B.shape[1]:
        raise ValueError(f"Incompatible LoRA shapes: A={tuple(A.shape)}, B={tuple(B.shape)}")

    A_f = A.detach().float()
    B_f = B.detach().float() * float(scale)
    r = min(A_f.shape[0], B_f.shape[1])
    if max_rank is not None:
        r = min(r, int(max_rank))

    # Reduced QR keeps the computation at O(d*r^2) plus a small r*r SVD.
    Qb, Rb = torch.linalg.qr(B_f, mode="reduced")
    Qa, Ra = torch.linalg.qr(A_f.T, mode="reduced")
    small = Rb @ Ra.T
    U_small, S, Vh_small = torch.linalg.svd(small, full_matrices=False)
    U = Qb @ U_small
    V = Qa @ Vh_small.T

    comps: List[CanonicalComponent] = []
    used = 0
    for k in range(S.numel()):
        sigma = S[k]
        if float(sigma.item()) <= eps:
            continue
        comps.append(
            CanonicalComponent(
                adapter_idx=-1,
                rank_idx=used,
                sigma=sigma.detach(),
                u=U[:, k].detach(),
                v=V[:, k].detach(),
            )
        )
        used += 1
        if used >= r:
            break
    return comps


def concatenate_rank_components(
    components: List[Tuple[torch.Tensor, torch.Tensor, float]],
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build B_cat and A_cat from weighted raw rank-1 components.

    Each component is (b, a, weight), representing weight * b a^T.
    To encode it as B_cat @ A_cat, use sqrt(abs(weight))*b and sign(weight)*sqrt(abs(weight))*a.
    """
    if not components:
        raise ValueError("No rank components to concatenate.")

    b_cols = []
    a_rows = []
    for b, a, weight in components:
        weight_f = float(weight)
        mag = abs(weight_f) ** 0.5
        sign = 1.0 if weight_f >= 0 else -1.0
        b_cols.append((b.to(device) * mag).reshape(-1, 1))
        a_rows.append((a.to(device) * (sign * mag)).reshape(1, -1))

    B = torch.cat(b_cols, dim=1)
    A = torch.cat(a_rows, dim=0)
    return B, A


def concatenate_canonical_components(
    components: List[Tuple[CanonicalComponent, float]],
    device: str | torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build B_cat and A_cat from weighted canonical SVD components.

    A component represents sigma * u v^T.  With a non-negative merge weight w,
    we encode w*sigma*u*v^T as:

        B_col = sqrt(w*sigma) * u,
        A_row = sqrt(w*sigma) * v^T.

    Negative weights are supported for completeness, although MIRAGE normally uses
    non-negative conflict-regularized weights.
    """
    if not components:
        raise ValueError("No canonical components to concatenate.")
    b_cols = []
    a_rows = []
    for comp, weight in components:
        w = float(weight)
        sigma = float(comp.sigma.item())
        mag = abs(w * sigma) ** 0.5
        sign = 1.0 if w >= 0 else -1.0
        b_cols.append((comp.u.to(device) * mag).reshape(-1, 1))
        a_rows.append((comp.v.to(device) * (sign * mag)).reshape(1, -1))
    return torch.cat(b_cols, dim=1), torch.cat(a_rows, dim=0)


def dense_delta_from_pair(A: torch.Tensor, B: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return float(scale) * (B @ A)


def compress_delta_svd(delta: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return A, B such that B @ A approximates delta with target rank.

    Warning: this materializes dense delta and can be expensive for large MLLM layers.
    """
    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    sqrt_s = torch.sqrt(torch.clamp(S_r, min=0))
    B = U_r * sqrt_s.reshape(1, -1)
    A = sqrt_s.reshape(-1, 1) * Vh_r
    return A.to(delta.dtype), B.to(delta.dtype)


def recompress_lora_factors_via_small_svd(A: torch.Tensor, B: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compress B@A to a target LoRA rank without a large dense SVD.

    This is the same QR-small-SVD trick used by canonical_lora_svd_from_factors.
    It returns A_new, B_new where B_new @ A_new approximates B @ A.
    """
    comps = canonical_lora_svd_from_factors(A=A, B=B, scale=1.0, max_rank=rank)
    if not comps:
        raise RuntimeError("Cannot recompress: all singular values are zero.")
    kept = comps[:rank]
    b_cols = []
    a_rows = []
    for comp in kept:
        sqrt_s = torch.sqrt(torch.clamp(comp.sigma.float(), min=0.0))
        b_cols.append((comp.u.float() * sqrt_s).reshape(-1, 1))
        a_rows.append((comp.v.float() * sqrt_s).reshape(1, -1))
    B_new = torch.cat(b_cols, dim=1)
    A_new = torch.cat(a_rows, dim=0)
    return A_new.to(A.dtype), B_new.to(B.dtype)
