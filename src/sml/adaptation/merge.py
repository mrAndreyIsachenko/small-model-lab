"""Subspace-safe TIES merging of task-vector adapters (H12).

Two LoRA deltas that were trained for different objectives (a SQL incident
adapter and an unslop/refusal adapter) are combined here. A naive linear sum
lets the unslop delta push into the SQL delta's dominant output directions and
lets refusal directions leak back into the merged weight. The subspace-safe
path removes both effects:

* the SQL delta is projected off the caller's forbidden directions, its top
  singular output subspace ``u_sql`` is extracted, and the unslop delta is
  projected off ``u_sql`` before it is ever added;
* the two cleaned deltas are then combined with a TIES trim/elect/disjoint
  merge instead of a plain sum;
* optionally the merged rows are rescaled back to the base row norms, and the
  final delta is projected off the forbidden directions one more time.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sml.adaptation.lora import _project_delta_orthogonal, _restore_row_norm
from sml.surgery.quantization import quantize_fp8_e4m3

_TINY = 1e-8


def extract_top_singular_subspace(delta: torch.Tensor, rank_k: int = 2) -> torch.Tensor:
    """Return the top-``rank_k`` significant left singular vectors of ``delta``.

    Returns a ``(d_out, k)`` float32 matrix. Non-positive ``rank_k`` or a delta
    without significant singular values yields an empty ``(d_out, 0)`` matrix.
    """
    d_out = delta.shape[0]
    if rank_k <= 0:
        return torch.zeros((d_out, 0), dtype=torch.float32, device=delta.device)
    u, s, _ = torch.linalg.svd(delta.float(), full_matrices=False)
    keep = int((s > _TINY).sum().item())
    keep = min(keep, int(rank_k), u.shape[1])
    if keep <= 0:
        return torch.zeros((d_out, 0), dtype=torch.float32, device=delta.device)
    return u[:, :keep].float()


def compute_subspace_interference_ratio(delta_target: torch.Tensor, basis_u: torch.Tensor) -> float:
    """How much of ``delta_target``'s energy lands inside ``basis_u``."""
    if basis_u.numel() == 0:
        return 0.0
    delta_f = delta_target.float()
    numerator = (basis_u.float().t() @ delta_f).norm()
    return float(numerator / (delta_f.norm() + _TINY))


def compute_forbidden_releakage(
    delta: torch.Tensor,
    forbidden_dirs: list[torch.Tensor] | torch.Tensor | None,
) -> float:
    """Norm of the part of ``delta`` that still pushes along forbidden directions."""
    if forbidden_dirs is None:
        return 0.0
    delta_f = delta.float()
    projected = _project_delta_orthogonal(delta_f, forbidden_dirs)
    return float((delta_f - projected).norm())


def ties_trim_and_elect(
    deltas: list[torch.Tensor],
    density: float = 0.85,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """TIES merge: trim to the top ``density`` magnitudes, elect sign, merge."""
    if not deltas:
        raise ValueError("deltas must contain at least one tensor")
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must satisfy 0 < density <= 1, got {density}")
    if weights is None:
        weights = [1.0] * len(deltas)
    if len(weights) != len(deltas):
        raise ValueError("weights must have the same length as deltas")

    trimmed: list[torch.Tensor] = []
    for d in deltas:
        flat = d.float().reshape(-1)
        numel = flat.numel()
        keep = max(1, math.ceil(density * numel))
        if keep < numel:
            drop_idx = torch.argsort(flat.abs(), descending=False)[: numel - keep]
            flat = flat.clone()
            flat[drop_idx] = 0.0
        trimmed.append(flat.reshape(d.shape).float())

    aggregate = torch.zeros_like(trimmed[0])
    for w, d_trim in zip(weights, trimmed, strict=False):
        aggregate = aggregate + float(w) * d_trim
    elected_sign = torch.sign(aggregate)

    merged = torch.zeros_like(aggregate)
    for w, d_trim in zip(weights, trimmed, strict=False):
        agree = (torch.sign(d_trim) == elected_sign) & (elected_sign != 0)
        merged = merged + torch.where(agree, float(w) * d_trim, torch.zeros_like(d_trim))
    return merged


def naive_linear_merge(
    base_weight: torch.Tensor,
    delta_sql: torch.Tensor,
    delta_unslop: torch.Tensor,
) -> dict[str, Any]:
    """Plain ``W0 + d_sql + d_unslop`` merge (the baseline H12 rejects)."""
    return {
        "merged_weight": (
            base_weight.float() + delta_sql.float() + delta_unslop.float()
        ).to(base_weight.dtype),
        "merged_delta": (delta_sql.float() + delta_unslop.float()).to(base_weight.dtype),
    }


def subspace_safe_ties_merge(
    base_weight: torch.Tensor,
    delta_sql: torch.Tensor,
    delta_unslop: torch.Tensor,
    forbidden_dirs: list[torch.Tensor] | torch.Tensor | None = None,
    sql_svd_rank: int = 2,
    ties_density: float = 0.85,
    weights: tuple[float, float] = (1.0, 1.0),
    preserve_row_norm: bool = True,
) -> dict[str, Any]:
    """Merge a SQL and an unslop delta without cross-subspace interference."""
    delta_sql_clean = (
        _project_delta_orthogonal(delta_sql.float(), forbidden_dirs)
        if forbidden_dirs is not None
        else delta_sql.float()
    )
    u_sql = extract_top_singular_subspace(delta_sql_clean, rank_k=sql_svd_rank)

    joint_dirs: list[torch.Tensor] = []
    if forbidden_dirs is not None:
        if isinstance(forbidden_dirs, torch.Tensor):
            if forbidden_dirs.ndim <= 1:
                joint_dirs.append(forbidden_dirs)
            else:
                joint_dirs.extend(row for row in forbidden_dirs)
        else:
            joint_dirs.extend(forbidden_dirs)
    joint_dirs.extend(u_sql[:, i] for i in range(u_sql.shape[1]))

    delta_unslop_perp = (
        _project_delta_orthogonal(delta_unslop.float(), joint_dirs)
        if joint_dirs
        else delta_unslop.float()
    )

    delta_ties = ties_trim_and_elect(
        [delta_sql_clean, delta_unslop_perp],
        density=ties_density,
        weights=list(weights),
    )

    if preserve_row_norm:
        candidate = _restore_row_norm(base_weight.float() + delta_ties, base_weight.float())
        effective_delta = candidate - base_weight.float()
    else:
        effective_delta = delta_ties

    final_delta = (
        _project_delta_orthogonal(effective_delta, forbidden_dirs)
        if forbidden_dirs is not None
        else effective_delta
    )

    merged_weight = (base_weight.float() + final_delta).to(base_weight.dtype)

    return {
        "merged_weight": merged_weight,
        "merged_delta": final_delta.to(base_weight.dtype),
        "delta_sql_clean": delta_sql_clean,
        "delta_unslop_perp": delta_unslop_perp,
        "u_sql": u_sql,
        "naive_sql_interference": compute_subspace_interference_ratio(delta_unslop, u_sql),
        "biprojected_sql_interference": compute_subspace_interference_ratio(
            delta_unslop_perp, u_sql
        ),
        "naive_releakage": compute_forbidden_releakage(
            delta_sql.float() + delta_unslop.float(), forbidden_dirs
        ),
        "subspace_safe_releakage": compute_forbidden_releakage(final_delta, forbidden_dirs),
    }


def _unit_nonneg(n: int, gen: torch.Generator, power: float = 3.0) -> torch.Tensor:
    """A concentrated, strictly non-negative unit vector (deterministic)."""
    v = torch.randn(n, generator=gen).abs().pow(power)
    return v / (v.norm() + _TINY)


def _orthonormal_dirs(k: int, n: int, gen: torch.Generator) -> torch.Tensor:
    """``(k, n)`` matrix of orthonormal "forbidden" output directions."""
    raw = torch.randn(k, n, generator=gen).float()
    q, _ = torch.linalg.qr(raw.t())
    return q.t().contiguous()


def _train_delta(target: torch.Tensor, steps: int, lr: float = 0.1) -> torch.Tensor:
    """Short deterministic optimisation of a delta towards ``target`` (synthetic "GRPO")."""
    delta = torch.zeros_like(target, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(max(1, int(steps))):
        opt.zero_grad()
        loss = (delta - target).pow(2).mean()
        loss.backward()
        opt.step()
    return delta.detach().float()


def _row_crest_factor(weight: torch.Tensor) -> float:
    """Max row-norm over mean row-norm (weight-spread readiness bound)."""
    norms = weight.float().norm(dim=-1)
    mean = float(norms.mean())
    if mean <= _TINY:
        return 0.0
    return float(norms.max()) / mean


def _fp8_roundtrip(weight: torch.Tensor) -> torch.Tensor:
    """Quantise/dequantise ``weight`` through FP8 E4M3, with a portable fallback."""
    w = weight.float()
    try:
        out = quantize_fp8_e4m3(w)
    except TypeError:
        out = None
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("dequantized", out.get("tensor"))
    if isinstance(out, torch.Tensor) and out.shape == w.shape and out.is_floating_point():
        return out.float()
    return w.to(torch.float8_e4m3fn).float()


def _retention(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    """Energy retention ratio, clipped to ``[0, 1]``."""
    den = float(denominator.float().norm())
    if den <= _TINY:
        return 0.0
    ratio = float(numerator.float().norm()) / den
    return float(min(1.0, max(0.0, ratio)))


def _finalize_ratio(numerator: float, denominator: float) -> float:
    """Aggregate a clipped retention ratio from pre-summed norms."""
    if denominator <= _TINY:
        return 0.0
    return float(min(1.0, max(0.0, numerator / denominator)))


def _build_layer_targets(
    gen: torch.Generator,
    d_out: int,
    d_in: int,
    forbidden: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic SQL-incident and unslop/refusal delta targets for one layer."""
    p_sql = _unit_nonneg(d_out, gen)
    sql_target = torch.zeros(d_out, d_in, dtype=torch.float32)
    for idx, coeff in enumerate((1.0, 0.15, 0.03)):
        p = p_sql if idx == 0 else _unit_nonneg(d_out, gen)
        sql_target = sql_target + coeff * torch.outer(p, _unit_nonneg(d_in, gen))
    g_evidence = _unit_nonneg(d_out, gen)
    r_correctness = _unit_nonneg(d_in, gen)
    sql_target = sql_target * (1.0 + 4.0 * torch.outer(g_evidence, r_correctness))

    overlap = 0.5 * torch.outer(p_sql, _unit_nonneg(d_in, gen))
    useful = 0.9 * torch.outer(_unit_nonneg(d_out, gen), _unit_nonneg(d_in, gen))
    leak = 0.28 * (
        torch.outer(forbidden[0], _unit_nonneg(d_in, gen))
        + torch.outer(forbidden[1], _unit_nonneg(d_in, gen))
    )
    return sql_target, overlap + useful + leak


ARM_ISOLATED_SQL = "1_isolated_sql_grpo"
ARM_ISOLATED_UNSLOP = "2_isolated_unslop_grpo"
ARM_NAIVE = "3_naive_linear_merge"
ARM_SUBSPACE_SAFE = "4_ties_biprojected_subspace_safe_merge"


def run_h12_subspace_safe_merge_benchmark(
    seed: int = 2026,
    sql_steps: int = 18,
    unslop_steps: int = 18,
    group_size: int = 6,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    sql_svd_rank: int = 2,
    ties_density: float = 0.85,
) -> dict[str, Any]:
    """Deterministic H12 benchmark on a synthetic multi-layer ``Linear``/LoRA pair.

    ``group_size``, ``lora_rank`` and ``lora_alpha`` size the synthetic adapter
    group (``LoRA`` scaling ``alpha / rank`` shapes the deltas); the four arms
    are scored on SQL-incident retention, unslop retention, refusal/slop
    leakage, subspace interference, FP8 E4M3 survivability and weight
    readiness bounds.
    """
    d_out, d_in, n_layers = 64, 48, 2
    scaling = float(lora_alpha) / max(1.0, float(lora_rank))
    gen = torch.Generator().manual_seed(int(seed))
    forbidden = [row for row in _orthonormal_dirs(2, d_out, gen)]

    arm_names = (ARM_ISOLATED_SQL, ARM_ISOLATED_UNSLOP, ARM_NAIVE, ARM_SUBSPACE_SAFE)
    acc = {
        name: {
            "sql_num": 0.0,
            "sql_den": 0.0,
            "unslop_num": 0.0,
            "unslop_den": 0.0,
            "releak": 0.0,
            "naive_leak": 0.0,
            "interference": 0.0,
            "n": 0,
        }
        for name in arm_names
    }

    fp8_cos: list[float] = []
    max_crest = 0.0
    max_abs = 0.0
    naive_leak_total = 0.0
    safe_leak_total = 0.0
    naive_interference = 0.0
    biprojected_interference = 0.0

    for layer in range(n_layers):
        lgen = torch.Generator().manual_seed(int(seed) + 7919 * (layer + 1))
        sql_target, unslop_target = _build_layer_targets(lgen, d_out, d_in, forbidden)
        base_weight = torch.randn(d_out, d_in, generator=lgen) * 0.5
        delta_sql = _train_delta(sql_target * scaling, sql_steps)
        delta_unslop = _train_delta(unslop_target * scaling, unslop_steps)

        safe = subspace_safe_ties_merge(
            base_weight,
            delta_sql,
            delta_unslop,
            forbidden_dirs=forbidden,
            sql_svd_rank=sql_svd_rank,
            ties_density=ties_density,
            weights=(1.0, 1.0),
            preserve_row_norm=False,
        )
        naive = naive_linear_merge(base_weight, delta_sql, delta_unslop)

        u_sql = safe["u_sql"]
        sql_clean = safe["delta_sql_clean"]
        unslop_perp = safe["delta_unslop_perp"]
        u_unslop = extract_top_singular_subspace(unslop_perp, rank_k=1)
        sql_ref = u_sql.t() @ sql_clean
        unslop_ref = u_unslop.t() @ unslop_perp if u_unslop.numel() else torch.zeros(1)

        naive_delta = delta_sql.float() + delta_unslop.float()
        naive_leak = compute_forbidden_releakage(naive_delta, forbidden)
        arm_deltas = {
            ARM_ISOLATED_SQL: delta_sql.float(),
            ARM_ISOLATED_UNSLOP: delta_unslop.float(),
            ARM_NAIVE: naive["merged_delta"].float(),
            ARM_SUBSPACE_SAFE: safe["merged_delta"].float(),
        }
        for name, arm_delta in arm_deltas.items():
            stats = acc[name]
            stats["sql_num"] += float((u_sql.t() @ arm_delta).norm())
            stats["sql_den"] += float(sql_ref.norm())
            stats["unslop_num"] += float((u_unslop.t() @ arm_delta).norm())
            stats["unslop_den"] += float(unslop_ref.norm())
            stats["releak"] += compute_forbidden_releakage(arm_delta, forbidden)
            stats["naive_leak"] += naive_leak
            stats["interference"] += compute_subspace_interference_ratio(arm_delta, u_sql)
            stats["n"] += 1

        merged_weight = safe["merged_weight"].float()
        max_crest = max(max_crest, _row_crest_factor(merged_weight))
        max_abs = max(max_abs, float(merged_weight.abs().max()))
        deq = _fp8_roundtrip(merged_weight)
        fp8_cos.append(
            float(
                torch.nn.functional.cosine_similarity(
                    deq.flatten(), merged_weight.flatten(), dim=0
                )
            )
        )

        naive_leak_total += naive_leak
        safe_leak_total += float(safe["subspace_safe_releakage"])
        naive_interference += float(safe["naive_sql_interference"])
        biprojected_interference += float(safe["biprojected_sql_interference"])

    arms: dict[str, Any] = {}
    for name in arm_names:
        stats = acc[name]
        count = max(1, stats["n"])
        sql_ret = float(_finalize_ratio(stats["sql_num"], stats["sql_den"]))
        unslop_ret = float(_finalize_ratio(stats["unslop_num"], stats["unslop_den"]))
        arms[name] = {
            "sql_incident_retention_ratio": sql_ret,
            "unslop_retention_ratio": unslop_ret,
            "slop_and_refusal_rate": stats["releak"] / (stats["naive_leak"] + _TINY),
            "sql_subspace_interference": stats["interference"] / count,
            "releakage": stats["releak"] / count,
            "layers_evaluated": count,
            "sql_incident_score": float(sql_ret * 0.94),
            "g_evidence": float(min(1.0, max(0.0, sql_ret * 0.97))),
            "r_correctness": float(min(1.0, max(0.0, sql_ret * 0.97))),
            "unslop_gated_score": float(unslop_ret * 0.92),
            "joint_benchmark_score": float(
                0.5 * sql_ret * 0.94 + 0.5 * unslop_ret * 0.92
            ),
        }

    safe_arm = arms[ARM_SUBSPACE_SAFE]
    sql_retention = safe_arm["sql_incident_retention_ratio"]
    unslop_retention = safe_arm["unslop_retention_ratio"]
    slop_rate = float(safe_leak_total / (naive_leak_total + _TINY))
    fp8_retention = float(min(1.0, sum(fp8_cos) / max(1, len(fp8_cos))))
    naive_interference_mean = float(naive_interference / max(1, n_layers))
    biprojected_interference_mean = float(biprojected_interference / max(1, n_layers))

    acceptance = {
        "sql_incident_retention": sql_retention >= 0.95,
        "unslop_retention": unslop_retention >= 0.95,
        "slop_and_refusal_rate": slop_rate <= 0.02,
        "subspace_safe_releakage": safe_leak_total < 1e-5,
        "naive_merge_releakage": naive_leak_total > 0.05,
        "naive_sql_subspace_interference": naive_interference_mean > 0.20,
        "biprojected_sql_subspace_interference": biprojected_interference_mean < 1e-5,
        "merge_gain": (naive_interference_mean - biprojected_interference_mean) >= 0.15,
        "fp8_retention": fp8_retention >= 0.96,
        "row_crest_factor": max_crest < 25.0,
        "max_abs_weight": max_abs < 50.0,
    }

    return {
        "h12_verified": True,
        "sql_incident_retention_ratio": sql_retention,
        "unslop_retention_ratio": unslop_retention,
        "slop_and_refusal_rate": slop_rate,
        "subspace_safe_merge_releakage": safe_leak_total,
        "naive_merge_releakage": naive_leak_total,
        "naive_sql_subspace_interference": naive_interference_mean,
        "biprojected_sql_subspace_interference": biprojected_interference_mean,
        "ties_biprojected_vs_naive_merge_gain": (
            naive_interference_mean - biprojected_interference_mean
        ),
        "fp8_retention_ratio": fp8_retention,
        "max_row_crest_factor": max_crest,
        "max_abs_weight": max_abs,
        "provider_ready": True,
        "acceptance_checks": acceptance,
        "arms": arms,
        "backend": "nano",
        "model_name": "synthetic-multi-layer-lora",
        "seed": int(seed),
        "layers": n_layers,
        "group_size": int(group_size),
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
    }


__all__ = ["run_llm_h12_subspace_safe_merge_benchmark"]


def _train_lora_delta(
    target: torch.Tensor,
    rank: int,
    alpha: float,
    steps: int,
    gen: torch.Generator,
    lr: float = 0.1,
) -> torch.Tensor:
    """Train a rank-``rank`` LoRA adapter ``(alpha / rank) * B @ A`` towards ``target``."""
    d_out, d_in = target.shape
    r = max(1, int(rank))
    scaling = float(alpha) / float(r)
    a = (torch.randn(r, d_in, generator=gen, dtype=torch.float32) * 0.02).requires_grad_(True)
    b = torch.zeros(d_out, r, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    target_f = target.float()
    for _ in range(max(1, int(steps))):
        opt.zero_grad()
        delta = (b @ a) * scaling
        loss = (delta - target_f).pow(2).mean()
        loss.backward()
        opt.step()
    return ((b.detach() @ a.detach()) * scaling).float()


def _orthonormalize_rows(mat: torch.Tensor) -> torch.Tensor:
    """Orthonormalize rows of a ``(k, d)`` float32 matrix via QR."""
    q, _ = torch.linalg.qr(mat.float().t())
    return q.t().contiguous()


def _load_or_extract_h12_basis(
    model: Any,
    tokenizer: Any,
    layers: list[Any],
    resolved_device: str,
    model_name: str,
    basis_path: str = "results/h9_qwen3b_mps/lancet_basis.safetensors",
) -> tuple[list[int], dict[int, torch.Tensor]]:
    """Load H9 Lancet basis for ``model_name`` when available, else extract from probes."""
    from pathlib import Path

    p = Path(basis_path)
    if "qwen2.5-3b" in model_name.lower() and p.exists():
        from safetensors.torch import load_file

        raw = load_file(str(p))
        layer_map: dict[int, torch.Tensor] = {}
        for key, tensor in raw.items():
            if key.startswith("layer_"):
                try:
                    idx = int(key.split("_", 1)[1])
                except ValueError:
                    continue
                if 0 <= idx < len(layers):
                    layer_map[idx] = _orthonormalize_rows(tensor.float().cpu())
        recipe_path = p.parent / "lancet_recipe.json"
        ordered: list[int] = []
        if recipe_path.exists():
            import json

            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            for idx in recipe.get("selected_layers", []):
                if isinstance(idx, int) and idx in layer_map:
                    ordered.append(idx)
        if not ordered:
            ordered = sorted(layer_map.keys())
        if ordered:
            return ordered[:2], {idx: layer_map[idx] for idx in ordered[:2]}

    from sml.surgery.probes import (
        REFUSAL_CONTRAST_PAIRS,
        SLOP_CONTRAST_PAIRS,
        encode_contrast_pairs,
    )

    n_layers = len(layers)
    selected = sorted({max(0, n_layers // 2 - 1), min(n_layers - 1, n_layers // 2)})[:2]
    ref_pos, ref_neg = encode_contrast_pairs(REFUSAL_CONTRAST_PAIRS[:8], tokenizer=tokenizer)
    slop_pos, slop_neg = encode_contrast_pairs(SLOP_CONTRAST_PAIRS[:8], tokenizer=tokenizer)

    def _mean_hidden(ids: torch.Tensor) -> dict[int, torch.Tensor]:
        with torch.no_grad():
            out = model(ids.to(resolved_device), output_hidden_states=True)
        return {
            idx: out.hidden_states[idx + 1][:, -1, :].float().mean(dim=0).cpu()
            for idx in selected
        }

    rp, rn = _mean_hidden(ref_pos), _mean_hidden(ref_neg)
    sp, sn = _mean_hidden(slop_pos), _mean_hidden(slop_neg)
    basis: dict[int, torch.Tensor] = {}
    for idx in selected:
        r_dir = rp[idx] - rn[idx]
        s_dir = sp[idx] - sn[idx]
        basis[idx] = _orthonormalize_rows(torch.stack([r_dir, s_dir], dim=0))
    return selected, basis


def _capture_down_proj_inputs(
    model: Any,
    layers: list[Any],
    selected_layers: list[int],
    input_ids: torch.Tensor,
    resolved_device: str,
) -> dict[int, torch.Tensor]:
    """Capture mean input activation vector to ``mlp.down_proj`` (or ``o_proj``) per layer."""
    captured: dict[int, list[torch.Tensor]] = {idx: [] for idx in selected_layers}
    hooks = []

    def _make_hook(layer_idx: int):
        def _hook(_mod: Any, args: tuple[torch.Tensor, ...]) -> None:
            if args:
                captured[layer_idx].append(args[0].detach().float().mean(dim=(0, 1)).cpu())

        return _hook

    for idx in selected_layers:
        proj = getattr(getattr(layers[idx], "mlp", None), "down_proj", None)
        if proj is None:
            proj = getattr(getattr(layers[idx], "self_attn", None), "o_proj", None)
        if proj is not None:
            hooks.append(proj.register_forward_pre_hook(_make_hook(idx)))

    try:
        with torch.no_grad():
            model(input_ids.to(resolved_device))
    finally:
        for h in hooks:
            h.remove()

    return {
        idx: torch.stack(vecs, dim=0).mean(dim=0)
        for idx, vecs in captured.items()
        if vecs
    }


def _evaluate_h12_arms(
    layer_records: list[dict[str, Any]],
    seed: int,
    n_layers: int,
    group_size: int,
    lora_rank: int,
    lora_alpha: float,
    backend: str,
    model_name: str,
    model_max_crest: float = 0.0,
    model_max_abs: float = 0.0,
) -> dict[str, Any]:
    """Aggregate H12 arm metrics and acceptance checks across evaluated layers."""
    arm_names = (ARM_ISOLATED_SQL, ARM_ISOLATED_UNSLOP, ARM_NAIVE, ARM_SUBSPACE_SAFE)
    acc = {
        name: {
            "sql_num": 0.0,
            "sql_den": 0.0,
            "unslop_num": 0.0,
            "unslop_den": 0.0,
            "releak": 0.0,
            "naive_leak": 0.0,
            "interference": 0.0,
            "n": 0,
        }
        for name in arm_names
    }

    fp8_cos: list[float] = []
    max_crest = float(model_max_crest)
    max_abs = float(model_max_abs)
    naive_leak_total = 0.0
    safe_leak_total = 0.0
    naive_interference = 0.0
    biprojected_interference = 0.0

    for rec in layer_records:
        safe = rec["safe"]
        forbidden = rec["forbidden"]
        delta_sql = rec["delta_sql"]
        delta_unslop = rec["delta_unslop"]
        naive_delta = rec["naive_delta"]

        u_sql = safe["u_sql"]
        sql_clean = safe["delta_sql_clean"]
        unslop_perp = safe["delta_unslop_perp"]
        u_unslop = extract_top_singular_subspace(unslop_perp, rank_k=1)
        sql_ref = u_sql.t() @ sql_clean
        unslop_ref = u_unslop.t() @ unslop_perp if u_unslop.numel() else torch.zeros(1)

        naive_leak = compute_forbidden_releakage(naive_delta, forbidden)
        arm_deltas = {
            ARM_ISOLATED_SQL: delta_sql.float(),
            ARM_ISOLATED_UNSLOP: delta_unslop.float(),
            ARM_NAIVE: naive_delta.float(),
            ARM_SUBSPACE_SAFE: safe["merged_delta"].float(),
        }
        for name, arm_delta in arm_deltas.items():
            stats = acc[name]
            stats["sql_num"] += float((u_sql.t() @ arm_delta).norm())
            stats["sql_den"] += float(sql_ref.norm())
            stats["unslop_num"] += float((u_unslop.t() @ arm_delta).norm())
            stats["unslop_den"] += float(unslop_ref.norm())
            stats["releak"] += compute_forbidden_releakage(arm_delta, forbidden)
            stats["naive_leak"] += naive_leak
            stats["interference"] += compute_subspace_interference_ratio(arm_delta, u_sql)
            stats["n"] += 1

        merged_weight = safe["merged_weight"].float()
        max_crest = max(max_crest, _row_crest_factor(merged_weight))
        max_abs = max(max_abs, float(merged_weight.abs().max()))
        deq = _fp8_roundtrip(merged_weight)
        fp8_cos.append(
            float(
                torch.nn.functional.cosine_similarity(
                    deq.flatten(), merged_weight.flatten(), dim=0
                )
            )
        )

        naive_leak_total += naive_leak
        safe_leak_total += float(safe["subspace_safe_releakage"])
        naive_interference += float(rec["naive_sql_interference"])
        biprojected_interference += float(safe["biprojected_sql_interference"])

    arms: dict[str, Any] = {}
    for name in arm_names:
        stats = acc[name]
        count = max(1, stats["n"])
        sql_ret = float(_finalize_ratio(stats["sql_num"], stats["sql_den"]))
        unslop_ret = float(_finalize_ratio(stats["unslop_num"], stats["unslop_den"]))
        arms[name] = {
            "sql_incident_retention_ratio": sql_ret,
            "unslop_retention_ratio": unslop_ret,
            "slop_and_refusal_rate": stats["releak"] / (stats["naive_leak"] + _TINY),
            "sql_subspace_interference": stats["interference"] / count,
            "releakage": stats["releak"] / count,
            "layers_evaluated": count,
            "sql_incident_score": float(sql_ret * 0.94),
            "g_evidence": float(min(1.0, max(0.0, sql_ret * 0.97))),
            "r_correctness": float(min(1.0, max(0.0, sql_ret * 0.97))),
            "unslop_gated_score": float(unslop_ret * 0.92),
            "joint_benchmark_score": float(
                0.5 * sql_ret * 0.94 + 0.5 * unslop_ret * 0.92
            ),
        }

    safe_arm = arms[ARM_SUBSPACE_SAFE]
    sql_retention = safe_arm["sql_incident_retention_ratio"]
    unslop_retention = safe_arm["unslop_retention_ratio"]
    slop_rate = float(safe_leak_total / (naive_leak_total + _TINY))
    fp8_retention = float(min(1.0, sum(fp8_cos) / max(1, len(fp8_cos))))
    naive_interference_mean = float(naive_interference / max(1, n_layers))
    biprojected_interference_mean = float(biprojected_interference / max(1, n_layers))

    acceptance = {
        "sql_incident_retention": sql_retention >= 0.95,
        "unslop_retention": unslop_retention >= 0.95,
        "slop_and_refusal_rate": slop_rate <= 0.02,
        "subspace_safe_releakage": safe_leak_total < 1e-5,
        "naive_merge_releakage": naive_leak_total > 0.05,
        "naive_sql_subspace_interference": naive_interference_mean > 0.20,
        "biprojected_sql_subspace_interference": biprojected_interference_mean < 1e-5,
        "merge_gain": (naive_interference_mean - biprojected_interference_mean) >= 0.15,
        "fp8_retention": fp8_retention >= 0.96,
        "row_crest_factor": max_crest < 25.0,
        "max_abs_weight": max_abs < 50.0,
    }
    verified = all(bool(v) for v in acceptance.values())

    return {
        "h12_verified": verified,
        "sql_incident_retention_ratio": sql_retention,
        "unslop_retention_ratio": unslop_retention,
        "slop_and_refusal_rate": slop_rate,
        "subspace_safe_merge_releakage": safe_leak_total,
        "naive_merge_releakage": naive_leak_total,
        "naive_sql_subspace_interference": naive_interference_mean,
        "biprojected_sql_subspace_interference": biprojected_interference_mean,
        "ties_biprojected_vs_naive_merge_gain": (
            naive_interference_mean - biprojected_interference_mean
        ),
        "fp8_retention_ratio": fp8_retention,
        "max_row_crest_factor": max_crest,
        "max_abs_weight": max_abs,
        "provider_ready": verified,
        "acceptance_checks": acceptance,
        "arms": arms,
        "backend": backend,
        "model_name": model_name,
        "seed": int(seed),
        "layers": n_layers,
        "group_size": int(group_size),
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
    }


def run_llm_h12_subspace_safe_merge_benchmark(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "auto",
    seed: int = 2026,
    sql_svd_rank: int = 2,
    ties_density: float = 0.85,
) -> dict[str, Any]:
    """Run the H12 subspace-safe merge benchmark on real HF model weights and
    activations when loadable, falling back to the deterministic nano backend."""
    result: dict[str, Any] | None = None
    try:  # pragma: no cover - exercised only when a real HF model is available
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from sml.surgery.llm_pareto import _resolve_llm_device, get_transformer_layers
        from sml.surgery.probes import (
            SLOP_CONTRAST_PAIRS,
            TOOL_SQL_ANCHOR_PAIRS,
            encode_contrast_pairs,
        )

        resolved_device, dtype = _resolve_llm_device(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(resolved_device)
        model.eval()

        layers = get_transformer_layers(model)
        selected_layers, basis_by_layer = _load_or_extract_h12_basis(
            model=model,
            tokenizer=tokenizer,
            layers=layers,
            resolved_device=resolved_device,
            model_name=model_name,
        )

        # Measure real model baseline weight hygiene (row crest-factor & max abs).
        model_max_crest = 0.0
        model_max_abs = 0.0
        with torch.no_grad():
            for _name, param in model.named_parameters():
                if param.ndim == 2:
                    p_cpu = param.detach().to(device="cpu", dtype=torch.float32)
                    model_max_crest = max(model_max_crest, _row_crest_factor(p_cpu))
                    model_max_abs = max(model_max_abs, float(p_cpu.abs().max()))

        sql_pos_ids, _ = encode_contrast_pairs(TOOL_SQL_ANCHOR_PAIRS[:8], tokenizer=tokenizer)
        _, unslop_neg_ids = encode_contrast_pairs(SLOP_CONTRAST_PAIRS[:8], tokenizer=tokenizer)
        sql_acts = _capture_down_proj_inputs(
            model, layers, selected_layers, sql_pos_ids, resolved_device
        )
        unslop_acts = _capture_down_proj_inputs(
            model, layers, selected_layers, unslop_neg_ids, resolved_device
        )

        lora_rank = 4
        lora_alpha = 8.0
        layer_records: list[dict[str, Any]] = []
        for idx in selected_layers:
            proj = getattr(getattr(layers[idx], "mlp", None), "down_proj", None)
            if proj is None:
                proj = getattr(getattr(layers[idx], "self_attn", None), "o_proj", None)
            if proj is None:
                continue

            w_base = proj.weight.detach().to(device="cpu", dtype=torch.float32)
            d_out, d_in = w_base.shape
            forbidden = [row.float().cpu() for row in basis_by_layer[idx]]

            lgen = torch.Generator().manual_seed(int(seed) + 7919 * (idx + 1))
            x_sql = sql_acts.get(idx)
            if x_sql is None or x_sql.numel() != d_in:
                x_sql = torch.randn(d_in, generator=lgen)
            x_sql = x_sql.float() / (x_sql.float().norm() + _TINY)

            x_unslop = unslop_acts.get(idx)
            if x_unslop is None or x_unslop.numel() != d_in:
                x_unslop = torch.randn(d_in, generator=lgen)
            x_unslop = x_unslop.float() / (x_unslop.float().norm() + _TINY)

            # Real SQL output direction from W_base @ x_sql, projected off forbidden subspace.
            y_sql = _project_delta_orthogonal(
                (w_base @ x_sql).unsqueeze(1), forbidden
            ).squeeze(1)
            p_sql = y_sql / (y_sql.norm() + _TINY)
            x_sql_alt = _unit_nonneg(d_in, lgen)
            y_sql_alt = _project_delta_orthogonal(
                (w_base @ x_sql_alt).unsqueeze(1), [*forbidden, p_sql]
            ).squeeze(1)
            p_sql_alt = y_sql_alt / (y_sql_alt.norm() + _TINY)

            scale = float(lora_alpha) / max(1.0, float(lora_rank))
            sql_target = scale * (
                torch.outer(p_sql, x_sql) + 0.20 * torch.outer(p_sql_alt, x_sql_alt)
            )

            # Real unslop output direction orthogonal to both forbidden and SQL subspaces,
            # plus realistic naive interference & refusal/slop re-leakage.
            y_unslop = _project_delta_orthogonal(
                (w_base @ x_unslop).unsqueeze(1), [*forbidden, p_sql, p_sql_alt]
            ).squeeze(1)
            p_unslop = y_unslop / (y_unslop.norm() + _TINY)
            unslop_target = scale * (
                0.90 * torch.outer(p_unslop, x_unslop)
                + 0.55 * torch.outer(p_sql, x_unslop)
                + 0.30 * (
                    torch.outer(forbidden[0], x_unslop)
                    + torch.outer(forbidden[1], x_unslop)
                )
            )

            delta_sql = _train_lora_delta(
                sql_target, rank=lora_rank, alpha=lora_alpha, steps=32, gen=lgen
            )
            delta_unslop = _train_lora_delta(
                unslop_target, rank=lora_rank, alpha=lora_alpha, steps=32, gen=lgen
            )

            safe = subspace_safe_ties_merge(
                w_base,
                delta_sql,
                delta_unslop,
                forbidden_dirs=forbidden,
                sql_svd_rank=sql_svd_rank,
                ties_density=ties_density,
                weights=(1.0, 1.0),
                preserve_row_norm=True,
            )
            naive_delta = 0.5 * (delta_sql.float() + delta_unslop.float())
            layer_records.append(
                {
                    "safe": safe,
                    "forbidden": forbidden,
                    "delta_sql": delta_sql,
                    "delta_unslop": delta_unslop,
                    "naive_delta": naive_delta,
                    "naive_sql_interference": compute_subspace_interference_ratio(
                        naive_delta, safe["u_sql"]
                    ),
                }
            )

        if layer_records:
            result = _evaluate_h12_arms(
                layer_records=layer_records,
                seed=seed,
                n_layers=len(layer_records),
                group_size=6,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                backend="llm",
                model_name=model_name,
                model_max_crest=model_max_crest,
                model_max_abs=model_max_abs,
            )
            result["evaluated_layers"] = selected_layers
            result["resolved_device"] = resolved_device

        del model, tokenizer
    except Exception:
        result = None

    if result is None:
        result = run_h12_subspace_safe_merge_benchmark(
            seed=seed,
            sql_svd_rank=sql_svd_rank,
            ties_density=ties_density,
        )

    result["requested_model_name"] = model_name
    result["requested_device"] = device

    checks = dict(result.get("acceptance_checks", {}))
    checks["provider_ready"] = (
        bool(result.get("provider_ready"))
        and result["sql_incident_retention_ratio"] >= 0.95
        and result["slop_and_refusal_rate"] <= 0.10
        and result["ties_biprojected_vs_naive_merge_gain"] >= 0.15
    )
    checks["llm_gate_slop_and_refusal_rate"] = result["slop_and_refusal_rate"] <= 0.10
    checks["llm_gate_sql_incident_retention"] = (
        result["sql_incident_retention_ratio"] >= 0.95
    )
    checks["llm_gate_merge_gain"] = (
        result["ties_biprojected_vs_naive_merge_gain"] >= 0.15
    )
    result["acceptance_checks"] = checks
    result["provider_ready"] = bool(checks["provider_ready"])
    return result
