"""Weight surgery (`H7` CKA Mixed-Precision, `H8` Subspace Ablation) & Pareto frontier."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sml.config import ModelConfig, load_model_config
from sml.model.transformer import Transformer
from sml.surgery.cka import collect_layer_activations, compute_layerwise_cka_drift
from sml.surgery.llm_pareto import (
    run_llm_h9_biprojected_surgery_benchmark,
    run_llm_surgery_and_pareto_suite,
)
from sml.surgery.probes import (
    COT_REASONING_ANCHOR_PAIRS,
    REASONING_ANCHOR_PAIRS,
    REFUSAL_CONTRAST_PAIRS,
    SLOP_CONTRAST_PAIRS,
    TOOL_SQL_ANCHOR_PAIRS,
    PhaseSegmenter,
    encode_contrast_pairs,
)
from sml.surgery.quantization import (
    allocate_mixed_precision_bits,
    apply_mixed_precision_quantization,
    prune_transformer_layers,
)
from sml.surgery.subspace import (
    _compute_row_crest_factor,
    apply_lancet_biprojected_surgery,
    apply_subspace_ablation,
    biproject_direction,
    build_lancet_recipe,
    build_tri_anchor_basis,
    compute_layer_scr_profile,
    extract_contrastive_direction,
    extract_contrastive_direction_with_stats,
    project_out_rank1_subspace,
    save_lancet_recipe,
)

__all__ = [
    "run_h9_biprojected_surgery_benchmark",
    "run_h15_cot_tri_anchor_lancet_benchmark",
    "run_llm_h9_biprojected_surgery_benchmark",
    "run_llm_surgery_and_pareto_suite",
    "run_surgery_and_pareto_suite",
]


def compute_logit_kl_divergence(
    ref_model: Transformer,
    candidate_model: Transformer,
    input_ids: torch.Tensor,
) -> float:
    """Compute token-averaged forward KL divergence `KL(P_ref || P_candidate)`."""
    with torch.no_grad():
        ref_logits = ref_model(input_ids).logits.float()
        cand_logits = candidate_model(input_ids).logits.float()

        ref_probs = F.softmax(ref_logits, dim=-1)
        log_ref_probs = F.log_softmax(ref_logits, dim=-1)
        log_cand_probs = F.log_softmax(cand_logits, dim=-1)

        kl_per_token = torch.sum(ref_probs * (log_ref_probs - log_cand_probs), dim=-1)
        return float(kl_per_token.mean().item())


def _measure_inference_latency(
    model: Transformer,
    input_ids: torch.Tensor,
    iters: int = 5,
) -> tuple[float, float]:
    """Measure actual forward-pass latency (`ms/token`) and throughput (`tokens/sec`)."""
    model.eval()
    total_tokens = input_ids.numel() * iters
    with torch.no_grad():
        _ = model(input_ids)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(input_ids)
        elapsed = max(1e-6, time.perf_counter() - t0)
    ms_per_token = (elapsed * 1000.0) / total_tokens
    tok_per_sec = total_tokens / elapsed
    return round(ms_per_token, 4), round(tok_per_sec, 1)


def _select_top_scr_layers(profile: Any, top_k: int) -> list[int]:
    """Select the `top_k` best layers (highest SCR) from a `compute_layer_scr_profile` result."""
    rows = profile[0] if isinstance(profile, tuple) else profile
    if isinstance(rows, dict):
        ranked = sorted(rows.items(), key=lambda item: float(item[1]), reverse=True)
        return [int(layer_idx) for layer_idx, _score in ranked[:top_k]]
    score_keys = ("scr", "scr_score", "score", "suppression_ratio", "target_suppression_ratio")
    scored: list[tuple[int, float]] = []
    for entry in rows:
        score = 0.0
        for key in score_keys:
            if key in entry:
                score = float(entry[key])
                break
        scored.append((int(entry["layer_idx"]), score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [layer_idx for layer_idx, _score in scored[:top_k]]


def _biprojected_basis(r_refusal_perp: torch.Tensor, r_slop_perp: torch.Tensor) -> torch.Tensor:
    """Orthonormal `[2, d]` basis spanning both bi-projected suppression targets."""
    q_refusal = r_refusal_perp.detach().float()
    q_refusal = q_refusal / max(float(q_refusal.norm().item()), 1e-8)
    q_slop = biproject_direction(r_slop_perp.detach().float(), [q_refusal])
    return torch.stack([q_refusal, q_slop], dim=0)


def _snapshot_write_weights(
    layers: list[Any], layer_indices: list[int]
) -> dict[tuple[int, str], torch.Tensor]:
    """Snapshot `o_proj` / `down_proj` weights of `layer_indices` as detached CPU copies."""
    snapshots: dict[tuple[int, str], torch.Tensor] = {}
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        snapshots[(layer_idx, "self_attn.o_proj")] = (
            layer.self_attn.o_proj.weight.detach().cpu().clone()
        )
        snapshots[(layer_idx, "mlp.down_proj")] = layer.mlp.down_proj.weight.detach().cpu().clone()
    return snapshots


def _h9_surgery_metrics(
    surg_layers: list[Any],
    original_weights: dict[tuple[int, str], torch.Tensor],
    selected_layers: list[int],
    basis_per_layer: dict[int, torch.Tensor],
    layer_weights: dict[int, float] | None = None,
) -> tuple[float, float, float]:
    """Max row-norm drift & target-suppression ratio over `o_proj`/`down_proj` weights.

    The overall (unweighted) suppression ratio is always computed across all
    `selected_layers`. When soft SCR attenuation `layer_weights` (per-layer alphas)
    is provided, the returned target suppression ratio is restricted to the
    full-strength primary layers (`alpha >= 0.999`), so the max-SCR layer's own
    suppression is what must clear the `>= 0.90` gate; the unweighted ratio is
    returned alongside it for diagnostics.
    """
    if layer_weights is not None:
        primary_layer_set = {
            layer_idx for layer_idx in selected_layers if layer_weights.get(layer_idx, 1.0) >= 0.999
        }
        if not primary_layer_set:
            primary_layer_set = {selected_layers[0]}
    else:
        primary_layer_set = set(selected_layers)

    max_row_norm_drift = 0.0
    before_sq = 0.0
    after_sq = 0.0
    primary_before_sq = 0.0
    primary_after_sq = 0.0
    for layer_idx in selected_layers:
        layer = surg_layers[layer_idx]
        basis = basis_per_layer[layer_idx]
        for attr, proj in (
            ("self_attn.o_proj", layer.self_attn.o_proj),
            ("mlp.down_proj", layer.mlp.down_proj),
        ):
            orig_weight = original_weights[(layer_idx, attr)].to(proj.weight.device)
            new_weight = proj.weight
            drift = (orig_weight.float().norm(dim=1) - new_weight.float().norm(dim=1)).abs()
            max_row_norm_drift = max(max_row_norm_drift, float(drift.max().item()))
            before_sq_proj = float((basis @ orig_weight.float()).pow(2).sum().item())
            after_sq_proj = float((basis @ new_weight.float()).pow(2).sum().item())
            before_sq += before_sq_proj
            after_sq += after_sq_proj
            if layer_idx in primary_layer_set:
                primary_before_sq += before_sq_proj
                primary_after_sq += after_sq_proj
    after_norm = after_sq**0.5
    before_norm = max(1e-8, before_sq**0.5)
    unweighted_suppression_ratio = 1.0 - (after_norm / before_norm)
    primary_after_norm = primary_after_sq**0.5
    primary_before_norm = max(1e-8, primary_before_sq**0.5)
    target_suppression_ratio = 1.0 - (primary_after_norm / primary_before_norm)
    return max_row_norm_drift, target_suppression_ratio, unweighted_suppression_ratio


def _compute_probes_sha256() -> str:
    """Deterministic SHA-256 fingerprint of the canonical probe pair sets."""
    payload = json.dumps(
        {
            "refusal": REFUSAL_CONTRAST_PAIRS,
            "slop": SLOP_CONTRAST_PAIRS,
            "reasoning_anchor": REASONING_ANCHOR_PAIRS,
            "tool_sql_anchor": TOOL_SQL_ANCHOR_PAIRS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _inject_deliberation_tag_ids(
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    pairs: list[tuple[str, str]],
    segmenter: PhaseSegmenter,
    vocab_size: int,
    start_id: int = 3,
    end_id: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splice deliberation tag token ids into hash-encoded probe sequences.

    The fallback hash encoder carries no ``<think>`` / ``</think>`` tags, so
    token-id phase masks would remain empty on sml-nano. For every sequence
    this scrubs accidental tag-id collisions, writes ``start_id`` at position 0
    and ``end_id`` at the proportional deliberation/answer boundary of the
    source text (midpoint when the text has no close tag), so
    ``deliberation_only`` and ``answer_only`` masks are populated.
    """
    pos_ids = pos_ids.clone()
    neg_ids = neg_ids.clone()
    fallback_id = (start_id + end_id) % max(5, vocab_size)
    rows: list[tuple[torch.Tensor, str]] = [
        *[(pos_ids[i], pos) for i, (pos, _neg) in enumerate(pairs)],
        *[(neg_ids[i], neg) for i, (_pos, neg) in enumerate(pairs)],
    ]
    for ids_row, text in rows:
        seq_len = int(ids_row.numel())
        if seq_len < 2:
            continue
        collide = (ids_row == start_id) | (ids_row == end_id)
        ids_row[collide] = fallback_id
        content_bytes = max(1, len(text.encode("utf-8")))
        close_idx = text.find(segmenter.deliberation_end)
        boundary_bytes = (
            close_idx + len(segmenter.deliberation_end)
            if close_idx != -1
            else content_bytes // 2
        )
        k = int(round(boundary_bytes / content_bytes * seq_len))
        k = min(max(k, 1), seq_len - 1)
        ids_row[0] = start_id
        ids_row[k] = end_id
    return pos_ids, neg_ids


def _orthonormality_error(basis: list[torch.Tensor]) -> float:
    """Max deviation of a stacked basis from an identity Gram matrix."""
    if not basis:
        return 0.0
    stacked = torch.stack([b.detach().float() for b in basis], dim=0)
    gram = stacked @ stacked.T
    return float((gram - torch.eye(gram.shape[0], device=gram.device)).abs().max().item())


def _max_row_crest_factor(model: Transformer, layer_indices: list[int]) -> float:
    """Peak-to-mean row-norm crest factor over `o_proj`/`down_proj` weights."""
    crest = 1.0
    for layer_idx in layer_indices:
        layer = model.layers[layer_idx]
        for proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
            crest = max(crest, _compute_row_crest_factor(proj.weight.detach().float()))
    return crest


def run_h9_biprojected_surgery_benchmark(
    seed: int = 2026,
    top_k_layers: int = 3,
    scale: float = 1.0,
    output_dir: Path | str | None = None,
    model_id: str = "Tenet-Nano-6L-Unslop-v1",
) -> dict[str, Any]:
    """Execute `H9` (Lancet bi-projected surgery) on a 6-layer sml-nano Transformer.

    Suppresses refusal & slop write directions while keeping them orthogonal to the
    reasoning-anchor direction and preserving row norms, then verifies that anchor-behaviour
    KL collateral collapses versus naive (unprojected, all-layer) surgery.
    """
    torch.manual_seed(seed)

    config = ModelConfig(
        name="sml-surgery-6l",
        vocab_size=512,
        d_model=128,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
    )
    model = Transformer(config)
    model.eval()

    refusal_pos_ids, refusal_neg_ids = encode_contrast_pairs(
        REFUSAL_CONTRAST_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )
    slop_pos_ids, slop_neg_ids = encode_contrast_pairs(
        SLOP_CONTRAST_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )
    anchor_pos_ids, anchor_neg_ids = encode_contrast_pairs(
        REASONING_ANCHOR_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )
    tool_sql_pos_ids, tool_sql_neg_ids = encode_contrast_pairs(
        TOOL_SQL_ANCHOR_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )

    # Per-layer contrastive directions, bi-projected to be orthogonal to the 2D
    # anchor subspace spanned by the reasoning & tool-SQL anchor directions.
    r_refusal_perp: list[torch.Tensor] = []
    r_slop_perp: list[torch.Tensor] = []
    a_reasoning_dirs: list[torch.Tensor] = []
    a_tool_sql_dirs: list[torch.Tensor] = []
    anchor_orthogonality_error = 0.0
    tool_sql_orthogonality_error = 0.0
    per_layer_ortho: dict[str, float] = {}
    per_layer_tool_sql_ortho: dict[str, float] = {}
    per_layer_evr_refusal: dict[str, float] = {}
    per_layer_evr_slop: dict[str, float] = {}
    evr_refusal_total = 0.0
    evr_slop_total = 0.0
    for layer_idx in range(config.n_layers):
        r_refusal, refusal_stats = extract_contrastive_direction_with_stats(
            model, refusal_pos_ids, refusal_neg_ids, layer_idx=layer_idx
        )
        r_slop, slop_stats = extract_contrastive_direction_with_stats(
            model, slop_pos_ids, slop_neg_ids, layer_idx=layer_idx
        )
        evr_refusal = float(refusal_stats["explained_variance_ratio_1"])
        evr_slop = float(slop_stats["explained_variance_ratio_1"])
        per_layer_evr_refusal[str(layer_idx)] = round(evr_refusal, 6)
        per_layer_evr_slop[str(layer_idx)] = round(evr_slop, 6)
        evr_refusal_total += evr_refusal
        evr_slop_total += evr_slop
        a_reasoning = extract_contrastive_direction(
            model, anchor_pos_ids, anchor_neg_ids, layer_idx=layer_idx
        )
        a_tool_sql = extract_contrastive_direction(
            model, tool_sql_pos_ids, tool_sql_neg_ids, layer_idx=layer_idx
        )
        anchor_pair = [a_reasoning, a_tool_sql]
        ref_perp = biproject_direction(r_refusal, anchor_pair)
        slop_perp = biproject_direction(r_slop, anchor_pair)
        layer_error = max(
            float((ref_perp @ a_reasoning).abs().item()),
            float((slop_perp @ a_reasoning).abs().item()),
        )
        layer_tool_sql_error = max(
            float((ref_perp @ a_tool_sql).abs().item()),
            float((slop_perp @ a_tool_sql).abs().item()),
        )
        per_layer_ortho[str(layer_idx)] = round(layer_error, 8)
        per_layer_tool_sql_ortho[str(layer_idx)] = round(layer_tool_sql_error, 8)
        anchor_orthogonality_error = max(anchor_orthogonality_error, layer_error)
        tool_sql_orthogonality_error = max(tool_sql_orthogonality_error, layer_tool_sql_error)
        r_refusal_perp.append(ref_perp)
        r_slop_perp.append(slop_perp)
        a_reasoning_dirs.append(a_reasoning)
        a_tool_sql_dirs.append(a_tool_sql)
    probe_evr_refusal = evr_refusal_total / max(1, config.n_layers)
    probe_evr_slop = evr_slop_total / max(1, config.n_layers)

    # SCR-guided layer selection on the bi-projected targets.
    target_dirs_per_layer = {
        idx: [r_refusal_perp[idx], r_slop_perp[idx]] for idx in range(config.n_layers)
    }
    scr_profile = compute_layer_scr_profile(
        model,
        target_dirs_per_layer=target_dirs_per_layer,
        anchor_input_ids=anchor_pos_ids,
        scale=scale,
        preserve_row_norm=True,
    )
    selected_layers = _select_top_scr_layers(scr_profile, top_k_layers)

    # Soft SCR attenuation: full strength (alpha=1.0) on the max-SCR layer, softer
    # alphas (>= 0.1) proportional to each layer's SCR on the remaining selections.
    scr_by_layer = {int(entry["layer_idx"]): float(entry["scr"]) for entry in scr_profile}
    max_scr = max((scr_by_layer[layer_idx] for layer_idx in selected_layers), default=1.0)
    layer_alpha_schedule = {
        int(layer_idx): round(
            float(min(1.0, max(0.1, scr_by_layer[layer_idx] / max(max_scr, 1e-12)))), 6
        )
        for layer_idx in selected_layers
    }

    basis_per_layer = {
        idx: _biprojected_basis(r_refusal_perp[idx], r_slop_perp[idx]) for idx in selected_layers
    }
    original_weights = _snapshot_write_weights(model.layers, selected_layers)

    anchors_per_layer = {
        idx: [a_reasoning_dirs[idx], a_tool_sql_dirs[idx]] for idx in range(config.n_layers)
    }

    # Serializable Lancet recipe: the shared builder re-derives the bi-projected
    # targets, SCR selection, soft weights and orthonormal bases for this model.
    recipe = build_lancet_recipe(
        model,
        target_dirs=target_dirs_per_layer,
        anchor_dirs=anchors_per_layer,
        anchor_input_ids=anchor_pos_ids,
        model_id=model_id,
        base_model=config.name,
        top_k_layers=top_k_layers,
        scale=scale,
        probes_sha256=_compute_probes_sha256(),
    )

    # Naive control: every layer, no anchor bi-projection, no row-norm preservation.
    naive_model = apply_lancet_biprojected_surgery(
        model,
        target_dirs=target_dirs_per_layer,
        anchor_dirs=None,
        layer_indices=None,
        scale=scale,
        preserve_row_norm=False,
    )
    # Lancet: anchor-bi-projected targets, SCR top-k layers, norm-preserving projection,
    # with soft SCR attenuation weights per layer.
    lancet_model = apply_lancet_biprojected_surgery(
        model,
        target_dirs=target_dirs_per_layer,
        anchor_dirs=anchors_per_layer,
        layer_indices=recipe.selected_layers,
        scale=scale,
        preserve_row_norm=True,
        layer_weights=recipe.layer_weights,
    )

    max_row_norm_drift, target_suppression_ratio, unweighted_suppression_ratio = (
        _h9_surgery_metrics(
            lancet_model.layers,
            original_weights,
            recipe.selected_layers,
            recipe.basis_per_layer,
            layer_weights=recipe.layer_weights,
        )
    )
    naive_row_norm_drift, naive_suppression_ratio, _ = _h9_surgery_metrics(
        naive_model.layers, original_weights, selected_layers, basis_per_layer
    )

    naive_kl_collateral = compute_logit_kl_divergence(model, naive_model, anchor_pos_ids)
    lancet_kl_collateral = compute_logit_kl_divergence(model, lancet_model, anchor_pos_ids)
    h9_kl_reduction_ratio = 1.0 - (lancet_kl_collateral / max(1e-8, naive_kl_collateral))

    h9_verified = bool(
        anchor_orthogonality_error < 1e-6
        and tool_sql_orthogonality_error < 1e-6
        and max_row_norm_drift < 1e-5
        and target_suppression_ratio >= 0.90
        and h9_kl_reduction_ratio >= 0.35
    )

    if output_dir is not None:
        save_lancet_recipe(recipe, output_dir)

    return {
        "h9_verified": h9_verified,
        "backend": "nano",
        "model_name": config.name,
        "probes_sha256": recipe.probes_sha256,
        "recipe_metadata": {
            "schema_version": recipe.schema_version,
            "probes_sha256": recipe.probes_sha256,
            "selected_layers": recipe.selected_layers,
            "d_model": recipe.d_model,
            "num_layers": recipe.num_layers,
        },
        "anchor_orthogonality_error": round(anchor_orthogonality_error, 8),
        "tool_sql_orthogonality_error": round(tool_sql_orthogonality_error, 8),
        "probe_evr_refusal": round(probe_evr_refusal, 6),
        "probe_evr_slop": round(probe_evr_slop, 6),
        "layer_alpha_schedule": layer_alpha_schedule,
        "max_row_norm_drift": round(max_row_norm_drift, 8),
        "target_suppression_ratio": round(target_suppression_ratio, 6),
        "naive_kl_collateral": round(naive_kl_collateral, 6),
        "lancet_kl_collateral": round(lancet_kl_collateral, 6),
        "h9_kl_reduction_ratio": round(h9_kl_reduction_ratio, 4),
        "selected_layers": selected_layers,
        "scr_profile": scr_profile,
        "diagnostics": {
            "top_k_layers": top_k_layers,
            "scale": scale,
            "per_layer_anchor_orthogonality": per_layer_ortho,
            "per_layer_tool_sql_orthogonality": per_layer_tool_sql_ortho,
            "per_layer_evr_refusal": per_layer_evr_refusal,
            "per_layer_evr_slop": per_layer_evr_slop,
            "unweighted_suppression_ratio": round(unweighted_suppression_ratio, 6),
            "naive_row_norm_drift": round(naive_row_norm_drift, 8),
            "naive_suppression_ratio": round(naive_suppression_ratio, 6),
        },
    }


def run_h15_cot_tri_anchor_lancet_benchmark(
    seed: int = 2026,
    top_k_layers: int = 3,
    scale: float = 1.0,
    lambda_cot: float = 0.5,
    output_dir: Path | str | None = None,
    model_id: str = "Tenet-Nano-6L-Unslop-v2",
) -> dict[str, Any]:
    """Execute `H15` (CoT Tri-Anchor Lancet surgery) on the sml-nano Transformer.

    Builds a Tri-Anchor basis per layer (CoT deliberation + reasoning + tool/SQL
    anchors), bi-projects the refusal & slop suppression targets against it so
    deliberation/answer phases are preserved, picks layers by CoT-aware SCR, and
    applies row-norm-preserving Lancet surgery. Verifies basis/target
    orthogonality, row-norm preservation and crest-factor stability, and that
    anchor & CoT KL collateral collapses versus naive all-layer surgery.
    """
    torch.manual_seed(seed)

    config = load_model_config("configs/model/sml_nano.yaml")
    model = Transformer(config)
    model.eval()

    segmenter = PhaseSegmenter.for_format("think")
    segmenter.deliberation_start_id = 3
    segmenter.deliberation_end_id = 4
    cot_tag_token_ids = [segmenter.deliberation_start_id, segmenter.deliberation_end_id]

    cot_pos_ids, cot_neg_ids = encode_contrast_pairs(
        COT_REASONING_ANCHOR_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )
    refusal_pos_ids, refusal_neg_ids = encode_contrast_pairs(
        REFUSAL_CONTRAST_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )
    slop_pos_ids, slop_neg_ids = encode_contrast_pairs(
        SLOP_CONTRAST_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )
    anchor_pos_ids, anchor_neg_ids = encode_contrast_pairs(
        REASONING_ANCHOR_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )
    tool_sql_pos_ids, tool_sql_neg_ids = encode_contrast_pairs(
        TOOL_SQL_ANCHOR_PAIRS, vocab_size=config.vocab_size, seq_len=32
    )

    # Inject token ids 3/4 so the deliberation_only / answer_only phase masks
    # are populated for the CoT, refusal and slop probe sets on sml-nano.
    cot_pos_ids, cot_neg_ids = _inject_deliberation_tag_ids(
        cot_pos_ids, cot_neg_ids, COT_REASONING_ANCHOR_PAIRS, segmenter, config.vocab_size
    )
    refusal_pos_ids, refusal_neg_ids = _inject_deliberation_tag_ids(
        refusal_pos_ids, refusal_neg_ids, REFUSAL_CONTRAST_PAIRS, segmenter, config.vocab_size
    )
    slop_pos_ids, slop_neg_ids = _inject_deliberation_tag_ids(
        slop_pos_ids, slop_neg_ids, SLOP_CONTRAST_PAIRS, segmenter, config.vocab_size
    )

    # Per-layer Tri-Anchor basis (CoT + reasoning + tool/SQL, orthonormalized) and
    # refusal/slop targets bi-projected against that basis.
    tri_anchor_orthogonality_error = 0.0
    biprojected_orthogonality_error = 0.0
    per_layer_tri_ortho: dict[str, float] = {}
    per_layer_biproj_ortho: dict[str, float] = {}
    basis_per_layer_all: dict[int, list[torch.Tensor]] = {}
    raw_target_dirs_per_layer: dict[int, list[torch.Tensor]] = {}
    target_dirs_per_layer: dict[int, list[torch.Tensor]] = {}
    for layer_idx in range(config.n_layers):
        a_cot = extract_contrastive_direction(
            model,
            cot_pos_ids,
            cot_neg_ids,
            layer_idx=layer_idx,
            pool_mode="deliberation_only",
            phase_segmenter=segmenter,
        )
        a_reasoning = extract_contrastive_direction(
            model, anchor_pos_ids, anchor_neg_ids, layer_idx=layer_idx
        )
        a_tool_sql = extract_contrastive_direction(
            model, tool_sql_pos_ids, tool_sql_neg_ids, layer_idx=layer_idx
        )
        basis = build_tri_anchor_basis(a_cot, a_reasoning, a_tool_sql)
        basis_per_layer_all[layer_idx] = basis

        r_refusal = extract_contrastive_direction(
            model,
            refusal_pos_ids,
            refusal_neg_ids,
            layer_idx=layer_idx,
            pool_mode="answer_only",
            phase_segmenter=segmenter,
        )
        r_slop = extract_contrastive_direction(
            model,
            slop_pos_ids,
            slop_neg_ids,
            layer_idx=layer_idx,
            pool_mode="answer_only",
            phase_segmenter=segmenter,
        )
        r_refusal_perp = biproject_direction(r_refusal, basis)
        r_slop_perp = biproject_direction(r_slop, basis)

        layer_tri_error = _orthonormality_error(basis)
        layer_biproj_error = max(
            float((r_refusal_perp @ b).abs().item()) for b in basis
        ) if basis else 0.0
        layer_biproj_error = max(
            layer_biproj_error,
            max((float((r_slop_perp @ b).abs().item()) for b in basis), default=0.0),
        )
        per_layer_tri_ortho[str(layer_idx)] = round(layer_tri_error, 8)
        per_layer_biproj_ortho[str(layer_idx)] = round(layer_biproj_error, 8)
        tri_anchor_orthogonality_error = max(tri_anchor_orthogonality_error, layer_tri_error)
        biprojected_orthogonality_error = max(
            biprojected_orthogonality_error, layer_biproj_error
        )
        target_dirs_per_layer[layer_idx] = [r_refusal_perp, r_slop_perp]
        raw_target_dirs_per_layer[layer_idx] = [r_refusal, r_slop]

    # CoT-aware SCR-guided layer selection: blended collateral weights the
    # CoT-side KL (with tag-logit slice) against the anchor-side KL.
    scr_profile = compute_layer_scr_profile(
        model,
        target_dirs_per_layer=target_dirs_per_layer,
        anchor_input_ids=anchor_pos_ids,
        scale=scale,
        preserve_row_norm=True,
        cot_input_ids=cot_pos_ids,
        cot_tag_token_ids=cot_tag_token_ids,
        lambda_cot=lambda_cot,
    )
    selected_layers = _select_top_scr_layers(scr_profile, top_k_layers)

    anchor_dirs_per_layer = {idx: basis_per_layer_all[idx] for idx in selected_layers}

    # Naive control: every layer, un-projected raw targets, no row-norm preservation.
    naive_model = apply_lancet_biprojected_surgery(
        model,
        target_dirs=raw_target_dirs_per_layer,
        anchor_dirs=None,
        layer_indices=None,
        scale=scale,
        preserve_row_norm=False,
    )
    # Lancet: Tri-Anchor bi-projected targets, CoT-aware SCR top-k layers,
    # norm-preserving projection.
    lancet_model = apply_lancet_biprojected_surgery(
        model,
        target_dirs=target_dirs_per_layer,
        anchor_dirs=anchor_dirs_per_layer,
        layer_indices=selected_layers,
        scale=scale,
        preserve_row_norm=True,
    )

    original_weights = _snapshot_write_weights(model.layers, selected_layers)
    max_row_norm_drift, _target_suppression_ratio, _unweighted_suppression_ratio = (
        _h9_surgery_metrics(
            lancet_model.layers,
            original_weights,
            selected_layers,
            {idx: torch.stack(basis_per_layer_all[idx]) for idx in selected_layers},
        )
    )
    max_row_crest_factor = _max_row_crest_factor(lancet_model, selected_layers)

    kl_anchor_collateral = compute_logit_kl_divergence(model, lancet_model, anchor_pos_ids)
    kl_cot_collateral = compute_logit_kl_divergence(model, lancet_model, cot_pos_ids)
    naive_kl_anchor = compute_logit_kl_divergence(model, naive_model, anchor_pos_ids)
    naive_kl_cot = compute_logit_kl_divergence(model, naive_model, cot_pos_ids)
    lancet_collateral = lambda_cot * kl_cot_collateral + (1.0 - lambda_cot) * kl_anchor_collateral
    naive_collateral = lambda_cot * naive_kl_cot + (1.0 - lambda_cot) * naive_kl_anchor
    kl_reduction_ratio = 1.0 - (lancet_collateral / max(1e-8, naive_collateral))

    h15_verified = bool(
        tri_anchor_orthogonality_error < 1e-6
        and biprojected_orthogonality_error < 1e-6
        and max_row_norm_drift < 1e-5
        and max_row_crest_factor < 25.0
        and kl_reduction_ratio >= 0.35
    )

    result: dict[str, Any] = {
        "h15_verified": h15_verified,
        "backend": "nano",
        "model_name": config.name,
        "model_id": model_id,
        "tri_anchor_orthogonality_error": round(tri_anchor_orthogonality_error, 8),
        "biprojected_orthogonality_error": round(biprojected_orthogonality_error, 8),
        "max_row_norm_drift": round(max_row_norm_drift, 8),
        "max_row_crest_factor": round(max_row_crest_factor, 6),
        "kl_anchor_collateral": round(kl_anchor_collateral, 6),
        "kl_cot_collateral": round(kl_cot_collateral, 6),
        "naive_kl_anchor_collateral": round(naive_kl_anchor, 6),
        "naive_kl_cot_collateral": round(naive_kl_cot, 6),
        "kl_reduction_ratio": round(kl_reduction_ratio, 4),
        "selected_layers": selected_layers,
        "scr_profile": scr_profile,
        "diagnostics": {
            "top_k_layers": top_k_layers,
            "scale": scale,
            "lambda_cot": lambda_cot,
            "cot_tag_token_ids": cot_tag_token_ids,
            "num_tri_anchor_dirs_per_layer": {
                str(idx): len(basis_per_layer_all[idx]) for idx in range(config.n_layers)
            },
            "per_layer_tri_anchor_orthogonality": per_layer_tri_ortho,
            "per_layer_biprojected_orthogonality": per_layer_biproj_ortho,
        },
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "h15_cot_tri_anchor_benchmark.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )

    return result


def run_surgery_and_pareto_suite(seed: int = 2026) -> dict[str, Any]:
    """Execute end-to-end `H7` (CKA Mixed-Precision RCO), `H8` (Subspace Projection), and Pareto."""
    torch.manual_seed(seed)

    config = ModelConfig(
        name="sml-surgery-6l",
        vocab_size=512,
        d_model=128,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
    )
    base_model = Transformer(config)
    base_model.eval()

    adapted_model = copy.deepcopy(base_model)
    adapted_model.eval()

    # Simulate domain adaptation concentrated in layers 2 and 4 (rank-4 LoRA + outliers)
    with torch.no_grad():
        for layer_idx in (2, 4):
            layer = adapted_model.layers[layer_idx]
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ):
                out_f, in_f = proj.weight.shape
                u = torch.randn(out_f, 4)
                v = torch.randn(4, in_f)
                # Structured rank-4 update with high-salience channels sensitive to low-bit rounding
                v[:, ::16] *= 4.5
                proj.weight.add_(torch.matmul(u, v) * 0.14)

        for layer_idx in (0, 1, 3, 5):
            layer = adapted_model.layers[layer_idx]
            for proj in (
                layer.self_attn.q_proj,
                layer.self_attn.o_proj,
                layer.mlp.down_proj,
            ):
                proj.weight.add_(torch.randn_like(proj.weight) * 0.002)

    calibration_ids = torch.randint(0, config.vocab_size, (8, 32))
    eval_ids = torch.randint(0, config.vocab_size, (8, 32))

    layer_drifts = compute_layerwise_cka_drift(base_model, adapted_model, calibration_ids)

    # H7: Representation-Constrained Compression (CKA-guided vs equal-bit bottom-k control)
    bits_cka = allocate_mixed_precision_bits(
        layer_drifts, high_bits=8, low_bits=4, top_k_high=2, strategy="cka_guided"
    )
    bits_bottom_k = allocate_mixed_precision_bits(
        layer_drifts, high_bits=8, low_bits=4, top_k_high=2, strategy="bottom_k_control"
    )
    bits_int4 = allocate_mixed_precision_bits(
        layer_drifts, high_bits=8, low_bits=4, top_k_high=2, strategy="uniform_low"
    )
    bits_fp8 = allocate_mixed_precision_bits(
        layer_drifts, high_bits=8, low_bits=4, top_k_high=2, strategy="uniform_high"
    )

    model_cka, stats_cka = apply_mixed_precision_quantization(
        adapted_model, bits_cka, group_size=32
    )
    model_bottom_k, stats_bottom_k = apply_mixed_precision_quantization(
        adapted_model, bits_bottom_k, group_size=32
    )
    model_int4, stats_int4 = apply_mixed_precision_quantization(
        adapted_model, bits_int4, group_size=32
    )
    model_fp8, stats_fp8 = apply_mixed_precision_quantization(
        adapted_model, bits_fp8, group_size=32
    )
    model_pruned_int4 = prune_transformer_layers(model_int4, drop_layer_indices=[0])

    kl_cka = round(compute_logit_kl_divergence(adapted_model, model_cka, eval_ids), 6)
    kl_bottom_k = round(compute_logit_kl_divergence(adapted_model, model_bottom_k, eval_ids), 6)
    kl_int4 = round(compute_logit_kl_divergence(adapted_model, model_int4, eval_ids), 6)
    kl_fp8 = round(compute_logit_kl_divergence(adapted_model, model_fp8, eval_ids), 6)
    kl_pruned = round(compute_logit_kl_divergence(adapted_model, model_pruned_int4, eval_ids), 6)

    h7_equal_budget = (
        stats_cka["average_bits_per_weight"] == stats_bottom_k["average_bits_per_weight"]
    )
    h7_kl_reduction_ratio = round(1.0 - (kl_cka / max(1e-8, kl_bottom_k)), 4)
    h7_verified = bool(h7_equal_budget and (kl_cka < kl_bottom_k * 0.80))

    # H8: Contrastive Rank-1 Subspace Projection W' = W - r (r^T W)
    pos_ids = torch.randint(0, config.vocab_size // 2, (4, 16))
    neg_ids = torch.randint(config.vocab_size // 2, config.vocab_size, (4, 16))
    r_dir = extract_contrastive_direction(adapted_model, pos_ids, neg_ids, layer_idx=5)

    w_orig = adapted_model.layers[5].self_attn.o_proj.weight
    w_proj = project_out_rank1_subspace(w_orig, r_dir)
    max_ortho_residual = float((r_dir @ w_proj.float()).abs().max().item())

    # Verify that projecting out r_dir from layer 5 zeroes out layer 5's write along r_dir
    ablated_model = apply_subspace_ablation(adapted_model, r_dir, layer_indices=[5], scale=1.0)
    act_before = collect_layer_activations(adapted_model, pos_ids)
    act_after = collect_layer_activations(ablated_model, pos_ids)

    # Layer 5 residual write delta is (h_5 - h_4)
    write_delta_before = act_before[5] - act_before[4]
    write_delta_after = act_after[5] - act_after[4]
    proj_before = float((write_delta_before @ r_dir).abs().mean().item())
    proj_after = float((write_delta_after @ r_dir).abs().mean().item())
    kl_ablation_collateral = round(
        compute_logit_kl_divergence(adapted_model, ablated_model, eval_ids), 6
    )

    h8_verified = bool(max_ortho_residual < 1e-5 and proj_after < 1e-5 and proj_before > 1e-3)

    # Measure inference latencies & build Pareto frontier
    lat_bf16, tps_bf16 = _measure_inference_latency(adapted_model, eval_ids)
    lat_fp8, tps_fp8 = _measure_inference_latency(model_fp8, eval_ids)
    lat_cka, tps_cka = _measure_inference_latency(model_cka, eval_ids)
    lat_bot, tps_bot = _measure_inference_latency(model_bottom_k, eval_ids)
    lat_int4, tps_int4 = _measure_inference_latency(model_int4, eval_ids)
    lat_pruned, tps_pruned = _measure_inference_latency(model_pruned_int4, eval_ids)

    raw_variants = [
        {
            "variant": "1_bf16_adapted",
            "avg_bits": 16.0,
            "compression_ratio": 1.0,
            "kl_vs_bf16": 0.0,
            "measured_cpu_ms_per_tok": lat_bf16,
            "measured_cpu_tok_s": tps_bf16,
        },
        {
            "variant": "2_fp8_uniform",
            "avg_bits": stats_fp8["average_bits_per_weight"],
            "compression_ratio": stats_fp8["compression_ratio_vs_bf16"],
            "kl_vs_bf16": kl_fp8,
            "measured_cpu_ms_per_tok": lat_fp8,
            "measured_cpu_tok_s": tps_fp8,
        },
        {
            "variant": "3_cka_guided_mixed_4_8",
            "avg_bits": stats_cka["average_bits_per_weight"],
            "compression_ratio": stats_cka["compression_ratio_vs_bf16"],
            "kl_vs_bf16": kl_cka,
            "measured_cpu_ms_per_tok": lat_cka,
            "measured_cpu_tok_s": tps_cka,
        },
        {
            "variant": "4_bottom_k_control_4_8",
            "avg_bits": stats_bottom_k["average_bits_per_weight"],
            "compression_ratio": stats_bottom_k["compression_ratio_vs_bf16"],
            "kl_vs_bf16": kl_bottom_k,
            "measured_cpu_ms_per_tok": lat_bot,
            "measured_cpu_tok_s": tps_bot,
        },
        {
            "variant": "5_uniform_int4",
            "avg_bits": stats_int4["average_bits_per_weight"],
            "compression_ratio": stats_int4["compression_ratio_vs_bf16"],
            "kl_vs_bf16": kl_int4,
            "measured_cpu_ms_per_tok": lat_int4,
            "measured_cpu_tok_s": tps_int4,
        },
        {
            "variant": "6_pruned_5layer_int4",
            "avg_bits": round(stats_int4["average_bits_per_weight"] * (5.0 / 6.0), 2),
            "compression_ratio": round(
                16.0 / (stats_int4["average_bits_per_weight"] * (5.0 / 6.0)), 2
            ),
            "kl_vs_bf16": kl_pruned,
            "measured_cpu_ms_per_tok": lat_pruned,
            "measured_cpu_tok_s": tps_pruned,
        },
    ]

    # Compute Pareto optimality on (avg_bits, kl_vs_bf16): a variant is Pareto-optimal iff no other
    # variant has both <= avg_bits and <= kl_vs_bf16 with at least one strict inequality.
    pareto_frontier: list[dict[str, Any]] = []
    for cand in raw_variants:
        dominated = False
        for other in raw_variants:
            if other["variant"] == cand["variant"]:
                continue
            if (
                other["avg_bits"] <= cand["avg_bits"]
                and other["kl_vs_bf16"] <= cand["kl_vs_bf16"]
                and (
                    other["avg_bits"] < cand["avg_bits"] or other["kl_vs_bf16"] < cand["kl_vs_bf16"]
                )
            ):
                dominated = True
                break
        entry = dict(cand)
        entry["is_pareto_optimal"] = not dominated
        pareto_frontier.append(entry)

    by_variant = {item["variant"]: item for item in pareto_frontier}
    pareto_dominance_verified = (
        by_variant["3_cka_guided_mixed_4_8"]["is_pareto_optimal"] is True
        and by_variant["4_bottom_k_control_4_8"]["is_pareto_optimal"] is False
    )

    return {
        "h7_verified": h7_verified,
        "h7_kl_reduction_ratio": h7_kl_reduction_ratio,
        "h8_verified": h8_verified,
        "h8_diagnostics": {
            "max_ortho_residual": round(max_ortho_residual, 8),
            "layer5_write_proj_before": round(proj_before, 6),
            "layer5_write_proj_after": round(proj_after, 8),
            "kl_ablation_collateral": kl_ablation_collateral,
        },
        "pareto_dominance_verified": pareto_dominance_verified,
        "layer_drifts": layer_drifts,
        "bit_allocations": {
            "cka_guided": bits_cka,
            "bottom_k_control": bits_bottom_k,
        },
        "quantization_stats": {
            "cka_guided": stats_cka,
            "bottom_k_control": stats_bottom_k,
            "uniform_low": stats_int4,
            "uniform_high": stats_fp8,
        },
        "kl_divergences": {
            "cka_guided": kl_cka,
            "bottom_k_control": kl_bottom_k,
            "uniform_low": kl_int4,
            "uniform_high": kl_fp8,
            "pruned_5layer_int4": kl_pruned,
        },
        "pareto_frontier": pareto_frontier,
    }
