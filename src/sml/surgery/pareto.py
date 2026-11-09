"""Weight surgery (`H7` CKA Mixed-Precision, `H8` Subspace Ablation) & Pareto frontier."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch
import torch.nn.functional as F

from sml.config import ModelConfig
from sml.model.transformer import Transformer
from sml.surgery.cka import collect_layer_activations, compute_layerwise_cka_drift
from sml.surgery.quantization import (
    allocate_mixed_precision_bits,
    apply_mixed_precision_quantization,
    prune_transformer_layers,
)
from sml.surgery.subspace import (
    apply_subspace_ablation,
    extract_contrastive_direction,
    project_out_rank1_subspace,
)


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
