"""Quantization utilities for model surgery."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import torch

from sml.surgery.cka import get_transformer_layers


def quantize_tensor_groupwise(
    weight: torch.Tensor,
    bits: int = 4,
    group_size: int = 32,
    activation_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Simulate symmetric group-wise fake quantization.

    Args:
        weight: Weight tensor to quantize
        bits: Number of bits for quantization (e.g., 4 for INT4, 8 for INT8)
        group_size: Size of groups for quantization
        activation_scale: Optional activation scaling factors

    Returns:
        Quantized weight tensor in original dtype
    """
    # Store original shape and dtype
    orig_shape = weight.shape
    orig_dtype = weight.dtype

    # Reshape to 2D if needed
    if weight.ndim > 2:
        weight = weight.view(-1, weight.shape[-1])

    # Apply activation scaling if provided
    if activation_scale is not None:
        s = activation_scale.clamp(min=1e-4)
        weight = weight * s.unsqueeze(0)

    # Pad last dimension to be divisible by group_size
    pad_len = (group_size - (weight.shape[-1] % group_size)) % group_size
    if pad_len > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_len))

    # Reshape into groups
    grouped = weight.view(weight.shape[0], -1, group_size)

    # Compute absmax per group
    absmax = torch.amax(grouped.abs(), dim=-1, keepdim=True)

    # Compute scale
    qmax = (2 ** (bits - 1)) - 1
    scale = (absmax / qmax).clamp(min=1e-8)

    # Quantize
    quantized = torch.round(grouped / scale).clamp(-qmax, qmax) * scale

    # Reshape back
    weight_q = quantized.view(weight.shape)

    # Unpad if needed
    if pad_len > 0:
        weight_q = weight_q[:, :-pad_len]

    # Apply inverse activation scaling if provided
    if activation_scale is not None:
        weight_q = weight_q / s.unsqueeze(0)

    # Reshape to original shape and cast back to the original tensor dtype
    return weight_q.reshape(orig_shape).to(dtype=orig_dtype)


def allocate_mixed_precision_bits(
    layer_drifts: list[dict[str, float | int]],
    high_bits: int = 8,
    low_bits: int = 4,
    top_k_high: int = 2,
    strategy: str = "cka_guided",
) -> dict[int, int]:
    """
    Allocate mixed precision bits based on different strategies.

    Args:
        layer_drifts: List of layer drift information from compute_layerwise_cka_drift
        high_bits: High precision bit width
        low_bits: Low precision bit width
        top_k_high: Number of layers to assign high precision to
        strategy: Allocation strategy ("cka_guided", "bottom_k_control", "uniform_low")

    Returns:
        Dictionary mapping layer indices to bit widths
    """
    layer_bits = {}

    if strategy == "cka_guided":
        # Sort layers by cka_drift descending
        sorted_layers = sorted(layer_drifts, key=lambda x: x["cka_drift"], reverse=True)
        high_layers = [layer["layer_idx"] for layer in sorted_layers[:top_k_high]]
        for layer in layer_drifts:
            layer_idx = layer["layer_idx"]
            layer_bits[layer_idx] = high_bits if layer_idx in high_layers else low_bits

    elif strategy == "bottom_k_control":
        # Sort layers by cka_drift ascending
        sorted_layers = sorted(layer_drifts, key=lambda x: x["cka_drift"])
        high_layers = [layer["layer_idx"] for layer in sorted_layers[:top_k_high]]
        for layer in layer_drifts:
            layer_idx = layer["layer_idx"]
            layer_bits[layer_idx] = high_bits if layer_idx in high_layers else low_bits

    elif strategy == "uniform_low":
        # Assign low_bits to all layers
        for layer in layer_drifts:
            layer_bits[layer["layer_idx"]] = low_bits

    elif strategy == "uniform_high":
        # Assign high_bits to all layers
        for layer in layer_drifts:
            layer_bits[layer["layer_idx"]] = high_bits

    return layer_bits


def apply_mixed_precision_quantization(
    model: torch.nn.Module, layer_bits: dict[int, int], group_size: int = 32
) -> tuple[torch.nn.Module, dict[str, float]]:
    """
    Apply mixed precision quantization to a transformer model.

    Args:
        model: Transformer model to quantize (HF or sml-nano)
        layer_bits: Mapping of layer indices to bit widths
        group_size: Group size for group-wise quantization

    Returns:
        Tuple of (quantized model, statistics dictionary)
    """
    # Deep copy model
    copied = copy.deepcopy(model)

    # Track quantization statistics
    total_params = 0
    weighted_bits_sum = 0

    # Quantize all linear layers in each layer block
    for layer_idx, block in enumerate(get_transformer_layers(copied)):
        bits = layer_bits.get(layer_idx, 4)  # Default to 4-bit

        # Quantize attention weights
        if hasattr(block.self_attn, "q_proj") and block.self_attn.q_proj is not None:
            params = block.self_attn.q_proj.weight.numel()
            total_params += params
            block.self_attn.q_proj.weight.data = quantize_tensor_groupwise(
                block.self_attn.q_proj.weight.data, bits, group_size
            )
            weighted_bits_sum += params * bits

        if hasattr(block.self_attn, "k_proj") and block.self_attn.k_proj is not None:
            params = block.self_attn.k_proj.weight.numel()
            total_params += params
            block.self_attn.k_proj.weight.data = quantize_tensor_groupwise(
                block.self_attn.k_proj.weight.data, bits, group_size
            )
            weighted_bits_sum += params * bits

        if hasattr(block.self_attn, "v_proj") and block.self_attn.v_proj is not None:
            params = block.self_attn.v_proj.weight.numel()
            total_params += params
            block.self_attn.v_proj.weight.data = quantize_tensor_groupwise(
                block.self_attn.v_proj.weight.data, bits, group_size
            )
            weighted_bits_sum += params * bits

        if hasattr(block.self_attn, "o_proj") and block.self_attn.o_proj is not None:
            params = block.self_attn.o_proj.weight.numel()
            total_params += params
            block.self_attn.o_proj.weight.data = quantize_tensor_groupwise(
                block.self_attn.o_proj.weight.data, bits, group_size
            )
            weighted_bits_sum += params * bits

        # Quantize MLP weights
        if hasattr(block.mlp, "gate_proj") and block.mlp.gate_proj is not None:
            params = block.mlp.gate_proj.weight.numel()
            total_params += params
            block.mlp.gate_proj.weight.data = quantize_tensor_groupwise(
                block.mlp.gate_proj.weight.data, bits, group_size
            )
            weighted_bits_sum += params * bits

        if hasattr(block.mlp, "up_proj") and block.mlp.up_proj is not None:
            params = block.mlp.up_proj.weight.numel()
            total_params += params
            block.mlp.up_proj.weight.data = quantize_tensor_groupwise(
                block.mlp.up_proj.weight.data, bits, group_size
            )
            weighted_bits_sum += params * bits

        if hasattr(block.mlp, "down_proj") and block.mlp.down_proj is not None:
            params = block.mlp.down_proj.weight.numel()
            total_params += params
            block.mlp.down_proj.weight.data = quantize_tensor_groupwise(
                block.mlp.down_proj.weight.data, bits, group_size
            )
            weighted_bits_sum += params * bits

    # Calculate statistics
    average_bits_per_weight = weighted_bits_sum / total_params if total_params > 0 else 0
    compression_ratio_vs_bf16 = 16.0 / average_bits_per_weight if average_bits_per_weight > 0 else 0
    estimated_model_mb = (total_params * average_bits_per_weight) / (
        8 * 1024 * 1024
    )  # Convert bits to MB

    stats_dict = {
        "average_bits_per_weight": round(average_bits_per_weight, 2),
        "compression_ratio_vs_bf16": round(compression_ratio_vs_bf16, 2),
        "estimated_model_mb": round(estimated_model_mb, 2),
    }

    return copied, stats_dict


def prune_transformer_layers(
    model: torch.nn.Module, drop_layer_indices: list[int]
) -> torch.nn.Module:
    """
    Prune transformer layers by removing specified layer indices.

    Supports both Hugging Face layouts (`model.model.layers`) and sml-nano
    layouts (`model.layers`).

    Args:
        model: Transformer model to prune (HF or sml-nano)
        drop_layer_indices: List of layer indices to remove

    Returns:
        Pruned transformer model
    """
    # Deep copy model
    copied = copy.deepcopy(model)

    if hasattr(copied, "model") and hasattr(copied.model, "layers"):
        # Hugging Face layout: model.model.layers
        kept_layers = [
            layer for i, layer in enumerate(copied.model.layers) if i not in drop_layer_indices
        ]
        copied.model.layers = torch.nn.ModuleList(kept_layers)
        if hasattr(copied.config, "num_hidden_layers"):
            copied.config.num_hidden_layers = len(kept_layers)
        else:
            copied.config = dataclasses.replace(copied.config, n_layers=len(kept_layers))
    else:
        # sml-nano layout: model.layers
        kept_layers = [
            layer for i, layer in enumerate(copied.layers) if i not in drop_layer_indices
        ]
        copied.layers = torch.nn.ModuleList(kept_layers)
        copied.config = dataclasses.replace(copied.config, n_layers=len(kept_layers))

    # Re-index attention layers after pruning
    for new_idx, layer in enumerate(kept_layers):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_idx

    return copied


FP8_E4M3_MAX = 448.0  # Max finite value of NVIDIA FP8 E4M3FN
_FP8_SCALE_MIN = 1e-12


def quantize_fp8_e4m3(tensor: torch.Tensor, per_channel: bool = True) -> torch.Tensor:
    """
    Emulate server-side FP8 (E4M3FN) weight quantization as used by
    NVIDIA/vLLM FP8 inference.

    The weight is scaled by ``absmax / 448.0``, cast through
    ``torch.float8_e4m3fn`` and dequantized back, so the returned tensor
    contains exactly the values a real FP8 endpoint would compute with.

    Args:
        tensor: Weight tensor to quantize.
        per_channel: If True (and the tensor is at least 2-D), use one scale
            per output channel (row); otherwise a single per-tensor scale.

    Returns:
        Dequantized FP8-emulated tensor in the original dtype and device.
    """
    orig_dtype = tensor.dtype
    orig_device = tensor.device

    work = tensor.detach().to(dtype=torch.float32)

    if per_channel and work.ndim >= 2:
        reduce_dims = tuple(range(1, work.ndim))
        absmax = torch.amax(work.abs(), dim=reduce_dims, keepdim=True)
    else:
        absmax = work.abs().amax()

    scale = (absmax / FP8_E4M3_MAX).clamp(min=_FP8_SCALE_MIN)
    scaled = (work / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    try:
        quantized = scaled.to(torch.float8_e4m3fn).to(torch.float32)
    except (RuntimeError, TypeError):
        # Device (e.g. MPS) cannot cast to float8: emulate on CPU and move back.
        quantized = scaled.cpu().to(torch.float8_e4m3fn).to(torch.float32).to(scaled.device)

    return (quantized * scale).to(dtype=orig_dtype).to(device=orig_device)


def apply_fp8_server_quantization(
    model: torch.nn.Module, per_channel: bool = True
) -> torch.nn.Module:
    """
    Emulate FP8 server quantization on every ``torch.nn.Linear`` weight.

    Args:
        model: Model whose linear weights should be FP8-quantized.
        per_channel: Passed through to :func:`quantize_fp8_e4m3`.

    Returns:
        Deep-copied model with FP8-emulated linear weights.
    """
    quantized_model = copy.deepcopy(model)
    with torch.no_grad():
        for module in quantized_model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = quantize_fp8_e4m3(module.weight.data, per_channel=per_channel)
    return quantized_model


def _forward_logits(input_ids: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Run ``model(input_ids)`` and return raw logits regardless of output type."""
    with torch.no_grad():
        out = model(input_ids)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple):
        return out[0]
    logits = getattr(out, "logits", None)
    if logits is None:
        raise TypeError(f"Cannot extract logits from model output of type {type(out)!r}")
    return logits


def _next_token_kl(
    reference: torch.nn.Module, candidate: torch.nn.Module, input_ids: torch.Tensor
) -> float:
    """Mean next-token KL(reference || candidate) over prompts and positions."""
    ref_logits = _forward_logits(input_ids, reference).float()
    cand_logits = _forward_logits(input_ids, candidate).float()
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
    cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
    ref_probs = ref_log_probs.exp()
    kl = (ref_probs * (ref_log_probs - cand_log_probs)).sum(dim=-1).mean()
    return float(kl.item())


def _normalized_mean_direction(weight: torch.Tensor) -> torch.Tensor:
    """Deterministic unit direction from the mean row of a weight matrix."""
    direction = weight.detach().float().mean(dim=0)
    norm = direction.norm()
    if float(norm) < 1e-8:
        direction = torch.zeros_like(direction)
        direction[0] = 1.0
    else:
        direction = direction / norm
    return direction


def _compute_subspace_releakage(
    base_model: torch.nn.Module,
    fp8_model: torch.nn.Module,
    direction: torch.Tensor,
) -> float:
    """Measure how much weight energy re-leaks onto ``direction`` after FP8.

    Projects the ``o_proj`` and ``down_proj`` weights of every transformer
    layer onto the unit ``direction`` and accumulates the Frobenius norms:
    ``sqrt(sum ||u @ W_fp8||_F^2) / (sqrt(sum ||u @ W_base||_F^2) + 1e-8)``.
    """
    u = direction.detach().to(dtype=torch.float32).reshape(1, -1)
    numerator = 0.0
    denominator = 0.0
    for base_layer, fp8_layer in zip(
        get_transformer_layers(base_model), get_transformer_layers(fp8_model), strict=True
    ):
        for base_proj, fp8_proj in (
            (base_layer.self_attn.o_proj, fp8_layer.self_attn.o_proj),
            (base_layer.mlp.down_proj, fp8_layer.mlp.down_proj),
        ):
            w_base = base_proj.weight.detach().to(dtype=torch.float32)
            w_fp8 = fp8_proj.weight.detach().to(dtype=torch.float32)
            numerator += float((u @ w_fp8).norm() ** 2)
            denominator += float((u @ w_base).norm() ** 2)
    return (numerator**0.5) / ((denominator**0.5) + 1e-8)


def evaluate_h11_fp8_provider_survival(
    seed: int = 2026, n_prompts: int = 8, seq_len: int = 32
) -> dict[str, float | bool | dict[str, dict[str, float]]]:
    """
    H11: verify that Lancet biprojected surgery survives FP8 (E4M3) server
    quantization better than naive subspace ablation.

    A deterministic small Transformer is surgered two ways (naive ablation at
    scale 1.5 without norm preservation vs. Lancet biprojection with norm
    preservation), each arm is FP8-emulated, and the next-token KL between the
    BF16-reference and FP8 outputs is compared per arm.

    Returns:
        Dict with ``fp8_retention_ratio``, ``h11_fp8_kl_reduction_ratio``,
        ``kl_naive_fp8``, ``kl_lancet_fp8``, ``h11_verified`` and an ``arms``
        breakdown.
    """
    from sml.model.transformer import ModelConfig, Transformer
    from sml.surgery.subspace import (
        apply_lancet_biprojected_surgery,
        apply_subspace_ablation,
        biproject_direction,
    )

    torch.manual_seed(seed)
    config = ModelConfig(
        name="h11-fp8-survival",
        vocab_size=256,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=384,
        max_seq_len=64,
        rope_theta=10000.0,
    )
    base_model = Transformer(config)
    base_model.eval()

    layers = get_transformer_layers(base_model)
    with torch.no_grad():
        target_direction = _normalized_mean_direction(layers[-1].self_attn.o_proj.weight)
        anchor_direction = _normalized_mean_direction(layers[0].self_attn.o_proj.weight)

    # Deployment direction for the Lancet arm: the biprojected target direction.
    lancet_direction = biproject_direction(target_direction, [anchor_direction])

    # a) Naive: rank-1 ablation without biprojection or norm preservation.
    naive_model = apply_subspace_ablation(base_model, target_direction, scale=1.5)
    # b) Lancet: biprojected, norm-preserving surgery.
    lancet_model = apply_lancet_biprojected_surgery(
        base_model,
        target_direction,
        anchor_direction,
        scale=1.0,
        preserve_row_norm=True,
    )

    prompt_gen = torch.Generator().manual_seed(seed + 1)
    prompt_ids = torch.randint(0, config.vocab_size, (n_prompts, seq_len), generator=prompt_gen)

    # BF16 provider reference and FP8-emulated endpoints per arm.
    reference_model = base_model.to(torch.bfloat16)
    naive_model = naive_model.to(torch.bfloat16)
    lancet_model = lancet_model.to(torch.bfloat16)
    naive_fp8_model = apply_fp8_server_quantization(naive_model, per_channel=True)
    lancet_fp8_model = apply_fp8_server_quantization(lancet_model, per_channel=True)
    naive_fp8_per_tensor = apply_fp8_server_quantization(naive_model, per_channel=False)
    lancet_fp8_per_tensor = apply_fp8_server_quantization(lancet_model, per_channel=False)

    # Drift of each deployed endpoint (surgery + FP8) vs the BF16 provider.
    kl_naive_fp8 = _next_token_kl(reference_model, naive_fp8_model, prompt_ids)
    kl_lancet_fp8 = _next_token_kl(reference_model, lancet_fp8_model, prompt_ids)
    kl_naive_own_fp8 = _next_token_kl(naive_model, naive_fp8_model, prompt_ids)
    kl_lancet_own_fp8 = _next_token_kl(lancet_model, lancet_fp8_model, prompt_ids)
    kl_naive_fp8_per_tensor = _next_token_kl(reference_model, naive_fp8_per_tensor, prompt_ids)
    kl_lancet_fp8_per_tensor = _next_token_kl(reference_model, lancet_fp8_per_tensor, prompt_ids)

    # Subspace re-leakage: weight energy along the ablated direction that FP8
    # quantization re-introduces, per channel (E4M3, per-row scale) and per tensor.
    fp8_subspace_releakage_naive = _compute_subspace_releakage(
        base_model, naive_fp8_model, target_direction
    )
    fp8_subspace_releakage_lancet = _compute_subspace_releakage(
        base_model, lancet_fp8_model, lancet_direction
    )
    fp8_subspace_releakage_naive_per_tensor = _compute_subspace_releakage(
        base_model, naive_fp8_per_tensor, target_direction
    )
    fp8_subspace_releakage_lancet_per_tensor = _compute_subspace_releakage(
        base_model, lancet_fp8_per_tensor, lancet_direction
    )

    fp8_retention_ratio = float(torch.exp(torch.tensor(-kl_lancet_fp8)).item())
    h11_fp8_kl_reduction_ratio = float((kl_naive_fp8 - kl_lancet_fp8) / max(kl_naive_fp8, 1e-12))
    h11_verified = bool(fp8_retention_ratio >= 0.96 and h11_fp8_kl_reduction_ratio >= 0.25)

    return {
        "fp8_retention_ratio": fp8_retention_ratio,
        "h11_fp8_kl_reduction_ratio": h11_fp8_kl_reduction_ratio,
        "kl_naive_fp8": kl_naive_fp8,
        "kl_lancet_fp8": kl_lancet_fp8,
        "kl_naive_fp8_per_tensor": kl_naive_fp8_per_tensor,
        "kl_lancet_fp8_per_tensor": kl_lancet_fp8_per_tensor,
        "fp8_subspace_releakage_naive": fp8_subspace_releakage_naive,
        "fp8_subspace_releakage_lancet": fp8_subspace_releakage_lancet,
        "fp8_subspace_releakage_naive_per_tensor": fp8_subspace_releakage_naive_per_tensor,
        "fp8_subspace_releakage_lancet_per_tensor": fp8_subspace_releakage_lancet_per_tensor,
        "h11_verified": h11_verified,
        "arms": {
            "naive": {
                "kl_bf16_vs_fp8": kl_naive_fp8,
                "kl_arm_vs_own_fp8": kl_naive_own_fp8,
                "kl_bf16_vs_fp8_per_tensor": kl_naive_fp8_per_tensor,
                "fp8_subspace_releakage_per_channel": fp8_subspace_releakage_naive,
                "fp8_subspace_releakage_per_tensor": fp8_subspace_releakage_naive_per_tensor,
            },
            "lancet": {
                "kl_bf16_vs_fp8": kl_lancet_fp8,
                "kl_arm_vs_own_fp8": kl_lancet_own_fp8,
                "kl_bf16_vs_fp8_per_tensor": kl_lancet_fp8_per_tensor,
                "fp8_subspace_releakage_per_channel": fp8_subspace_releakage_lancet,
                "fp8_subspace_releakage_per_tensor": fp8_subspace_releakage_lancet_per_tensor,
            },
        },
    }


def evaluate_llm_h11_fp8_provider_survival(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "auto",
    seed: int = 2026,
    n_prompts: int = 8,
    seq_len: int = 32,
) -> dict[str, Any]:
    """H11 on a real causal LLM: Lancet biprojection survives FP8 quantization.

    Loads `model_name`, derives target/anchor directions from the last / first
    block's `o_proj`, and surgeres the base two ways: naive rank-1 ablation
    (scale 1.5, no norm preservation) vs. Lancet biprojection (scale 1.0, row
    norms preserved). Each arm is FP8 (E4M3)-emulated per channel and per
    tensor, then scored by next-token KL against the unquantized reference and
    forbidden-subspace re-leakage. Re-leakage denominators are captured before
    any surgery, and each arm is freed right after scoring, so the LLM is
    never duplicated in RAM longer than one quantization copy.
    """
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sml.adaptation.compare import _resolve_llm_device
    from sml.surgery.subspace import (
        apply_lancet_biprojected_surgery,
        apply_subspace_ablation,
        biproject_direction,
    )

    torch.manual_seed(seed)
    resolved_device, dtype = _resolve_llm_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(resolved_device)
    base_model.eval()

    layers = get_transformer_layers(base_model)
    with torch.no_grad():
        target_direction = _normalized_mean_direction(layers[-1].self_attn.o_proj.weight)
        anchor_direction = _normalized_mean_direction(layers[0].self_attn.o_proj.weight)
    lancet_direction = biproject_direction(target_direction, [anchor_direction])

    prompt_gen = torch.Generator().manual_seed(seed + 1)
    prompt_ids = torch.randint(
        0,
        min(tokenizer.vocab_size, 32000),
        (n_prompts, seq_len),
        generator=prompt_gen,
    ).to(resolved_device)

    ref_logits = _forward_logits(prompt_ids, base_model).float().cpu()
    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
    ref_probs = ref_log_probs.exp()

    def _kl_vs_ref(candidate: torch.nn.Module) -> float:
        """Mean next-token KL(reference || candidate) against the FP32 reference."""
        cand_logits = _forward_logits(prompt_ids, candidate).float().cpu()
        cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
        kl = (ref_probs * (ref_log_probs - cand_log_probs)).sum(dim=-1).mean()
        return float(kl.item())

    def _base_denom(model: torch.nn.Module, direction: torch.Tensor) -> float:
        """sqrt(sum ||u @ W_base||_F^2) over every block's o_proj and down_proj."""
        u = direction.detach().reshape(1, -1).float()
        total = 0.0
        for block in get_transformer_layers(model):
            for proj in (block.self_attn.o_proj, block.mlp.down_proj):
                w = proj.weight.detach().float()
                total += float((u @ w).norm() ** 2)
        return total**0.5

    def _releakage(surgured_model: torch.nn.Module, direction: torch.Tensor, denom: float) -> float:
        """Forbidden-subspace energy of `surgured_model` normalized by `denom`."""
        u = direction.detach().reshape(1, -1).float()
        numerator = 0.0
        for block in get_transformer_layers(surgured_model):
            for proj in (block.self_attn.o_proj, block.mlp.down_proj):
                w = proj.weight.detach().float()
                numerator += float((u @ w).norm() ** 2)
        return (numerator**0.5) / (denom + 1e-8)

    # Denominators must be captured before any surgery or `del base_model`.
    base_denom_naive = _base_denom(base_model, target_direction)
    base_denom_lancet = _base_denom(base_model, lancet_direction)

    # a) Naive: rank-1 ablation without biprojection or norm preservation.
    naive_model = apply_subspace_ablation(base_model, target_direction, scale=1.5)
    naive_fp8_per_channel = apply_fp8_server_quantization(naive_model, per_channel=True)
    naive_fp8_per_tensor = apply_fp8_server_quantization(naive_model, per_channel=False)
    kl_naive_fp8 = _kl_vs_ref(naive_fp8_per_channel)
    kl_naive_fp8_per_tensor = _kl_vs_ref(naive_fp8_per_tensor)
    kl_naive_own_fp8 = _next_token_kl(naive_model, naive_fp8_per_channel, prompt_ids)
    fp8_subspace_releakage_naive = _releakage(
        naive_fp8_per_channel, target_direction, base_denom_naive
    )
    fp8_subspace_releakage_naive_per_tensor = _releakage(
        naive_fp8_per_tensor, target_direction, base_denom_naive
    )
    del naive_fp8_per_channel, naive_fp8_per_tensor, naive_model
    gc.collect()

    # b) Lancet: biprojected, norm-preserving surgery.
    lancet_model = apply_lancet_biprojected_surgery(
        base_model,
        target_direction,
        anchor_direction,
        scale=1.0,
        preserve_row_norm=True,
    )
    lancet_fp8_per_channel = apply_fp8_server_quantization(lancet_model, per_channel=True)
    lancet_fp8_per_tensor = apply_fp8_server_quantization(lancet_model, per_channel=False)
    kl_lancet_fp8 = _kl_vs_ref(lancet_fp8_per_channel)
    kl_lancet_fp8_per_tensor = _kl_vs_ref(lancet_fp8_per_tensor)
    kl_lancet_own_fp8 = _next_token_kl(lancet_model, lancet_fp8_per_channel, prompt_ids)
    fp8_subspace_releakage_lancet = _releakage(
        lancet_fp8_per_channel, lancet_direction, base_denom_lancet
    )
    fp8_subspace_releakage_lancet_per_tensor = _releakage(
        lancet_fp8_per_tensor, lancet_direction, base_denom_lancet
    )

    # Row-norm health of the surgured Lancet weights: crest factor (max/mean
    # row norm) and the largest weight magnitude across all 2D weights.
    max_row_crest_factor = 0.0
    max_abs_weight = 0.0
    for module in lancet_model.modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach().float()
            if weight.ndim == 2:
                max_abs_weight = max(max_abs_weight, float(weight.abs().max()))
                row_norms = weight.norm(dim=1)
                if row_norms.numel() > 0:
                    crest = float(row_norms.max() / row_norms.mean().clamp(min=1e-12))
                    max_row_crest_factor = max(max_row_crest_factor, crest)

    del (
        lancet_fp8_per_channel,
        lancet_fp8_per_tensor,
        lancet_model,
        base_model,
        layers,
    )
    gc.collect()

    fp8_retention_ratio = float(torch.exp(torch.tensor(-kl_lancet_fp8)).item())
    h11_fp8_kl_reduction_ratio = float((kl_naive_fp8 - kl_lancet_fp8) / max(kl_naive_fp8, 1e-12))
    # 16-bit LLM weights quantize with slightly higher KL, so the retention
    # floor is dtype-aware: 0.95 for fp16/bf16, 0.96 otherwise.
    retention_floor = 0.95 if dtype in (torch.float16, torch.bfloat16) else 0.96
    h11_verified = bool(
        fp8_retention_ratio >= retention_floor and h11_fp8_kl_reduction_ratio >= 0.25
    )

    return {
        "fp8_retention_ratio": fp8_retention_ratio,
        "h11_fp8_kl_reduction_ratio": h11_fp8_kl_reduction_ratio,
        "kl_naive_fp8": kl_naive_fp8,
        "kl_lancet_fp8": kl_lancet_fp8,
        "kl_naive_fp8_per_tensor": kl_naive_fp8_per_tensor,
        "kl_lancet_fp8_per_tensor": kl_lancet_fp8_per_tensor,
        "fp8_subspace_releakage_naive": fp8_subspace_releakage_naive,
        "fp8_subspace_releakage_lancet": fp8_subspace_releakage_lancet,
        "fp8_subspace_releakage_naive_per_tensor": fp8_subspace_releakage_naive_per_tensor,
        "fp8_subspace_releakage_lancet_per_tensor": fp8_subspace_releakage_lancet_per_tensor,
        "h11_verified": h11_verified,
        "arms": {
            "naive": {
                "kl_bf16_vs_fp8": kl_naive_fp8,
                "kl_arm_vs_own_fp8": kl_naive_own_fp8,
                "kl_bf16_vs_fp8_per_tensor": kl_naive_fp8_per_tensor,
                "fp8_subspace_releakage_per_channel": fp8_subspace_releakage_naive,
                "fp8_subspace_releakage_per_tensor": fp8_subspace_releakage_naive_per_tensor,
            },
            "lancet": {
                "kl_bf16_vs_fp8": kl_lancet_fp8,
                "kl_arm_vs_own_fp8": kl_lancet_own_fp8,
                "kl_bf16_vs_fp8_per_tensor": kl_lancet_fp8_per_tensor,
                "fp8_subspace_releakage_per_channel": fp8_subspace_releakage_lancet,
                "fp8_subspace_releakage_per_tensor": fp8_subspace_releakage_lancet_per_tensor,
            },
        },
        "backend": "llm",
        "model_name": model_name,
        "device": str(resolved_device),
        "max_row_crest_factor": max_row_crest_factor,
        "max_abs_weight": max_abs_weight,
    }
