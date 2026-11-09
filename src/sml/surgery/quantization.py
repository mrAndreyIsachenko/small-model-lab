"""Quantization utilities for model surgery."""

from __future__ import annotations

import copy
import dataclasses

import torch

from sml.model.transformer import Transformer


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

    # Reshape to original shape and convert to original dtype
    return weight_q.view(orig_shape).to(orig_dtype)


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
    model: Transformer, layer_bits: dict[int, int], group_size: int = 32
) -> tuple[Transformer, dict[str, float]]:
    """
    Apply mixed precision quantization to a transformer model.

    Args:
        model: Transformer model to quantize
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
    for layer_idx, block in enumerate(copied.layers):
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


def prune_transformer_layers(model: Transformer, drop_layer_indices: list[int]) -> Transformer:
    """
    Prune transformer layers by removing specified layer indices.

    Args:
        model: Transformer model to prune
        drop_layer_indices: List of layer indices to remove

    Returns:
        Pruned transformer model
    """
    # Deep copy model
    copied = copy.deepcopy(model)

    # Filter out layers whose index is in drop_layer_indices
    kept_layers = [layer for i, layer in enumerate(copied.layers) if i not in drop_layer_indices]

    # Update model layers and config
    copied.layers = torch.nn.ModuleList(kept_layers)
    copied.config = dataclasses.replace(copied.config, n_layers=len(copied.layers))

    return copied
