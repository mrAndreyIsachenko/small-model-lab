"""Subspace manipulation utilities for model surgery."""

from __future__ import annotations

import copy

import torch

from sml.model.transformer import Transformer
from sml.surgery.cka import collect_layer_activations


def extract_contrastive_direction(
    model: Transformer, positive_ids: torch.Tensor, negative_ids: torch.Tensor, layer_idx: int = -1
) -> torch.Tensor:
    """
    Extract a contrastive direction between positive and negative examples.

    Args:
        model: Transformer model instance
        positive_ids: Positive input tensor of shape (B, T)
        negative_ids: Negative input tensor of shape (B, T)
        layer_idx: Index of layer to extract activations from

    Returns:
        Unit-normalized contrastive direction vector of shape (D,)
    """
    # Collect activations for positive examples
    pos_activations = collect_layer_activations(model, positive_ids)
    pos_mean = pos_activations[layer_idx].mean(dim=0)  # Mean over tokens

    # Collect activations for negative examples
    neg_activations = collect_layer_activations(model, negative_ids)
    neg_mean = neg_activations[layer_idx].mean(dim=0)  # Mean over tokens

    # Compute difference and normalize
    diff = pos_mean - neg_mean
    direction = diff / (diff.norm(p=2) + 1e-8)

    return direction


def project_out_rank1_subspace(
    weight: torch.Tensor, direction: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    """
    Project out a rank-1 subspace component from a weight matrix.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        direction: Direction vector of shape (out_features,)
        scale: Scaling factor for projection

    Returns:
        Modified weight tensor of same shape as input
    """
    # Normalize direction to unit norm
    r = direction.to(weight.device, dtype=torch.float32)
    r = r / (r.norm(p=2) + 1e-8)

    # Compute projection matrix
    proj = torch.outer(r, r @ weight.float())

    # Apply projection and return in original dtype
    return (weight.float() - scale * proj).to(weight.dtype)


def apply_subspace_ablation(
    model: Transformer,
    direction: torch.Tensor,
    layer_indices: list[int] | None = None,
    scale: float = 1.0,
) -> Transformer:
    """
    Apply subspace ablation to specific layers of a model.

    Args:
        model: Transformer model instance
        direction: Direction vector to project out
        layer_indices: Indices of layers to modify (None means all layers)
        scale: Scaling factor for projection

    Returns:
        Modified model with subspace ablation applied
    """
    # Create deep copy of model
    modified_model = copy.deepcopy(model)

    # Determine which layers to modify
    if layer_indices is None:
        layer_indices = list(range(len(modified_model.layers)))

    # Apply projection to selected layers
    with torch.no_grad():
        for i in layer_indices:
            layer = modified_model.layers[i]

            # Update attention output projection
            layer.self_attn.o_proj.weight = torch.nn.Parameter(
                project_out_rank1_subspace(layer.self_attn.o_proj.weight, direction, scale)
            )

            # Update MLP down projection
            layer.mlp.down_proj.weight = torch.nn.Parameter(
                project_out_rank1_subspace(layer.mlp.down_proj.weight, direction, scale)
            )

    return modified_model
