"""Linear CKA computation and layer activation collection utilities."""

from __future__ import annotations

import torch

from sml.model.transformer import Transformer


def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute Linear CKA similarity between two activation matrices.

    Args:
        x: Tensor of shape (B*T, D) or (B, T, D)
        y: Tensor of shape (B*T, D) or (B, T, D)
        eps: Small constant for numerical stability

    Returns:
        float: Linear CKA similarity value in [0.0, 1.0]
    """
    # Flatten leading dimensions if 3D
    if x.ndim == 3:
        x = x.view(-1, x.shape[-1]).to(torch.float32)
    if y.ndim == 3:
        y = y.view(-1, y.shape[-1]).to(torch.float32)

    # Center features column-wise
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # Compute linear CKA via Frobenius norm identity
    xx = torch.matmul(x.t(), x)
    yy = torch.matmul(y.t(), y)
    xy = torch.matmul(x.t(), y)

    numerator = torch.trace(torch.matmul(xy.t(), xy))
    denominator = torch.trace(torch.matmul(xx.t(), xx)) * torch.trace(torch.matmul(yy.t(), yy))

    cka = numerator / torch.sqrt(denominator + eps)
    return float(torch.clamp(cka, 0.0, 1.0))


def collect_layer_activations(model: Transformer, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """
    Collect activations from each layer of the transformer model.

    Args:
        model: Transformer model instance
        input_ids: Input tensor of shape (B, T)

    Returns:
        List of activation tensors, one per layer, each of shape (B*T, D)
    """
    activations = []

    def hook_fn(module, input, output):
        # TransformerBlock forward returns (x, present_kv), so we take output[0]
        if isinstance(output, tuple):
            activations.append(output[0].detach().view(-1, output[0].shape[-1]).to(torch.float32))
        else:
            activations.append(output.detach().view(-1, output.shape[-1]).to(torch.float32))

    # Register forward hooks on all layers
    hooks = []
    for layer in model.layers:
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Run model forward pass
    with torch.no_grad():
        model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def compute_layerwise_cka_drift(
    base_model: Transformer, adapted_model: Transformer, input_ids: torch.Tensor
) -> list[dict[str, float | int]]:
    """Compute per-block CKA similarity and drift between `base_model` and `adapted_model`."""
    base_deltas: list[torch.Tensor] = []
    adapted_deltas: list[torch.Tensor] = []

    def make_delta_hook(store: list[torch.Tensor]):
        def _hook(module, inputs, output):
            out_t = output[0] if isinstance(output, tuple) else output
            in_t = inputs[0]
            delta = (out_t - in_t).detach().view(-1, out_t.shape[-1]).to(torch.float32)
            store.append(delta)

        return _hook

    hooks = [
        layer.register_forward_hook(make_delta_hook(base_deltas)) for layer in base_model.layers
    ]
    with torch.no_grad():
        base_model(input_ids)
    for h in hooks:
        h.remove()

    hooks = [
        layer.register_forward_hook(make_delta_hook(adapted_deltas))
        for layer in adapted_model.layers
    ]
    with torch.no_grad():
        adapted_model(input_ids)
    for h in hooks:
        h.remove()

    results: list[dict[str, float | int]] = []
    for i, (base_d, adapted_d) in enumerate(zip(base_deltas, adapted_deltas, strict=False)):
        sim = linear_cka(base_d, adapted_d)
        results.append(
            {"layer_idx": i, "cka_similarity": round(sim, 6), "cka_drift": round(1.0 - sim, 6)}
        )

    return results
