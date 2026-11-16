"""Linear CKA computation and layer activation collection utilities."""

from __future__ import annotations

import torch


def get_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """
    Return the transformer layer stack for a model.

    Supports both Hugging Face causal LM layouts (`model.model.layers`) and
    sml-nano layouts (`model.layers`).

    Args:
        model: Transformer model instance (HF or sml-nano)

    Returns:
        ModuleList of transformer layers
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    return model.layers


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


def collect_layer_activations(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> list[torch.Tensor]:
    """
    Collect activations from each layer of the transformer model.

    Args:
        model: Transformer model instance (HF or sml-nano)
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
    for layer in get_transformer_layers(model):
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
    base_model: torch.nn.Module,
    adapted_model: torch.nn.Module,
    input_ids: torch.Tensor | None = None,
    *,
    prompt_inputs: list[torch.Tensor] | None = None,
) -> list[dict[str, float | int]]:
    """
    Compute per-block CKA similarity and drift between `base_model` and `adapted_model`.

    Supports two mutually exclusive input modes:
    - `input_ids`: single batched tensor of shape (B, T) (backwards-compatible path).
    - `prompt_inputs`: list of per-prompt tensors, each of shape (1, seq_len).
      Forward passes run per prompt without padding; per-layer residual deltas
      (out - in) are flattened to [tokens, hidden] and concatenated across
      prompts before computing CKA.
    """
    if prompt_inputs is not None:
        base_deltas = _collect_prompt_deltas(base_model, prompt_inputs)
        adapted_deltas = _collect_prompt_deltas(adapted_model, prompt_inputs)
    else:
        if input_ids is None:
            raise ValueError("Either `input_ids` or `prompt_inputs` must be provided.")
        base_deltas = _collect_deltas(base_model, input_ids)
        adapted_deltas = _collect_deltas(adapted_model, input_ids)

    results: list[dict[str, float | int]] = []
    for i, (base_d, adapted_d) in enumerate(zip(base_deltas, adapted_deltas, strict=False)):
        sim = linear_cka(base_d, adapted_d)
        results.append(
            {"layer_idx": i, "cka_similarity": round(sim, 6), "cka_drift": round(1.0 - sim, 6)}
        )

    return results


def _make_delta_hook(store: list[torch.Tensor]):
    """Build a forward hook that records per-layer residual deltas (out - in)."""

    def _hook(module, inputs, output):
        out_t = output[0] if isinstance(output, tuple) else output
        in_t = inputs[0]
        delta = (out_t - in_t).detach().view(-1, out_t.shape[-1]).to(torch.float32)
        store.append(delta)

    return _hook


def _collect_deltas(model: torch.nn.Module, input_ids: torch.Tensor) -> list[torch.Tensor]:
    """Run a single forward pass and return per-layer residual deltas, one per layer."""
    device = next(model.parameters()).device
    deltas: list[torch.Tensor] = []

    hooks = [
        layer.register_forward_hook(_make_delta_hook(deltas))
        for layer in get_transformer_layers(model)
    ]
    try:
        with torch.no_grad():
            model(input_ids.to(device))
    finally:
        for h in hooks:
            h.remove()

    return deltas


def _collect_prompt_deltas(
    model: torch.nn.Module, prompt_inputs: list[torch.Tensor]
) -> list[torch.Tensor]:
    """
    Run forward passes over each prompt (no padding) and return per-layer residual
    deltas concatenated across prompts, each of shape [tokens, hidden].
    """
    device = next(model.parameters()).device
    layers = get_transformer_layers(model)
    per_layer: list[list[torch.Tensor]] = [[] for _ in layers]

    def make_prompt_delta_hook(layer_idx: int):
        def _hook(module, inputs, output):
            out_t = output[0] if isinstance(output, tuple) else output
            in_t = inputs[0]
            delta = (out_t - in_t).detach().view(-1, out_t.shape[-1]).to(torch.float32)
            per_layer[layer_idx].append(delta)

        return _hook

    # Register hooks once; each prompt's forward pass appends one chunk per layer.
    hooks = [
        layer.register_forward_hook(make_prompt_delta_hook(i)) for i, layer in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            for prompt in prompt_inputs:
                model(prompt.to(device))
    finally:
        for h in hooks:
            h.remove()

    return [torch.cat(chunks, dim=0) for chunks in per_layer]


def compute_blockwise_quantization_drift(
    model: torch.nn.Module,
    prompt_inputs: list[torch.Tensor],
    quant_bits: int = 4,
    group_size: int = 32,
) -> list[dict[str, float | int]]:
    """
    Compute per-layer leave-one-layer-quantized forward-KL drift (the H7 metric).

    For each transformer layer, all `torch.nn.Linear` weights are quantized in-place
    with group-wise fake quantization while every other layer keeps its original
    weights. The token-averaged forward KL divergence between the clean model's
    output distribution and the leave-one-out quantized model's distribution is
    averaged over a small set of calibration prompts. Original weights are always
    restored in a `finally` block.
    """
    # Imported lazily: `sml.surgery.quantization` imports `get_transformer_layers`
    # from this module, so a module-level import here would be circular.
    from sml.surgery.quantization import quantize_tensor_groupwise

    device = next(model.parameters()).device
    layers = get_transformer_layers(model)
    cal_inputs = [p.to(device) for p in prompt_inputs[:8]]

    def _logits(m: torch.nn.Module, p: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return float32 logits."""
        try:
            return m(p, use_cache=False).logits.float()
        except TypeError:
            return m(p).logits.float()

    with torch.no_grad():
        ref_log_probs = [
            torch.nn.functional.log_softmax(_logits(model, p), dim=-1)
            for p in cal_inputs
        ]
        ref_probs = [lp.exp() for lp in ref_log_probs]

    results: list[dict[str, float | int]] = []
    for i, layer in enumerate(layers):
        linears = [m for m in layer.modules() if isinstance(m, torch.nn.Linear)]
        originals = {id(m): m.weight.data.clone() for m in linears}
        try:
            for m in linears:
                m.weight.data = quantize_tensor_groupwise(
                    originals[id(m)], bits=quant_bits, group_size=group_size
                )
            kl_per_prompt: list[torch.Tensor] = []
            with torch.no_grad():
                for j, p in enumerate(cal_inputs):
                    quant_log_probs = torch.nn.functional.log_softmax(
                        _logits(model, p), dim=-1
                    )
                    kl = (
                        (ref_probs[j] * (ref_log_probs[j] - quant_log_probs))
                        .sum(dim=-1)
                        .mean()
                        .clamp_min(0.0)
                    )
                    kl_per_prompt.append(kl)
            mean_kl = (
                float(torch.stack(kl_per_prompt).mean()) if kl_per_prompt else 0.0
            )
        finally:
            for m in linears:
                m.weight.data = originals[id(m)]

        results.append(
            {
                "layer_idx": i,
                "leave_one_out_kl": round(mean_kl, 6),
                "cka_drift": round(mean_kl, 6),
            }
        )

    return results
