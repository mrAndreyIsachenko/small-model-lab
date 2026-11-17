"""Subspace manipulation utilities for model surgery."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from sml.model.transformer import Transformer
from sml.surgery.cka import collect_layer_activations, get_transformer_layers


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


def compute_probe_spectral_stats(diff_matrix: torch.Tensor) -> dict[str, float]:
    """
    Compute spectral statistics of a per-example difference matrix.

    Given ``Delta_H`` of shape (B, D) holding per-example activation
    differences, compute the singular values of the (float32) matrix and
    summarize how concentrated the differences are along a single direction:

    - ``explained_variance_ratio_1``: share of total squared singular-value
      energy captured by the top singular value.
    - ``singular_value_1``: the top singular value itself.
    - ``spectral_rank_effective``: participation-ratio style effective rank,
      ``(sum(s^2))^2 / sum(s^4)``; 1.0 for a rank-1 matrix, up to B for a
      flat spectrum.

    Args:
        diff_matrix: Difference matrix of shape (B, D).

    Returns:
        Dict with the three statistics above; zeros when the matrix is
        empty or has no energy.
    """
    if diff_matrix.numel() == 0:
        return {
            "explained_variance_ratio_1": 0.0,
            "singular_value_1": 0.0,
            "spectral_rank_effective": 0.0,
        }

    s = torch.linalg.svdvals(diff_matrix.float())
    if s.numel() == 0:
        return {
            "explained_variance_ratio_1": 0.0,
            "singular_value_1": 0.0,
            "spectral_rank_effective": 0.0,
        }

    return {
        "explained_variance_ratio_1": float((s[0] ** 2) / (torch.sum(s**2) + 1e-8)),
        "singular_value_1": float(s[0]),
        "spectral_rank_effective": float((torch.sum(s**2) ** 2) / (torch.sum(s**4) + 1e-8)),
    }


def extract_contrastive_direction_with_stats(
    model: Transformer, positive_ids: torch.Tensor, negative_ids: torch.Tensor, layer_idx: int = -1
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Extract a contrastive direction plus spectral stats of the differences.

    Like ``extract_contrastive_direction``, but instead of averaging
    activations across examples first, it keeps a per-example mean over
    sequence tokens so ``Delta_H = mean_pos - mean_neg`` has shape (B, D).
    The returned direction is the unit-normalized mean of ``Delta_H``,
    matching ``extract_contrastive_direction``; the statistics describe how
    concentrated the per-example differences are around that direction.

    Args:
        model: Transformer model instance
        positive_ids: Positive input tensor of shape (B, T)
        negative_ids: Negative input tensor of shape (B, T)
        layer_idx: Index of layer to extract activations from

    Returns:
        Tuple ``(direction, stats)`` where ``direction`` is a unit-norm
        vector of shape (D,) and ``stats`` comes from
        ``compute_probe_spectral_stats``.
    """
    # Collect activations for positive examples. `collect_layer_activations`
    # returns flattened (B*T, D) tensors, so restore the (B, T, D) layout from
    # the input's shape; the mean over sequence tokens (dim=1) gives (B, D).
    pos_activations = collect_layer_activations(model, positive_ids)
    batch_size = int(positive_ids.shape[0])
    seq_len = int(positive_ids.shape[1]) if positive_ids.dim() > 1 else 1
    pos_mean = pos_activations[layer_idx].reshape(batch_size, seq_len, -1).mean(dim=1)  # (B, D)

    # Collect activations for negative examples
    neg_activations = collect_layer_activations(model, negative_ids)
    neg_mean = neg_activations[layer_idx].reshape(batch_size, seq_len, -1).mean(dim=1)  # (B, D)

    # Per-example difference matrix
    delta_h = pos_mean - neg_mean
    stats = compute_probe_spectral_stats(delta_h)

    # Mean direction, normalized to unit L2 norm (matching
    # extract_contrastive_direction).
    mean_dir = delta_h.mean(dim=0)
    direction = mean_dir / (mean_dir.norm(p=2) + 1e-8)

    return direction, stats


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

    # Apply projection and return in original dtype and device
    return (weight.float() - scale * proj).to(device=weight.device, dtype=weight.dtype)


def _orthonormalize_basis(
    vectors: list[torch.Tensor] | torch.Tensor, device: torch.device | None = None
) -> list[torch.Tensor]:
    """
    Orthonormalize a set of vectors with modified Gram-Schmidt in float32.

    Args:
        vectors: List of vectors (any shape, flattened) or a tensor whose
            first dimension enumerates the vectors.
        device: Optional device to move vectors to before orthonormalization.

    Returns:
        List of orthonormal float32 vectors. Linearly dependent or
        near-zero vectors are dropped.
    """
    if isinstance(vectors, torch.Tensor):
        if vectors.ndim == 1:
            candidates: list[torch.Tensor] = [vectors]
        else:
            candidates = [vectors[i] for i in range(vectors.shape[0])]
    else:
        candidates = [torch.as_tensor(v) for v in vectors]

    basis: list[torch.Tensor] = []
    for vec in candidates:
        if device is not None:
            v = vec.detach().to(device=device, dtype=torch.float32)
        else:
            v = vec.detach().to(dtype=torch.float32)
        v = v.reshape(-1)

        # Modified Gram-Schmidt: subtract projections sequentially against
        # the already-orthonormalized basis vectors.
        for b in basis:
            v = v - (v @ b) * b

        norm = float(v.norm(p=2))
        if norm > 1e-6:
            basis.append(v / norm)

    return basis


def biproject_direction(
    target_dir: torch.Tensor, anchor_dirs: list[torch.Tensor] | torch.Tensor
) -> torch.Tensor:
    """
    Remove all anchor components from a direction, then renormalize.

    The result is the residual of ``target_dir`` after projecting out the
    subspace spanned by ``anchor_dirs``, normalized to unit L2 norm.

    Args:
        target_dir: Direction vector to biproject.
        anchor_dirs: Anchor directions whose span should be removed.

    Returns:
        Unit-norm float32 residual tensor on ``target_dir.device``.
    """
    device = target_dir.device
    basis = _orthonormalize_basis(anchor_dirs, device=device)

    t = target_dir.detach().to(dtype=torch.float32).reshape(-1)
    for b in basis:
        t = t - (t @ b) * b

    norm = float(t.norm(p=2))
    if norm > 1e-8:
        t = t / norm

    return t


def project_out_subspace_norm_preserving(
    weight: torch.Tensor,
    directions: list[torch.Tensor] | torch.Tensor,
    scale: float = 1.0,
    preserve_row_norm: bool = True,
) -> torch.Tensor:
    """
    Project a multi-dimensional subspace out of a weight matrix, keeping row norms.

    Args:
        weight: Weight tensor of shape (out_features, in_features).
        directions: Directions spanning the subspace to remove.
        scale: Scaling factor for the projection.
        preserve_row_norm: If True, rescale each row of the projected weight
            so its L2 norm matches the corresponding original row.

    Returns:
        Modified weight tensor in the original device and dtype.
    """
    orig_device = weight.device
    orig_dtype = weight.dtype

    w_f = weight.detach().to(dtype=torch.float32)
    basis = _orthonormalize_basis(directions, device=orig_device)

    w_proj = w_f.clone()
    for r in basis:
        w_proj = w_proj - scale * torch.outer(r, r @ w_f)

    if preserve_row_norm:
        orig_row_norms = w_f.norm(p=2, dim=-1)
        new_row_norms = w_proj.norm(p=2, dim=-1)
        w_proj = w_proj * (orig_row_norms / (new_row_norms + 1e-8)).unsqueeze(-1)

    return w_proj.to(device=orig_device, dtype=orig_dtype)


def _as_direction_list(value: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
    """Normalize a single per-layer direction spec into a list of vectors."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value]
        return [value[i] for i in range(value.shape[0])]
    return [torch.as_tensor(v) for v in value]


def _resolve_per_layer_directions(
    directions: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    num_layers: int,
) -> dict[int, list[torch.Tensor]]:
    """
    Resolve a direction spec into a per-layer mapping ``{layer_idx: [dirs]}``.

    Accepted forms:
        - dict mapping layer index -> single vector, list of vectors, or
          tensor whose rows are vectors.
        - tensor of shape (D,): one direction applied to every layer.
        - tensor of shape (L, D): row ``i`` is the direction for layer ``i``.
        - list of length L: element ``i`` holds the directions for layer ``i``.
    """
    if isinstance(directions, dict):
        return {int(k): _as_direction_list(v) for k, v in directions.items()}

    if isinstance(directions, torch.Tensor):
        if directions.ndim == 1:
            return {i: [directions] for i in range(num_layers)}
        return {i: [directions[i]] for i in range(min(directions.shape[0], num_layers))}

    resolved = {}
    for i, entry in enumerate(directions):
        if i >= num_layers:
            break
        resolved[i] = _as_direction_list(entry)
    return resolved


def _forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return raw logits (handles HF and sml-nano)."""
    out = model(input_ids)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


def compute_layer_scr_profile(
    model: torch.nn.Module,
    target_dirs_per_layer: (
        dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor
    ),
    anchor_input_ids: torch.Tensor,
    candidate_layers: list[int] | None = None,
    scale: float = 1.0,
    preserve_row_norm: bool = True,
) -> list[dict[str, float | int]]:
    """
    Score candidate layers by signal-to-collateral ratio (SCR) for surgery.

    For each candidate layer, temporarily apply norm-preserving subspace
    projection to ``self_attn.o_proj`` and ``mlp.down_proj`` (weights are
    restored afterward, so no per-layer deepcopy is needed), then measure:

    - ``signal``: sum of squared removed target-projection energy on both
      weights.
    - ``kl_collateral``: forward KL divergence between original and modified
      model logits on ``anchor_input_ids``.
    - ``scr``: ``signal / (kl_collateral + 1e-6)``.

    Args:
        model: Transformer model instance (HF or sml-nano).
        target_dirs_per_layer: Target directions per layer.
        anchor_input_ids: Token ids used to measure collateral damage.
        candidate_layers: Layers to score (None means all layers).
        scale: Projection scale passed through to the projection.
        preserve_row_norm: Passed through to the projection.

    Returns:
        List of dicts with keys ``layer_idx``, ``signal``, ``kl_collateral``,
        and ``scr``, sorted descending by ``scr``.
    """
    layers = get_transformer_layers(model)
    num_layers = len(layers)
    if candidate_layers is None:
        candidate_layers = list(range(num_layers))

    target_map = _resolve_per_layer_directions(target_dirs_per_layer, num_layers)
    model_device = next(model.parameters()).device
    anchor_input_ids = anchor_input_ids.to(model_device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        orig_logits = _forward_logits(model, anchor_input_ids)
        orig_log_probs = torch.log_softmax(orig_logits.float(), dim=-1)
        orig_probs = orig_log_probs.exp()

        profile: list[dict[str, float | int]] = []
        try:
            for idx in candidate_layers:
                layer = layers[idx]
                dirs = target_map.get(idx, [])
                basis = _orthonormalize_basis(dirs, device=model_device)

                # Signal: removed target projection energy on both weights.
                signal = 0.0
                for weight in (layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight):
                    w_f = weight.detach().to(dtype=torch.float32)
                    for r in basis:
                        signal += float((scale * (r @ w_f)).pow(2).sum())

                # Temporarily apply the projection, measure KL, then restore.
                orig_attn_weight = layer.self_attn.o_proj.weight
                orig_mlp_weight = layer.mlp.down_proj.weight
                layer.self_attn.o_proj.weight = torch.nn.Parameter(
                    project_out_subspace_norm_preserving(
                        orig_attn_weight, dirs, scale=scale, preserve_row_norm=preserve_row_norm
                    ),
                    requires_grad=False,
                )
                layer.mlp.down_proj.weight = torch.nn.Parameter(
                    project_out_subspace_norm_preserving(
                        orig_mlp_weight, dirs, scale=scale, preserve_row_norm=preserve_row_norm
                    ),
                    requires_grad=False,
                )
                try:
                    mod_logits = _forward_logits(model, anchor_input_ids)
                    mod_log_probs = torch.log_softmax(mod_logits.float(), dim=-1)
                    kl = float((orig_probs * (orig_log_probs - mod_log_probs)).sum(dim=-1).mean())
                finally:
                    layer.self_attn.o_proj.weight = orig_attn_weight
                    layer.mlp.down_proj.weight = orig_mlp_weight

                scr = signal / (kl + 1e-6)
                profile.append(
                    {
                        "layer_idx": int(idx),
                        "signal": float(signal),
                        "kl_collateral": float(kl),
                        "scr": float(scr),
                    }
                )
        finally:
            if was_training:
                model.train()

    return sorted(profile, key=lambda entry: entry["scr"], reverse=True)


def apply_subspace_ablation(
    model: torch.nn.Module,
    direction: torch.Tensor,
    layer_indices: list[int] | None = None,
    scale: float = 1.0,
) -> torch.nn.Module:
    """
    Apply subspace ablation to specific layers of a model.

    Args:
        model: Transformer model instance (HF or sml-nano)
        direction: Direction vector to project out
        layer_indices: Indices of layers to modify (None means all layers)
        scale: Scaling factor for projection

    Returns:
        Modified model with subspace ablation applied
    """
    # Create deep copy of model
    modified_model = copy.deepcopy(model)
    layers = get_transformer_layers(modified_model)

    # Determine which layers to modify
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    # Apply projection to selected layers
    with torch.no_grad():
        for i in layer_indices:
            layer = layers[i]

            # Update attention output projection
            layer.self_attn.o_proj.weight = torch.nn.Parameter(
                project_out_rank1_subspace(layer.self_attn.o_proj.weight, direction, scale)
            )

            # Update MLP down projection
            layer.mlp.down_proj.weight = torch.nn.Parameter(
                project_out_rank1_subspace(layer.mlp.down_proj.weight, direction, scale)
            )

    return modified_model


def apply_lancet_biprojected_surgery(
    model: torch.nn.Module,
    target_dirs: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    anchor_dirs: (
        dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor | None
    ) = None,
    layer_indices: list[int] | None = None,
    scale: float = 1.0,
    preserve_row_norm: bool = True,
    layer_weights: dict[int, float] | None = None,
) -> torch.nn.Module:
    """
    Apply Lancet-style biprojected subspace surgery to selected layers.

    Each target direction is first biprojected against the anchor directions
    for that layer (removing components that overlap with the anchors, e.g.
    reasoning directions we want to preserve), then the resulting residual
    directions are projected out of ``self_attn.o_proj`` and ``mlp.down_proj``
    with row-norm preservation.

    Args:
        model: Transformer model instance (HF or sml-nano).
        target_dirs: Target directions per layer.
        anchor_dirs: Optional anchor directions per layer to protect.
        layer_indices: Indices of layers to modify (None means all layers).
        scale: Scaling factor for the projection.
        preserve_row_norm: Whether to preserve row L2 norms of the weights.
        layer_weights: Optional mapping of layer index -> multiplier applied
            on top of ``scale`` for that layer; layers absent from the dict
            use the base ``scale`` (multiplier 1.0).

    Returns:
        Modified (deep-copied) model with biprojected ablation applied.
    """
    modified_model = copy.deepcopy(model)
    layers = get_transformer_layers(modified_model)
    num_layers = len(layers)

    if layer_indices is None:
        layer_indices = list(range(num_layers))

    target_map = _resolve_per_layer_directions(target_dirs, num_layers)
    anchor_map = (
        _resolve_per_layer_directions(anchor_dirs, num_layers) if anchor_dirs is not None else {}
    )

    with torch.no_grad():
        for idx in layer_indices:
            layer = layers[idx]

            target_layer_dirs = target_map.get(idx, [])
            anchor_layer_dirs = anchor_map.get(idx, [])

            # Biproject each target direction against the anchors so the
            # ablated subspace does not overlap the protected one.
            if anchor_layer_dirs:
                resolved_dirs = [
                    biproject_direction(d, anchor_layer_dirs) for d in target_layer_dirs
                ]
            else:
                resolved_dirs = list(target_layer_dirs)

            # Per-layer scaling multiplier on top of the base ``scale``.
            layer_scale = scale
            if layer_weights is not None:
                layer_scale = scale * float(layer_weights.get(idx, 1.0))

            orig_attn_weight = layer.self_attn.o_proj.weight
            orig_mlp_weight = layer.mlp.down_proj.weight

            layer.self_attn.o_proj.weight = torch.nn.Parameter(
                project_out_subspace_norm_preserving(
                    orig_attn_weight,
                    resolved_dirs,
                    scale=layer_scale,
                    preserve_row_norm=preserve_row_norm,
                ),
                requires_grad=orig_attn_weight.requires_grad,
            )
            layer.mlp.down_proj.weight = torch.nn.Parameter(
                project_out_subspace_norm_preserving(
                    orig_mlp_weight,
                    resolved_dirs,
                    scale=layer_scale,
                    preserve_row_norm=preserve_row_norm,
                ),
                requires_grad=orig_mlp_weight.requires_grad,
            )

    return modified_model


@dataclass(frozen=True)
class LancetRecipe:
    """Serializable Lancet surgery recipe: biprojected bases plus per-layer weights."""

    model_id: str
    base_model: str
    extraction_mode: str
    d_model: int
    num_layers: int
    scale: float
    top_k_layers: int
    selected_layers: list[int]
    layer_weights: dict[int, float]
    basis_per_layer: dict[int, torch.Tensor]
    probes_sha256: str
    schema_version: str = "1.0"


def build_lancet_recipe(
    model: torch.nn.Module,
    target_dirs: dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor,
    anchor_dirs: (
        dict[int, list[torch.Tensor] | torch.Tensor] | list[torch.Tensor] | torch.Tensor | None
    ),
    anchor_input_ids: torch.Tensor,
    *,
    model_id: str,
    base_model: str,
    extraction_mode: str = "activation_contrastive",
    top_k_layers: int = 3,
    scale: float = 1.0,
    probes_sha256: str = "",
    preserve_row_norm: bool = True,
) -> LancetRecipe:
    """
    Build a serializable Lancet recipe from probe directions and anchors.

    Each layer's target directions are biprojected against its anchor
    directions, candidate layers are scored with ``compute_layer_scr_profile``,
    the ``top_k_layers`` best layers are selected (descending by SCR) with soft
    SCR-derived weights, and an orthonormal CPU float32 basis is stored for
    every selected layer.

    Args:
        model: Transformer model instance (HF or sml-nano).
        target_dirs: Target directions per layer.
        anchor_dirs: Optional anchor directions per layer to protect.
        anchor_input_ids: Token ids used to measure collateral damage in the
            SCR profile.
        model_id: Identifier of the model the recipe targets.
        base_model: Base (pre-surgery) model identifier.
        extraction_mode: How the target directions were extracted.
        top_k_layers: Number of highest-SCR layers to keep.
        scale: Base projection scale for the surgery.
        probes_sha256: Hash of the probe set used for extraction.
        preserve_row_norm: Passed through to the SCR profile projection.

    Returns:
        A frozen ``LancetRecipe`` ready for ``save_lancet_recipe`` /
        ``apply_lancet_recipe``.
    """
    layers = get_transformer_layers(model)
    num_layers = len(layers)

    target_map = _resolve_per_layer_directions(target_dirs, num_layers)
    anchor_map = (
        _resolve_per_layer_directions(anchor_dirs, num_layers) if anchor_dirs is not None else {}
    )

    # Biproject each layer's target directions against its anchor directions so
    # the stored bases do not overlap the protected subspaces.
    effective_dirs: dict[int, list[torch.Tensor]] = {}
    for idx, layer_dirs in target_map.items():
        anchor_layer_dirs = anchor_map.get(idx, [])
        if anchor_layer_dirs:
            effective_dirs[idx] = [biproject_direction(d, anchor_layer_dirs) for d in layer_dirs]
        else:
            effective_dirs[idx] = list(layer_dirs)

    candidate_layers = sorted(idx for idx, dirs in effective_dirs.items() if dirs)
    if not candidate_layers:
        raise ValueError("No target directions resolved for any layer")

    profile = compute_layer_scr_profile(
        model,
        {idx: effective_dirs[idx] for idx in candidate_layers},
        anchor_input_ids,
        candidate_layers=candidate_layers,
        scale=scale,
        preserve_row_norm=preserve_row_norm,
    )

    top_rows = profile[: max(1, int(top_k_layers))]
    selected_layers = [int(row["layer_idx"]) for row in top_rows]
    max_scr = max(float(row["scr"]) for row in profile)
    layer_weights = {
        int(row["layer_idx"]): round(min(1.0, max(0.1, float(row["scr"]) / max(max_scr, 1e-12))), 6)
        for row in top_rows
    }

    basis_per_layer: dict[int, torch.Tensor] = {}
    d_model: int | None = None
    for idx in selected_layers:
        basis_vectors = _orthonormalize_basis(effective_dirs[idx], device=torch.device("cpu"))
        if not basis_vectors:
            raise ValueError(f"Layer {idx} has no usable directions after biprojection")
        basis = torch.stack(basis_vectors, dim=0).contiguous()
        d_model = int(basis.shape[1])
        basis_per_layer[idx] = basis

    if d_model is None:
        raise ValueError("Failed to build a basis for any selected layer")

    return LancetRecipe(
        model_id=model_id,
        base_model=base_model,
        extraction_mode=extraction_mode,
        d_model=d_model,
        num_layers=num_layers,
        scale=float(scale),
        top_k_layers=int(top_k_layers),
        selected_layers=selected_layers,
        layer_weights=layer_weights,
        basis_per_layer=basis_per_layer,
        probes_sha256=probes_sha256,
    )


def save_lancet_recipe(recipe: LancetRecipe, output_dir: Path | str) -> Path:
    """
    Persist a recipe as ``lancet_recipe.json`` + ``lancet_basis.safetensors``.

    Args:
        recipe: Recipe produced by ``build_lancet_recipe``.
        output_dir: Directory to create; existing files are overwritten.

    Returns:
        The resolved output directory as a ``Path``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "schema_version": recipe.schema_version,
        "model_id": recipe.model_id,
        "base_model": recipe.base_model,
        "extraction_mode": recipe.extraction_mode,
        "d_model": int(recipe.d_model),
        "num_layers": int(recipe.num_layers),
        "scale": float(recipe.scale),
        "top_k_layers": int(recipe.top_k_layers),
        "selected_layers": [int(idx) for idx in recipe.selected_layers],
        "layer_weights": {str(idx): float(w) for idx, w in recipe.layer_weights.items()},
        "probes_sha256": recipe.probes_sha256,
    }
    metadata_path = output_path / "lancet_recipe.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    basis_tensors = {
        f"layer_{idx}": basis.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for idx, basis in recipe.basis_per_layer.items()
    }
    basis_path = output_path / "lancet_basis.safetensors"
    save_file(basis_tensors, str(basis_path))

    return output_path


def load_lancet_recipe(recipe_dir: Path | str) -> LancetRecipe:
    """
    Load and validate a recipe saved by ``save_lancet_recipe``.

    Args:
        recipe_dir: Directory holding ``lancet_recipe.json`` and
            ``lancet_basis.safetensors``.

    Returns:
        The reconstructed ``LancetRecipe`` with CPU float32 bases.

    Raises:
        ValueError: On missing files, unsupported schema, invalid layer
            indices or weights, malformed bases, non-finite entries, or bases
            whose rows are not unit L2 norm.
    """
    dir_path = Path(recipe_dir)
    metadata_path = dir_path / "lancet_recipe.json"
    basis_path = dir_path / "lancet_basis.safetensors"
    if not metadata_path.is_file():
        raise ValueError(f"Missing recipe metadata file: {metadata_path}")
    if not basis_path.is_file():
        raise ValueError(f"Missing recipe basis file: {basis_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported recipe schema_version: {metadata.get('schema_version')!r}")

    try:
        model_id = str(metadata["model_id"])
        base_model = str(metadata["base_model"])
        extraction_mode = str(metadata["extraction_mode"])
        d_model = int(metadata["d_model"])
        num_layers = int(metadata["num_layers"])
        scale = float(metadata["scale"])
        top_k_layers = int(metadata["top_k_layers"])
        probes_sha256 = str(metadata.get("probes_sha256", ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid lancet_recipe.json metadata: {exc}") from exc

    if d_model <= 0 or num_layers <= 0:
        raise ValueError(f"Invalid d_model/num_layers: {d_model}/{num_layers}")

    raw_layers = metadata.get("selected_layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("selected_layers must be a non-empty list of layer indices")
    selected_layers = [int(idx) for idx in raw_layers]
    if any(idx < 0 or idx >= num_layers for idx in selected_layers):
        raise ValueError(f"selected_layers {selected_layers} out of range [0, {num_layers})")

    raw_weights = metadata.get("layer_weights")
    if not isinstance(raw_weights, dict):
        raise ValueError("layer_weights must be an object with stringified layer keys")
    layer_weights: dict[int, float] = {}
    for key, value in raw_weights.items():
        weight = float(value)
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"layer_weights[{key!r}] must be in (0, 1]; got {weight}")
        layer_weights[int(key)] = weight
    missing = [idx for idx in selected_layers if idx not in layer_weights]
    if missing:
        raise ValueError(f"layer_weights missing selected layers: {missing}")

    basis_tensors = load_file(str(basis_path))
    basis_per_layer: dict[int, torch.Tensor] = {}
    for idx in selected_layers:
        key = f"layer_{idx}"
        if key not in basis_tensors:
            raise ValueError(f"Missing basis tensor {key!r} in lancet_basis.safetensors")
        basis = basis_tensors[key].to(device="cpu", dtype=torch.float32)
        if basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] != d_model:
            raise ValueError(
                f"Basis {key!r} must have shape (k, {d_model}) with k >= 1; "
                f"got {tuple(basis.shape)}"
            )
        if not bool(torch.isfinite(basis).all()):
            raise ValueError(f"Basis {key!r} contains non-finite values")
        row_norm_diff = (basis.norm(p=2, dim=-1) - 1.0).abs()
        if float(row_norm_diff.max()) >= 1e-5:
            raise ValueError(f"Basis {key!r} rows are not unit L2 norm")
        basis_per_layer[idx] = basis.contiguous()

    return LancetRecipe(
        model_id=model_id,
        base_model=base_model,
        extraction_mode=extraction_mode,
        d_model=d_model,
        num_layers=num_layers,
        scale=scale,
        top_k_layers=top_k_layers,
        selected_layers=selected_layers,
        layer_weights=layer_weights,
        basis_per_layer=basis_per_layer,
        probes_sha256=probes_sha256,
    )


def apply_lancet_recipe(
    model: torch.nn.Module, recipe: LancetRecipe, *, in_place: bool = False
) -> torch.nn.Module:
    """
    Apply a saved Lancet recipe to a model.

    Each selected layer's ``self_attn.o_proj`` and ``mlp.down_proj`` weights
    get the stored orthonormal basis projected out with row-norm preservation,
    scaled by ``recipe.scale * recipe.layer_weights.get(idx, 1.0)``.

    Args:
        model: Transformer model instance (HF or sml-nano).
        recipe: Recipe produced by ``build_lancet_recipe`` / ``load_lancet_recipe``.
        in_place: If True, modify ``model`` directly instead of deep-copying.

    Returns:
        The modified model (a deep copy unless ``in_place=True``).
    """
    target_model = model if in_place else copy.deepcopy(model)
    layers = get_transformer_layers(target_model)

    with torch.no_grad():
        for idx in recipe.selected_layers:
            layer = layers[idx]
            basis = recipe.basis_per_layer[idx]
            layer_scale = recipe.scale * float(recipe.layer_weights.get(idx, 1.0))

            orig_attn_weight = layer.self_attn.o_proj.weight
            orig_mlp_weight = layer.mlp.down_proj.weight
            layer.self_attn.o_proj.weight = torch.nn.Parameter(
                project_out_subspace_norm_preserving(
                    orig_attn_weight, basis, scale=layer_scale, preserve_row_norm=True
                ),
                requires_grad=orig_attn_weight.requires_grad,
            )
            layer.mlp.down_proj.weight = torch.nn.Parameter(
                project_out_subspace_norm_preserving(
                    orig_mlp_weight, basis, scale=layer_scale, preserve_row_norm=True
                ),
                requires_grad=orig_mlp_weight.requires_grad,
            )

    return target_model
