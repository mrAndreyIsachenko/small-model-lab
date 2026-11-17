"""From-scratch Low-Rank Adaptation (`LoRA`) and weight merging in pure PyTorch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _project_delta_orthogonal(
    delta: torch.Tensor,
    forbidden_dirs: list[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Project ``delta`` onto the orthogonal complement of ``forbidden_dirs``.

    The forbidden directions are orthonormalized in float32 with modified
    Gram-Schmidt and stacked into ``U_perp`` of shape ``(d_out, k)`` (where
    ``d_out == delta.shape[0]``); their component is then removed from
    ``delta``: ``delta <- delta - U_perp @ (U_perp.T @ delta)``.

    An empty ``forbidden_dirs`` returns ``delta`` unchanged; a direction whose
    flattened dimension mismatches ``delta.shape[0]`` raises ``ValueError``.
    """
    if isinstance(forbidden_dirs, torch.Tensor):
        rows = [forbidden_dirs] if forbidden_dirs.ndim <= 1 else [row for row in forbidden_dirs]
    else:
        rows = []
        for vec in forbidden_dirs:
            if vec.ndim > 1:
                rows.extend(row for row in vec)
            else:
                rows.append(vec)
    if not rows:
        return delta

    d_out = delta.shape[0]
    vectors: list[torch.Tensor] = []
    for vec in rows:
        flat = vec.reshape(-1).float()
        if flat.shape[0] != d_out:
            raise ValueError(
                f"forbidden_dirs dimension mismatch: expected {d_out}, got {flat.shape[0]}"
            )
        vectors.append(flat)

    # Modified Gram-Schmidt orthonormalization (float32); skip near-zero vectors.
    basis: list[torch.Tensor] = []
    for vec in vectors:
        v = vec.clone()
        for b in basis:
            v = v - (b @ v) * b
        norm = v.norm()
        if float(norm) > 1e-8:
            basis.append(v / norm)
    if not basis:
        return delta

    u_perp = torch.stack(basis, dim=1)  # (d_out, k)
    delta_f = delta.float()
    projected = delta_f - u_perp @ (u_perp.t() @ delta_f)
    return projected.to(delta.dtype)


def _restore_row_norm(merged: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale each row of ``merged`` to match the row norms of ``reference``."""
    ref_norms = reference.float().norm(dim=-1, keepdim=True)
    new_norms = merged.float().norm(dim=-1, keepdim=True)
    return (merged.float() * (ref_norms / (new_norms + 1e-8))).to(merged.dtype)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around `nn.Linear`: $W' = W_0 + (\\alpha / r) B A$."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        dev = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_a = nn.Parameter(
            torch.empty(rank, base_layer.in_features, device=dev, dtype=dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, device=dev, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))

        self.scaling = alpha / max(1, rank)
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scaling
        return base_out + lora_out

    def merge(
        self,
        forbidden_dirs: list[torch.Tensor] | torch.Tensor | None = None,
        preserve_row_norm: bool = True,
    ) -> nn.Linear:
        """Fold `(alpha / r) * B @ A` into `base_layer.weight` for zero-overhead inference.

        When ``forbidden_dirs`` is given, the LoRA delta is first projected onto
        the orthogonal complement of those directions; with ``preserve_row_norm``
        the merged rows are then rescaled to the base weight's row norms.
        """
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        with torch.no_grad():
            delta = (self.lora_b @ self.lora_a) * self.scaling
            if forbidden_dirs is not None:
                delta = _project_delta_orthogonal(delta, forbidden_dirs)
            merged_weight = self.base_layer.weight + delta
            if forbidden_dirs is not None and preserve_row_norm:
                merged_weight = _restore_row_norm(merged_weight, self.base_layer.weight)
            merged.weight.copy_(merged_weight)
            merged.weight.requires_grad = False
            if self.base_layer.bias is not None and merged.bias is not None:
                merged.bias.copy_(self.base_layer.bias)
                merged.bias.requires_grad = False
        return merged


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_transformer(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
) -> nn.Module:
    """Freeze base parameters and wrap target `nn.Linear` projections with `LoRALinear`."""
    for param in model.parameters():
        param.requires_grad = False

    for name, module in list(model.named_modules()):
        leaf_name = name.split(".")[-1]
        if isinstance(module, nn.Linear) and leaf_name in target_modules:
            _set_submodule(
                model,
                name,
                LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
            )
    return model


def _forbidden_dirs_match(
    dirs: list[torch.Tensor] | torch.Tensor,
    d_out: int,
) -> bool:
    """True when every forbidden direction can be applied to a ``(d_out, ...)`` delta."""
    if isinstance(dirs, torch.Tensor):
        return dirs.numel() > 0 and dirs.shape[-1] == d_out
    return bool(dirs) and all(vec.shape[-1] == d_out for vec in dirs)


def _resolve_forbidden_dirs(
    forbidden_dirs: (
        list[torch.Tensor] | torch.Tensor | dict[int, list[torch.Tensor] | torch.Tensor] | None
    ),
    module_name: str,
    d_out: int,
) -> list[torch.Tensor] | torch.Tensor | None:
    """Pick the forbidden dirs applicable to ``module_name``, or ``None``.

    Flat lists/tensors apply whenever their last dimension equals ``d_out``;
    dicts are keyed by layer index extracted from the module name (e.g.
    ``'layers.3.self_attn.o_proj'`` -> ``3``). Inapplicable or dimension-
    mismatched entries resolve to ``None`` (no projection).
    """
    dirs: list[torch.Tensor] | torch.Tensor | None
    if forbidden_dirs is None:
        return None
    if isinstance(forbidden_dirs, dict):
        dirs = None
        for part in module_name.split("."):
            if part.isdigit() and int(part) in forbidden_dirs:
                dirs = forbidden_dirs[int(part)]
                break
        if dirs is None:
            return None
    else:
        dirs = forbidden_dirs
    if _forbidden_dirs_match(dirs, d_out):
        return dirs
    return None


def merge_lora_weights(
    model: nn.Module,
    forbidden_dirs: (
        list[torch.Tensor] | torch.Tensor | dict[int, list[torch.Tensor] | torch.Tensor] | None
    ) = None,
    preserve_row_norm: bool = True,
) -> nn.Module:
    """Replace all `LoRALinear` layers in `model` with merged `nn.Linear` layers.

    When ``forbidden_dirs`` applies to a layer, its delta is projected onto the
    orthogonal complement of those directions before merging, and (with
    ``preserve_row_norm``) the merged rows are rescaled to the base weight's
    row norms.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            dirs = _resolve_forbidden_dirs(forbidden_dirs, name, module.base_layer.weight.shape[0])
            _set_submodule(model, name, module.merge(dirs, preserve_row_norm=preserve_row_norm))
    return model


def merge_lora_weights_in_place(
    model: nn.Module,
    forbidden_dirs: (
        list[torch.Tensor] | torch.Tensor | dict[int, list[torch.Tensor] | torch.Tensor] | None
    ) = None,
    preserve_row_norm: bool = True,
) -> nn.Module:
    """Fold each LoRA delta into its base weight in-place and unwrap the adapter.

    Unlike `merge_lora_weights` (which rebuilds fresh `nn.Linear` layers), this
    keeps the original `base_layer` module and adds `(alpha / r) * B @ A`
    directly to `base_layer.weight`, so the merged layer preserves identity,
    device, and dtype. Every `LoRALinear` wrapper is replaced by its
    `base_layer`, leaving zero LoRA parameters behind.

    When ``forbidden_dirs`` applies to a layer, its delta is projected onto the
    orthogonal complement of those directions before being added, and (with
    ``preserve_row_norm``) the merged rows are rescaled to the pre-merge base
    weight's row norms.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            dirs = _resolve_forbidden_dirs(forbidden_dirs, name, module.base_layer.weight.shape[0])
            with torch.no_grad():
                delta = (module.lora_b @ module.lora_a) * module.scaling
                if dirs is not None:
                    delta = _project_delta_orthogonal(delta, dirs)
                    merged_weight = module.base_layer.weight + delta
                    if preserve_row_norm:
                        merged_weight = _restore_row_norm(merged_weight, module.base_layer.weight)
                    module.base_layer.weight.copy_(merged_weight)
                else:
                    module.base_layer.weight.add_(delta)
                module.base_layer.weight.requires_grad = False
                if module.base_layer.bias is not None:
                    module.base_layer.bias.requires_grad = False
            _set_submodule(model, name, module.base_layer)
    return model


def lora_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    """Return trainable vs. total parameter counts and trainable percentage."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (100.0 * trainable_params / total_params) if total_params > 0 else 0.0
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": round(trainable_pct, 4),
    }
