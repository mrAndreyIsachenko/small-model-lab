"""From-scratch Low-Rank Adaptation (`LoRA`) and weight merging in pure PyTorch."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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

    def merge(self) -> nn.Linear:
        """Fold `(alpha / r) * B @ A` into `base_layer.weight` for zero-overhead inference."""
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
            device=self.base_layer.weight.device,
            dtype=self.base_layer.weight.dtype,
        )
        with torch.no_grad():
            delta = (self.lora_b @ self.lora_a) * self.scaling
            merged.weight.copy_(self.base_layer.weight + delta)
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


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Replace all `LoRALinear` layers in `model` with merged `nn.Linear` layers."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            _set_submodule(model, name, module.merge())
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
