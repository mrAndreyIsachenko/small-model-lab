"""Normalization layers (RMSNorm and LayerNorm factory)."""

from __future__ import annotations

import torch
import torch.nn as nn

from sml.config import NormType


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization computed in fp32 for stability."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rrms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = (x_fp32 * rrms).to(dtype=x.dtype)
        return normed * self.weight


def build_norm(d_model: int, norm_type: NormType = "rmsnorm", eps: float = 1e-5) -> nn.Module:
    """Instantiate RMSNorm (modern Llama/Qwen) or LayerNorm with bias (GPT-2 compat)."""
    if norm_type == "rmsnorm":
        return RMSNorm(d_model=d_model, eps=eps)
    if norm_type == "layernorm":
        return nn.LayerNorm(normalized_shape=d_model, eps=eps, elementwise_affine=True)
    raise ValueError(f"Unsupported norm_type: {norm_type}")
