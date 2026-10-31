"""Feed-forward MLP blocks: SwiGLU (Llama/Qwen) and NewGELU (GPT-2 compat)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sml.config import ModelConfig


def gpt2_new_gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact OpenAI GPT-2 / Hugging Face NewGELUActivation formula."""
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


class SwiGLUMLP(nn.Module):
    """Llama/Qwen-style SwiGLU feed-forward network (gate_proj, up_proj, down_proj)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GELUMLP(nn.Module):
    """GPT-2 compatible 2-layer MLP using NewGELU activation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(gpt2_new_gelu(self.up_proj(x)))


def build_mlp(config: ModelConfig) -> nn.Module:
    if config.mlp_type == "swiglu":
        return SwiGLUMLP(config)
    if config.mlp_type == "gelu":
        return GELUMLP(config)
    raise ValueError(f"Unsupported mlp_type: {config.mlp_type}")
