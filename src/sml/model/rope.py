"""Rotary Position Embeddings (RoPE) with relative-distance dot-product invariance."""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting in half: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply pre-sliced (T, head_dim) or (B, 1, T, head_dim) rotary embeddings to Q and K.

    Args:
        q: Query tensor of shape (B, n_heads, T, head_dim)
        k: Key tensor of shape (B, n_kv_heads, T, head_dim)
        cos: Cosine frequencies broadcastable to (B, 1, T, head_dim)
        sin: Sine frequencies broadcastable to (B, 1, T, head_dim)
    """
    q_fp32 = q.float()
    k_fp32 = k.float()
    q_rot = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_rot = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_rot.to(dtype=q.dtype), k_rot.to(dtype=k.dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed Rotary Position Embedding cache (non-persistent buffers)."""

    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        cos, sin = self._build_cache(max_seq_len=max_seq_len, device=torch.device("cpu"))
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build_cache(
        self, max_seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ar = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (ar / self.head_dim))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        offset: int = 0,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) broadcastable to (B, 1, seq_len, head_dim)."""
        needed_len = offset + seq_len
        if position_ids is not None:
            needed_len = max(needed_len, int(position_ids.max().item()) + 1)

        if needed_len > self.cos_cached.shape[0] or self.cos_cached.device != device:
            new_max = max(needed_len, self.max_seq_len)
            cos, sin = self._build_cache(max_seq_len=new_max, device=device)
            self.cos_cached = cos
            self.sin_cached = sin

        if position_ids is not None:
            cos_pos = self.cos_cached[position_ids].unsqueeze(1)
            sin_pos = self.sin_cached[position_ids].unsqueeze(1)
            return cos_pos, sin_pos

        cos_slice = self.cos_cached[offset : offset + seq_len].view(1, 1, seq_len, self.head_dim)
        sin_slice = self.sin_cached[offset : offset + seq_len].view(1, 1, seq_len, self.head_dim)
        return cos_slice, sin_slice
