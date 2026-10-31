"""Grouped-Query Attention (GQA) with KV-cache, intra-document masking, and SDPA/naive parity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from sml.config import ModelConfig
from sml.model.rope import RotaryEmbedding, apply_rotary_pos_emb

AttnBackend = Literal["sdpa", "naive"]


@dataclass
class KVCache:
    """Per-layer Key/Value cache for autoregressive decoding."""

    k_cache: list[torch.Tensor | None] = field(default_factory=list)
    v_cache: list[torch.Tensor | None] = field(default_factory=list)

    @classmethod
    def empty(cls, n_layers: int) -> KVCache:
        return cls(k_cache=[None] * n_layers, v_cache=[None] * n_layers)

    @property
    def seq_len(self) -> int:
        first = self.k_cache[0] if self.k_cache else None
        return 0 if first is None else int(first.shape[2])

    def layer_seq_len(self, layer_idx: int) -> int:
        """Return cached sequence length for `layer_idx` prior to updating that layer."""
        cached = self.k_cache[layer_idx] if self.k_cache else None
        return 0 if cached is None else int(cached.shape[2])

    def update(
        self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append (B, n_kv_heads, T_new, head_dim) to cache and return full (k, v)."""
        k_prev = self.k_cache[layer_idx]
        v_prev = self.v_cache[layer_idx]
        if k_prev is None or v_prev is None:
            k_full = k_new
            v_full = v_new
        else:
            k_full = torch.cat([k_prev, k_new], dim=2)
            v_full = torch.cat([v_prev, v_new], dim=2)
        self.k_cache[layer_idx] = k_full
        self.v_cache[layer_idx] = v_full
        return k_full, v_full


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads from (B, n_kv_heads, T, head_dim) to (B, n_heads, T, head_dim)."""
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
    )


def compute_doc_ids_from_eos(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Assign monotonically increasing document IDs within packed sequences separated by <|eos|>.

    The <|eos|> token belongs to the document it terminates; the token immediately
    following <|eos|> starts a new document ID.
    """
    is_prev_eos = F.pad(input_ids[:, :-1] == eos_token_id, (1, 0), value=False)
    return is_prev_eos.to(dtype=torch.int64).cumsum(dim=1)


def build_causal_doc_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    offset: int = 0,
    doc_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct boolean mask (True = allowed to attend) of shape (B or 1, 1, q_len, kv_len)."""
    q_idx = torch.arange(q_len, device=device).unsqueeze(1) + offset
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    causal = kv_idx <= q_idx

    if doc_ids is None:
        return causal.view(1, 1, q_len, kv_len)

    if doc_ids.shape[1] < kv_len:
        raise ValueError(f"doc_ids length ({doc_ids.shape[1]}) < kv_len ({kv_len})")

    q_docs = doc_ids[:, offset : offset + q_len].unsqueeze(2)
    kv_docs = doc_ids[:, :kv_len].unsqueeze(1)
    same_doc = q_docs == kv_docs
    return (causal.unsqueeze(0) & same_doc).unsqueeze(1)


class GroupedQueryAttention(nn.Module):
    """Multi-Head / Grouped-Query Causal Self-Attention with Llama/Qwen projection naming."""

    def __init__(self, config: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.kv_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.record_attn_weights: bool = False
        self.last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding | None = None,
        kv_cache: KVCache | None = None,
        doc_ids: torch.Tensor | None = None,
        backend: AttnBackend = "sdpa",
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        offset = 0 if kv_cache is None else kv_cache.layer_seq_len(self.layer_idx)

        q = self.q_proj(x).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if rotary_emb is not None:
            cos, sin = rotary_emb(seq_len=q_len, device=x.device, offset=offset)
            q, k = apply_rotary_pos_emb(q, k, cos=cos, sin=sin)

        if kv_cache is not None:
            k, v = kv_cache.update(self.layer_idx, k, v)

        kv_len = k.shape[2]
        k_exp = repeat_kv(k, self.n_rep)
        v_exp = repeat_kv(v, self.n_rep)

        if backend == "naive" or self.record_attn_weights:
            attn_out = self._naive_attention(
                q=q,
                k=k_exp,
                v=v_exp,
                q_len=q_len,
                kv_len=kv_len,
                offset=offset,
                doc_ids=doc_ids,
            )
        elif backend == "sdpa":
            if doc_ids is None and offset == 0 and q_len == kv_len:
                attn_out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, attn_mask=None, dropout_p=0.0, is_causal=True
                )
            else:
                allow_mask = build_causal_doc_mask(
                    q_len=q_len,
                    kv_len=kv_len,
                    device=x.device,
                    offset=offset,
                    doc_ids=doc_ids,
                )
                attn_out = F.scaled_dot_product_attention(
                    q, k_exp, v_exp, attn_mask=allow_mask, dropout_p=0.0, is_causal=False
                )
        else:
            raise ValueError(f"Unsupported attention backend: {backend}")

        merged = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        return self.o_proj(merged)

    def _naive_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_len: int,
        kv_len: int,
        offset: int,
        doc_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        allow_mask = build_causal_doc_mask(
            q_len=q_len,
            kv_len=kv_len,
            device=q.device,
            offset=offset,
            doc_ids=doc_ids,
        )
        scores = scores.masked_fill(~allow_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        if self.record_attn_weights:
            self.last_attn_weights = weights.detach()
        return torch.matmul(weights.to(dtype=v.dtype), v)
