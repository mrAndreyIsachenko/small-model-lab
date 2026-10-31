"""Transformer model layers, RoPE, GQA attention, SwiGLU, and GPT-2 reference adapter."""

from sml.model.attention import (
    GroupedQueryAttention,
    KVCache,
    build_causal_doc_mask,
    compute_doc_ids_from_eos,
    repeat_kv,
)
from sml.model.gpt2_adapter import load_gpt2_weights_into_sml
from sml.model.mlp import GELUMLP, SwiGLUMLP, build_mlp, gpt2_new_gelu
from sml.model.norms import RMSNorm, build_norm
from sml.model.rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from sml.model.transformer import Transformer, TransformerBlock, TransformerOutput

__all__ = [
    "GELUMLP",
    "GroupedQueryAttention",
    "KVCache",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUMLP",
    "Transformer",
    "TransformerBlock",
    "TransformerOutput",
    "apply_rotary_pos_emb",
    "build_causal_doc_mask",
    "build_mlp",
    "build_norm",
    "compute_doc_ids_from_eos",
    "gpt2_new_gelu",
    "load_gpt2_weights_into_sml",
    "repeat_kv",
    "rotate_half",
]
