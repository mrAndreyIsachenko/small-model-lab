"""Isolated weight converter from Hugging Face GPT2LMHeadModel into SML Transformer (`H1`)."""

from __future__ import annotations

import torch

from sml.model.transformer import Transformer


@torch.no_grad()
def load_gpt2_weights_into_sml(
    sml_model: Transformer,
    hf_state_dict: dict[str, torch.Tensor],
) -> Transformer:
    """Copy OpenAI / Hugging Face GPT-2 state_dict into an SML Transformer in gpt2-compat mode.

    Hugging Face GPT-2 uses `Conv1D` modules whose weight matrices have shape
    `(in_features, out_features)` (transposed relative to `nn.Linear`) and fuses
    Q, K, V into a single `c_attn` projection of width `3 * d_model`.
    """
    cfg = sml_model.config
    if cfg.pos_encoding != "learned" or cfg.norm_type != "layernorm" or cfg.mlp_type != "gelu":
        raise ValueError(
            "sml_model must be configured with pos_encoding='learned', "
            "norm_type='layernorm', and mlp_type='gelu' for GPT-2 weight loading."
        )

    def _get(key: str) -> torch.Tensor:
        if key in hf_state_dict:
            return hf_state_dict[key]
        prefixed = f"transformer.{key}"
        if prefixed in hf_state_dict:
            return hf_state_dict[prefixed]
        raise KeyError(f"Missing GPT-2 weight key: {key}")

    sml_model.embed_tokens.weight.copy_(_get("wte.weight"))
    assert sml_model.pos_embed is not None
    sml_model.pos_embed.weight.copy_(_get("wpe.weight"))

    d = cfg.d_model
    for i, block in enumerate(sml_model.layers):
        block.input_layernorm.weight.copy_(_get(f"h.{i}.ln_1.weight"))
        block.input_layernorm.bias.copy_(_get(f"h.{i}.ln_1.bias"))

        c_attn_w = _get(f"h.{i}.attn.c_attn.weight")
        c_attn_b = _get(f"h.{i}.attn.c_attn.bias")
        q_w, k_w, v_w = c_attn_w.split(d, dim=1)
        q_b, k_b, v_b = c_attn_b.split(d, dim=0)

        block.self_attn.q_proj.weight.copy_(q_w.t().contiguous())
        block.self_attn.q_proj.bias.copy_(q_b)
        block.self_attn.k_proj.weight.copy_(k_w.t().contiguous())
        block.self_attn.k_proj.bias.copy_(k_b)
        block.self_attn.v_proj.weight.copy_(v_w.t().contiguous())
        block.self_attn.v_proj.bias.copy_(v_b)

        block.self_attn.o_proj.weight.copy_(_get(f"h.{i}.attn.c_proj.weight").t().contiguous())
        block.self_attn.o_proj.bias.copy_(_get(f"h.{i}.attn.c_proj.bias"))

        block.post_attention_layernorm.weight.copy_(_get(f"h.{i}.ln_2.weight"))
        block.post_attention_layernorm.bias.copy_(_get(f"h.{i}.ln_2.bias"))

        block.mlp.up_proj.weight.copy_(_get(f"h.{i}.mlp.c_fc.weight").t().contiguous())
        block.mlp.up_proj.bias.copy_(_get(f"h.{i}.mlp.c_fc.bias"))
        block.mlp.down_proj.weight.copy_(_get(f"h.{i}.mlp.c_proj.weight").t().contiguous())
        block.mlp.down_proj.bias.copy_(_get(f"h.{i}.mlp.c_proj.bias"))

    sml_model.norm.weight.copy_(_get("ln_f.weight"))
    sml_model.norm.bias.copy_(_get("ln_f.bias"))

    if not cfg.tie_embeddings:
        sml_model.lm_head.weight.copy_(_get("wte.weight"))

    return sml_model
