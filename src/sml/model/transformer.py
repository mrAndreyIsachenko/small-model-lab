"""Decoder-only Transformer model supporting modern Llama/Qwen blocks and GPT-2 parity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sml.config import ModelConfig
from sml.model.attention import AttnBackend, GroupedQueryAttention, KVCache
from sml.model.mlp import build_mlp
from sml.model.norms import build_norm
from sml.model.rope import RotaryEmbedding


@dataclass
class TransformerOutput:
    """Structured forward output of `Transformer`."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    kv_cache: KVCache | None = None


class TransformerBlock(nn.Module):
    """Pre-norm Transformer decoder block with Llama/Qwen-compatible submodule naming."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = build_norm(
            d_model=config.d_model, norm_type=config.norm_type, eps=config.norm_eps
        )
        self.self_attn = GroupedQueryAttention(config=config, layer_idx=layer_idx)
        self.post_attention_layernorm = build_norm(
            d_model=config.d_model, norm_type=config.norm_type, eps=config.norm_eps
        )
        self.mlp = build_mlp(config)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: RotaryEmbedding | None = None,
        kv_cache: KVCache | None = None,
        doc_ids: torch.Tensor | None = None,
        attn_backend: AttnBackend = "sdpa",
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x),
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
            doc_ids=doc_ids,
            backend=attn_backend,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Transformer(nn.Module):
    """Decoder-only causal language model implemented from scratch in PyTorch."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        if config.pos_encoding == "learned":
            self.pos_embed: nn.Embedding | None = nn.Embedding(config.max_seq_len, config.d_model)
            self.rotary_emb: RotaryEmbedding | None = None
        else:
            self.pos_embed = None
            self.rotary_emb = RotaryEmbedding(
                head_dim=config.head_dim,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta,
            )

        self.layers = nn.ModuleList(
            [TransformerBlock(config=config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm = build_norm(
            d_model=config.d_model, norm_type=config.norm_type, eps=config.norm_eps
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        residual_std = std / math.sqrt(2.0 * self.config.n_layers)

        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed.weight, mean=0.0, std=std)
        if not self.config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

        for block in self.layers:
            assert isinstance(block, TransformerBlock)
            for proj in (
                block.self_attn.q_proj,
                block.self_attn.k_proj,
                block.self_attn.v_proj,
            ):
                nn.init.normal_(proj.weight, mean=0.0, std=std)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

            nn.init.normal_(block.self_attn.o_proj.weight, mean=0.0, std=residual_std)
            if block.self_attn.o_proj.bias is not None:
                nn.init.zeros_(block.self_attn.o_proj.bias)

            for name, param in block.mlp.named_parameters():
                if name.endswith("bias"):
                    nn.init.zeros_(param)
                elif "down_proj" in name:
                    nn.init.normal_(param, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(param, mean=0.0, std=std)

    def num_parameters(self, non_embedding_only: bool = False) -> int:
        """Count actual trainable parameters in the instantiated PyTorch module."""
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding_only:
            return total
        emb = self.embed_tokens.weight.numel()
        if not self.config.tie_embeddings:
            emb += self.lm_head.weight.numel()
        return total - emb

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        attn_backend: AttnBackend = "sdpa",
        ignore_index: int = -100,
    ) -> TransformerOutput:
        _bsz, seq_len = input_ids.shape
        offset = 0 if kv_cache is None else kv_cache.seq_len

        h = self.embed_tokens(input_ids)
        if self.pos_embed is not None:
            if offset + seq_len > self.config.max_seq_len:
                raise ValueError(
                    f"Sequence length {offset + seq_len} exceeds learned max_seq_len "
                    f"{self.config.max_seq_len}"
                )
            pos_ids = torch.arange(offset, offset + seq_len, device=input_ids.device)
            h = h + self.pos_embed(pos_ids).unsqueeze(0)

        for layer in self.layers:
            h = layer(
                h,
                rotary_emb=self.rotary_emb,
                kv_cache=kv_cache,
                doc_ids=doc_ids,
                attn_backend=attn_backend,
            )

        h = self.norm(h)
        logits = self.lm_head(h)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )

        return TransformerOutput(logits=logits, loss=loss, kv_cache=kv_cache)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        use_kv_cache: bool = True,
        eos_token_id: int | None = None,
        attn_backend: AttnBackend = "sdpa",
    ) -> torch.Tensor:
        """Autoregressively generate tokens with optional KV-cache."""
        generated = input_ids.clone()
        cache = KVCache.empty(self.config.n_layers) if use_kv_cache else None

        for step in range(max_new_tokens):
            if use_kv_cache:
                step_input = generated if step == 0 else generated[:, -1:]
            else:
                step_input = generated[:, -self.config.max_seq_len :]

            out = self.forward(
                input_ids=step_input,
                kv_cache=cache,
                attn_backend=attn_backend,
            )
            next_logits = out.logits[:, -1, :].float()

            if temperature <= 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                    scaled[scaled < v[:, [-1]]] = float("-inf")
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

        return generated
