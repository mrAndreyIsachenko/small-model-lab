"""Unit tests for M1: Transformer architecture, H1 GPT-2 parity, RoPE, GQA, KV-cache, and hooks."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from sml.config import ModelConfig, load_model_config
from sml.hooks import ActivationHookManager
from sml.kernels import (
    compute_rmsnorm_hbm_bytes,
    compute_swiglu_hbm_bytes,
    fused_rmsnorm,
    fused_swiglu,
    is_triton_available,
    ref_rmsnorm,
    ref_swiglu,
    simulate_tiled_rmsnorm_fwd_bwd,
    simulate_tiled_swiglu_fwd_bwd,
)
from sml.model import (
    KVCache,
    RotaryEmbedding,
    Transformer,
    apply_rotary_pos_emb,
    compute_doc_ids_from_eos,
    load_gpt2_weights_into_sml,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_instantiated_parameter_counts_match_closed_form() -> None:
    """Verify instantiated PyTorch parameter counts match closed-form equations exactly."""
    for yaml_name in ("sml_nano.yaml", "sml_xs.yaml", "sml_100m.yaml"):
        cfg = load_model_config(REPO_ROOT / "configs/model" / yaml_name)
        model = Transformer(cfg)
        assert model.num_parameters(non_embedding_only=False) == cfg.total_params
        assert model.num_parameters(non_embedding_only=True) == cfg.non_embedding_params


def test_rope_relative_position_invariance() -> None:
    """Verify <RoPE(q, m), RoPE(k, n)> depends only on relative offset (m - n)."""
    torch.manual_seed(42)
    head_dim = 64
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=512, theta=10000.0)

    q_vec = torch.randn(1, 1, 1, head_dim, dtype=torch.float32)
    k_vec = torch.randn(1, 1, 1, head_dim, dtype=torch.float32)

    m, n = 17, 5
    delta = 123

    cos_m, sin_m = rope(seq_len=1, device=q_vec.device, offset=m)
    cos_n, sin_n = rope(seq_len=1, device=k_vec.device, offset=n)
    q_m, _ = apply_rotary_pos_emb(q_vec, q_vec, cos_m, sin_m)
    _, k_n = apply_rotary_pos_emb(k_vec, k_vec, cos_n, sin_n)
    dot_base = (q_m * k_n).sum()

    cos_md, sin_md = rope(seq_len=1, device=q_vec.device, offset=m + delta)
    cos_nd, sin_nd = rope(seq_len=1, device=k_vec.device, offset=n + delta)
    q_md, _ = apply_rotary_pos_emb(q_vec, q_vec, cos_md, sin_md)
    _, k_nd = apply_rotary_pos_emb(k_vec, k_vec, cos_nd, sin_nd)
    dot_shifted = (q_md * k_nd).sum()

    assert torch.allclose(dot_base, dot_shifted, atol=1e-5)


def test_causality_and_intra_document_mask_isolation() -> None:
    """Verify H1: no future leakage and zero cross-document leakage across <|eos|> boundaries."""
    torch.manual_seed(1337)
    cfg = load_model_config(REPO_ROOT / "configs/model/sml_nano.yaml")
    model = Transformer(cfg).eval()

    # 1. Causal future isolation: mutating tokens > t must not affect logits <= t
    seq_a = torch.randint(0, cfg.vocab_size, (2, 16))
    seq_b = seq_a.clone()
    seq_b[:, 8:] = torch.randint(0, cfg.vocab_size, (2, 8))

    out_a = model(seq_a, attn_backend="naive").logits
    out_b = model(seq_b, attn_backend="naive").logits
    assert torch.max(torch.abs(out_a[:, :8, :] - out_b[:, :8, :])).item() == 0.0

    # 2. Intra-document isolation: packed [doc0, EOS, doc1] must block doc0 -> doc1 leakage
    eos_id = 2
    packed_1 = torch.tensor([[10, 11, 12, eos_id, 20, 21, 22, 23]], dtype=torch.long)
    packed_2 = torch.tensor([[99, 98, 97, eos_id, 20, 21, 22, 23]], dtype=torch.long)

    doc_ids = compute_doc_ids_from_eos(packed_1, eos_token_id=eos_id)
    assert doc_ids.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]

    for backend in ("naive", "sdpa"):
        logits_1 = model(packed_1, doc_ids=doc_ids, attn_backend=backend).logits
        logits_2 = model(packed_2, doc_ids=doc_ids, attn_backend=backend).logits
        # Positions 4..7 belong to doc 1 and cannot attend to positions 0..3 (doc 0)
        doc1_diff = torch.max(torch.abs(logits_1[:, 4:, :] - logits_2[:, 4:, :])).item()
        assert doc1_diff < 1e-6


def test_naive_vs_sdpa_and_kv_cache_equivalence() -> None:
    """Verify H1: naive attention == SDPA and KV-cache decoding == full-sequence forward."""
    torch.manual_seed(7)
    cfg = load_model_config(REPO_ROOT / "configs/model/sml_nano.yaml")
    model = Transformer(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
    doc_ids = compute_doc_ids_from_eos(input_ids, eos_token_id=1)

    logits_naive = model(input_ids, doc_ids=doc_ids, attn_backend="naive").logits
    logits_sdpa = model(input_ids, doc_ids=doc_ids, attn_backend="sdpa").logits
    assert torch.max(torch.abs(logits_naive - logits_sdpa)).item() < 1e-5

    # Step-by-step KV-cache vs full forward
    full_logits = model(input_ids, attn_backend="sdpa").logits
    cache = KVCache.empty(cfg.n_layers)
    prefill = model(input_ids[:, :6], kv_cache=cache, attn_backend="sdpa").logits
    step_logits = [prefill]
    for pos in range(6, 12):
        out_step = model(input_ids[:, pos : pos + 1], kv_cache=cache, attn_backend="sdpa").logits
        step_logits.append(out_step)
    cached_logits = torch.cat(step_logits, dim=1)
    assert torch.max(torch.abs(full_logits - cached_logits)).item() < 1e-5

    # Greedy generation parity with and without KV-cache
    prompt = input_ids[:, :5]
    gen_cached = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_kv_cache=True)
    gen_uncached = model.generate(prompt, max_new_tokens=6, temperature=0.0, use_kv_cache=False)
    assert torch.equal(gen_cached, gen_uncached)


def test_gpt2_compat_logit_parity_with_huggingface() -> None:
    """Verify H1: SML in gpt2-compat mode reproduces Hugging Face GPT2LMHeadModel logits < 1e-4."""
    torch.manual_seed(99)
    hf_cfg = GPT2Config(
        vocab_size=1024,
        n_positions=128,
        n_embd=128,
        n_layer=4,
        n_head=4,
        n_inner=512,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
    )
    hf_model = GPT2LMHeadModel(hf_cfg).eval()

    sml_cfg = ModelConfig(
        name="gpt2-test",
        vocab_size=1024,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        d_ff=512,
        max_seq_len=128,
        norm_type="layernorm",
        pos_encoding="learned",
        mlp_type="gelu",
        bias=True,
        tie_embeddings=True,
        norm_eps=1e-5,
    )
    sml_model = Transformer(sml_cfg).eval()
    load_gpt2_weights_into_sml(sml_model, hf_model.state_dict())

    input_ids = torch.randint(0, 1024, (2, 32))
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        sml_logits_sdpa = sml_model(input_ids, attn_backend="sdpa").logits
        sml_logits_naive = sml_model(input_ids, attn_backend="naive").logits

    diff_sdpa = torch.max(torch.abs(hf_logits - sml_logits_sdpa)).item()
    diff_naive = torch.max(torch.abs(hf_logits - sml_logits_naive)).item()
    assert diff_sdpa < 1e-4, f"SDPA GPT-2 logit diff {diff_sdpa} >= 1e-4"
    assert diff_naive < 1e-4, f"Naive GPT-2 logit diff {diff_naive} >= 1e-4"


def test_initial_loss_and_single_batch_overfit() -> None:
    """Verify step-0 loss ~ ln(V) and single-batch memorization to loss < 0.05."""
    torch.manual_seed(2026)
    cfg = ModelConfig(
        name="sml-tiny-overfit",
        vocab_size=2048,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        max_seq_len=64,
    )
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))

    init_out = model(x, targets=y)
    assert init_out.loss is not None
    expected_init = math.log(cfg.vocab_size)
    assert abs(init_out.loss.item() - expected_init) < 0.25

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    final_loss = init_out.loss.item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        out = model(x, targets=y)
        assert out.loss is not None
        out.loss.backward()
        optimizer.step()
        final_loss = out.loss.item()
        if final_loss < 0.02:
            break

    assert final_loss < 0.05, f"Failed to overfit single batch: final_loss={final_loss}"


def test_activation_hooks_capture_and_identity_patch() -> None:
    """Verify ActivationHookManager captures activations and identity-patching preserves logits."""
    torch.manual_seed(11)
    cfg = load_model_config(REPO_ROOT / "configs/model/sml_nano.yaml")
    model = Transformer(cfg).eval()
    hooks = ActivationHookManager(model)

    x = torch.randint(0, cfg.vocab_size, (2, 10))
    target_modules = ["layers.0.self_attn", "layers.1.mlp", "layers.2"]

    with hooks.capture(target_modules) as acts:
        base_logits = model(x).logits

    for name in target_modules:
        assert name in acts
        assert acts[name].shape == (2, 10, cfg.d_model)

    saved_mlp = acts["layers.1.mlp"].clone()

    # Identity patch: replacing layers.1.mlp with its own output must produce identical logits
    with hooks.patch({"layers.1.mlp": saved_mlp}):
        patched_id_logits = model(x).logits
    assert torch.max(torch.abs(base_logits - patched_id_logits)).item() == 0.0

    # Zero-ablation patch: zeroing layers.1.mlp must alter logits
    with hooks.patch({"layers.1.mlp": lambda t: torch.zeros_like(t)}):
        patched_zero_logits = model(x).logits
    assert torch.max(torch.abs(base_logits - patched_zero_logits)).item() > 1e-3


def test_triton_fused_rmsnorm_and_swiglu_forward_backward_parity() -> None:
    """Verify tiled SRAM block-masked RMSNorm & SwiGLU forward/backward match PyTorch < 1e-5."""
    torch.manual_seed(314)
    # Test non-power-of-2 dimensions (768 -> BLOCK_SIZE 1024, 1408 -> BLOCK_SIZE 2048)
    for d_model, d_ff in ((256, 768), (768, 1408)):
        x = torch.randn(3, 11, d_model, dtype=torch.float32, requires_grad=True)
        w = torch.randn(d_model, dtype=torch.float32, requires_grad=True)
        dy = torch.randn_like(x)

        y_ref = ref_rmsnorm(x, w, eps=1e-5)
        y_ref.backward(dy)
        assert x.grad is not None and w.grad is not None
        dx_ref, dw_ref = x.grad.clone(), w.grad.clone()

        y_tiled, dx_tiled, dw_tiled = simulate_tiled_rmsnorm_fwd_bwd(x, w, dy, eps=1e-5)
        assert torch.max(torch.abs(y_ref - y_tiled)).item() < 1e-5
        assert torch.max(torch.abs(dx_ref - dx_tiled)).item() < 1e-5
        assert torch.max(torch.abs(dw_ref - dw_tiled)).item() < 1e-5

        # SwiGLU forward + analytical backward parity
        gate = torch.randn(3, 11, d_ff, dtype=torch.float32, requires_grad=True)
        up = torch.randn(3, 11, d_ff, dtype=torch.float32, requires_grad=True)
        dout = torch.randn_like(gate)

        out_ref = ref_swiglu(gate, up)
        out_ref.backward(dout)
        assert gate.grad is not None and up.grad is not None
        dgate_ref, dup_ref = gate.grad.clone(), up.grad.clone()

        out_tiled, dgate_tiled, dup_tiled = simulate_tiled_swiglu_fwd_bwd(
            gate, up, dout, block_size=256
        )
        assert torch.max(torch.abs(out_ref - out_tiled)).item() < 1e-5
        assert torch.max(torch.abs(dgate_ref - dgate_tiled)).item() < 1e-5
        assert torch.max(torch.abs(dup_ref - dup_tiled)).item() < 1e-5

        # Auto dispatcher parity
        assert torch.allclose(fused_rmsnorm(x.detach(), w.detach(), backend="auto"), y_ref.detach())
        assert torch.allclose(
            fused_swiglu(gate.detach(), up.detach(), backend="auto"), out_ref.detach()
        )

    # Verify HBM traffic closed-form calculators
    assert compute_rmsnorm_hbm_bytes(num_tokens=1024, d_model=768, bytes_per_elem=2) == (
        (2 * 1024 * 768 + 768) * 2 + 1024 * 4
    )
    assert compute_swiglu_hbm_bytes(num_tokens=1024, d_ff=2048, bytes_per_elem=2) == (
        3 * 1024 * 2048 * 2
    )

    # If CUDA + Triton are present, also run the hardware @triton.jit kernels directly
    if is_triton_available():
        x_cuda = torch.randn(4, 16, 768, device="cuda", dtype=torch.float32, requires_grad=True)
        w_cuda = torch.randn(768, device="cuda", dtype=torch.float32, requires_grad=True)
        y_triton = fused_rmsnorm(x_cuda, w_cuda, backend="triton")
        y_torch = ref_rmsnorm(x_cuda, w_cuda)
        assert torch.max(torch.abs(y_triton - y_torch)).item() < 1e-5
