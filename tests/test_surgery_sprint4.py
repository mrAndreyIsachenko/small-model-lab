"""Tests for Sprint 4 weight surgery (`H7` CKA Mixed-Precision, `H8` Subspace Ablation) & Pareto."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sml.cli import main
from sml.config import ModelConfig
from sml.model import Transformer
from sml.surgery import (
    allocate_mixed_precision_bits,
    apply_mixed_precision_quantization,
    apply_subspace_ablation,
    collect_layer_activations,
    compute_layerwise_cka_drift,
    extract_contrastive_direction,
    linear_cka,
    project_out_rank1_subspace,
    prune_transformer_layers,
    quantize_tensor_groupwise,
    run_surgery_and_pareto_suite,
)


def test_linear_cka_invariance_and_layerwise_drift() -> None:
    torch.manual_seed(42)
    x = torch.randn(32, 64)
    y = torch.randn(32, 64)

    assert abs(linear_cka(x, x) - 1.0) < 1e-5
    assert abs(linear_cka(x, y) - linear_cka(y, x)) < 1e-5

    q, _ = torch.linalg.qr(torch.randn(64, 64))
    assert abs(linear_cka(x, y) - linear_cka(x, y @ q)) < 1e-4

    cfg = ModelConfig(
        name="tiny-cka",
        vocab_size=128,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
    )
    base = Transformer(cfg)
    adapted = copy.deepcopy(base)
    with torch.no_grad():
        adapted.layers[2].mlp.down_proj.weight.add_(
            torch.randn_like(adapted.layers[2].mlp.down_proj.weight) * 0.5
        )

    ids = torch.randint(0, 128, (4, 16))
    drifts = compute_layerwise_cka_drift(base, adapted, ids)
    assert len(drifts) == 4
    assert drifts[0]["cka_drift"] < 1e-5
    assert drifts[2]["cka_drift"] > drifts[1]["cka_drift"]


def test_quantize_groupwise_mixed_precision_and_pruning() -> None:
    torch.manual_seed(42)
    w = torch.randn(32, 64)
    act_scale = torch.linspace(0.5, 2.0, 64)

    w_q8 = quantize_tensor_groupwise(w, bits=8, group_size=32, activation_scale=act_scale)
    w_q4 = quantize_tensor_groupwise(w, bits=4, group_size=32, activation_scale=act_scale)
    assert w_q8.shape == w.shape
    assert F.mse_loss(w_q8, w).item() < F.mse_loss(w_q4, w).item()

    drifts = [
        {"layer_idx": 0, "cka_similarity": 0.99, "cka_drift": 0.01},
        {"layer_idx": 1, "cka_similarity": 0.60, "cka_drift": 0.40},
        {"layer_idx": 2, "cka_similarity": 0.95, "cka_drift": 0.05},
        {"layer_idx": 3, "cka_similarity": 0.50, "cka_drift": 0.50},
    ]
    bits_cka = allocate_mixed_precision_bits(
        drifts, high_bits=8, low_bits=4, top_k_high=2, strategy="cka_guided"
    )
    bits_bot = allocate_mixed_precision_bits(
        drifts, high_bits=8, low_bits=4, top_k_high=2, strategy="bottom_k_control"
    )
    assert bits_cka == {0: 4, 1: 8, 2: 4, 3: 8}
    assert bits_bot == {0: 8, 1: 4, 2: 8, 3: 4}

    cfg = ModelConfig(
        name="tiny-quant",
        vocab_size=128,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
    )
    model = Transformer(cfg)
    q_model, stats = apply_mixed_precision_quantization(model, bits_cka, group_size=32)
    assert stats["average_bits_per_weight"] == 6.0

    pruned = prune_transformer_layers(q_model, drop_layer_indices=[1])
    assert len(pruned.layers) == 3
    assert pruned.config.n_layers == 3


def test_rank1_subspace_projection_and_ablation() -> None:
    torch.manual_seed(42)
    w = torch.randn(64, 128)
    r = torch.randn(64)
    w_proj = project_out_rank1_subspace(w, r)

    r_unit = r / r.norm(p=2)
    max_ortho = float((r_unit @ w_proj).abs().max().item())
    assert max_ortho < 1e-5

    cfg = ModelConfig(
        name="tiny-subspace",
        vocab_size=128,
        d_model=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
    )
    model = Transformer(cfg)
    pos_ids = torch.randint(0, 64, (2, 12))
    neg_ids = torch.randint(64, 128, (2, 12))
    direction = extract_contrastive_direction(model, pos_ids, neg_ids, layer_idx=2)

    ablated = apply_subspace_ablation(model, direction, layer_indices=[2])
    act_after = collect_layer_activations(ablated, pos_ids)
    write_delta_after = act_after[2] - act_after[1]
    proj_after = float((write_delta_after @ direction).abs().max().item())
    assert proj_after < 1e-5


def test_run_surgery_and_pareto_suite_and_cli(tmp_path: Path) -> None:
    res = run_surgery_and_pareto_suite(seed=2026)
    assert res["h7_verified"] is True
    assert res["h8_verified"] is True
    assert res["pareto_dominance_verified"] is True

    rc = main(
        [
            "run-surgery",
            "--experiment",
            "experiments/h7_h8_surgery_pareto.yaml",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "test_h7_h8_run",
        ]
    )
    assert rc == 0
    saved = json.loads(
        (tmp_path / "test_h7_h8_run" / "surgery_pareto.json").read_text(encoding="utf-8")
    )
    assert saved["h7_verified"] is True
    assert saved["h8_verified"] is True
