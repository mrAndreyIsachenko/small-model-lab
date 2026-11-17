"""Tests for Sprint 5 Batch 3: H9 Lancet bi-projected norm-preserving subspace surgery.

Covers the `H9` acceptance criteria from ADR-005
(`docs/decisions/005-lancet-biprojected-surgery.md`):

- Deterministic contrastive probe families and encoding (`sml.surgery.probes`).
- Anchor biprojection (`biproject_direction`): unit norm, float32, and
  ``anchor_orthogonality_error < 1e-6`` against single and multiple anchors.
- Norm-preserving multi-direction projection (`project_out_subspace_norm_preserving`):
  ``max_row_norm_drift < 1e-5`` with row-norm preservation, >= 99% target removal without.
- SCR layer profiling and Lancet surgery on a tiny 4-layer Transformer.
- The end-to-end `H9` benchmark (`run_h9_biprojected_surgery_benchmark`) and the
  `sml verify-h9` CLI bundle artifacts.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from sml.cli import main
from sml.config import ModelConfig
from sml.model import Transformer
from sml.surgery import (
    REASONING_ANCHOR_PAIRS,
    REFUSAL_CONTRAST_PAIRS,
    SLOP_CONTRAST_PAIRS,
    apply_lancet_biprojected_surgery,
    apply_lancet_recipe,
    biproject_direction,
    build_lancet_recipe,
    compute_layer_scr_profile,
    encode_contrast_pairs,
    load_lancet_recipe,
    project_out_subspace_norm_preserving,
    run_h9_biprojected_surgery_benchmark,
    save_lancet_recipe,
)
from sml.surgery.probes import TOOL_SQL_ANCHOR_PAIRS
from sml.surgery.subspace import (
    compute_probe_spectral_stats,
    extract_contrastive_direction_with_stats,
)


def _tiny_config() -> ModelConfig:
    """Small 4-layer Transformer used for surgery unit tests."""
    return ModelConfig(
        name="sml-h9-tiny",
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=2,
        n_kv_heads=1,
        d_ff=128,
        max_seq_len=32,
    )


class _MockTokenizer:
    """Minimal HF-style tokenizer exercising the `encode_contrast_pairs` tokenizer path."""

    eos_token = "<eos>"
    pad_token = None

    def __call__(
        self,
        texts: list[str],
        truncation: bool = True,
        max_length: int = 32,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        del truncation, padding, return_tensors  # signature parity with HF tokenizers
        rows: list[list[int]] = []
        for text in texts:
            ids = [ord(char) % 251 + 1 for char in text[:max_length]]
            rows.append(ids + [0] * (max_length - len(ids)))
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


def test_probe_families_have_twelve_pairs() -> None:
    """Each probe family is a fixed set of 12 non-trivial (positive, negative) pairs."""
    for pairs in (REFUSAL_CONTRAST_PAIRS, SLOP_CONTRAST_PAIRS, REASONING_ANCHOR_PAIRS):
        assert len(pairs) == 12
        for positive, negative in pairs:
            assert isinstance(positive, str) and positive
            assert isinstance(negative, str) and negative
            assert positive != negative


def test_encode_contrast_pairs_deterministic_and_tokenizer_compatible() -> None:
    """Hash-encoded probes are deterministic (12, 32) long tensors; a mock tokenizer works."""
    pos_ids, neg_ids = encode_contrast_pairs(REFUSAL_CONTRAST_PAIRS, vocab_size=256, seq_len=32)
    assert pos_ids.shape == neg_ids.shape == (12, 32)
    assert pos_ids.dtype == torch.long
    assert neg_ids.dtype == torch.long
    assert int(pos_ids.max()) < 256
    assert int(neg_ids.max()) < 256

    pos_again, neg_again = encode_contrast_pairs(REFUSAL_CONTRAST_PAIRS, vocab_size=256, seq_len=32)
    assert torch.equal(pos_ids, pos_again)
    assert torch.equal(neg_ids, neg_again)

    tokenizer = _MockTokenizer()
    tok_pos, tok_neg = encode_contrast_pairs(SLOP_CONTRAST_PAIRS, tokenizer=tokenizer)
    assert tok_pos.shape == tok_neg.shape == (12, 32)
    assert tok_pos.dtype == torch.long
    assert tok_neg.dtype == torch.long
    assert tokenizer.pad_token == tokenizer.eos_token  # pad-token fallback applied


def test_tool_sql_anchor_pairs_and_encoding() -> None:
    """The tool/SQL anchor family has 12 pairs covering the four tools and SQL."""
    assert len(TOOL_SQL_ANCHOR_PAIRS) == 12
    joined = "\n".join(text for pair in TOOL_SQL_ANCHOR_PAIRS for text in pair)
    for needle in (
        "query_warehouse",
        "get_metric_lineage",
        "inspect_changes",
        "submit_diagnosis",
        "SELECT",
        "GROUP BY",
        "JOIN",
    ):
        assert needle in joined

    pos_ids, neg_ids = encode_contrast_pairs(TOOL_SQL_ANCHOR_PAIRS, vocab_size=256, seq_len=32)
    assert pos_ids.shape == neg_ids.shape == (12, 32)
    assert pos_ids.dtype == torch.long
    assert neg_ids.dtype == torch.long


def test_probe_spectral_stats_and_extraction() -> None:
    """Rank-1 probe spectra collapse to one component; extraction returns unit directions."""
    torch.manual_seed(19)
    vec = torch.randn(64)
    rank_one = vec[:, None] @ vec[None, :]
    stats = compute_probe_spectral_stats(rank_one)
    assert float(stats["explained_variance_ratio_1"]) > 0.99
    assert abs(float(stats["spectral_rank_effective"]) - 1.0) < 1e-3

    model = Transformer(_tiny_config()).eval()
    pos_ids, neg_ids = encode_contrast_pairs(TOOL_SQL_ANCHOR_PAIRS, vocab_size=256, seq_len=32)
    direction, extraction_stats = extract_contrastive_direction_with_stats(model, pos_ids, neg_ids)
    assert direction.shape == (64,)
    assert abs(float(direction.norm(p=2)) - 1.0) < 1e-5
    assert 0.0 < float(extraction_stats["explained_variance_ratio_1"]) <= 1.0


def test_biproject_direction_unit_norm_and_anchor_orthogonal() -> None:
    """Biprojected residuals are unit-norm float32 vectors orthogonal to the anchor span."""
    torch.manual_seed(7)
    d_model = 64

    # Single anchor: residual drops the anchor component and stays unit norm.
    anchor = torch.randn(d_model)
    anchor = anchor / anchor.norm(p=2)
    target = 0.7 * anchor + 0.7 * torch.randn(d_model)
    residual = biproject_direction(target, [anchor])
    assert residual.dtype == torch.float32
    assert abs(float(residual.norm(p=2)) - 1.0) < 1e-5
    assert abs(float(residual @ anchor)) < 1e-6  # anchor_orthogonality_error

    # Multiple anchors: residual is orthogonal to the full anchor span.
    anchors = torch.randn(3, d_model)
    anchors = anchors / anchors.norm(p=2, dim=-1, keepdim=True)
    target_multi = 2.0 * anchors[0] - anchors[1] + torch.randn(d_model)
    residual_multi = biproject_direction(target_multi, anchors)
    assert residual_multi.dtype == torch.float32
    assert abs(float(residual_multi.norm(p=2)) - 1.0) < 1e-5
    anchor_orthogonality_error = max(abs(float(residual_multi @ a)) for a in anchors)
    assert anchor_orthogonality_error < 1e-6


def test_project_out_subspace_norm_preserving() -> None:
    """Row norms survive projection (< 1e-5 drift); without preservation >= 99% is removed."""
    torch.manual_seed(11)
    weight = torch.randn(64, 128)
    directions = torch.randn(3, 64)

    kept = project_out_subspace_norm_preserving(
        weight, directions, scale=1.0, preserve_row_norm=True
    )
    assert kept.shape == weight.shape
    assert kept.dtype == weight.dtype
    max_row_norm_drift = float(
        (weight.float().norm(p=2, dim=-1) - kept.float().norm(p=2, dim=-1)).abs().max()
    )
    assert max_row_norm_drift < 1e-5

    cut = project_out_subspace_norm_preserving(
        weight, directions, scale=1.0, preserve_row_norm=False
    )
    basis, _ = torch.linalg.qr(directions.T.float().contiguous())
    before = float((basis.T @ weight.float()).pow(2).sum())
    after = float((basis.T @ cut.float()).pow(2).sum())
    removed_fraction = 1.0 - after / before
    assert removed_fraction >= 0.99


def test_compute_layer_scr_profile_on_tiny_transformer() -> None:
    """SCR profile restores weights, ranks layers by signal / (kl_collateral + 1e-6)."""
    torch.manual_seed(13)
    model = Transformer(_tiny_config()).eval()
    target_dirs = {layer_idx: [torch.randn(64), torch.randn(64)] for layer_idx in range(4)}
    anchor_ids = torch.randint(0, 256, (2, 16))

    weight_before = model.layers[2].mlp.down_proj.weight.detach().clone()
    profile = compute_layer_scr_profile(model, target_dirs, anchor_input_ids=anchor_ids)
    assert torch.equal(model.layers[2].mlp.down_proj.weight.detach(), weight_before)

    assert len(profile) == 4
    for row in profile:
        assert set(row) == {"layer_idx", "signal", "kl_collateral", "scr"}
        assert 0 <= row["layer_idx"] < 4
        assert row["signal"] >= 0.0
        assert row["kl_collateral"] >= -1e-6
        assert abs(row["scr"] - row["signal"] / (row["kl_collateral"] + 1e-6)) < 1e-9
    scr_values = [row["scr"] for row in profile]
    assert scr_values == sorted(scr_values, reverse=True)

    subset = compute_layer_scr_profile(
        model, target_dirs, anchor_input_ids=anchor_ids, candidate_layers=[1, 2]
    )
    assert len(subset) == 2
    assert {row["layer_idx"] for row in subset} == {1, 2}


def test_apply_lancet_biprojected_surgery_on_tiny_transformer() -> None:
    """Lancet surgery deep-copies, preserves row norms, suppresses targets, spares anchors."""
    torch.manual_seed(17)
    model = Transformer(_tiny_config()).eval()
    anchor = torch.randn(64)
    anchor = anchor / anchor.norm(p=2)
    targets = {idx: [0.7 * anchor + torch.randn(64)] for idx in range(4)}
    anchors = {idx: [anchor] for idx in range(4)}

    original_weights = {
        (idx, name): proj.weight.detach().clone()
        for idx in range(4)
        for name, proj in (
            ("self_attn.o_proj", model.layers[idx].self_attn.o_proj),
            ("mlp.down_proj", model.layers[idx].mlp.down_proj),
        )
    }
    surg = apply_lancet_biprojected_surgery(
        model,
        target_dirs=targets,
        anchor_dirs=anchors,
        layer_indices=[1, 3],
        scale=1.0,
        preserve_row_norm=True,
    )
    assert surg is not model

    # Original model untouched; untouched layers identical.
    for (idx, name), snapshot in original_weights.items():
        proj = getattr(surg.layers[idx], name.split(".")[0])
        proj = getattr(proj, name.split(".")[1])
        if idx not in (1, 3):
            assert torch.equal(proj.weight, snapshot)
        assert torch.equal(
            model.layers[idx].self_attn.o_proj.weight,
            original_weights[(idx, "self_attn.o_proj")],
        )

    # Modified layers: row norms preserved and bi-projected targets suppressed.
    max_row_norm_drift = 0.0
    for idx in (1, 3):
        r_eff = biproject_direction(targets[idx][0], anchors[idx])
        assert abs(float(r_eff @ anchor)) < 1e-6  # anchor_orthogonality_error
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            module, attr = name.split(".")
            proj = getattr(getattr(surg.layers[idx], module), attr)
            snapshot = original_weights[(idx, name)]
            drift = (
                snapshot.detach().float().norm(dim=1) - proj.weight.detach().float().norm(dim=1)
            ).abs()
            max_row_norm_drift = max(max_row_norm_drift, float(drift.max().item()))
            energy_before = float((r_eff @ snapshot.detach().float()).pow(2).sum().item())
            energy_after = float((r_eff @ proj.weight.detach().float()).pow(2).sum().item())
            assert energy_after <= 0.01 * max(energy_before, 1e-12)
    assert max_row_norm_drift < 1e-5

    # Naive control (no anchors, no norm preservation) shrinks row norms by > 1e-5.
    naive = apply_lancet_biprojected_surgery(
        model,
        target_dirs=targets,
        anchor_dirs=None,
        layer_indices=[1, 3],
        scale=1.0,
        preserve_row_norm=False,
    )
    naive_drift = 0.0
    for idx in (1, 3):
        for name in ("self_attn.o_proj", "mlp.down_proj"):
            module, attr = name.split(".")
            proj = getattr(getattr(naive.layers[idx], module), attr)
            snapshot = original_weights[(idx, name)]
            drift = (snapshot.float().norm(dim=1) - proj.weight.float().norm(dim=1)).abs()
            naive_drift = max(naive_drift, float(drift.detach().max()))
    assert naive_drift > 1e-5


def test_apply_lancet_soft_scr_layer_weights() -> None:
    """Per-layer alpha weights soften layer 0 (0.25) more than layer 1 (1.0)."""
    torch.manual_seed(23)
    model = Transformer(_tiny_config()).eval()
    anchor = torch.randn(64)
    anchor = anchor / anchor.norm(p=2)
    targets = {idx: [0.7 * anchor + torch.randn(64)] for idx in (0, 1)}
    anchors = {idx: [anchor] for idx in (0, 1)}
    weights_before = {
        idx: model.layers[idx].mlp.down_proj.weight.detach().clone() for idx in (0, 1)
    }

    surg = apply_lancet_biprojected_surgery(
        model,
        target_dirs=targets,
        anchor_dirs=anchors,
        layer_indices=[0, 1],
        scale=1.0,
        layer_weights={0: 0.25, 1: 1.0},
        preserve_row_norm=True,
    )
    assert surg is not model

    residual_energy = {}
    max_row_norm_drift = 0.0
    for idx in (0, 1):
        r_eff = biproject_direction(targets[idx][0], anchors[idx])
        assert abs(float(r_eff @ anchor)) < 1e-6
        weight = surg.layers[idx].mlp.down_proj.weight.detach().float()
        residual_energy[idx] = float((r_eff @ weight).pow(2).sum().item())
        drift = (weights_before[idx].float().norm(dim=1) - weight.norm(dim=1)).abs().max()
        max_row_norm_drift = max(max_row_norm_drift, float(drift.item()))
    assert residual_energy[0] > residual_energy[1]
    assert max_row_norm_drift < 1e-5


def test_run_h9_biprojected_surgery_benchmark_seed_2026() -> None:
    """The seeded benchmark meets every H9 acceptance criterion from ADR-005."""
    res = run_h9_biprojected_surgery_benchmark(seed=2026)
    assert res["h9_verified"] is True
    assert res["anchor_orthogonality_error"] < 1e-6
    assert res["max_row_norm_drift"] < 1e-5
    assert res["target_suppression_ratio"] >= 0.90
    assert res["h9_kl_reduction_ratio"] >= 0.35
    assert res["tool_sql_orthogonality_error"] < 1e-6
    assert res["probe_evr_refusal"] > 0.0
    assert res["probe_evr_slop"] > 0.0
    schedule = res["layer_alpha_schedule"]
    alpha_values = list(schedule.values()) if isinstance(schedule, dict) else list(schedule)
    assert alpha_values
    assert all(0.1 <= value <= 1.0 for value in alpha_values)
    assert max(alpha_values) == 1.0


def test_verify_h9_cli_writes_valid_bundle(tmp_path: Path) -> None:
    """`sml verify-h9` exits 0 and writes valid surgery_h9.json / final_metrics.json."""
    rc = main(
        [
            "verify-h9",
            "--experiment",
            "experiments/h9_biprojected_surgery.yaml",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "test_h9_cli_run",
        ]
    )
    assert rc == 0

    run_dir = tmp_path / "test_h9_cli_run"
    surgery_path = run_dir / "surgery_h9.json"
    final_metrics_path = run_dir / "final_metrics.json"
    assert surgery_path.is_file()
    assert final_metrics_path.is_file()

    saved = json.loads(surgery_path.read_text(encoding="utf-8"))
    assert saved["h9_verified"] is True
    assert saved["anchor_orthogonality_error"] < 1e-6
    assert saved["max_row_norm_drift"] < 1e-5
    assert saved["target_suppression_ratio"] >= 0.90
    assert saved["h9_kl_reduction_ratio"] >= 0.35
    assert len(saved["selected_layers"]) == 3
    assert saved["tool_sql_orthogonality_error"] < 1e-6
    assert saved["probe_evr_refusal"] > 0.0
    assert saved["probe_evr_slop"] > 0.0
    saved_schedule = saved["layer_alpha_schedule"]
    saved_alpha_values = (
        list(saved_schedule.values()) if isinstance(saved_schedule, dict) else list(saved_schedule)
    )
    assert saved_alpha_values
    assert all(0.1 <= value <= 1.0 for value in saved_alpha_values)
    assert max(saved_alpha_values) == 1.0

    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
    assert final_metrics["h9_verified"] is True
    assert final_metrics["backend"] == "nano"
    assert len(final_metrics["selected_layers"]) == 3


def _proj(model: torch.nn.Module, layer_idx: int, path: str) -> torch.nn.Module:
    """Return the projection module at ``model.layers[layer_idx].<path>``."""
    child, attr = path.split(".")
    return getattr(getattr(model.layers[layer_idx], child), attr)


def test_lancet_recipe_save_load_and_deterministic_reconstruction(tmp_path: Path) -> None:
    """Recipes round-trip, match direct surgery, and re-ablate merged LoRAs."""
    torch.manual_seed(29)
    model = Transformer(_tiny_config()).eval()
    anchor = torch.randn(64)
    anchor = anchor / anchor.norm(p=2)
    targets = {idx: [0.7 * anchor + torch.randn(64), torch.randn(64)] for idx in range(4)}
    anchors = {idx: [anchor] for idx in range(4)}
    anchor_ids = torch.randint(0, 256, (2, 16))
    paths = ("self_attn.o_proj", "mlp.down_proj")

    original_weights = {
        (idx, path): _proj(model, idx, path).weight.detach().clone()
        for idx in range(4)
        for path in paths
    }

    recipe = build_lancet_recipe(
        model,
        target_dirs=targets,
        anchor_dirs=anchors,
        anchor_input_ids=anchor_ids,
        model_id="sml-h9-tiny",
        base_model="sml-h9-base",
        top_k_layers=3,
        scale=1.0,
        probes_sha256="a" * 64,
    )
    assert recipe.schema_version == "1.0"
    assert recipe.d_model == 64 and recipe.num_layers == 4
    assert 1 <= len(recipe.selected_layers) <= 3
    assert set(recipe.layer_weights) == set(recipe.selected_layers)
    for basis in recipe.basis_per_layer.values():
        assert basis.shape[1] == 64 and basis.shape[0] >= 1
        gram = basis @ basis.T
        assert float((gram - torch.eye(basis.shape[0])).abs().max()) < 1e-5

    recipe_dir = save_lancet_recipe(recipe, tmp_path / "lancet")
    loaded = load_lancet_recipe(recipe_dir)
    assert loaded.model_id == "sml-h9-tiny" and loaded.probes_sha256 == "a" * 64
    assert loaded.selected_layers == recipe.selected_layers
    assert loaded.layer_weights == recipe.layer_weights
    for idx, basis in recipe.basis_per_layer.items():
        assert torch.equal(loaded.basis_per_layer[idx], basis)

    # (a) Round-tripped recipe == direct biprojected surgery on the same layers.
    direct = apply_lancet_biprojected_surgery(
        model,
        target_dirs=targets,
        anchor_dirs=anchors,
        layer_indices=list(recipe.selected_layers),
        scale=recipe.scale,
        preserve_row_norm=True,
        layer_weights=dict(recipe.layer_weights),
    )
    from_recipe = apply_lancet_recipe(model, loaded)
    max_abs_diff = 0.0
    for idx in recipe.selected_layers:
        for path in paths:
            diff = (
                (
                    _proj(direct, idx, path).weight.detach().float()
                    - _proj(from_recipe, idx, path).weight.detach().float()
                )
                .abs()
                .max()
            )
            max_abs_diff = max(max_abs_diff, float(diff))
    assert max_abs_diff < 1e-6

    # (b) Row norms preserved and anchors spared.
    max_row_norm_drift = 0.0
    anchor_orthogonality_error = 0.0
    for idx in recipe.selected_layers:
        residual = biproject_direction(targets[idx][0], anchors[idx])
        anchor_orthogonality_error = max(anchor_orthogonality_error, abs(float(residual @ anchor)))
        for path in paths:
            snapshot = original_weights[(idx, path)]
            weight = _proj(from_recipe, idx, path).weight.detach().float()
            drift = (snapshot.float().norm(dim=1) - weight.norm(dim=1)).abs()
            max_row_norm_drift = max(max_row_norm_drift, float(drift.detach().max()))
    assert max_row_norm_drift < 1e-5
    assert anchor_orthogonality_error < 1e-6

    # (c) The recipe bundle stays compact.
    total_bytes = (recipe_dir / "lancet_recipe.json").stat().st_size + (
        recipe_dir / "lancet_basis.safetensors"
    ).stat().st_size
    assert total_bytes < 512 * 1024

    # (d) LoRA-merge guard: a merged LoRA that re-introduces the ablated
    # subspace along the stored basis is re-ablated by re-applying the recipe.
    full_scale = replace(recipe, layer_weights={idx: 1.0 for idx in recipe.selected_layers})
    merged = apply_lancet_recipe(model, full_scale)
    lora_strength = 1e-3
    for idx in recipe.selected_layers:
        basis = full_scale.basis_per_layer[idx]
        for path in paths:
            proj = _proj(merged, idx, path)
            snapshot = original_weights[(idx, path)].float()
            lora_delta = lora_strength * (basis.T @ (basis @ snapshot))
            perturbed = proj.weight.detach().float() + lora_delta
            row_scale = snapshot.norm(dim=1) / perturbed.norm(dim=1)
            proj.weight = torch.nn.Parameter(
                (perturbed * row_scale.unsqueeze(-1)).to(proj.weight.dtype),
                requires_grad=proj.weight.requires_grad,
            )

    merged = apply_lancet_recipe(merged, full_scale, in_place=True)
    projection_energy = 0.0
    max_merge_row_norm_drift = 0.0
    for idx in recipe.selected_layers:
        basis = full_scale.basis_per_layer[idx]
        for path in paths:
            weight = _proj(merged, idx, path).weight.detach().float()
            projection_energy += float((basis @ weight).pow(2).sum())
            snapshot = original_weights[(idx, path)]
            drift = (snapshot.float().norm(dim=1) - weight.norm(dim=1)).abs()
            drift_value = float(drift.detach().max())
            max_merge_row_norm_drift = max(max_merge_row_norm_drift, drift_value)
    assert projection_energy < 1e-5
    assert max_merge_row_norm_drift < 1e-5
