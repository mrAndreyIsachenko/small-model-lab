"""H13 streaming shard surgery/merge tests on sml-nano.

Covers:

* ``apply_lancet_recipe_streaming`` equivalence with the in-memory
  ``apply_lancet_recipe`` (plus ``cleanup_input_shards`` behavior);
* ``stream_subspace_safe_ties_merge`` equivalence with the in-memory
  ``subspace_safe_ties_merge`` for both dict and safetensors-directory
  delta maps;
* the ``provider_ready`` crest-factor gate on an injected outlier weight;
* ``run_h13_streaming_shard_benchmark`` and the ``verify-h13`` CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sml.adaptation.merge import (
    _compatible_forbidden_dirs,
    _load_shard_dir_tensors,
    _low_rank_delta,
    _orthonormalize_rows,
    _row_crest_factor,
    _save_sharded_state_dict,
    run_h13_streaming_shard_benchmark,
    stream_subspace_safe_ties_merge,
    subspace_safe_ties_merge,
)
from sml.config import load_model_config
from sml.model.transformer import Transformer
from sml.surgery.subspace import (
    LancetRecipe,
    _parse_target_layer_weight_key,
    apply_lancet_recipe,
    apply_lancet_recipe_streaming,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NANO_CONFIG = REPO_ROOT / "configs" / "model" / "sml_nano.yaml"


def _build_nano(seed: int = 2026):
    """Build the sml-nano transformer in eval mode (deterministic)."""
    torch.manual_seed(int(seed))
    cfg = load_model_config(str(NANO_CONFIG))
    model = Transformer(cfg)
    model.eval()
    return cfg, model


def _build_recipe(cfg, seed: int, model_id: str) -> LancetRecipe:
    """A synthetic-orthonormal Lancet recipe covering every nano layer."""
    n_layers = int(cfg.n_layers)
    d_model = int(cfg.d_model)
    basis_gen = torch.Generator().manual_seed(int(seed))
    basis_per_layer = {
        idx: _orthonormalize_rows(torch.randn(2, d_model, generator=basis_gen))
        for idx in range(n_layers)
    }
    return LancetRecipe(
        model_id=model_id,
        base_model="sml-nano",
        extraction_mode="synthetic_orthonormal",
        d_model=d_model,
        num_layers=n_layers,
        scale=1.0,
        top_k_layers=n_layers,
        selected_layers=list(range(n_layers)),
        layer_weights={idx: 1.0 for idx in range(n_layers)},
        basis_per_layer=basis_per_layer,
        probes_sha256="",
    )


def _max_row_norm_drift(streamed: torch.Tensor, reference: torch.Tensor) -> float:
    """Max per-row (last-dim) norm drift between two aligned tensors."""
    out = streamed.float()
    ref = reference.float()
    if out.ndim == 0:
        return float((out - ref).abs())
    return float((out.norm(dim=-1) - ref.norm(dim=-1)).abs().max())


def test_streaming_lancet_surgery_matches_in_memory_nano(tmp_path):
    """Streamed Lancet surgery must reproduce the in-memory surgery exactly."""
    seed = 2026
    cfg, model = _build_nano(seed)
    recipe = _build_recipe(cfg, seed, model_id="sml-nano-h13-surgery-test")

    ckpt_dir = tmp_path / "checkpoint"
    surg_dir = tmp_path / "surgery"
    surg_cleanup_dir = tmp_path / "surgery_cleanup"

    state_dict = {
        key: value.detach().to("cpu").clone().contiguous()
        for key, value in model.state_dict().items()
    }
    shard_files = _save_sharded_state_dict(state_dict, ckpt_dir, num_shards=3)
    assert len(shard_files) == 3

    surg_model = apply_lancet_recipe(model, recipe)
    surg_res = apply_lancet_recipe_streaming(ckpt_dir, surg_dir, recipe)

    reference = {
        key: value.detach().to("cpu").float() for key, value in surg_model.state_dict().items()
    }
    streamed = _load_shard_dir_tensors(surg_dir)
    assert set(streamed.keys()) == set(reference.keys())

    max_abs_diff = 0.0
    max_row_norm_drift = 0.0
    for key, ref in reference.items():
        out = streamed[key]
        assert tuple(out.shape) == tuple(ref.shape), key
        max_abs_diff = max(max_abs_diff, float((out.float() - ref.float()).abs().max()))
        max_row_norm_drift = max(max_row_norm_drift, _max_row_norm_drift(out, ref))

    assert max_abs_diff < 1e-6
    assert max_row_norm_drift < 1e-5
    assert float(surg_res["max_row_norm_drift"]) < 1e-5
    assert int(surg_res["shards_processed"]) == 3
    assert float(surg_res["peak_memory_gb"]) < 16.0
    assert surg_res["provider_ready"] is True

    # Re-run from the same checkpoint with cleanup_input_shards=True: the
    # source shards must be removed as they are processed, and the cleaned
    # output must match the first streamed pass.
    surg_res_cleanup = apply_lancet_recipe_streaming(
        ckpt_dir, surg_cleanup_dir, recipe, cleanup_input_shards=True
    )
    assert int(surg_res_cleanup["shards_processed"]) == 3
    for name in shard_files:
        assert not (ckpt_dir / name).exists(), f"source shard not removed: {name}"

    cleaned = _load_shard_dir_tensors(surg_cleanup_dir)
    assert set(cleaned.keys()) == set(streamed.keys())
    cleanup_diff = max(
        float((cleaned[key].float() - streamed[key].float()).abs().max()) for key in streamed
    )
    assert cleanup_diff < 1e-6


def test_streaming_ties_merge_matches_in_memory_nano(tmp_path):
    """Streamed subspace-safe TIES merge must match the in-memory merge."""
    seed = 4026
    cfg, model = _build_nano(seed)
    state_dict = {
        key: value.detach().to("cpu").clone().contiguous()
        for key, value in model.state_dict().items()
    }
    ckpt_dir = tmp_path / "checkpoint"
    _save_sharded_state_dict(state_dict, ckpt_dir, num_shards=3)

    d_model = int(cfg.d_model)
    n_layers = int(cfg.n_layers)
    basis_gen = torch.Generator().manual_seed(seed + 7)
    forbidden_dirs_per_layer = {
        idx: _orthonormalize_rows(torch.randn(2, d_model, generator=basis_gen))
        for idx in range(n_layers)
    }

    delta_gen = torch.Generator().manual_seed(seed + 13)
    delta_sql: dict[str, torch.Tensor] = {}
    delta_unslop: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        parsed = _parse_target_layer_weight_key(key)
        if parsed is None or tensor.ndim != 2:
            continue
        shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        delta_sql[key] = _low_rank_delta(shape, delta_gen)
        delta_unslop[key] = _low_rank_delta(shape, delta_gen)
    assert delta_sql, "sml-nano checkpoint exposed no o_proj/down_proj targets"

    merged_reference: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key not in delta_sql:
            continue
        parsed = _parse_target_layer_weight_key(key)
        forbidden = None
        if parsed is not None:
            forbidden = _compatible_forbidden_dirs(
                forbidden_dirs_per_layer.get(int(parsed[0])), int(tensor.shape[0])
            )
        res = subspace_safe_ties_merge(
            base_weight=tensor,
            delta_sql=delta_sql[key],
            delta_unslop=delta_unslop[key],
            forbidden_dirs=forbidden,
            sql_svd_rank=2,
            ties_density=0.85,
            weights=(1.0, 1.0),
            preserve_row_norm=True,
        )
        merged_reference[key] = res["merged_weight"].to(dtype=tensor.dtype)
    assert merged_reference

    def _compare(streamed: dict[str, torch.Tensor]) -> float:
        max_abs_diff = 0.0
        for key, tensor in streamed.items():
            ref = merged_reference.get(key)
            if ref is None:
                ref = state_dict[key]
            if tuple(tensor.shape) != tuple(ref.shape):
                return float("inf")
            max_abs_diff = max(max_abs_diff, float((tensor.float() - ref.float()).abs().max()))
        for key in merged_reference:
            assert key in streamed, f"merged tensor missing from streamed output: {key}"
        return max_abs_diff

    # Variant 1: in-memory dict delta maps.
    merged_dict_dir = tmp_path / "merged_dict"
    merge_res = stream_subspace_safe_ties_merge(
        ckpt_dir,
        merged_dict_dir,
        delta_sql_map=delta_sql,
        delta_unslop_map=delta_unslop,
        forbidden_dirs_per_layer=forbidden_dirs_per_layer,
    )
    streamed_dict = _load_shard_dir_tensors(merged_dict_dir)
    assert _compare(streamed_dict) < 1e-6
    assert int(merge_res["shards_processed"]) == 3
    assert int(merge_res["tensors_merged"]) == len(merged_reference)
    assert float(merge_res["max_row_crest_factor"]) < 25.0
    assert merge_res["provider_ready"] is True
    assert float(merge_res["peak_memory_gb"]) < 16.0

    # Variant 2: safetensors directory delta maps (lazily read per key).
    delta_sql_dir = tmp_path / "delta_sql"
    delta_unslop_dir = tmp_path / "delta_unslop"
    _save_sharded_state_dict(delta_sql, delta_sql_dir, num_shards=2)
    _save_sharded_state_dict(delta_unslop, delta_unslop_dir, num_shards=2)

    merged_st_dir = tmp_path / "merged_safetensors"
    merge_res_st = stream_subspace_safe_ties_merge(
        ckpt_dir,
        merged_st_dir,
        delta_sql_map=delta_sql_dir,
        delta_unslop_map=delta_unslop_dir,
        forbidden_dirs_per_layer=forbidden_dirs_per_layer,
    )
    streamed_st = _load_shard_dir_tensors(merged_st_dir)
    assert _compare(streamed_st) < 1e-6
    assert int(merge_res_st["shards_processed"]) == 3
    assert float(merge_res_st["max_row_crest_factor"]) < 25.0
    assert merge_res_st["provider_ready"] is True
    assert float(merge_res_st["peak_memory_gb"]) < 16.0

    # Both delta-map variants must agree with each other too.
    cross_diff = max(
        float((streamed_dict[key].float() - streamed_st[key].float()).abs().max())
        for key in streamed_dict
    )
    assert cross_diff < 1e-6


def test_streaming_merge_flags_high_crest_factor(tmp_path):
    """An extreme outlier row must trip the provider-readiness gate."""
    gen = torch.Generator().manual_seed(4242)
    outlier = torch.zeros(64, 32)
    outlier[0] = 5.0
    outlier[1:] = 0.01 * torch.randn(63, 32, generator=gen)
    normal = torch.randn(64, 32, generator=gen)
    state = {
        "outlier_proj.weight": outlier.contiguous(),
        "normal_proj.weight": normal.contiguous(),
    }
    assert _row_crest_factor(outlier) >= 25.0

    ckpt_dir = tmp_path / "checkpoint"
    _save_sharded_state_dict(state, ckpt_dir, num_shards=2)

    shape = (64, 32)
    delta_sql = {"normal_proj.weight": _low_rank_delta(shape, gen)}
    delta_unslop = {"normal_proj.weight": _low_rank_delta(shape, gen)}

    out_dir = tmp_path / "merged"
    res = stream_subspace_safe_ties_merge(
        ckpt_dir,
        out_dir,
        delta_sql_map=delta_sql,
        delta_unslop_map=delta_unslop,
    )

    assert int(res["tensors_merged"]) == 1
    assert int(res["shards_processed"]) == 2
    assert float(res["max_row_crest_factor"]) >= 25.0
    assert res["provider_ready"] is False

    # The untouched outlier weight is carried through bit-exactly, so the
    # reported crest factor is the outlier's.
    streamed = _load_shard_dir_tensors(out_dir)
    assert float((streamed["outlier_proj.weight"].float() - outlier.float()).abs().max()) == 0.0


def test_run_h13_benchmark_and_cli_verify_h13(tmp_path, monkeypatch):
    """run_h13_streaming_shard_benchmark and `verify-h13` both verify H13."""
    # The benchmark loads configs/model/sml_nano.yaml relative to the CWD.
    monkeypatch.chdir(REPO_ROOT)

    result = run_h13_streaming_shard_benchmark()
    assert result["h13_verified"] is True
    assert result["acceptance_checks"]["provider_ready"] is True
    assert int(result["shards_processed"]) == int(result["num_shards"]) == 3
    assert float(result["peak_memory_gb"]) < 16.0

    from sml.cli import main as cli_main

    try:
        exit_code = cli_main(["verify-h13", "--output-dir", str(tmp_path), "--backend", "nano"])
    except SystemExit as exc:  # argparse-style exits
        exit_code = 0 if exc.code is None else int(exc.code)
    assert exit_code == 0

    report = tmp_path / "streaming_shard_surgery.json"
    if not report.exists():
        candidates = sorted(tmp_path.rglob("streaming_shard_surgery.json"))
        assert candidates, "verify-h13 did not write streaming_shard_surgery.json"
        report = candidates[0]
    assert report.is_file()

    payload = json.loads(report.read_text(encoding="utf-8"))
    if "h13_verified" in payload:
        assert payload["h13_verified"] is True
