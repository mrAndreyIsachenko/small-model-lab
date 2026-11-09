"""Unit tests for M0: model parameter math, experiment configs, tracking, and report compilation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sml.cli import main as sml_main
from sml.config import (
    ModelConfig,
    load_experiment_spec,
    load_model_config,
)
from sml.tracking.cost import compute_run_cost, estimate_experiment_compute
from sml.tracking.report import compile_markdown_report

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_exact_closed_form_parameter_counts() -> None:
    """Verify exact closed-form parameter counts for all Sprint 1 model configs."""
    nano = load_model_config(REPO_ROOT / "configs/model/sml_nano.yaml")
    assert nano.embedding_params == 8_388_608
    assert nano.non_embedding_params == 3_148_032
    assert nano.total_params == 11_536_640
    assert nano.kv_cache_bytes_per_token(2) == 2 * 4 * (2 * 64) * 2

    xs = load_model_config(REPO_ROOT / "configs/model/sml_xs.yaml")
    assert xs.embedding_params == 16_777_216
    assert xs.non_embedding_params == 23_601_664
    assert xs.total_params == 40_378_880

    base = load_model_config(REPO_ROOT / "configs/model/sml_100m.yaml")
    assert base.embedding_params == 25_165_824
    assert base.non_embedding_params == 75_516_672
    assert base.total_params == 100_682_496
    assert base.kv_cache_bytes_per_token(2) == 2 * 12 * (4 * 64) * 2  # 12,288 B/tok

    gpt2 = load_model_config(REPO_ROOT / "configs/model/gpt2_124m.yaml")
    assert gpt2.embedding_params == 38_597_376
    assert gpt2.positional_embedding_params == 786_432
    assert gpt2.non_embedding_params == 85_842_432
    # Exact parameter count of OpenAI GPT-2 Small (124M)
    assert gpt2.total_params == 124_439_808


def test_invalid_head_divisibility_rejected() -> None:
    with pytest.raises(ValueError, match="divisible by n_heads"):
        ModelConfig(
            name="bad-heads",
            vocab_size=32768,
            d_model=768,
            n_layers=12,
            n_heads=10,
            n_kv_heads=5,
            d_ff=2048,
        )

    with pytest.raises(ValueError, match="divisible by n_kv_heads"):
        ModelConfig(
            name="bad-kv",
            vocab_size=32768,
            d_model=768,
            n_layers=12,
            n_heads=12,
            n_kv_heads=5,
            d_ff=2048,
        )


def test_all_experiment_specs_load_and_compute_budget() -> None:
    exp_dir = REPO_ROOT / "experiments"
    yaml_files = sorted(exp_dir.glob("*.yaml"))
    assert len(yaml_files) >= 4

    for exp_path in yaml_files:
        spec = load_experiment_spec(exp_path, repo_root=REPO_ROOT)
        assert spec.experiment_id
        assert spec.hypothesis_id in {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"}
        assert spec.train.grad_accum_steps >= 1

        est = estimate_experiment_compute(spec)
        assert est["planned_tokens"] > 0
        assert est["flops_per_token"] > 0
        assert est["total_flops"] > 0
        assert est["estimated_clean_gpu_hours"] > 0


def test_cost_accounting_overhead_math() -> None:
    spec = load_experiment_spec(REPO_ROOT / "experiments/h3_wsd_100m.yaml", repo_root=REPO_ROOT)
    rec = compute_run_cost(
        hardware=spec.hardware,
        active_seconds=3600.0,
        billed_seconds=4500.0,
        num_gpus=1,
        estimated_pre_run_cost_usd=10.0,
    )
    assert rec.active_gpu_hours == pytest.approx(1.0)
    assert rec.billed_gpu_hours == pytest.approx(1.25)
    assert rec.infrastructure_overhead_fraction == pytest.approx(0.20)
    assert rec.billed_cost_usd == pytest.approx(1.25 * spec.hardware.hourly_rate_usd, rel=1e-4)


def test_m0_dry_run_and_report_compiler_isolation(tmp_path: Path) -> None:
    """Verify M0 exit criterion: dry-run creates bundle and report enforces mode isolation."""
    results_dir = tmp_path / "results"
    report_measured = tmp_path / "reports" / "measured_only.md"
    report_with_dry = tmp_path / "reports" / "with_dry_run.md"

    rc = sml_main(
        [
            "dry-run",
            "--experiment",
            str(REPO_ROOT / "experiments/h3_wsd_100m.yaml"),
            "--output-dir",
            str(results_dir),
            "--run-id",
            "dry_run_test",
        ]
    )
    assert rc == 0

    run_dir = results_dir / "dry_run_test"
    for expected_file in (
        "run.json",
        "config.yaml",
        "environment.json",
        "metrics.jsonl",
        "final_metrics.json",
        "cost.json",
        "summary.md",
    ):
        assert (run_dir / expected_file).exists(), f"Missing {expected_file}"

    # 1. Default report MUST exclude dry-runs so synthetic numbers never pollute official tables
    md_measured = compile_markdown_report(
        results_dir=results_dir,
        output_path=report_measured,
        include_dry_runs=False,
    )
    assert "No matching completed run bundles found." in md_measured
    assert "dry_run_test" not in md_measured

    # 2. Explicit --include-dry-runs compiles the full markdown table from the run bundle
    md_dry = compile_markdown_report(
        results_dir=results_dir,
        output_path=report_with_dry,
        include_dry_runs=True,
    )
    assert "| `dry_run_test` | `H3` | `dry_run` | `sml-100m` | `75,516,672` |" in md_dry

    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["status"] == "completed"
    assert run_json["mode"] == "dry_run"
