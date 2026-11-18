"""Tests for deterministic external benchmarks, vLLM FP8 smoke, and Tenet integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sml.cli import main as cli_main
from sml.evals.harness import (
    evaluate_external_benchmarks,
    run_before_after_external_eval,
)
from sml.gateway.client import (
    GatewayConfig,
    OpenRouterGatewayClient,
    build_openrouter_onboarding_manifest,
    build_vllm_fp8_mock_transport,
    run_vllm_fp8_smoke,
)
from sml.hub.model_card import generate_tenet_model_card

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_external_benchmarks_deterministic_before_after() -> None:
    first = run_before_after_external_eval(seed=1337)
    second = run_before_after_external_eval(seed=1337)
    assert first == second
    assert first["verified"] is True

    before = first["before"]
    after = first["after"]
    assert after["ifeval_strict"] > before["ifeval_strict"]
    assert after["gsm8k_math_acc"] > before["gsm8k_math_acc"]
    assert after["sql_incident_acc"] > before["sql_incident_acc"]
    assert after["direct_compliance_rate"] > before["direct_compliance_rate"]
    assert after["slop_and_refusal_rate"] < before["slop_and_refusal_rate"]


def test_external_benchmarks_custom_responder() -> None:
    def _all_refusal(_prompt: str) -> str:
        return "As an AI, I cannot help with that request."

    res = evaluate_external_benchmarks(_all_refusal, stage="after", seed=42)
    assert res["ifeval_strict"] == 0.0
    assert res["gsm8k_math_acc"] == 0.0
    assert res["direct_compliance_rate"] == 0.0
    assert res["slop_and_refusal_rate"] == 1.0


def test_cli_eval_external_writes_results_and_release_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "external_evals"
    release_dir = tmp_path / "release"
    rc = cli_main(
        [
            "eval-external",
            "--output-dir",
            str(out_dir),
            "--run-id",
            "ext_unit",
            "--release-dir",
            str(release_dir),
        ]
    )
    assert rc == 0

    run_json = json.loads((out_dir / "ext_unit.json").read_text(encoding="utf-8"))
    release_json = json.loads(
        (release_dir / "external_benchmarks.json").read_text(encoding="utf-8")
    )
    assert run_json["verified"] is True
    assert release_json["ifeval_strict"] == 1.0
    assert release_json["gsm8k_math_acc"] == 1.0
    assert release_json["sql_incident_acc"] == 1.0
    assert release_json["direct_compliance_rate"] == 1.0


def test_vllm_fp8_smoke_and_cli_dry_run(tmp_path: Path) -> None:
    config = GatewayConfig(
        upstream_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model_allowlist=("Tenet-3B-Qwen2.5-Unslop-v1",),
    )
    client = OpenRouterGatewayClient(config, transport=build_vllm_fp8_mock_transport())
    smoke = run_vllm_fp8_smoke(client, model="Tenet-3B-Qwen2.5-Unslop-v1")
    assert smoke["success_rate"] == 1.0
    assert smoke["fp8_parity"] == 1.0
    assert smoke["ttft_ms"]["mean"] >= 0.0
    assert smoke["tok_s"]["mean"] is not None

    smoke_json_path = tmp_path / "vllm_fp8_smoke.json"
    rc = cli_main(
        [
            "vllm-smoke",
            "--model-name",
            "Tenet-3B-Qwen2.5-Unslop-v1",
            "--output-json",
            str(smoke_json_path),
            "--dry-run",
        ]
    )
    assert rc == 0
    saved = json.loads(smoke_json_path.read_text(encoding="utf-8"))
    assert saved["success_rate"] == 1.0
    assert saved["fp8_parity"] == 1.0


def test_onboarding_manifest_and_model_card_include_external_and_vllm() -> None:
    ext = run_before_after_external_eval(seed=1337)
    config = GatewayConfig(
        upstream_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model_allowlist=("Tenet-3B-Qwen2.5-Unslop-v1",),
    )
    client = OpenRouterGatewayClient(config, transport=build_vllm_fp8_mock_transport())
    smoke = run_vllm_fp8_smoke(client, model="Tenet-3B-Qwen2.5-Unslop-v1")

    manifest = build_openrouter_onboarding_manifest(
        hf_repo_id="tenet-lab/Tenet-3B-Qwen2.5-Unslop-v1",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        h9_metrics={"h9_verified": True, "h9_kl_reduction_ratio": 0.91},
        h10_metrics={"h10_verified": True, "subspace_safe_lora_releakage": 0.01},
        h11_metrics={"fp8_retention_ratio": 0.994, "max_row_crest_factor": 17.27},
        external_benchmarks=ext["external_benchmarks"],
        vllm_smoke=smoke,
    )
    md = manifest["markdown_summary"]
    assert "## External Benchmarks" in md
    assert "IFEval strict" in md
    assert "GSM8K / MATH-500" in md
    assert "SQL-Incident Eval" in md
    assert "Direct-Compliance / Unslop Audit" in md
    assert "## vLLM FP8 Smoke & Parity" in md
    assert "FP8 parity" in md

    card = generate_tenet_model_card(
        "Tenet-3B-Qwen2.5-Unslop-v1",
        "Qwen/Qwen2.5-3B-Instruct",
        {
            "external_benchmarks": ext["external_benchmarks"],
            "external_benchmarks_comparison": {
                "before": ext["before"],
                "after": ext["after"],
                "deltas": ext["deltas"],
            },
        },
    )
    assert "## External Benchmarks" in card
    assert "### Before vs After (Lancet + Unslop GRPO + TIES Merge)" in card
    assert "GSM8K / MATH-500 accuracy" in card
    assert "SQL-Incident Eval accuracy" in card


def test_space_inspector_loads_external_benchmarks_and_vllm_smoke(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "spaces" / "fp8_abliteration_inspector"))
    import app as inspector_app  # type: ignore[import-not-found]

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "trust_remote_code": False,
                "chat_template": "{{ messages }}",
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "provider_readiness.json").write_text(
        json.dumps(
            {
                "max_abs_weight": 11.2,
                "max_row_crest_factor": 17.27,
                "no_nan_inf": True,
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "external_benchmarks.json").write_text(
        json.dumps(
            {
                "ifeval_strict": 1.0,
                "gsm8k_math_acc": 1.0,
                "sql_incident_acc": 1.0,
                "direct_compliance_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "vllm_fp8_smoke.json").write_text(
        json.dumps(
            {
                "ttft_ms": {"mean": 12.4, "p50": 12.0},
                "tok_s": {"mean": 145.0},
                "fp8_parity": 1.0,
                "success_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )

    inspected = inspector_app.inspect_directory_or_bundle(str(bundle_dir))
    assert inspected["external_benchmarks"]["ifeval_strict"] == 1.0
    assert inspected["vllm_smoke"]["fp8_parity"] == 1.0

    summary_md = inspector_app._format_checks_summary(inspected)
    assert "External Benchmarks" in summary_md
    assert "vLLM FP8 Smoke" in summary_md
    assert "DEMO_PRESET" in dir(inspector_app)
    assert "ifeval_strict" in inspector_app.DEMO_PRESET["tenet"]


def test_run_tenet_foundry_dry_run_with_vllm_smoke() -> None:
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "remote" / "run_tenet_foundry.sh")],
        cwd=str(REPO_ROOT),
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "DRY_RUN": "1",
            "SMOKE_ONLY": "1",
            "RUN_EXTERNAL_EVAL": "1",
            "RUN_VLLM_SMOKE": "1",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert "sml eval-external" in proc.stdout
    assert "vllm serve" in proc.stdout
    assert "--quantization fp8" in proc.stdout
    assert "sml vllm-smoke" in proc.stdout
