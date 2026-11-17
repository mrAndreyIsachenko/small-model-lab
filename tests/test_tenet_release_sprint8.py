"""Sprint 8 Tenet release tests: OpenRouter gateway routing, onboarding manifest, release bundle."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from sml.cli import main as cli_main
from sml.gateway.client import (
    OPENROUTER_TENET_PROVIDERS,
    GatewayConfig,
    GatewayError,
    OpenRouterGatewayClient,
    build_openrouter_onboarding_manifest,
)
from sml.hub.release import build_tenet_release_bundle

FOUNDRY_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "remote" / "run_tenet_foundry.sh"
)

H9_METRICS: dict[str, Any] = {
    "h9_verified": True,
    "h9_kl_reduction_ratio": 0.42,
    "target_suppression_ratio": 0.9,
    "naive_kl_collateral": 0.03,
    "lancet_kl_collateral": 0.01,
    "max_row_norm_drift": 0.02,
    "anchor_orthogonality_error": 1e-6,
}

H10_METRICS: dict[str, Any] = {
    "h10_verified": True,
    "subspace_safe_lora_releakage": 0.01,
    "unprojected_lora_releakage": 0.4,
    "post_merge_suppression_retention": 0.95,
    "h10_gated_vs_naive_gain": 0.2,
    "h10_post_surgery_gain": 0.1,
    "merged_lora_count": 4,
    "length_collapse_rate": 0.0,
}

H11_METRICS: dict[str, Any] = {
    "h11_verified": True,
    "fp8_retention_ratio": 0.98,
    "h11_fp8_kl_reduction_ratio": 0.5,
    "fp8_subspace_releakage_lancet": 0.01,
    "fp8_subspace_releakage_naive": 0.3,
    "max_row_crest_factor": 2.5,
    "max_abs_weight": 0.8,
}


def _make_client() -> tuple[OpenRouterGatewayClient, dict[str, Any]]:
    """Build a gateway client whose transport records the outbound request."""
    captured: dict[str, Any] = {}

    def transport(request: urllib.request.Request, timeout: int) -> dict[str, Any]:
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return {
            "id": "resp-1",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    config = GatewayConfig(api_key="test-key")
    return OpenRouterGatewayClient(config, transport=transport), captured


def _chat_payload() -> dict[str, Any]:
    return {"model": "qwen/qwen3-coder", "messages": [{"role": "user", "content": "hello"}]}


def test_openrouter_tenet_providers_constant():
    assert OPENROUTER_TENET_PROVIDERS == ["Featherless", "Chutes", "Novita", "DeepInfra"]


def test_chat_completions_injects_provider_order_without_mutating_payload():
    client, captured = _make_client()
    payload = _chat_payload()
    original = copy.deepcopy(payload)

    response = client.chat_completions(
        payload, request_id="req-42", provider_order=OPENROUTER_TENET_PROVIDERS
    )

    assert captured["body"]["provider"] == {
        "order": OPENROUTER_TENET_PROVIDERS,
        "allow_fallbacks": True,
    }
    assert captured["body"]["model"] == "qwen/qwen3-coder"
    assert payload == original
    assert "provider" not in payload
    assert response["id"] == "resp-1"
    assert client.audit_log[-1]["status"] == 200


@pytest.mark.parametrize("bad_order", [[], [""], ["Featherless", ""]])
def test_chat_completions_rejects_invalid_provider_order(bad_order):
    client, captured = _make_client()

    with pytest.raises(GatewayError) as exc_info:
        client.chat_completions(_chat_payload(), provider_order=bad_order)

    assert exc_info.value.code == "invalid_provider_order"
    assert exc_info.value.status == 400
    assert captured == {}


def test_provider_order_does_not_bypass_forbidden_key_filter():
    client, captured = _make_client()
    payload = {
        **_chat_payload(),
        "credentials": {"password": "hunter2"},
    }

    with pytest.raises(GatewayError) as exc_info:
        client.chat_completions(payload, provider_order=OPENROUTER_TENET_PROVIDERS)

    assert exc_info.value.code == "raw_body_fields_not_allowed"
    assert exc_info.value.status == 403
    assert captured == {}


def test_provider_order_does_not_bypass_secret_leak_filter():
    client, captured = _make_client()
    payload = {
        "model": "qwen/qwen3-coder",
        "messages": [{"role": "user", "content": "use this key sk-or-v1-abcdefghij0123456789"}],
    }

    with pytest.raises(GatewayError) as exc_info:
        client.chat_completions(payload, provider_order=OPENROUTER_TENET_PROVIDERS)

    assert exc_info.value.code == "secret_leak_detected"
    assert exc_info.value.status == 403
    assert captured == {}


def test_build_openrouter_onboarding_manifest_keys_and_proof():
    manifest = build_openrouter_onboarding_manifest(
        hf_repo_id="tenet-lab/Tenet-3B-Qwen2.5-Unslop-v1",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        h9_metrics=H9_METRICS,
        h10_metrics=H10_METRICS,
        h11_metrics=H11_METRICS,
    )

    expected_keys = {
        "hf_repo_id",
        "base_model",
        "providers",
        "vllm_launch_args",
        "fp8_and_crest_proof",
        "architectural_compatibility",
        "markdown_summary",
    }
    assert set(manifest) == expected_keys
    assert manifest["hf_repo_id"] == "tenet-lab/Tenet-3B-Qwen2.5-Unslop-v1"
    assert manifest["base_model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert manifest["providers"] == OPENROUTER_TENET_PROVIDERS
    assert manifest["vllm_launch_args"]["quantization"] == "fp8"
    assert manifest["fp8_and_crest_proof"]["h11_fp8_provider_audit"]["max_row_crest_factor"] == 2.5
    summary = manifest["markdown_summary"]
    assert "FP8 Re-Leakage & Row Crest-Factor proof" in summary
    assert "vllm serve" in summary
    assert "Tenet-3B-Qwen2.5-Unslop-v1" in summary


def test_build_tenet_release_bundle_nano(tmp_path: Path):
    result = build_tenet_release_bundle(
        output_dir=tmp_path,
        release_id="Tenet-3B-Qwen2.5-Unslop-v1",
        backend="nano",
        grpo_steps=2,
        n_prompts=2,
        seq_len=16,
    )

    assert result["ready"] is True
    release_dir = Path(result["release_dir"])
    assert release_dir.is_dir()
    for artifact in (
        "README.md",
        "config.json",
        "tokenizer_config.json",
        "openrouter_onboarding.json",
        "OPENROUTER_ONBOARDING.md",
        "release_manifest.json",
    ):
        assert (release_dir / artifact).is_file(), artifact
    weights_path = Path(result["artifacts"]["weights"])
    assert weights_path.is_file()
    onboarding = json.loads((release_dir / "openrouter_onboarding.json").read_text())
    assert onboarding["providers"] == OPENROUTER_TENET_PROVIDERS


def test_cli_build_tenet_prints_ready_json(tmp_path: Path, capsys: pytest.CaptureFixture):
    exit_code = cli_main(
        [
            "build-tenet",
            "--output-dir",
            str(tmp_path),
            "--release-id",
            "Tenet-12B-Nemo-Unslop-v1",
            "--model-name",
            "mistralai/Mistral-Nemo-Instruct-2407",
            "--grpo-steps",
            "3",
            "--n-prompts",
            "2",
            "--seq-len",
            "16",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ready"] is True
    assert summary["release_id"] == "Tenet-12B-Nemo-Unslop-v1"
    assert summary["backend"] == "nano"


def test_run_tenet_foundry_script_dry_run():
    assert FOUNDRY_SCRIPT.is_file()
    assert os.access(FOUNDRY_SCRIPT, os.X_OK)
    syntax = subprocess.run(
        ["bash", "-n", str(FOUNDRY_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr

    env = {**os.environ, "DRY_RUN": "1", "TENET_PROFILE": "12b-h100", "UPLOAD_TO_HF": "1"}
    proc = subprocess.run(
        ["bash", str(FOUNDRY_SCRIPT)], capture_output=True, text=True, env=env, check=False
    )

    assert proc.returncode == 0, proc.stderr
    assert "Tenet-12B-Nemo-Unslop-v1" in proc.stdout
    assert "uv run sml build-tenet" in proc.stdout
    assert "huggingface-cli upload" in proc.stdout
