"""Tests for Sprint 7 (H11): FP8 provider survival and provider readiness.

Covers the `H11` acceptance criteria and the hub publishing utilities:

- FP8 (E4M3FN) fake quantization emulation (`quantize_fp8_e4m3`): shape /
  dtype preservation, finiteness, and dequantized bounds within the source
  ``absmax`` for per-channel and per-tensor scales.
- Server-side FP8 quantization (`apply_fp8_server_quantization`) deep-copies
  the model instead of mutating it.
- The seeded `H11` benchmark (`evaluate_h11_fp8_provider_survival`):
  ``h11_verified`` is True, ``fp8_retention_ratio >= 0.96`` and
  ``h11_fp8_kl_reduction_ratio >= 0.25``.
- Provider readiness validation (`verify_provider_readiness`): standard
  architectures, no ``trust_remote_code`` / ``auto_map``, a usable chat
  template, and clean weights (no unmerged LoRA, NaN/Inf, or FP8-hostile
  outliers).
- Lancet recipe provenance (`verify_recipe_provenance`): the
  ``lancet_recipe.json`` / ``lancet_basis.safetensors`` pair matches the
  checkpoint dimensions and carries unit-norm basis rows (ADR-006).
- Tenet model card rendering (`generate_tenet_model_card`), including the
  intended-use and ADR-006 reconstruction sections.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import save_file
from torch import nn

from sml.hub.model_card import generate_tenet_model_card
from sml.hub.validator import (
    MAX_ABS_WEIGHT_THRESHOLD,
    MAX_ROW_CREST_FACTOR_THRESHOLD,
    RECIPE_BASIS_FILENAME,
    RECIPE_FILENAME,
    RECIPE_SCHEMA_VERSION,
    STANDARD_ARCHITECTURES,
    verify_provider_readiness,
    verify_recipe_provenance,
)
from sml.surgery.quantization import (
    FP8_E4M3_MAX,
    apply_fp8_server_quantization,
    evaluate_h11_fp8_provider_survival,
    quantize_fp8_e4m3,
)

CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_checkpoint(
    root: Path,
    architectures: list[str] | None = None,
    config_extra: dict[str, Any] | None = None,
    tokenizer_config: dict[str, Any] | None = None,
    weights: dict[str, torch.Tensor] | None = None,
    omit: tuple[str, ...] = (),
) -> Path:
    """Materialize a minimal checkpoint directory under ``root``."""
    root.mkdir(parents=True, exist_ok=True)

    if "config" not in omit:
        config: dict[str, Any] = {
            "model_type": "qwen2",
            "architectures": architectures if architectures is not None else ["Qwen2ForCausalLM"],
        }
        config.update(config_extra or {})
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")

    if "tokenizer" not in omit:
        tok = tokenizer_config if tokenizer_config is not None else {"chat_template": CHAT_TEMPLATE}
        (root / "tokenizer_config.json").write_text(json.dumps(tok), encoding="utf-8")

    if "weights" not in omit:
        tensors = (
            weights
            if weights is not None
            else {
                "model.embed_tokens.weight": torch.randn(16, 32) * 0.02,
            }
        )
        torch.save(tensors, root / "weights.pt")

    return root


# ---------------------------------------------------------------------------
# quantize_fp8_e4m3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_quantize_fp8_e4m3_preserves_shape_dtype_and_is_finite(
    per_channel: bool, dtype: torch.dtype
) -> None:
    """FP8 emulation keeps shape, dtype, device, and finiteness for both modes."""
    torch.manual_seed(2026)
    tensor = (torch.randn(8, 16, dtype=torch.float32) * 3.0).to(dtype)
    out = quantize_fp8_e4m3(tensor, per_channel=per_channel)

    assert out.shape == tensor.shape
    assert out.dtype == dtype
    assert out.device == tensor.device
    assert bool(torch.isfinite(out).all())
    assert not torch.equal(out, tensor)  # quantization is lossy for random floats


@pytest.mark.parametrize("per_channel", [True, False])
def test_quantize_fp8_e4m3_bounds_within_absmax(per_channel: bool) -> None:
    """Dequantized magnitudes never exceed the source tensor's ``absmax``."""
    torch.manual_seed(11)
    tensor = torch.randn(6, 24) * 7.5
    absmax = float(tensor.abs().max())

    out = quantize_fp8_e4m3(tensor, per_channel=per_channel)
    assert bool(torch.isfinite(out).all())
    # 448 * (absmax / 448) reproduces absmax up to float32 rounding.
    assert float(out.abs().max()) <= absmax * (1.0 + 1e-5) + 1e-12


def test_quantize_fp8_e4m3_per_channel_rows_bounded_by_row_absmax() -> None:
    """With per-channel scales every row is bounded by its own row absmax."""
    torch.manual_seed(23)
    tensor = torch.randn(5, 12) * 20.0
    out = quantize_fp8_e4m3(tensor, per_channel=True)

    row_absmax = tensor.abs().amax(dim=1, keepdim=True)
    assert bool((out.abs() <= row_absmax * (1.0 + 1e-5) + 1e-12).all())
    # The dominant entry of each row survives quantization intact.
    assert torch.allclose(out.abs().amax(dim=1), tensor.abs().amax(dim=1), rtol=1e-3)


def test_quantize_fp8_e4m3_per_tensor_uses_single_global_scale() -> None:
    """Per-tensor mode bounds all entries by the global absmax, whatever the shape."""
    torch.manual_seed(31)
    tensor = torch.randn(2, 3, 8) * 2.5
    absmax = float(tensor.abs().max())

    out = quantize_fp8_e4m3(tensor, per_channel=False)
    assert out.shape == tensor.shape
    assert float(out.abs().max()) <= absmax * (1.0 + 1e-5) + 1e-12
    assert bool(torch.isfinite(out).all())


def test_quantize_fp8_e4m3_per_channel_helps_small_magnitude_rows() -> None:
    """Per-channel scales track small rows; a single global scale crushes them."""
    tensor = torch.stack([torch.linspace(50.0, 100.0, 16), torch.linspace(0.0005, 0.001, 16)])
    small_absmax = float(tensor[1].abs().max())

    per_channel = quantize_fp8_e4m3(tensor, per_channel=True)
    per_tensor = quantize_fp8_e4m3(tensor, per_channel=False)

    ch_err = abs(float(per_channel[1].abs().max()) - small_absmax) / small_absmax
    pt_err = abs(float(per_tensor[1].abs().max()) - small_absmax) / small_absmax
    assert ch_err < 0.05
    assert pt_err > 0.10
    # The large row is reproduced well in either mode.
    assert abs(float(per_tensor[0].abs().max()) - 100.0) < 0.05


def test_quantize_fp8_e4m3_edge_cases_stay_finite_and_bounded() -> None:
    """Zeros, tiny subnormal-scale inputs, and 1-D tensors stay finite and bounded."""
    zero = torch.zeros(4, 4)
    out_zero = quantize_fp8_e4m3(zero, per_channel=True)
    assert bool(torch.isfinite(out_zero).all())
    assert float(out_zero.abs().max()) == 0.0

    tiny = torch.tensor([[1e-15, 2e-15], [0.0, -1.5e-15]])
    out_tiny = quantize_fp8_e4m3(tiny, per_channel=False)
    assert bool(torch.isfinite(out_tiny).all())
    assert float(out_tiny.abs().max()) <= 2e-15 * (1.0 + 1e-5) + 1e-20

    flat = torch.randn(32) * 4.0
    out_flat = quantize_fp8_e4m3(flat, per_channel=True)  # 1-D falls back to per-tensor
    assert out_flat.shape == flat.shape
    assert float(out_flat.abs().max()) <= float(flat.abs().max()) * (1.0 + 1e-5) + 1e-12


def test_quantize_fp8_e4m3_max_value_maps_to_e4m3_grid() -> None:
    """The per-channel scale factor ``absmax / 448`` maps values onto the E4M3 grid."""
    tensor = torch.linspace(-30.0, 30.0, 64).reshape(4, 16)
    out = quantize_fp8_e4m3(tensor, per_channel=True)
    peak = float(tensor.abs().max())
    assert abs(float(out.abs().max()) - peak) <= 1e-5 * peak
    # Every dequantized value scaled back by row_scale is an exact float8_e4m3fn representable.
    row_scale = tensor.abs().amax(dim=1, keepdim=True) / FP8_E4M3_MAX
    scaled = out.float() / row_scale
    roundtrip = scaled.to(torch.float8_e4m3fn).to(torch.float32)
    assert bool(torch.isclose(scaled, roundtrip, atol=1e-5, rtol=1e-5).all())


# ---------------------------------------------------------------------------
# apply_fp8_server_quantization
# ---------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    """Small module mixing Linear layers with a non-quantized LayerNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fc1 -> LayerNorm -> fc2."""
        return self.fc2(self.norm(self.fc1(x)))


def _linear_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }


@pytest.mark.parametrize("per_channel", [True, False])
def test_apply_fp8_server_quantization_does_not_mutate_original_model(
    per_channel: bool,
) -> None:
    """Quantization deep-copies: the source model keeps its exact weights."""
    torch.manual_seed(2026)
    model = _TinyMLP().eval()
    before_linear = _linear_weights(model)
    before_norm = model.norm.weight.detach().clone()
    before_output = model(torch.ones(2, 16)).detach().clone()

    quantized = apply_fp8_server_quantization(model, per_channel=per_channel)

    assert quantized is not model
    for name, snapshot in before_linear.items():
        module = dict(model.named_modules())[name]
        assert torch.equal(module.weight.detach(), snapshot), name
    assert torch.equal(model.norm.weight.detach(), before_norm)
    assert torch.equal(model(torch.ones(2, 16)).detach(), before_output)


@pytest.mark.parametrize("per_channel", [True, False])
def test_apply_fp8_server_quantization_quantizes_every_linear(per_channel: bool) -> None:
    """Every Linear weight is FP8-requantized; non-Linear modules stay untouched."""
    torch.manual_seed(7)
    model = _TinyMLP().eval()
    quantized = apply_fp8_server_quantization(model, per_channel=per_channel)

    for name, module in quantized.named_modules():
        if isinstance(module, nn.Linear):
            original = dict(model.named_modules())[name].weight.detach()
            weight = module.weight.detach()
            assert weight.shape == original.shape
            assert weight.dtype == original.dtype
            assert bool(torch.isfinite(weight).all())
            assert float(weight.abs().max()) <= float(original.abs().max()) * (1.0 + 1e-5) + 1e-12
            assert not torch.equal(weight, original)
        elif hasattr(module, "weight"):
            original = dict(model.named_modules())[name].weight
            assert torch.equal(module.weight.detach(), original.detach())


# ---------------------------------------------------------------------------
# evaluate_h11_fp8_provider_survival
# ---------------------------------------------------------------------------


def test_evaluate_h11_fp8_provider_survival_meets_acceptance_criteria() -> None:
    """Lancet surgery survives FP8 (E4M3) deployment: H11 acceptance criteria hold."""
    result = evaluate_h11_fp8_provider_survival()

    assert result["h11_verified"] is True
    assert result["fp8_retention_ratio"] >= 0.96
    assert result["h11_fp8_kl_reduction_ratio"] >= 0.25

    kl_naive = result["kl_naive_fp8"]
    kl_lancet = result["kl_lancet_fp8"]
    assert kl_naive >= 0.0
    assert kl_lancet >= 0.0
    assert kl_lancet < kl_naive  # Lancet arm degrades strictly less under FP8


def test_evaluate_h11_fp8_provider_survival_report_is_complete_and_deterministic() -> None:
    """The report exposes per-arm breakdowns and is reproducible for a fixed seed."""
    first = evaluate_h11_fp8_provider_survival(seed=2026)
    second = evaluate_h11_fp8_provider_survival(seed=2026)

    assert set(first) == {
        "fp8_retention_ratio",
        "h11_fp8_kl_reduction_ratio",
        "kl_naive_fp8",
        "kl_lancet_fp8",
        "kl_naive_fp8_per_tensor",
        "kl_lancet_fp8_per_tensor",
        "fp8_subspace_releakage_naive",
        "fp8_subspace_releakage_lancet",
        "fp8_subspace_releakage_naive_per_tensor",
        "fp8_subspace_releakage_lancet_per_tensor",
        "h11_verified",
        "arms",
    }
    assert set(first["arms"]) == {"naive", "lancet"}
    for arm in first["arms"].values():
        assert set(arm) == {
            "kl_bf16_vs_fp8",
            "kl_arm_vs_own_fp8",
            "kl_bf16_vs_fp8_per_tensor",
            "fp8_subspace_releakage_per_channel",
            "fp8_subspace_releakage_per_tensor",
        }
        assert arm["kl_bf16_vs_fp8"] >= 0.0
        assert arm["kl_arm_vs_own_fp8"] >= 0.0
        assert arm["kl_bf16_vs_fp8_per_tensor"] >= 0.0
        assert arm["fp8_subspace_releakage_per_channel"] >= 0.0
        assert arm["fp8_subspace_releakage_per_tensor"] >= 0.0

    assert first == second  # seeded run is fully deterministic
    assert first["h11_verified"] is True
    assert first["fp8_retention_ratio"] >= 0.96


def test_h11_fp8_subspace_releakage_lancet_below_naive() -> None:
    """Lancet keeps FP8 subspace releakage below Naive for both scale modes."""
    result = evaluate_h11_fp8_provider_survival(seed=2026)

    assert result["fp8_subspace_releakage_lancet"] < result["fp8_subspace_releakage_naive"]
    assert result["fp8_subspace_releakage_lancet"] < 0.10
    assert (
        result["fp8_subspace_releakage_lancet_per_tensor"]
        < result["fp8_subspace_releakage_naive_per_tensor"]
    )


def test_h11_per_channel_vs_per_tensor_metrics() -> None:
    """Per-tensor KL and per-arm releakage metrics are finite non-negative floats."""
    result = evaluate_h11_fp8_provider_survival(seed=2026)

    kl_naive_pt = result["kl_naive_fp8_per_tensor"]
    kl_lancet_pt = result["kl_lancet_fp8_per_tensor"]
    for value in (kl_naive_pt, kl_lancet_pt):
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value >= 0.0
    assert kl_lancet_pt < kl_naive_pt  # Lancet wins on per-tensor KL too

    for arm in result["arms"].values():
        for key in (
            "fp8_subspace_releakage_per_channel",
            "fp8_subspace_releakage_per_tensor",
        ):
            value = arm[key]
            assert isinstance(value, float)
            assert math.isfinite(value)
            assert value >= 0.0


# ---------------------------------------------------------------------------
# verify_provider_readiness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", sorted(STANDARD_ARCHITECTURES))
def test_verify_provider_readiness_accepts_valid_standard_bundle(
    tmp_path: Path, architecture: str
) -> None:
    """A clean standard checkpoint with template and weights is provider ready."""
    checkpoint = _write_checkpoint(tmp_path / architecture, architectures=[architecture])

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is True
    assert report["errors"] == []
    assert report["checks"] == {
        "standard_architecture": True,
        "no_trust_remote_code": True,
        "valid_chat_template": True,
        "no_nan_inf": True,
        "no_unmerged_lora": True,
        "weight_outliers_ok": True,
        "acceptable_row_crest_factor": True,
    }
    assert report["architectures"] == [architecture]
    assert report["weight_files_checked"] == 1
    assert report["max_abs_weight"] < MAX_ABS_WEIGHT_THRESHOLD
    assert report["max_row_crest_factor"] < MAX_ROW_CREST_FACTOR_THRESHOLD


def test_verify_provider_readiness_rejects_auto_map(tmp_path: Path) -> None:
    """``auto_map`` implies remote code and blocks provider deployment."""
    checkpoint = _write_checkpoint(
        tmp_path,
        config_extra={"auto_map": {"AutoModelForCausalLM": "modeling.CustomModel"}},
    )

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["no_trust_remote_code"] is False
    assert any("auto_map" in error for error in report["errors"])


def test_verify_provider_readiness_rejects_trust_remote_code(tmp_path: Path) -> None:
    """``trust_remote_code=True`` is rejected wherever it appears."""
    checkpoint = _write_checkpoint(
        tmp_path,
        config_extra={"trust_remote_code": True},
        tokenizer_config={"chat_template": CHAT_TEMPLATE, "trust_remote_code": True},
    )

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["no_trust_remote_code"] is False
    assert report["checks"]["valid_chat_template"] is False
    assert any("trust_remote_code" in error for error in report["errors"])


def test_verify_provider_readiness_rejects_non_standard_architecture(
    tmp_path: Path,
) -> None:
    """Custom architectures require remote code support and are not provider ready."""
    checkpoint = _write_checkpoint(tmp_path, architectures=["TenetCustomForCausalLM"])

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["standard_architecture"] is False
    assert report["checks"]["no_trust_remote_code"] is True  # untouched by this failure
    assert report["architectures"] == ["TenetCustomForCausalLM"]


def test_verify_provider_readiness_rejects_missing_chat_template(tmp_path: Path) -> None:
    """An absent or empty chat template fails the template check."""
    cases = ({}, {"chat_template": ""}, {"chat_template": "   "})
    for index, tokenizer_config in enumerate(cases):
        checkpoint = _write_checkpoint(
            tmp_path / f"case_{index}", tokenizer_config=tokenizer_config
        )

        report = verify_provider_readiness(checkpoint)

        assert report["ready"] is False, tokenizer_config
        assert report["checks"]["valid_chat_template"] is False, tokenizer_config


def test_verify_provider_readiness_rejects_unmerged_lora(tmp_path: Path) -> None:
    """Unmerged ``lora_A`` / ``lora_B`` adapters in the weights are rejected."""
    checkpoint = _write_checkpoint(
        tmp_path,
        weights={
            "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8) * 0.02,
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 8) * 0.02,
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(8, 4) * 0.02,
        },
    )

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["no_unmerged_lora"] is False
    assert sum("lora" in error for error in report["errors"]) == 2
    assert report["checks"]["no_nan_inf"] is True  # adapters are structurally fine


def test_verify_provider_readiness_rejects_nan_and_inf_weights(tmp_path: Path) -> None:
    """NaN / Inf weight values fail the numerical sanity check."""
    checkpoint = _write_checkpoint(
        tmp_path,
        weights={
            "model.embed_tokens.weight": torch.randn(8, 8) * 0.02,
            "bad.layer.weight": torch.tensor([[float("nan"), 0.5], [1.0, 2.0]]),
            "worse.layer.weight": torch.tensor([[float("inf"), -1.0]]),
        },
    )

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["no_nan_inf"] is False
    assert len(report["errors"]) == 2
    assert not any("lora" in error for error in report["errors"])


def test_verify_provider_readiness_rejects_extreme_weight_outliers(tmp_path: Path) -> None:
    """Weights at or beyond the FP8-hostile magnitude threshold are rejected."""
    checkpoint = _write_checkpoint(
        tmp_path,
        weights={
            "model.embed_tokens.weight": torch.randn(8, 8) * 0.02,
            "outlier.weight": torch.full((4, 4), 75.0),
        },
    )

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["weight_outliers_ok"] is False
    assert report["max_abs_weight"] >= MAX_ABS_WEIGHT_THRESHOLD
    assert report["checks"]["no_nan_inf"] is True  # 75.0 is finite, just too large


def test_verify_provider_readiness_rejects_high_row_crest_factor(tmp_path: Path) -> None:
    """One spiky row entry inflates the crest factor while staying benign in size."""
    spike = torch.zeros(1, 1024)
    spike[0, 0] = 10.0  # peak / row RMS = 10 / (10 / sqrt(1024)) ~= 32.0
    checkpoint = _write_checkpoint(tmp_path, weights={"spike.weight": spike})

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["weight_outliers_ok"] is True  # 10.0 is a benign magnitude
    assert report["checks"]["acceptable_row_crest_factor"] is False
    assert report["max_row_crest_factor"] >= MAX_ROW_CREST_FACTOR_THRESHOLD
    assert report["max_abs_weight"] < MAX_ABS_WEIGHT_THRESHOLD


def test_verify_provider_readiness_rejects_missing_or_invalid_json(tmp_path: Path) -> None:
    """Missing or unparseable config / tokenizer files fail their checks."""
    checkpoint = _write_checkpoint(tmp_path, omit=("config", "tokenizer"))
    (tmp_path / "tokenizer_config.json").write_text("{not json", encoding="utf-8")

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["standard_architecture"] is False
    assert report["checks"]["no_trust_remote_code"] is False
    assert report["checks"]["valid_chat_template"] is False
    assert any("config.json not found" in error for error in report["errors"])
    assert any("tokenizer_config.json could not be parsed" in error for error in report["errors"])
    # Weights are still inspected when the metadata is broken.
    assert report["checks"]["no_nan_inf"] is True
    assert report["weight_files_checked"] == 1


# ---------------------------------------------------------------------------
# generate_tenet_model_card
# ---------------------------------------------------------------------------

MODEL_ID = "Tenet-12B-Nemo-Unslop-v1"
BASE_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
METRICS: dict[str, Any] = {
    "h9_bf16": 0.1234,
    "h9_fp8": 0.1401,
    "h9_naive": 0.42,
    "h10_bf16": 0.71,
    "h10_fp8": 0.69,
    "h10_naive": 0.55,
    "h11_bf16": 0.88,
    "h11_fp8": 0.86,
    "h11_naive": 0.61,
}


def _frontmatter(card: str) -> dict[str, Any]:
    """Parse the YAML frontmatter block of a rendered model card."""
    lines = card.splitlines()
    assert lines[0] == "---"
    closing = lines.index("---", 1)
    parsed = yaml.safe_load("\n".join(lines[1:closing]))
    assert isinstance(parsed, dict)
    return parsed


def test_generate_tenet_model_card_has_valid_yaml_frontmatter() -> None:
    """The card opens with parseable YAML frontmatter carrying tags and base model."""
    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, METRICS)

    assert card.startswith("---\n")
    frontmatter = _frontmatter(card)
    assert frontmatter["license"] == "apache-2.0"
    assert frontmatter["base_model"] == BASE_MODEL
    assert frontmatter["pipeline_tag"] == "text-generation"
    for tag in ("tenet", "lancet", "unslop", "fp8", "vllm", "openrouter", "grpo"):
        assert tag in frontmatter["tags"]


def test_generate_tenet_model_card_documents_lancet_and_h9_h10_h11() -> None:
    """The card explains the Lancet engine and reports H9 / H10 / H11 benchmarks."""
    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, METRICS)

    assert f"# {MODEL_ID}" in card
    assert "Lancet" in card
    assert "bi-projected" in card
    assert "norm-preserving" in card

    for header in ("H9", "H10", "H11"):
        assert f"### {header} - " in card
    # Formatted metric values land in the benchmark tables.
    assert "| 0.12 |" in card
    assert "| 0.86 |" in card


def test_generate_tenet_model_card_includes_deployment_sections() -> None:
    """vLLM, Ollama/GGUF, MLX and OpenRouter deployment recipes are present."""
    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, METRICS)

    assert f"vllm serve {MODEL_ID} --quantization fp8" in card
    assert "### Ollama / GGUF" in card
    assert "FROM ./Tenet-12B-Nemo-Unslop-v1-Q4_K_M.gguf" in card
    assert f"ollama create {MODEL_ID.lower()} -f Modelfile" in card
    assert "### MLX" in card
    assert "mlx_lm.generate" in card
    assert "### OpenRouter" in card
    assert "openrouter.ai/api/v1" in card
    assert "trust_remote_code" in card  # promised absent for providers


@pytest.mark.parametrize(
    "model_id",
    [
        "tenet-12B-Nemo-Unslop-v1",
        "Tenet-12B-Nemo",
        "Tenet-B-Nemo-Unslop-v1",
        "Tenet-12B--Unslop-v1",
        "Tenet-12B-Nemo-Unslop-vX",
        "MyModel-12B-Nemo-Unslop-v1",
        "",
    ],
)
def test_generate_tenet_model_card_rejects_invalid_model_id(model_id: str) -> None:
    """Non-canonical Tenet names raise ``ValueError``."""
    with pytest.raises(ValueError, match="canonical Tenet naming scheme"):
        generate_tenet_model_card(model_id, BASE_MODEL, METRICS)


def test_generate_tenet_model_card_falls_back_to_na_placeholders() -> None:
    """Missing metric keys render as ``n/a`` instead of crashing."""
    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, {})
    assert card.count("| n/a |") == 9


RECIPE_D_MODEL = 32
RECIPE_NUM_LAYERS = 4
RECIPE_SELECTED_LAYERS = [0, 2]
RECIPE_RANK = 8
PROBES_SHA256 = "d" * 64


@pytest.mark.parametrize(
    ("model_id", "base_model"),
    [
        ("Tenet-32B-Qwen2.5-Unslop-v1", "Qwen/Qwen2.5-32B-Instruct"),
        ("Tenet-70B-Llama3.3-Unslop-v1", "meta-llama/Llama-3.3-70B-Instruct"),
    ],
)
def test_generate_tenet_model_card_documents_safety_and_reconstruction(
    model_id: str, base_model: str
) -> None:
    """Larger Tenet sizes carry the ADR-006 safety and reconstruction sections."""
    card = generate_tenet_model_card(model_id, base_model, METRICS)

    assert "## Intended Use & Safety Alignment Scope" in card
    assert "## Reproducible Reconstruction (ADR-006)" in card


def test_generate_tenet_model_card_renders_recipe_metadata() -> None:
    """``recipe_metadata`` lands the probe checksum and steered layers in the card."""
    recipe_metadata = {"probes_sha256": PROBES_SHA256, "selected_layers": [7, 17, 27]}

    card = generate_tenet_model_card(MODEL_ID, BASE_MODEL, METRICS, recipe_metadata=recipe_metadata)

    assert "## Reproducible Reconstruction (ADR-006)" in card
    assert f"(`probes_sha256`): `{PROBES_SHA256}`" in card
    assert "(`selected_layers`): 7, 17, 27" in card


# ---------------------------------------------------------------------------
# verify_recipe_provenance
# ---------------------------------------------------------------------------


def _write_recipe_checkpoint(
    root: Path,
    *,
    hidden_size: int = RECIPE_D_MODEL,
    num_hidden_layers: int = RECIPE_NUM_LAYERS,
    basis: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Materialize a checkpoint shipping a consistent Lancet recipe pair."""
    checkpoint = _write_checkpoint(
        root,
        config_extra={
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
        },
    )

    recipe = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "base_model": "Qwen/Qwen2.5-32B-Instruct",
        "d_model": RECIPE_D_MODEL,
        "num_layers": RECIPE_NUM_LAYERS,
        "selected_layers": RECIPE_SELECTED_LAYERS,
        "rank": RECIPE_RANK,
        "layer_weights": {str(index): 1.0 for index in RECIPE_SELECTED_LAYERS},
        "probes_sha256": PROBES_SHA256,
    }
    (checkpoint / RECIPE_FILENAME).write_text(json.dumps(recipe), encoding="utf-8")

    if basis is None:
        basis = {
            f"layer_{index}": torch.eye(RECIPE_RANK, RECIPE_D_MODEL)
            for index in RECIPE_SELECTED_LAYERS
        }
    save_file(basis, str(checkpoint / RECIPE_BASIS_FILENAME))
    return checkpoint


def test_verify_recipe_provenance_accepts_consistent_recipe_bundle(
    tmp_path: Path,
) -> None:
    """A dimension-matched recipe pair with orthonormal rows verifies cleanly."""
    checkpoint = _write_recipe_checkpoint(tmp_path / "valid")

    result = verify_recipe_provenance(checkpoint)

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["recipe"]["d_model"] == RECIPE_D_MODEL
    assert result["recipe"]["selected_layers"] == RECIPE_SELECTED_LAYERS

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is True
    assert report["errors"] == []
    assert report["checks"]["recipe_provenance"] is True


def test_verify_recipe_provenance_rejects_hidden_size_mismatch(tmp_path: Path) -> None:
    """A config ``hidden_size`` disagreeing with recipe ``d_model`` is rejected."""
    checkpoint = _write_recipe_checkpoint(tmp_path / "mismatch", hidden_size=64)

    result = verify_recipe_provenance(checkpoint)

    assert result["valid"] is False
    assert any("d_model" in error for error in result["errors"])

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["recipe_provenance"] is False


def test_verify_recipe_provenance_rejects_non_unit_basis_rows(tmp_path: Path) -> None:
    """Basis tensors whose rows drift from unit L2 norm fail provenance."""
    basis = {
        f"layer_{index}": torch.eye(RECIPE_RANK, RECIPE_D_MODEL) for index in RECIPE_SELECTED_LAYERS
    }
    basis["layer_0"] = basis["layer_0"] * 2.0
    checkpoint = _write_recipe_checkpoint(tmp_path / "bad_rows", basis=basis)

    result = verify_recipe_provenance(checkpoint)

    assert result["valid"] is False
    assert any("unit L2 norm" in error for error in result["errors"])

    report = verify_provider_readiness(checkpoint)

    assert report["ready"] is False
    assert report["checks"]["recipe_provenance"] is False
