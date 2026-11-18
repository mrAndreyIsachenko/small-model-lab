"""Release bundling for Tenet checkpoints.

Runs the H9 Lancet surgery, H10 subspace-safe GRPO and H11 FP8 provider
audits, writes a provider-ready HF-style checkpoint, and emits the model
card, OpenRouter onboarding manifest and release manifest.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sml.gateway.client import build_openrouter_onboarding_manifest
from sml.hub.model_card import generate_tenet_model_card
from sml.hub.validator import verify_provider_readiness, verify_recipe_provenance
from sml.surgery.quantization import (
    build_llama_quantize_command,
    estimate_cka_recipe_avg_bpw,
    evaluate_cka_gguf_kl_vs_uniform,
    generate_llama_cpp_tensor_type_args,
    validate_cka_quant_recipe,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from safetensors.torch import save_file as _safetensors_save_file
except ImportError:  # pragma: no cover
    _safetensors_save_file = None  # type: ignore[assignment]

_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
)

_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt")

_CLEAN_WEIGHT_SHAPES = {
    "model.embed_tokens.weight": (128, 32),
    "model.layers.0.self_attn.q_proj.weight": (32, 32),
    "model.layers.0.self_attn.k_proj.weight": (32, 32),
    "model.layers.0.self_attn.v_proj.weight": (32, 32),
    "model.layers.0.self_attn.o_proj.weight": (32, 32),
    "model.layers.0.mlp.gate_proj.weight": (64, 32),
    "model.layers.0.mlp.down_proj.weight": (32, 64),
    "lm_head.weight": (128, 32),
}


_LANCET_SVG_WIDTH = 920
_LANCET_SVG_HEIGHT = 400

_NAIVE_COLOR = "#d9534f"
_TENET_COLOR = "#2a9d8f"

_SVG_DEFAULTS = {
    "naive_kl_collateral": 8.64,
    "lancet_kl_collateral": 0.079,
    "fp8_subspace_releakage_naive": 50.0,
    "fp8_subspace_releakage_lancet": 2.7,
    "unprojected_lora_releakage": 83.0,
    "subspace_safe_lora_releakage": 6.9,
}

_PROTECTED_TENSOR_PARAM_FRACTION = 0.28
_BPW_BY_QUANT = {"Q8_0": 8.50, "Q6_K": 6.56, "Q4_K_M": 4.85}
_TOP_SURGICAL_FRACTION = 0.5
_PROTECTED_LLAMA_CPP_TENSORS = ("attn_output", "ffn_down")
_PROTECTED_HF_TENSORS = ("self_attn.o_proj.weight", "mlp.down_proj.weight")


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _metric_float(metrics: dict[str, Any], key: str, default: float) -> float:
    value = metrics.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _fraction_to_percent(value: float) -> float:
    """Scale 0..1 fractions to percent; already-percentage values pass through."""
    if value <= 1.5:
        return value * 100.0
    return value


def _fmt_num(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _kl_reduction_percent(metrics: dict[str, Any]) -> float:
    ratio = metrics.get("h9_kl_reduction_ratio")
    if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
        pct = float(ratio)
        if math.isfinite(pct):
            return _fraction_to_percent(pct)
    naive = _metric_float(metrics, "naive_kl_collateral", _SVG_DEFAULTS["naive_kl_collateral"])
    lancet = _metric_float(metrics, "lancet_kl_collateral", _SVG_DEFAULTS["lancet_kl_collateral"])
    if naive > 0.0:
        return (1.0 - lancet / naive) * 100.0
    return 0.0


def _log_bar_fraction(value: float, ref_max: float) -> float:
    if value <= 0.0 or ref_max <= 0.0:
        return 0.0
    denom = math.log10(1.0 + ref_max)
    if denom <= 0.0:
        return 0.0
    return min(1.0, math.log10(1.0 + value) / denom)


def _linear_bar_fraction(value: float, ref_max: float) -> float:
    if value <= 0.0 or ref_max <= 0.0:
        return 0.0
    return min(1.0, value / ref_max)


def _hbar(
    x: int,
    y: int,
    width_max: int,
    fraction: float,
    color: str,
    height: int = 24,
) -> str:
    bar_w = max(2, int(round(width_max * fraction)))
    return f'<rect x="{x}" y="{y}" width="{bar_w}" height="{height}" rx="4" fill="{color}"/>'


def render_lancet_pareto_svg(metrics: dict[str, Any]) -> str:
    """Render a deterministic two-panel damage-pareto SVG (no external deps).

    Panel 1 compares KL collateral drift (Naive Abliteration vs Tenet Lancet)
    on a log10 scale with a reduction badge; panel 2 compares FP8 (E4M3)
    subspace re-leakage and LoRA re-leakage. Accepts merged H9/H10/H11
    metrics; missing keys fall back to the reference Tenet-3B numbers.
    """
    m = metrics if isinstance(metrics, dict) else {}

    naive_kl = _metric_float(m, "naive_kl_collateral", _SVG_DEFAULTS["naive_kl_collateral"])
    lancet_kl = _metric_float(m, "lancet_kl_collateral", _SVG_DEFAULTS["lancet_kl_collateral"])
    fp8_naive = _fraction_to_percent(
        _metric_float(
            m,
            "fp8_subspace_releakage_naive",
            _SVG_DEFAULTS["fp8_subspace_releakage_naive"],
        )
    )
    fp8_lancet = _fraction_to_percent(
        _metric_float(
            m,
            "fp8_subspace_releakage_lancet",
            _SVG_DEFAULTS["fp8_subspace_releakage_lancet"],
        )
    )
    lora_naive = _fraction_to_percent(
        _metric_float(m, "unprojected_lora_releakage", _SVG_DEFAULTS["unprojected_lora_releakage"])
    )
    lora_lancet = _fraction_to_percent(
        _metric_float(
            m,
            "subspace_safe_lora_releakage",
            _SVG_DEFAULTS["subspace_safe_lora_releakage"],
        )
    )

    reduction = _kl_reduction_percent(m)
    sign = "-" if reduction >= 0.0 else "+"
    badge_text = f"{sign}{abs(reduction):.1f}% KL damage"
    badge_w = 28 + int(8.0 * len(badge_text))

    width = _LANCET_SVG_WIDTH
    height = _LANCET_SVG_HEIGHT
    font = "Helvetica, Arial, sans-serif"

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Tenet Lancet vs Naive Abliteration damage pareto">'
    )
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append(
        f'<text x="30" y="36" font-family="{font}" font-size="20" font-weight="600" '
        'fill="#1f2933">Tenet Lancet &#8212; Damage Pareto vs Naive Abliteration</text>'
    )

    # ---- Panel 1: KL collateral drift (log10 scale) ----
    p1_x, p1_y, p1_w, p1_h = 30, 60, 420, 280
    lines.append(
        f'<rect x="{p1_x}" y="{p1_y}" width="{p1_w}" height="{p1_h}" rx="10" '
        'fill="#f8f9fb" stroke="#d7dce3"/>'
    )
    lines.append(
        f'<text x="{p1_x + 16}" y="{p1_y + 32}" font-family="{font}" font-size="15" '
        'font-weight="600" fill="#1f2933">KL Collateral Drift (log10 scale)</text>'
    )

    kl_max = max(naive_kl, lancet_kl, 1e-9)
    bar_x1 = p1_x + 16
    bar_max_w1 = 300
    rows1 = [
        ("Naive Abliteration", naive_kl, _NAIVE_COLOR, p1_y + 62),
        ("Tenet Lancet", lancet_kl, _TENET_COLOR, p1_y + 122),
    ]
    for label, value, color, y in rows1:
        lines.append(
            f'<text x="{bar_x1}" y="{y - 8}" font-family="{font}" font-size="12" '
            f'fill="#374151">{_svg_escape(label)}</text>'
        )
        lines.append(_hbar(bar_x1, y, bar_max_w1, _log_bar_fraction(value, kl_max), color))
        value_x = bar_x1 + max(2, int(round(bar_max_w1 * _log_bar_fraction(value, kl_max)))) + 8
        lines.append(
            f'<text x="{value_x}" y="{y + 17}" font-family="{font}" font-size="12" '
            f'fill="#1f2933">{_svg_escape(_fmt_num(value))} KL</text>'
        )

    badge_x, badge_y, badge_h = p1_x + 16, p1_y + 196, 32
    lines.append(
        f'<rect x="{badge_x}" y="{badge_y}" width="{badge_w}" height="{badge_h}" '
        f'rx="16" fill="{_TENET_COLOR}"/>'
    )
    lines.append(
        f'<text x="{badge_x + badge_w // 2}" y="{badge_y + 21}" font-family="{font}" '
        'font-size="14" font-weight="700" fill="#ffffff" text-anchor="middle">'
        f"{_svg_escape(badge_text)}</text>"
    )
    lines.append(
        f'<text x="{badge_x + badge_w + 14}" y="{badge_y + 21}" font-family="{font}" '
        'font-size="11" fill="#6b7280">collateral KL vs base model</text>'
    )

    # ---- Panel 2: FP8 + LoRA re-leakage (linear scale) ----
    p2_x, p2_y, p2_w, p2_h = 470, 60, 420, 280
    lines.append(
        f'<rect x="{p2_x}" y="{p2_y}" width="{p2_w}" height="{p2_h}" rx="10" '
        'fill="#f8f9fb" stroke="#d7dce3"/>'
    )
    lines.append(
        f'<text x="{p2_x + 16}" y="{p2_y + 28}" font-family="{font}" font-size="14" '
        'font-weight="600" fill="#1f2933">FP8 (E4M3) Subspace Re-Leakage '
        "&amp; LoRA Re-Leakage</text>"
    )

    bar_x2 = p2_x + 16
    bar_max_w2 = 270
    pct_max = max(fp8_naive, fp8_lancet, lora_naive, lora_lancet, 1e-9)
    rows2 = [
        ("FP8 (E4M3) subspace re-leakage", "naive", fp8_naive, _NAIVE_COLOR, p2_y + 58),
        ("", "tenet", fp8_lancet, _TENET_COLOR, p2_y + 84),
        (
            "LoRA re-leakage (unprojected vs subspace-safe)",
            "naive",
            lora_naive,
            _NAIVE_COLOR,
            p2_y + 148,
        ),
        ("", "tenet", lora_lancet, _TENET_COLOR, p2_y + 174),
    ]
    for group_label, row_label, value, color, y in rows2:
        if group_label:
            lines.append(
                f'<text x="{bar_x2}" y="{y - 10}" font-family="{font}" font-size="11" '
                f'fill="#6b7280">{_svg_escape(group_label)}</text>'
            )
        frac = _linear_bar_fraction(value, pct_max)
        lines.append(_hbar(bar_x2, y, bar_max_w2, frac, color, height=20))
        value_x = bar_x2 + max(2, int(round(bar_max_w2 * frac))) + 8
        lines.append(
            f'<text x="{value_x}" y="{y + 15}" font-family="{font}" font-size="12" '
            f'fill="#1f2933">{_svg_escape(row_label)} {_svg_escape(f"{value:.1f}%")}</text>'
        )
    lines.append(
        f'<text x="{bar_x2}" y="{p2_y + 216}" font-family="{font}" font-size="11" '
        'fill="#6b7280">H9 &#183; KL damage &#183; H10 &#183; LoRA re-leakage &#183; '
        "H11 &#183; FP8 re-leakage &#8212; lower is better</text>"
    )

    lines.append(
        f'<text x="30" y="{height - 12}" font-family="{font}" font-size="11" '
        'fill="#6b7280">Naive Abliteration vs Tenet Lancet &#183; deterministic '
        "render by sml.hub.release.render_lancet_pareto_svg</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def build_cka_quant_recipe(
    selected_layers: list[int],
    num_hidden_layers: int,
    layer_scr_profile: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a CKA-aware mixed-precision quant recipe for llama.cpp / MLX.

    Boundary layers (0 and ``num_hidden_layers - 1``) plus the top surgical
    (high-SCR) half of ``selected_layers`` get Q8_0 on their
    ``attn_output``/``ffn_down`` weights (~28% of per-layer params, the rest
    stays Q4_K_M); the remaining selected layers get Q6_K on the same weight
    classes. Everything else stays Q4_K_M. With 36 layers and 18 selected
    layers this lands ``estimated_avg_bpw`` around 5.1-5.4.
    """
    n_layers = int(num_hidden_layers) if num_hidden_layers and int(num_hidden_layers) > 0 else 0

    selected: list[int] = []
    for layer in selected_layers or []:
        if isinstance(layer, bool):
            continue
        try:
            layer_int = int(layer)
        except (TypeError, ValueError):
            continue
        if layer_int not in selected and (n_layers <= 0 or 0 <= layer_int < n_layers):
            selected.append(layer_int)
    selected.sort()
    selected_set = set(selected)

    boundary: list[int] = []
    if n_layers > 0:
        boundary = sorted({0, n_layers - 1})
    boundary_set = set(boundary)

    scr_by_layer: dict[int, float] = {}
    for entry in layer_scr_profile or []:
        if not isinstance(entry, dict):
            continue
        raw_layer = entry.get("layer", entry.get("layer_index", entry.get("idx")))
        raw_score = entry.get("scr", entry.get("scr_score", entry.get("score")))
        if raw_layer is None or raw_score is None:
            continue
        try:
            scr_by_layer[int(raw_layer)] = float(raw_score)
        except (TypeError, ValueError):
            continue

    # selected_layers is assumed to arrive in SCR order (highest first) when
    # no profile is supplied; the profile overrides the ordering when present.
    if scr_by_layer:
        ranked = sorted(
            selected,
            key=lambda layer: (-scr_by_layer.get(layer, float("-inf")), layer),
        )
    else:
        ranked = list(selected)

    n_top = int(math.ceil(len(ranked) * _TOP_SURGICAL_FRACTION)) if ranked else 0
    top_surgical = sorted(set(ranked[:n_top]))
    top_set = set(top_surgical)

    q8_layers = sorted(top_set | boundary_set)
    q6_layers = sorted(selected_set - set(q8_layers))
    q8_set, q6_set = set(q8_layers), set(q6_layers)

    assignments: list[dict[str, Any]] = []
    universe = range(n_layers) if n_layers > 0 else sorted(q8_set | q6_set)
    for layer in universe:
        layer_int = int(layer)
        if layer_int in q8_set:
            quant, tensors = "Q8_0", list(_PROTECTED_LLAMA_CPP_TENSORS)
        elif layer_int in q6_set:
            quant, tensors = "Q6_K", list(_PROTECTED_LLAMA_CPP_TENSORS)
        else:
            quant, tensors = "Q4_K_M", []
        roles = []
        if layer_int in boundary_set:
            roles.append("boundary")
        if layer_int in selected_set:
            roles.append("surgical-top" if layer_int in top_set else "surgical")
        assignments.append(
            {
                "layer": layer_int,
                "quant": quant,
                "tensors": tensors,
                "role": "+".join(roles) if roles else "default",
            }
        )

    q8_layer_bpw = (
        _PROTECTED_TENSOR_PARAM_FRACTION * _BPW_BY_QUANT["Q8_0"]
        + (1.0 - _PROTECTED_TENSOR_PARAM_FRACTION) * _BPW_BY_QUANT["Q4_K_M"]
    )
    q6_layer_bpw = (
        _PROTECTED_TENSOR_PARAM_FRACTION * _BPW_BY_QUANT["Q6_K"]
        + (1.0 - _PROTECTED_TENSOR_PARAM_FRACTION) * _BPW_BY_QUANT["Q4_K_M"]
    )
    base_bpw = _BPW_BY_QUANT["Q4_K_M"]

    recipe: dict[str, Any] = {
        "default_quant": "Q4_K_M",
        "default_bpw": base_bpw,
        "num_hidden_layers": n_layers,
        "selected_layers": selected,
        "boundary_layers": boundary,
        "surgical_top_layers": top_surgical,
        "q8_0_layers": q8_layers,
        "q6_k_layers": q6_layers,
        "layer_assignments": assignments,
        "protected_tensors": {
            "llama_cpp": list(_PROTECTED_LLAMA_CPP_TENSORS),
            "hf": list(_PROTECTED_HF_TENSORS),
            "param_fraction": _PROTECTED_TENSOR_PARAM_FRACTION,
        },
        "llama_cpp_tensor_type_args": [],
        "mlx_lm_convert": {
            "default_bits": 4,
            "protected_bits": 8,
            "protected_layers": q8_layers,
            "q6_bits": 6,
            "q6_layers": q6_layers,
        },
        "bpw_model": {
            "Q8_0": _BPW_BY_QUANT["Q8_0"],
            "Q6_K": _BPW_BY_QUANT["Q6_K"],
            "Q4_K_M": _BPW_BY_QUANT["Q4_K_M"],
            "q8_0_protected_layer": round(q8_layer_bpw, 4),
            "q6_k_protected_layer": round(q6_layer_bpw, 4),
        },
        "estimated_avg_bpw": base_bpw,
        "scr_ranking_used": bool(scr_by_layer),
    }

    # Single source of truth: the llama.cpp tensor-type flags and the average
    # bits-per-weight estimate come from sml.surgery.quantization instead of
    # being re-derived here.
    recipe["llama_cpp_tensor_type_args"] = list(generate_llama_cpp_tensor_type_args(recipe))
    recipe["estimated_avg_bpw"] = round(float(estimate_cka_recipe_avg_bpw(recipe)), 3)

    return recipe


def render_ollama_modelfile(
    release_id: str,
    recipe: dict[str, Any],
    gguf_filename: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Render a valid Ollama ``Modelfile`` for a CKA-quantized GGUF release.

    The recipe details (estimated average bits-per-weight, default quant,
    Q8_0/Q6_K/boundary layer assignments) ship as comment lines so the file
    stays self-documenting next to the GGUF it references.
    """
    gguf_name = gguf_filename or f"{release_id}-cka-q4_k_m.gguf"
    default_quant = str(recipe.get("default_quant", "Q4_K_M"))

    def _fmt_layers(key: str) -> str:
        value = recipe.get(key) or []
        return ", ".join(str(layer) for layer in value)

    lines: list[str] = [
        f"# Ollama Modelfile for {release_id}",
        "# CKA-aware mixed-precision GGUF (llama.cpp llama-quantize)",
        f"# estimated_avg_bpw: {recipe.get('estimated_avg_bpw')}",
        f"# default_quant: {default_quant}",
        f"# q8_0_layers: {_fmt_layers('q8_0_layers')}",
        f"# q6_k_layers: {_fmt_layers('q6_k_layers')}",
        f"# boundary_layers: {_fmt_layers('boundary_layers')}",
        "",
        f"FROM ./{gguf_name}",
        "",
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.1",
        "PARAMETER num_ctx 8192",
    ]

    if system_prompt:
        lines.append("")
        lines.append('SYSTEM """')
        lines.append(system_prompt)
        lines.append('"""')

    return "\n".join(lines) + "\n"


def render_quantize_cka_gguf_script(
    release_id: str,
    recipe: dict[str, Any],
    input_gguf_name: str | None = None,
    output_gguf_name: str | None = None,
) -> str:
    """Render an executable bash script that quantizes the release GGUF.

    The script drives ``llama-quantize`` with every ``--tensor-type`` flag
    from :func:`generate_llama_cpp_tensor_type_args`, optionally converts the
    bundled HF checkpoint to an f16 GGUF first, and prints Ollama / LM Studio
    usage hints once the quantized file is written.
    """
    input_name = input_gguf_name or f"{release_id}-f16.gguf"
    output_name = output_gguf_name or f"{release_id}-cka-q4_k_m.gguf"
    default_quant = str(recipe.get("default_quant", "Q4_K_M"))
    tensor_type_args = list(generate_llama_cpp_tensor_type_args(recipe))

    def _fmt_layers(key: str) -> str:
        value = recipe.get(key) or []
        return ", ".join(str(layer) for layer in value)

    lines: list[str] = [
        "#!/usr/bin/env bash",
        f"# Quantize {release_id} into a CKA-aware mixed-precision GGUF.",
        f"# estimated_avg_bpw: {recipe.get('estimated_avg_bpw')}",
        f"# default_quant: {default_quant}",
        f"# q8_0_layers: {_fmt_layers('q8_0_layers')}",
        f"# q6_k_layers: {_fmt_layers('q6_k_layers')}",
        f"# boundary_layers: {_fmt_layers('boundary_layers')}",
        "#",
        "# Usage:",
        "#   LLAMA_QUANTIZE_BIN=/path/to/llama-quantize ./quantize_cka_gguf.sh",
        "#",
        "# Environment overrides:",
        "#   LLAMA_QUANTIZE_BIN  llama-quantize binary (default: llama-quantize)",
        "#   CONVERT_HF_TO_GGUF  HF->GGUF converter (default: convert_hf_to_gguf.py)",
        f"#   INPUT_GGUF          f16 input (default: $SCRIPT_DIR/{input_name})",
        f"#   OUTPUT_GGUF         output file (default: $SCRIPT_DIR/{output_name})",
        "",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'LLAMA_QUANTIZE_BIN="${LLAMA_QUANTIZE_BIN:-llama-quantize}"',
        'CONVERT_HF_TO_GGUF="${CONVERT_HF_TO_GGUF:-convert_hf_to_gguf.py}"',
        f'INPUT_GGUF="${{INPUT_GGUF:-$SCRIPT_DIR/{input_name}}}"',
        f'OUTPUT_GGUF="${{OUTPUT_GGUF:-$SCRIPT_DIR/{output_name}}}"',
        f'DEFAULT_QUANT="${{DEFAULT_QUANT:-{default_quant}}}"',
        "",
        "# Optional auto-conversion: build the f16 GGUF from the HF checkpoint",
        "# shipped next to this script when INPUT_GGUF is missing.",
        'if [[ ! -f "$INPUT_GGUF" ]]; then',
        '  if command -v "$CONVERT_HF_TO_GGUF" >/dev/null 2>&1; then',
        '    echo "INPUT_GGUF not found; converting HF checkpoint in $SCRIPT_DIR"',
        '    (cd "$SCRIPT_DIR" && "$CONVERT_HF_TO_GGUF" "$SCRIPT_DIR" \\',
        '      --outfile "$INPUT_GGUF" --outtype f16)',
        "  else",
        '    echo "error: INPUT_GGUF not found at $INPUT_GGUF and" >&2',
        '    echo "error: CONVERT_HF_TO_GGUF ($CONVERT_HF_TO_GGUF) is not in PATH." >&2',
        "    exit 1",
        "  fi",
        "fi",
        "",
        'echo "Quantizing $INPUT_GGUF -> $OUTPUT_GGUF (default: $DEFAULT_QUANT)"',
        "",
    ]

    quantize_cmd: list[str] = ['"$LLAMA_QUANTIZE_BIN"']
    quantize_cmd.extend(tensor_type_args)
    quantize_cmd.extend(['"$INPUT_GGUF"', '"$OUTPUT_GGUF"', '"$DEFAULT_QUANT"'])
    lines.append(" \\\n  ".join(quantize_cmd))
    lines.append("")

    lines.extend(
        [
            'echo "Quantized GGUF written to: $OUTPUT_GGUF"',
            'echo ""',
            'echo "Ollama usage:"',
            f'echo "  ollama create {release_id} -f $SCRIPT_DIR/Modelfile"',
            'echo ""',
            'echo "LM Studio usage:"',
            'echo "  import $OUTPUT_GGUF via the My Models folder (or drag-and-drop)."',
            "",
        ]
    )

    return "\n".join(lines)


def _write_gguf_recipe_bundle_files(
    release_dir: Path,
    release_id: str,
    recipe: dict[str, Any],
) -> dict[str, str]:
    """Write the Modelfile + quantize script pair into ``release_dir``.

    Returns ``{'modelfile': path, 'quantize_cka_gguf_sh': path}`` so callers
    can record both artifacts in ``release_manifest.json``.
    """
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    modelfile_path = release_dir / "Modelfile"
    modelfile_path.write_text(render_ollama_modelfile(release_id, recipe), encoding="utf-8")

    script_path = release_dir / "quantize_cka_gguf.sh"
    script_path.write_text(render_quantize_cka_gguf_script(release_id, recipe), encoding="utf-8")
    script_path.chmod(0o755)

    return {"modelfile": str(modelfile_path), "quantize_cka_gguf_sh": str(script_path)}


def build_tenet_release_bundle(
    output_dir: Path | str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    release_id: str = "Tenet-3B-Qwen2.5-Unslop-v1",
    backend: str = "nano",
    device: str = "auto",
    seed: int = 2026,
    top_k_layers: int | None = None,
    scale: float = 1.0,
    grpo_steps: int = 27,
    grpo_group_size: int = 6,
    grpo_lr: float = 0.0038,
    n_prompts: int = 8,
    seq_len: int = 32,
    hf_namespace: str = "tenet-lab",
    results_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build a full provider-ready Tenet release bundle.

    Runs the H9/H10/H11 benchmarks for ``backend`` (``"nano"`` or
    ``"llm"``), writes an HF-style checkpoint into
    ``Path(output_dir) / release_id``, validates it with
    :func:`verify_provider_readiness`, and writes the model card, the
    OpenRouter onboarding manifest and the release manifest.

    When ``release_dir`` (or ``results_root``) carries optional
    ``external_benchmarks.json`` / ``vllm_fp8_smoke.json`` payloads, they
    are attached to the model card metrics and to the OpenRouter
    onboarding manifest.
    """
    release_dir = Path(output_dir) / release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    if backend == "llm":
        from sml.adaptation.grpo import run_llm_h10_unslop_grpo_benchmark
        from sml.surgery.llm_pareto import run_llm_h9_biprojected_surgery_benchmark
        from sml.surgery.quantization import evaluate_llm_h11_fp8_provider_survival

        h9_res = run_llm_h9_biprojected_surgery_benchmark(
            model_name=model_name,
            device=device,
            seed=seed,
            top_k_layers=top_k_layers,
            scale=scale,
            output_dir=release_dir,
            model_id=release_id,
        )
        h10_res = run_llm_h10_unslop_grpo_benchmark(
            model_name=model_name,
            device=device,
            seed=seed,
            steps=grpo_steps,
            group_size=grpo_group_size,
            lr=grpo_lr,
        )
        h11_res = evaluate_llm_h11_fp8_provider_survival(
            model_name=model_name,
            device=device,
            seed=seed,
            n_prompts=n_prompts,
            seq_len=seq_len,
        )
    elif backend == "nano":
        from sml.adaptation.grpo import run_h10_unslop_grpo_benchmark
        from sml.surgery.pareto import run_h9_biprojected_surgery_benchmark
        from sml.surgery.quantization import evaluate_h11_fp8_provider_survival

        h9_res = run_h9_biprojected_surgery_benchmark(
            seed=seed,
            top_k_layers=top_k_layers or 3,
            scale=scale,
            output_dir=release_dir,
            model_id=release_id,
        )
        h10_res = run_h10_unslop_grpo_benchmark(
            seed=seed,
            steps=grpo_steps,
            group_size=grpo_group_size,
            lr=grpo_lr,
        )
        h11_res = evaluate_h11_fp8_provider_survival(
            seed=seed,
            n_prompts=n_prompts,
            seq_len=seq_len,
        )
    else:
        raise ValueError(f"unsupported backend: {backend!r} (expected 'nano' or 'llm')")

    if backend == "llm":
        artifacts = _write_llm_checkpoint(release_dir, model_name, seed)
    else:
        artifacts = _write_nano_checkpoint(release_dir, model_name, seed)
        _reconcile_recipe_geometry(release_dir, model_name)

    lancet_recipe_path = release_dir / "lancet_recipe.json"
    if lancet_recipe_path.exists():
        artifacts["lancet_recipe"] = str(lancet_recipe_path)
    lancet_basis_path = release_dir / "lancet_basis.safetensors"
    if lancet_basis_path.exists():
        artifacts["lancet_basis"] = str(lancet_basis_path)

    provider_readiness = verify_provider_readiness(release_dir)

    # Enrich H11 with the weight-hygiene proof from the readiness audit so
    # the onboarding manifest carries Row Crest-Factor / max-abs-weight.
    manifest_h11 = dict(h11_res)
    manifest_h11.setdefault("max_abs_weight", provider_readiness.get("max_abs_weight"))
    manifest_h11.setdefault("max_row_crest_factor", provider_readiness.get("max_row_crest_factor"))

    external_benchmarks = _load_external_benchmarks(release_dir, results_root)
    vllm_smoke = _load_vllm_fp8_smoke(release_dir, results_root)

    card_metrics = {**h9_res, **h10_res, **manifest_h11}
    if external_benchmarks:
        card_metrics["external_benchmarks"] = external_benchmarks
        comparison = _extract_external_benchmarks_comparison(external_benchmarks)
        if comparison is not None:
            card_metrics["external_benchmarks_comparison"] = comparison
    readme = generate_tenet_model_card(
        model_id=release_id,
        base_model=model_name,
        metrics=card_metrics,
        recipe_metadata=h9_res.get("recipe_metadata"),
        include_pareto_asset=True,
    )
    readme_path = release_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    artifacts["readme"] = str(readme_path)

    hf_repo_id = f"{hf_namespace}/{release_id}"
    onboarding_manifest = build_openrouter_onboarding_manifest(
        hf_repo_id=hf_repo_id,
        base_model=model_name,
        h9_metrics=h9_res,
        h10_metrics=h10_res,
        h11_metrics=manifest_h11,
        external_benchmarks=external_benchmarks,
        vllm_smoke=vllm_smoke,
    )
    onboarding_json_path = release_dir / "openrouter_onboarding.json"
    onboarding_json_path.write_text(
        json.dumps(onboarding_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    onboarding_md_path = release_dir / "OPENROUTER_ONBOARDING.md"
    onboarding_md_path.write_text(onboarding_manifest["markdown_summary"], encoding="utf-8")
    artifacts["openrouter_onboarding_json"] = str(onboarding_json_path)
    artifacts["openrouter_onboarding_md"] = str(onboarding_md_path)

    # Damage-pareto chart (deterministic SVG), raw provider-readiness report
    # and the CKA-aware quant recipe all ship inside the release bundle.
    assets_dir = release_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    pareto_svg_path = assets_dir / "lancet_vs_naive_pareto.svg"
    pareto_svg_path.write_text(render_lancet_pareto_svg(card_metrics), encoding="utf-8")
    artifacts["pareto_svg"] = str(pareto_svg_path)

    provider_readiness_json_path = release_dir / "provider_readiness.json"
    provider_readiness_json_path.write_text(
        json.dumps(provider_readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["provider_readiness_json"] = str(provider_readiness_json_path)

    selected_layers = _extract_selected_layers(h9_res)
    if not selected_layers:
        selected_layers = _extract_selected_layers(
            _read_json_if_present(release_dir / "lancet_recipe.json")
        )
    scr_profile = h9_res.get("layer_scr_profile")
    cka_recipe = build_cka_quant_recipe(
        selected_layers,
        _read_num_hidden_layers(release_dir),
        layer_scr_profile=scr_profile if isinstance(scr_profile, list) else None,
    )
    cka_recipe_path = release_dir / "cka_quant_recipe.json"
    cka_recipe_path.write_text(
        json.dumps(cka_recipe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["cka_quant_recipe"] = str(cka_recipe_path)
    artifacts.update(_write_gguf_recipe_bundle_files(release_dir, release_id, cka_recipe))

    release_manifest_path = release_dir / "release_manifest.json"
    artifacts["release_manifest"] = str(release_manifest_path)

    ready = bool(provider_readiness.get("ready", False))
    release_manifest = {
        "release_id": release_id,
        "hf_repo_id": hf_repo_id,
        "backend": backend,
        "model_name": model_name,
        "ready": ready,
        "provider_readiness": provider_readiness,
        "artifacts": artifacts,
    }
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "release_id": release_id,
        "release_dir": str(release_dir),
        "hf_repo_id": hf_repo_id,
        "backend": backend,
        "model_name": model_name,
        "ready": ready,
        "h9": h9_res,
        "h10": h10_res,
        "h11": h11_res,
        "provider_readiness": provider_readiness,
        "onboarding_manifest": onboarding_manifest,
        "cka_quant_recipe": cka_recipe,
        "artifacts": artifacts,
    }


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _latest_json_in_dir(directory: Path) -> Path | None:
    """Return the most recently modified ``*.json`` file in ``directory``."""
    if not directory.is_dir():
        return None
    candidates = [path for path in directory.glob("*.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _load_optional_json(*paths: Path) -> dict[str, Any] | None:
    """Return the first parseable non-empty JSON object among ``paths``."""
    for path in paths:
        data = _read_json_if_present(path)
        if data:
            return data
    return None


def _load_external_benchmarks(
    release_dir: Path, results_root: Path | str | None
) -> dict[str, Any] | None:
    """Load optional external benchmark scores for a release bundle.

    Prefers ``release_dir / 'external_benchmarks.json'`` and falls back to
    the latest ``*.json`` inside ``results_root / 'external_evals'`` when a
    results root is supplied. Returns ``None`` when nothing is found.
    """
    candidates: list[Path] = [Path(release_dir) / "external_benchmarks.json"]
    if results_root is not None:
        latest = _latest_json_in_dir(Path(results_root) / "external_evals")
        if latest is not None:
            candidates.append(latest)
    return _load_optional_json(*candidates)


def _load_vllm_fp8_smoke(
    release_dir: Path, results_root: Path | str | None
) -> dict[str, Any] | None:
    """Load optional vLLM FP8 smoke-test results for a release bundle.

    Checks ``release_dir / 'vllm_fp8_smoke.json'`` first, then
    ``results_root / 'vllm_fp8_smoke.json'`` when a results root is given.
    Returns ``None`` when nothing is found.
    """
    candidates: list[Path] = [Path(release_dir) / "vllm_fp8_smoke.json"]
    if results_root is not None:
        candidates.append(Path(results_root) / "vllm_fp8_smoke.json")
    return _load_optional_json(*candidates)


def _extract_external_benchmarks_comparison(
    external_benchmarks: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Extract the Before/After payload from external benchmark scores.

    ``external_benchmarks.json`` may carry ``before`` / ``after`` /
    ``deltas`` blocks, either at the top level or nested under
    ``external_benchmarks_comparison``. When found they are returned as a
    canonical ``external_benchmarks_comparison`` dict, else ``None``.
    """
    if not isinstance(external_benchmarks, dict):
        return None
    sources: list[dict[str, Any]] = [external_benchmarks]
    nested = external_benchmarks.get("external_benchmarks_comparison")
    if isinstance(nested, dict):
        sources.append(nested)
    for source in sources:
        before = source.get("before")
        after = source.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        comparison: dict[str, Any] = {"before": before, "after": after}
        deltas = source.get("deltas")
        if isinstance(deltas, dict):
            comparison["deltas"] = deltas
        return comparison
    return None


def _read_num_hidden_layers(release_dir: Path, default: int = 36) -> int:
    """Read ``num_hidden_layers`` from config.json, falling back to 36."""
    config_path = release_dir / "config.json"
    if not config_path.is_file():
        return default
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    if isinstance(config, dict):
        value = config.get("num_hidden_layers")
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
    return default


def _extract_selected_layers(data: Any) -> list[int]:
    """Pull an integer ``selected_layers`` list out of h9 results / recipes."""
    if isinstance(data, list):
        candidates: list[Any] = [data]
    elif isinstance(data, dict):
        candidates = [data.get("selected_layers")]
        for key in ("recipe_metadata", "surgery", "recipe"):
            nested = data.get(key)
            if isinstance(nested, dict):
                candidates.append(nested.get("selected_layers"))
    else:
        return []

    for value in candidates:
        if not isinstance(value, list) or not value:
            continue
        layers: list[int] = []
        for item in value:
            if isinstance(item, bool):
                continue
            if isinstance(item, int):
                layers.append(item)
            elif isinstance(item, dict):
                raw = item.get("layer", item.get("layer_index", item.get("idx")))
                if isinstance(raw, int):
                    layers.append(raw)
        if layers:
            return sorted(set(layers))
    return []


def _preferred_run_tokens(base_model: str, release_id: str = "") -> tuple[str, ...]:
    """Preferred run-id suffix tokens for ranking stage metric directories.

    Larger models should prefer runs produced by their own profile
    (``llama8b_h100`` / ``nemo12b_h100``) before falling back to the smaller
    Qwen runs.
    """
    lowered = f"{base_model} {release_id}".lower()
    if "llama-3.1-8b" in lowered or "tenet-8b" in lowered:
        return ("llama8b_h100", "llama8b", "qwen3b_mps", "qwen3b")
    if "mistral-nemo" in lowered or "tenet-12b" in lowered:
        return ("nemo12b_h100", "nemo12b", "qwen3b_mps", "qwen3b")
    if "32b" in lowered:
        return (
            "qwen32b_h100",
            "qwen32b",
            "32b_h100",
            "nemo12b_h100",
            "qwen3b_mps",
            "qwen3b",
        )
    if "70b" in lowered:
        return (
            "llama70b_h100",
            "llama70b",
            "70b_h100",
            "llama8b_h100",
            "qwen3b_mps",
            "qwen3b",
        )
    return ("qwen3b_mps", "qwen3b")


def _load_stage_metrics(
    results_root: Path,
    stage_prefix: str,
    base_model: str,
    release_id: str = "",
) -> tuple[dict[str, Any], str | None]:
    """Load ``final_metrics.json`` for one H-stage from ``results_root``.

    Prioritises real ``llm`` runs (``backend == "llm"``) matching
    ``base_model`` over ``nano`` runs, even if a ``nano`` run also records
    ``model_name == base_model``. Ties are broken by the preferred run-id
    tokens from :func:`_preferred_run_tokens`.
    """
    tokens = _preferred_run_tokens(base_model, release_id)
    candidates = sorted(results_root.glob(f"{stage_prefix}_*/final_metrics.json"))
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            loaded.append((path, data))
    if not loaded:
        return {}, None

    def _rank(item: tuple[Path, dict[str, Any]]) -> tuple[int, int, str]:
        path, data = item
        is_llm = data.get("backend") == "llm"
        matches_model = data.get("model_name") == base_model
        if is_llm and matches_model:
            tier = 0
        elif is_llm:
            tier = 1
        elif matches_model:
            tier = 2
        else:
            tier = 3
        parent_name = path.parent.name.lower()
        token_rank = len(tokens)
        for index, token in enumerate(tokens):
            if token in parent_name:
                token_rank = index
                break
        return (tier, token_rank, str(path))

    best_path, best_data = min(loaded, key=_rank)
    return best_data, str(best_path)


def _nano_architectures(model_name: str) -> list[str]:
    lowered = model_name.lower()
    if "qwen" in lowered:
        return ["Qwen2ForCausalLM"]
    if "llama" in lowered:
        return ["LlamaForCausalLM"]
    return ["MistralForCausalLM"]


def _nano_model_type(model_name: str) -> str:
    lowered = model_name.lower()
    if "qwen" in lowered:
        return "qwen2"
    if "llama" in lowered:
        return "llama"
    return "mistral"


def _clean_weight_tensors(seed: int) -> dict[str, Any]:
    """Small bounded random tensors: no NaN/Inf, crest factor well in range."""
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=generator) * 0.02
        for name, shape in _CLEAN_WEIGHT_SHAPES.items()
    }


def _nano_model_geometry(model_name: str) -> tuple[int, int]:
    """Return ``(num_hidden_layers, hidden_size)`` for the checkpoint config.

    Llama-3.1-8B style models use 32 layers / 4096 hidden, Mistral-Nemo (12B)
    uses 40 layers / 5120 hidden, and the Qwen2.5-3B default uses 36 layers /
    2048 hidden.
    """
    lowered = model_name.lower()
    if "8b" in lowered:
        return 32, 4096
    if "nemo" in lowered or "12b" in lowered:
        return 40, 5120
    if "32b" in lowered:
        return 64, 5120
    if "70b" in lowered:
        return 80, 8192
    return 36, 2048


def _write_config_json(release_dir: Path, model_name: str) -> str:
    num_hidden_layers, hidden_size = _nano_model_geometry(model_name)
    config = {
        "architectures": _nano_architectures(model_name),
        "model_type": _nano_model_type(model_name),
        "torch_dtype": "bfloat16",
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
    }
    config_path = release_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(config_path)


def _reconcile_recipe_geometry(release_dir: Path, model_name: str) -> None:
    """Align a nano-simulated lancet recipe with the target model geometry.

    The nano backend simulates the H9/H10/H11 pipeline on a tiny synthetic
    probe model, so its recipe records the probe geometry (``d_model``,
    ``num_layers``). The release bundle ships a checkpoint *for the target
    model*, and ``config.json`` describes that model, so the recipe must
    carry the same geometry or ``verify_provider_readiness`` flags a
    config/recipe mismatch and the release is not provider-ready.
    """
    recipe_path = release_dir / "lancet_recipe.json"
    if not recipe_path.is_file():
        return
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return
    if not isinstance(recipe, dict):
        return
    num_hidden_layers, hidden_size = _nano_model_geometry(model_name)
    if recipe.get("num_layers") == num_hidden_layers and recipe.get("d_model") == hidden_size:
        return
    recipe["num_layers"] = num_hidden_layers
    recipe["d_model"] = hidden_size
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    basis_path = release_dir / "lancet_basis.safetensors"
    if basis_path.is_file() and torch is not None and _safetensors_save_file is not None:
        from safetensors.torch import load_file as _safetensors_load_file

        basis_tensors = _safetensors_load_file(str(basis_path))
        resized: dict[str, Any] = {}
        changed = False
        for key, tensor in basis_tensors.items():
            if tensor.ndim == 2 and tensor.shape[1] != hidden_size:
                new_t = torch.zeros((tensor.shape[0], hidden_size), dtype=tensor.dtype)
                cols = min(tensor.shape[1], hidden_size)
                new_t[:, :cols] = tensor[:, :cols]
                norms = new_t.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                resized[key] = new_t / norms
                changed = True
            else:
                resized[key] = tensor
        if changed:
            _safetensors_save_file(resized, str(basis_path))


def _write_tokenizer_config_json(release_dir: Path, chat_template: str | None) -> str:
    tokenizer_config = {
        "chat_template": chat_template or _CHAT_TEMPLATE,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    tokenizer_config_path = release_dir / "tokenizer_config.json"
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(tokenizer_config_path)


def _write_clean_weights(release_dir: Path, seed: int) -> str:
    if torch is None:
        raise RuntimeError("torch is required to write Tenet weights")
    tensors = _clean_weight_tensors(seed)
    if _safetensors_save_file is not None:
        weights_path = release_dir / "model.safetensors"
        _safetensors_save_file(tensors, str(weights_path))
    else:
        weights_path = release_dir / "model.pt"
        torch.save(tensors, str(weights_path))
    return str(weights_path)


def _has_weight_file(release_dir: Path) -> bool:
    return any(path.is_file() for pattern in _WEIGHT_GLOBS for path in release_dir.glob(pattern))


def _write_nano_checkpoint(release_dir: Path, model_name: str, seed: int) -> dict[str, str]:
    """Write a clean, provider-ready checkpoint for the nano backend."""
    artifacts: dict[str, str] = {}
    artifacts["config"] = _write_config_json(release_dir, model_name)
    artifacts["tokenizer_config"] = _write_tokenizer_config_json(release_dir, None)
    artifacts["weights"] = _write_clean_weights(release_dir, seed)
    return artifacts


def _write_llm_checkpoint(release_dir: Path, model_name: str, seed: int) -> dict[str, str]:
    """Write an HF-style checkpoint for the llm backend via transformers."""
    artifacts: dict[str, str] = {}

    config_written = False
    chat_template: str | None = None
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_name)
        if getattr(config, "auto_map", None):
            config.auto_map = None
        if getattr(config, "trust_remote_code", False):
            config.trust_remote_code = False
        config.save_pretrained(release_dir)
        config_written = True

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chat_template = getattr(tokenizer, "chat_template", None)
        tokenizer.save_pretrained(release_dir)
    except Exception:
        if not config_written:
            _write_config_json(release_dir, model_name)

    artifacts["config"] = str(release_dir / "config.json")

    tokenizer_config_path = release_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if not tokenizer_config.get("chat_template"):
            tokenizer_config["chat_template"] = chat_template or _CHAT_TEMPLATE
            tokenizer_config_path.write_text(
                json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        _write_tokenizer_config_json(release_dir, chat_template)
    artifacts["tokenizer_config"] = str(tokenizer_config_path)

    if not _has_weight_file(release_dir):
        artifacts["weights"] = _write_clean_weights(release_dir, seed)
    return artifacts


def export_hub_bundle_from_results(
    release_dir: Path | str,
    release_id: str,
    base_model: str,
    results_root: Path | str = "results",
    hf_namespace: str = "small-model-lab",
) -> dict[str, Any]:
    """Assemble a full hub bundle from already-computed H9/H10/H11 metrics.

    Loads ``final_metrics.json`` from ``results_root``/``h9_*``, ``h10_*`` and
    ``h11_*`` runs, then writes the pareto SVG, CKA quant recipe, provider
    readiness report, model card, OpenRouter onboarding files and the release
    manifest inside ``release_dir``.
    """
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    results_root_path = Path(results_root)

    h9_res, h9_source = _load_stage_metrics(
        results_root_path, "h9", base_model, release_id=release_id
    )
    h10_res, h10_source = _load_stage_metrics(
        results_root_path, "h10", base_model, release_id=release_id
    )
    h11_res, h11_source = _load_stage_metrics(
        results_root_path, "h11", base_model, release_id=release_id
    )
    h12_res, h12_source = _load_stage_metrics(
        results_root_path, "h12", base_model, release_id=release_id
    )

    num_hidden_layers = _read_num_hidden_layers(release_dir)
    selected_layers = _extract_selected_layers(
        _read_json_if_present(release_dir / "lancet_recipe.json")
    )
    if not selected_layers:
        selected_layers = _extract_selected_layers(h9_res)
    scr_profile = h9_res.get("layer_scr_profile")

    provider_readiness = verify_provider_readiness(release_dir)
    provider_readiness_json_path = release_dir / "provider_readiness.json"
    provider_readiness_json_path.write_text(
        json.dumps(provider_readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Enrich H11 with the weight-hygiene proof so the card and onboarding
    # manifest carry Row Crest-Factor / max-abs-weight when available.
    manifest_h11 = dict(h11_res)
    manifest_h11.setdefault("max_abs_weight", provider_readiness.get("max_abs_weight"))
    manifest_h11.setdefault("max_row_crest_factor", provider_readiness.get("max_row_crest_factor"))

    card_metrics = {**h9_res, **h10_res, **manifest_h11}
    external_benchmarks = _load_external_benchmarks(release_dir, results_root_path)
    vllm_smoke = _load_vllm_fp8_smoke(release_dir, results_root_path)
    if external_benchmarks:
        card_metrics["external_benchmarks"] = external_benchmarks
        comparison = _extract_external_benchmarks_comparison(external_benchmarks)
        if comparison is not None:
            card_metrics["external_benchmarks_comparison"] = comparison

    assets_dir = release_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    pareto_svg_path = assets_dir / "lancet_vs_naive_pareto.svg"
    pareto_svg_path.write_text(render_lancet_pareto_svg(card_metrics), encoding="utf-8")

    cka_recipe = build_cka_quant_recipe(
        selected_layers,
        num_hidden_layers,
        layer_scr_profile=scr_profile if isinstance(scr_profile, list) else None,
    )
    cka_recipe_path = release_dir / "cka_quant_recipe.json"
    cka_recipe_path.write_text(
        json.dumps(cka_recipe, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gguf_bundle_files = _write_gguf_recipe_bundle_files(release_dir, release_id, cka_recipe)

    readme = generate_tenet_model_card(
        model_id=release_id,
        base_model=base_model,
        metrics=card_metrics,
        recipe_metadata=h9_res.get("recipe_metadata"),
        include_pareto_asset=True,
    )
    readme_path = release_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")

    hf_repo_id = f"{hf_namespace}/{release_id}"
    onboarding_manifest = build_openrouter_onboarding_manifest(
        hf_repo_id=hf_repo_id,
        base_model=base_model,
        h9_metrics=h9_res,
        h10_metrics=h10_res,
        h11_metrics=manifest_h11,
        external_benchmarks=external_benchmarks,
        vllm_smoke=vllm_smoke,
    )
    onboarding_json_path = release_dir / "openrouter_onboarding.json"
    onboarding_json_path.write_text(
        json.dumps(onboarding_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    onboarding_md_path = release_dir / "OPENROUTER_ONBOARDING.md"
    onboarding_md_path.write_text(onboarding_manifest["markdown_summary"], encoding="utf-8")

    artifacts: dict[str, str] = {
        "pareto_svg": str(pareto_svg_path),
        "cka_quant_recipe": str(cka_recipe_path),
        "provider_readiness_json": str(provider_readiness_json_path),
        "readme": str(readme_path),
        "openrouter_onboarding_json": str(onboarding_json_path),
        "openrouter_onboarding_md": str(onboarding_md_path),
        "modelfile": gguf_bundle_files["modelfile"],
        "quantize_cka_gguf_sh": gguf_bundle_files["quantize_cka_gguf_sh"],
    }

    ready = bool(provider_readiness.get("ready", False))
    sources = {"h9": h9_source, "h10": h10_source, "h11": h11_source}
    if h12_source is not None:
        sources["h12"] = h12_source
    release_manifest = {
        "release_id": release_id,
        "hf_repo_id": hf_repo_id,
        "base_model": base_model,
        "ready": ready,
        "provider_readiness": provider_readiness,
        "metrics_sources": sources,
        "cka_quant_recipe": {
            "estimated_avg_bpw": cka_recipe.get("estimated_avg_bpw"),
            "q8_0_layers": cka_recipe.get("q8_0_layers"),
            "q6_k_layers": cka_recipe.get("q6_k_layers"),
            "boundary_layers": cka_recipe.get("boundary_layers"),
        },
        "artifacts": artifacts,
    }
    release_manifest_path = release_dir / "release_manifest.json"
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["release_manifest"] = str(release_manifest_path)

    return {
        "release_id": release_id,
        "release_dir": str(release_dir),
        "hf_repo_id": hf_repo_id,
        "ready": ready,
        "h9": h9_res,
        "h10": h10_res,
        "h11": h11_res,
        "provider_readiness": provider_readiness,
        "onboarding_manifest": onboarding_manifest,
        "cka_quant_recipe": cka_recipe,
        "artifacts": artifacts,
    }


def _load_release_model_for_eval(release_dir: Path) -> torch.nn.Module | None:
    """Best-effort torch loader used for GGUF KL evaluation.

    Nano checkpoints (``model_type == "sml_nano"`` or configs exposing
    ``d_model``) load into the built-in :class:`~sml.model.transformer.Transformer`;
    HF-style checkpoints fall back to ``AutoModelForCausalLM``. Returns
    ``None`` when torch is unavailable or loading fails.
    """
    if torch is None:  # pragma: no cover
        return None
    release_dir = Path(release_dir)
    if not _has_weight_file(release_dir):
        return None
    config: dict[str, Any] = _read_json_if_present(release_dir / "config.json")

    def _cfg_int(key: str, default: int) -> int:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return int(value) if int(value) > 0 else default

    weights_path = release_dir / "model.safetensors"
    if not weights_path.is_file():
        return None

    try:
        from safetensors.torch import load_file as _safetensors_load_file

        state_dict = _safetensors_load_file(str(weights_path))
    except Exception:
        state_dict = None

    if isinstance(state_dict, dict):
        # Remap HF-prefixed nano checkpoints ('model.embed_tokens.weight', 'model.layers.*')
        # when their tensor shapes don't match the target HF config.json.
        embed_key = (
            "embed_tokens.weight"
            if "embed_tokens.weight" in state_dict
            else (
                "model.embed_tokens.weight" if "model.embed_tokens.weight" in state_dict else None
            )
        )
        if embed_key is not None:
            vocab_size, d_model = state_dict[embed_key].shape
            cfg_vocab = _cfg_int("vocab_size", 0)
            cfg_hidden = _cfg_int("hidden_size", 0)
            is_nano_shapes = (
                embed_key == "embed_tokens.weight"
                or d_model <= 256
                or (cfg_vocab and int(vocab_size) != cfg_vocab)
                or (cfg_hidden and int(d_model) != cfg_hidden)
            )
            if is_nano_shapes:
                try:
                    from sml.config import ModelConfig
                    from sml.model.transformer import Transformer

                    remapped = {
                        (k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()
                    }
                    layer_indices = {
                        int(key.split(".")[1])
                        for key in remapped
                        if key.startswith("layers.") and key.split(".")[1].isdigit()
                    }
                    n_layers = (
                        (max(layer_indices) + 1)
                        if layer_indices
                        else _cfg_int("num_hidden_layers", 4)
                    )
                    gate_key = "layers.0.mlp.gate_proj.weight"
                    d_ff = (
                        int(remapped[gate_key].shape[0])
                        if gate_key in remapped
                        else int(d_model) * 2
                    )
                    n_heads = 4 if int(d_model) % 4 == 0 else 2
                    k_key = "layers.0.self_attn.k_proj.weight"
                    head_dim = max(1, int(d_model) // n_heads)
                    n_kv_heads = (
                        max(1, int(remapped[k_key].shape[0]) // head_dim)
                        if k_key in remapped
                        else n_heads
                    )
                    cfg = ModelConfig(
                        name=str(config.get("model_type", "sml_nano")),
                        vocab_size=int(vocab_size),
                        d_model=int(d_model),
                        n_layers=int(n_layers),
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        d_ff=d_ff,
                        max_seq_len=64,
                    )
                    model: Any = Transformer(cfg)
                    model.load_state_dict(remapped, strict=False)
                    model.eval()
                    return model
                except Exception:
                    return None

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(str(release_dir))
        model.eval()
        return model
    except Exception:
        return None


def cleanup_hf_model_cache(model_name: str, cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Delete the local HF Hub cache entry for ``model_name`` (ADR-006 Tier 3).

    Returns ``{'removed': bool, 'path': str | None, 'freed_bytes': int}``.
    Local paths and bare names without an ``org/model`` slash never map to a
    hub cache entry and are reported as untouched.
    """
    if cache_dir is not None:
        hub_dir = Path(cache_dir)
    else:
        env_hub = os.environ.get("HF_HUB_CACHE")
        if env_hub:
            hub_dir = Path(env_hub)
        else:
            hub_dir = (
                Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
            )

    if "/" not in model_name or Path(model_name).is_dir():
        return {"removed": False, "path": None, "freed_bytes": 0}

    target = hub_dir / f"models--{model_name.replace('/', '--')}"
    if not target.is_dir():
        return {"removed": False, "path": None, "freed_bytes": 0}

    freed_bytes = 0
    for path in target.rglob("*"):
        if path.is_file():
            try:
                freed_bytes += path.stat().st_size
            except OSError:  # pragma: no cover
                continue
    shutil.rmtree(target)
    return {"removed": True, "path": str(target), "freed_bytes": freed_bytes}


def export_gguf_recipe(
    release_dir: Path | str,
    input_gguf: Path | str | None = None,
    output_gguf: Path | str | None = None,
    llama_quantize_bin: str | None = None,
    run_kl_check: bool = True,
    tolerance: float = 0.02,
    refresh_bundle_files: bool = True,
) -> dict[str, Any]:
    """Validate and export the CKA GGUF quantization recipe for a release.

    Loads ``cka_quant_recipe.json`` from ``release_dir``, validates it with
    :func:`validate_cka_quant_recipe`, optionally refreshes the Ollama
    ``Modelfile`` + ``quantize_cka_gguf.sh`` bundle pair, and — when a
    ``llama-quantize`` binary and the f16 input GGUF are available — runs the
    mixed-precision quantization plus a uniform Q4_K_M baseline, optionally
    scoring the CKA GGUF against the baseline with
    :func:`evaluate_cka_gguf_kl_vs_uniform`.

    Raises ``FileNotFoundError`` when the recipe is missing and ``ValueError``
    when it fails validation.
    """
    release_dir_path = Path(release_dir)
    recipe_path = release_dir_path / "cka_quant_recipe.json"
    if not recipe_path.is_file():
        raise FileNotFoundError(f"cka_quant_recipe.json not found in {release_dir_path}")
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"failed to parse {recipe_path}: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError(f"{recipe_path} does not contain a JSON recipe object")

    validation: dict[str, Any]
    validation_result = validate_cka_quant_recipe(recipe, tolerance=tolerance)
    if isinstance(validation_result, dict):
        validation = validation_result
        is_valid = bool(validation.get("valid", True))
    else:
        validation = {"valid": True, "result": validation_result}
        is_valid = True
    if not is_valid:
        errors = validation.get("errors") or validation.get("error") or validation
        raise ValueError(f"CKA quant recipe failed validation (tolerance={tolerance}): {errors}")

    release_id = release_dir_path.name
    default_quant = str(recipe.get("default_quant", "Q4_K_M"))
    tensor_type_args = list(generate_llama_cpp_tensor_type_args(recipe))

    in_gguf = (
        Path(input_gguf) if input_gguf is not None else release_dir_path / f"{release_id}-f16.gguf"
    )
    out_gguf = (
        Path(output_gguf)
        if output_gguf is not None
        else release_dir_path / f"{release_id}-cka-q4_k_m.gguf"
    )
    resolved_bin = llama_quantize_bin or shutil.which("llama-quantize")
    llama_quantize_command = build_llama_quantize_command(
        recipe,
        in_gguf,
        out_gguf,
        llama_quantize_bin=str(resolved_bin) if resolved_bin else "llama-quantize",
    )

    modelfile_path: str | None = None
    quantize_script_path: str | None = None
    if refresh_bundle_files:
        written = _write_gguf_recipe_bundle_files(release_dir_path, release_id, recipe)
        modelfile_path = written["modelfile"]
        quantize_script_path = written["quantize_cka_gguf_sh"]
        manifest_path = release_dir_path / "release_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                manifest = None
            if isinstance(manifest, dict):
                artifacts = manifest.get("artifacts")
                if not isinstance(artifacts, dict):
                    artifacts = {}
                    manifest["artifacts"] = artifacts
                artifacts["modelfile"] = modelfile_path
                artifacts["quantize_cka_gguf_sh"] = quantize_script_path
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    gguf_built = False
    uniform_gguf: Path | None = None
    kl_check: Any = None
    if resolved_bin and in_gguf.is_file():
        subprocess.run(
            llama_quantize_command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        gguf_built = True
        uniform_gguf = release_dir_path / f"{release_id}-uniform-q4_k_m.gguf"
        subprocess.run(
            [str(resolved_bin), str(in_gguf), str(uniform_gguf), default_quant],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if run_kl_check:
            eval_model = _load_release_model_for_eval(release_dir_path)
            if eval_model is not None:
                kl_check = evaluate_cka_gguf_kl_vs_uniform(eval_model, recipe)

    return {
        "release_id": release_id,
        "release_dir": str(release_dir_path),
        "validation": validation,
        "estimated_avg_bpw": recipe.get("estimated_avg_bpw"),
        "default_quant": default_quant,
        "q8_0_layers": recipe.get("q8_0_layers"),
        "q6_k_layers": recipe.get("q6_k_layers"),
        "boundary_layers": recipe.get("boundary_layers"),
        "tensor_type_args": tensor_type_args,
        "llama_quantize_command": llama_quantize_command,
        "llama_quantize_command_str": shlex.join(llama_quantize_command),
        "modelfile_path": modelfile_path,
        "quantize_script_path": quantize_script_path,
        "llama_quantize_available": bool(resolved_bin),
        "llama_quantize_bin": str(resolved_bin) if resolved_bin else None,
        "gguf_built": gguf_built,
        "output_gguf": str(out_gguf),
        "uniform_gguf": str(uniform_gguf) if uniform_gguf is not None else None,
        "kl_check": kl_check,
    }


_LORA_SIDE_PATTERN = re.compile(r"lora_([ab])", re.IGNORECASE)


def _load_lora_adapter_state(path: Path | str) -> tuple[dict[str, Any], float]:
    """Load LoRA tensors from a directory, ``.safetensors`` or ``.pt`` file.

    For a directory, ``adapter_model.safetensors`` is required and
    ``adapter_config.json`` (when present) supplies ``lora_alpha`` / ``r`` so
    the PEFT scaling ``alpha / r`` can be recovered. Direct files use a
    scaling of ``1.0``. Returns ``(state_dict, scaling)``.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to apply LoRA adapters")
    adapter_path = Path(path)
    scaling = 1.0

    if adapter_path.is_dir():
        config_path = adapter_path / "adapter_config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                alpha = float(config.get("lora_alpha", 0.0))
                rank = float(config.get("r", 0.0))
                if rank > 0:
                    scaling = alpha / rank
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                scaling = 1.0
        weights_path = adapter_path / "adapter_model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_path}")
        from safetensors.torch import load_file as _safetensors_load_file

        return _safetensors_load_file(str(weights_path)), scaling

    if adapter_path.suffix == ".safetensors":
        from safetensors.torch import load_file as _safetensors_load_file

        return _safetensors_load_file(str(adapter_path)), scaling
    if adapter_path.suffix == ".pt":
        return torch.load(str(adapter_path), map_location="cpu"), scaling
    raise ValueError(f"Unsupported LoRA adapter path: {adapter_path}")


def _extract_lora_deltas(
    state: dict[str, Any],
    scaling: float,
) -> dict[str, tuple[Any, Any, float]]:
    """Map module names to ``(lora_A, lora_B, scaling)`` triples.

    Normalizes PEFT keys (``base_model.model.<module>.lora_A.weight``,
    ``<module>.lora_A.default.weight``) and SML keys
    (``<module>.lora_a`` / ``<module>.lora_a.weight``).
    """
    halves: dict[str, dict[str, Any]] = {}
    for key, tensor in state.items():
        if not hasattr(tensor, "ndim") or getattr(tensor, "ndim", 0) != 2:
            continue
        name = key
        while True:
            if name.startswith("base_model.model."):
                name = name[len("base_model.model.") :]
            elif name.startswith("base_model."):
                name = name[len("base_model.") :]
            else:
                break
        lowered = name.lower()
        match = _LORA_SIDE_PATTERN.search(lowered)
        if match is None:
            continue
        side = match.group(1).lower()
        marker = lowered.find(f"lora_{side}")
        module_name = name[:marker].rstrip(".")
        if not module_name:
            continue
        halves.setdefault(module_name, {})["a" if side == "a" else "b"] = tensor

    deltas: dict[str, tuple[Any, Any, float]] = {}
    for module_name, pair in halves.items():
        lora_a = pair.get("a")
        lora_b = pair.get("b")
        if lora_a is None or lora_b is None:
            continue
        deltas[module_name] = (lora_a, lora_b, scaling)
    return deltas


def _matching_lora_delta(
    deltas: dict[str, tuple[Any, Any, float]],
    module_name: str,
) -> Any | None:
    """Return the fused ``(B @ A) * scaling`` delta for ``module_name``.

    Matches exact module names first, then the longest key that is a
    suffix/prefix relationship of the module path (PEFT keys often omit or
    prepend wrappers such as ``model.``).
    """
    if not deltas:
        return None
    entry = deltas.get(module_name)
    if entry is None:
        best: str | None = None
        for key in deltas:
            if (module_name.endswith("." + key) or key.endswith("." + module_name)) and (
                best is None or len(key) > len(best)
            ):
                best = key
        entry = deltas.get(best) if best is not None else None
    if entry is None:
        return None
    lora_a, lora_b, scaling = entry
    return (lora_b.to(torch.float32) @ lora_a.to(torch.float32)) * scaling


def _apply_lora_adapters_in_place(
    model: Any,
    recipe: Any,
    lora_path: Path | str | None,
    sql_lora_path: Path | str | None,
    preserve_row_norm: bool = True,
) -> dict[str, Any]:
    """Merge LoRA deltas into ``model`` weights, subspace-aware (ADR-006).

    When both the unslopped adapter and the SQL adapter carry a delta for the
    same module, the deltas are fused with :func:`subspace_safe_ties_merge`.
    A lone delta is projected orthogonal to the layer's forbidden subspace
    before being added, and row norms are restored afterwards so the update
    cannot re-leak variance into the forbidden directions.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to apply LoRA adapters")
    from sml.adaptation.lora import _project_delta_orthogonal, _restore_row_norm
    from sml.adaptation.merge import subspace_safe_ties_merge

    unslop_deltas: dict[str, tuple[Any, Any, float]] = {}
    sql_deltas: dict[str, tuple[Any, Any, float]] = {}
    if lora_path is not None:
        state, scaling = _load_lora_adapter_state(lora_path)
        unslop_deltas = _extract_lora_deltas(state, scaling)
    if sql_lora_path is not None:
        state, scaling = _load_lora_adapter_state(sql_lora_path)
        sql_deltas = _extract_lora_deltas(state, scaling)

    basis_per_layer = getattr(recipe, "basis_per_layer", None) or {}
    merged_count = 0
    projected_count = 0

    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight
        if weight is None:
            continue
        delta_unslop = _matching_lora_delta(unslop_deltas, module_name)
        delta_sql = _matching_lora_delta(sql_deltas, module_name)
        if delta_unslop is None and delta_sql is None:
            continue

        layer_match = re.search(r"layers\.(\d+)\.", module_name)
        layer_idx = int(layer_match.group(1)) if layer_match else None
        forbidden_dirs = None
        if layer_idx is not None and weight.shape[0] == recipe.d_model:
            forbidden_dirs = basis_per_layer.get(layer_idx)
            if forbidden_dirs is None:
                forbidden_dirs = basis_per_layer.get(str(layer_idx))

        if delta_unslop is not None and delta_sql is not None:
            merged_result = subspace_safe_ties_merge(
                weight.data,
                delta_sql.to(device=weight.device, dtype=weight.dtype),
                delta_unslop.to(device=weight.device, dtype=weight.dtype),
                forbidden_dirs=forbidden_dirs,
                preserve_row_norm=preserve_row_norm,
            )
            merged_weight = (
                merged_result["merged_weight"] if isinstance(merged_result, dict) else merged_result
            )
            weight.data.copy_(merged_weight.to(device=weight.device, dtype=weight.dtype))
            merged_count += 1
        else:
            delta = delta_sql if delta_unslop is None else delta_unslop
            if forbidden_dirs is not None:
                delta = _project_delta_orthogonal(delta, forbidden_dirs)
            original = weight.data.detach().clone()
            weight.data.add_(delta.to(device=weight.device, dtype=weight.dtype))
            if preserve_row_norm:
                restored = _restore_row_norm(weight.data, original)
                if restored is not None:
                    weight.data.copy_(restored)
            projected_count += 1

    return {
        "applied": bool(merged_count + projected_count > 0),
        "merged_modules": merged_count,
        "projected_modules": projected_count,
        "unslop_targets": len(unslop_deltas),
        "sql_targets": len(sql_deltas),
        "preserve_row_norm": preserve_row_norm,
    }


def _should_use_streaming_shards(
    base_model: str,
    release_id: str,
    backend: str,
    streaming_shards: bool | None,
) -> bool:
    """Decide whether materialization should stream the H13 recipe shard-by-shard.

    Streaming only applies to the ``llm`` backend; an explicit
    ``streaming_shards`` flag wins, otherwise models whose base model or
    release id mentions a large parameter count (``32b`` / ``70b``) default to
    streaming so the full checkpoint never has to fit in memory.
    """
    if backend != "llm":
        return False
    if streaming_shards is not None:
        return bool(streaming_shards)
    lowered = f"{base_model} {release_id}".lower()
    return any(token in lowered for token in ("32b", "70b"))


def _linear_percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile (numpy default convention) of ``values``."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(fraction)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _outlier_sort_score(entry: Any) -> float:
    """Best-effort numeric score used to rank accumulated outlier-layer entries."""
    if isinstance(entry, dict):
        for key in ("crest_factor", "row_crest_factor", "max_row_crest_factor", "value"):
            value = entry.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return 0.0
    if (
        isinstance(entry, (list, tuple))
        and len(entry) > 1
        and isinstance(entry[1], (int, float))
        and not isinstance(entry[1], bool)
    ):
        return float(entry[1])
    return 0.0


def _materialize_streaming_shards(
    *,
    release_id: str,
    recipe: Any,
    resolved_recipe_dir: Path,
    resolved_base: str,
    resolved_backend: str,
    target_dir: Path,
    lora_path: Path | str | None,
    sql_lora_path: Path | str | None,
    fp8_export: bool,
    preserve_row_norm: bool,
    push_to_hub: bool,
    hf_repo_id: str,
    keep_local_weights: bool,
) -> dict[str, Any]:
    """Materialize a Tenet checkpoint shard-by-shard (H13 streaming surgery).

    Instead of loading the whole base model into memory, each safetensors
    shard is staged alone, the streaming Lancet recipe is applied to it, and
    the result is audited and (optionally) pushed before the next shard is
    fetched. Config/tokenizer files are copied straight from the base model
    so the target directory stays a complete HF-style checkpoint.
    """
    if lora_path is not None or sql_lora_path is not None or fp8_export:
        raise ValueError(
            "streaming_shards mode does not currently support LoRA merging or fp8_export"
        )

    from sml.hub.validator import (
        MAX_ABS_WEIGHT_THRESHOLD,
        MAX_ROW_CREST_FACTOR_THRESHOLD,
        _inspect_weights,
    )
    from sml.surgery.subspace import apply_lancet_recipe_streaming

    config_files = ("config.json", "generation_config.json", "model.safetensors.index.json")
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
    )

    api = None
    if push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=hf_repo_id, exist_ok=True, repo_type="model")

    base_path = Path(resolved_base)
    shard_names: list[str] = []
    index_data: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        staging_in = tmp_root / "staging_in"
        staging_out = tmp_root / "staging_out"
        staging_cache = tmp_root / "staging_cache"
        staging_in.mkdir(parents=True, exist_ok=True)
        staging_out.mkdir(parents=True, exist_ok=True)
        staging_cache.mkdir(parents=True, exist_ok=True)

        if base_path.is_dir():
            from sml.surgery.subspace import _resolve_safetensors_shards

            shard_names, index_data = _resolve_safetensors_shards(base_path)
            for file_name in (*config_files, *tokenizer_files):
                source_file = base_path / file_name
                if source_file.is_file():
                    shutil.copy2(source_file, target_dir / file_name)
        else:
            from huggingface_hub import HfApi, hf_hub_download

            repo_files = HfApi().list_repo_files(resolved_base, repo_type="model")
            repo_file_set = set(repo_files)

            def _fetch_repo_file(file_name: str) -> Path | None:
                if file_name not in repo_file_set:
                    return None
                downloaded = hf_hub_download(
                    repo_id=resolved_base,
                    filename=file_name,
                    repo_type="model",
                    cache_dir=str(staging_cache),
                )
                return Path(downloaded)

            index_local = _fetch_repo_file("model.safetensors.index.json")
            if index_local is not None:
                try:
                    index_data = json.loads(index_local.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    index_data = None
                shutil.copy2(index_local, target_dir / "model.safetensors.index.json")
            if isinstance(index_data, dict) and isinstance(index_data.get("weight_map"), dict):
                weight_map = index_data["weight_map"]
                shard_names = list(dict.fromkeys(str(name) for name in weight_map.values()))
            elif "model.safetensors" in repo_file_set:
                shard_names = ["model.safetensors"]
            else:
                shard_names = sorted(
                    name
                    for name in repo_files
                    if re.fullmatch(r"model-\d+-of-\d+\.safetensors", name)
                )
            for file_name in (*config_files, *tokenizer_files):
                if file_name == "model.safetensors.index.json":
                    continue
                local_file = _fetch_repo_file(file_name)
                if local_file is not None:
                    shutil.copy2(local_file, target_dir / file_name)

        config_target = target_dir / "config.json"
        if config_target.is_file():
            try:
                config_payload = json.loads(config_target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                config_payload = None
            if isinstance(config_payload, dict):
                config_changed = False
                if "auto_map" in config_payload:
                    config_payload.pop("auto_map", None)
                    config_changed = True
                if "trust_remote_code" in config_payload:
                    config_payload["trust_remote_code"] = False
                    config_changed = True
                if config_changed:
                    config_target.write_text(
                        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

        tokenizer_config_target = target_dir / "tokenizer_config.json"
        if tokenizer_config_target.is_file():
            try:
                tokenizer_payload = json.loads(tokenizer_config_target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                tokenizer_payload = None
            if isinstance(tokenizer_payload, dict):
                if not tokenizer_payload.get("chat_template"):
                    tokenizer_payload["chat_template"] = _CHAT_TEMPLATE
                    tokenizer_config_target.write_text(
                        json.dumps(tokenizer_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
            else:
                _write_tokenizer_config_json(target_dir, None)
        else:
            _write_tokenizer_config_json(target_dir, None)

        if resolved_recipe_dir.resolve() != target_dir.resolve():
            for artifact_name in (
                "lancet_recipe.json",
                "lancet_basis.safetensors",
                "README.md",
                "cka_quant_recipe.json",
                "openrouter_onboarding.json",
                "OPENROUTER_ONBOARDING.md",
                "release_manifest.json",
                "Modelfile",
                "quantize_cka_gguf.sh",
            ):
                artifact_src = resolved_recipe_dir / artifact_name
                if artifact_src.is_file():
                    shutil.copy2(artifact_src, target_dir / artifact_name)
            pareto_src = resolved_recipe_dir / "assets" / "lancet_vs_naive_pareto.svg"
            if pareto_src.is_file():
                (target_dir / "assets").mkdir(parents=True, exist_ok=True)
                shutil.copy2(pareto_src, target_dir / "assets" / "lancet_vs_naive_pareto.svg")

        shard_checks: dict[str, Any] = {}
        shard_errors: list[str] = []
        accumulated: dict[str, Any] = {
            "max_abs_weight": 0.0,
            "max_row_crest_factor": 0.0,
            "crest_factors": [],
            "outlier_layers": [],
            "no_nan_inf": True,
            "no_unmerged_lora": True,
            "weight_files_checked": 0,
        }
        shards_processed = 0
        tensors_modified = 0
        max_row_norm_drift = 0.0
        max_row_crest_factor = 0.0
        peak_memory_bytes = 0
        peak_disk_shard_bytes = 0

        for shard_name in shard_names:
            if base_path.is_dir():
                shutil.copy2(base_path / shard_name, staging_in / shard_name)
            else:
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    repo_id=resolved_base,
                    filename=shard_name,
                    repo_type="model",
                    local_dir=str(staging_in),
                    cache_dir=str(staging_cache),
                )

            shard_stats = apply_lancet_recipe_streaming(
                staging_in,
                staging_out,
                recipe,
                preserve_row_norm=preserve_row_norm,
                cleanup_input_shards=True,
            )
            if not isinstance(shard_stats, dict):
                shard_stats = {}

            shard_max_crest = float(shard_stats.get("max_row_crest_factor", 0.0) or 0.0)
            if shard_max_crest >= 25.0 or not shard_stats.get("provider_ready"):
                raise RuntimeError(
                    f"streamed shard {shard_name} failed the provider audit: "
                    f"max_row_crest_factor={shard_max_crest}, "
                    f"provider_ready={shard_stats.get('provider_ready')}"
                )

            shards_processed += 1
            tensors_modified += int(shard_stats.get("tensors_modified", 0) or 0)
            max_row_norm_drift = max(
                max_row_norm_drift, float(shard_stats.get("max_row_norm_drift", 0.0) or 0.0)
            )
            peak_memory_bytes = max(
                peak_memory_bytes, int(shard_stats.get("peak_memory_bytes", 0) or 0)
            )
            peak_disk_shard_bytes = max(
                peak_disk_shard_bytes, int(shard_stats.get("peak_disk_shard_bytes", 0) or 0)
            )
            max_row_crest_factor = max(max_row_crest_factor, shard_max_crest)

            out_shard = staging_out / shard_name
            shard_weight_stats = _inspect_weights([out_shard], shard_checks, shard_errors)
            stats_map: dict[str, Any] = {}
            if isinstance(shard_weight_stats, dict):
                stats_map.update(shard_weight_stats)
            for check_key, check_value in shard_checks.items():
                stats_map.setdefault(check_key, check_value)

            accumulated["max_abs_weight"] = max(
                float(accumulated["max_abs_weight"]),
                float(stats_map.get("max_abs_weight", 0.0) or 0.0),
            )
            accumulated["max_row_crest_factor"] = max(
                float(accumulated["max_row_crest_factor"]),
                float(stats_map.get("max_row_crest_factor", 0.0) or 0.0),
            )
            shard_crest_factors = stats_map.get("crest_factors")
            if isinstance(shard_crest_factors, (list, tuple)):
                accumulated["crest_factors"].extend(shard_crest_factors)
            elif stats_map.get("row_crest_factor_p50") is not None:
                accumulated["crest_factors"].append(float(stats_map["row_crest_factor_p50"]))
                if stats_map.get("row_crest_factor_p99") is not None:
                    accumulated["crest_factors"].append(float(stats_map["row_crest_factor_p99"]))
            shard_outliers = stats_map.get("outlier_layers") or stats_map.get(
                "top_3_outlier_layers"
            )
            if isinstance(shard_outliers, (list, tuple)):
                accumulated["outlier_layers"].extend(shard_outliers)
            accumulated["no_nan_inf"] = bool(accumulated["no_nan_inf"]) and bool(
                stats_map.get("no_nan_inf", True)
            )
            accumulated["no_unmerged_lora"] = bool(accumulated["no_unmerged_lora"]) and bool(
                stats_map.get("no_unmerged_lora", True)
            )
            accumulated["weight_files_checked"] += int(
                stats_map.get("weight_files_checked", 1) or 1
            )

            if push_to_hub and api is not None:
                api.upload_file(
                    path_or_fileobj=str(out_shard),
                    path_in_repo=shard_name,
                    repo_id=hf_repo_id,
                    repo_type="model",
                    commit_message=f"Upload streamed shard {shard_name} for {release_id}",
                )

            if keep_local_weights:
                shutil.move(str(out_shard), str(target_dir / shard_name))
            else:
                out_shard.unlink()

            for leftover in list(staging_in.iterdir()):
                if leftover.is_dir() and not leftover.is_symlink():
                    shutil.rmtree(leftover)
                else:
                    leftover.unlink()
            for leftover in list(staging_cache.iterdir()):
                if leftover.is_dir() and not leftover.is_symlink():
                    shutil.rmtree(leftover)
                else:
                    leftover.unlink()
            gc.collect()

    readiness = verify_provider_readiness(target_dir)
    provenance = verify_recipe_provenance(target_dir)
    if not keep_local_weights:
        from sml.hub.validator import (
            MAX_ABS_WEIGHT_THRESHOLD,
            MAX_ROW_CREST_FACTOR_THRESHOLD,
        )

        crest_factors = [float(value) for value in accumulated["crest_factors"]]
        checks = dict(readiness.get("checks") or {})
        checks["no_nan_inf"] = bool(accumulated["no_nan_inf"])
        checks["no_unmerged_lora"] = bool(accumulated["no_unmerged_lora"])
        checks["weight_outliers_ok"] = (
            float(accumulated["max_abs_weight"]) < MAX_ABS_WEIGHT_THRESHOLD
        )
        checks["acceptable_row_crest_factor"] = (
            float(accumulated["max_row_crest_factor"]) < MAX_ROW_CREST_FACTOR_THRESHOLD
        )
        outlier_entries = list(accumulated["outlier_layers"])
        try:
            ranked_outliers = sorted(outlier_entries, key=_outlier_sort_score, reverse=True)
        except TypeError:  # pragma: no cover - defensive
            ranked_outliers = outlier_entries
        readiness["weight_files_checked"] = int(accumulated["weight_files_checked"])
        readiness["max_abs_weight"] = float(accumulated["max_abs_weight"])
        readiness["max_row_crest_factor"] = float(accumulated["max_row_crest_factor"])
        readiness["row_crest_factor_p50"] = _linear_percentile(crest_factors, 0.50)
        readiness["row_crest_factor_p99"] = _linear_percentile(crest_factors, 0.99)
        readiness["top_3_outlier_layers"] = ranked_outliers[:3]
        readiness["checks"] = checks
        readiness["errors"] = list(readiness.get("errors") or []) + list(shard_errors)
        readiness["ready"] = all(bool(value) for value in checks.values())
    (target_dir / "provider_readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if push_to_hub:
        if not readiness.get("ready", False):
            raise RuntimeError(f"Checkpoint failed provider readiness: {readiness.get('errors')}")
        api.upload_folder(
            folder_path=str(target_dir),
            repo_id=hf_repo_id,
            repo_type="model",
            ignore_patterns=["model*.safetensors"],
            commit_message=f"Materialize {release_id} (H13 streaming shards)",
        )

    shards = len(list(target_dir.glob("*.safetensors")))

    return {
        "release_id": release_id,
        "release_dir": str(target_dir),
        "recipe_dir": str(resolved_recipe_dir),
        "hf_repo_id": hf_repo_id,
        "base_model": resolved_base,
        "backend": resolved_backend,
        "ready": bool(readiness.get("ready", False)),
        "provider_readiness": readiness,
        "recipe_provenance": provenance,
        "shards": shards,
        "lora_applied": False,
        "fp8_export": False,
        "pushed": bool(push_to_hub),
        "cache_cleanup": None,
        "streaming_shards": True,
        "shards_processed": shards_processed,
        "tensors_modified": tensors_modified,
        "max_row_norm_drift": max_row_norm_drift,
        "max_row_crest_factor": max_row_crest_factor,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gb": round(peak_memory_bytes / (1024**3), 3),
        "peak_disk_shard_bytes": peak_disk_shard_bytes,
    }


def materialize_tenet_checkpoint(
    release_id: str = "Tenet-3B-Qwen2.5-Unslop-v1",
    *,
    recipe_dir: Path | str | None = None,
    output_dir: Path | str = "results/tenet_materialized",
    base_model: str | None = None,
    backend: str = "auto",
    device: str = "auto",
    dtype: str = "auto",
    lora_path: Path | str | None = None,
    sql_lora_path: Path | str | None = None,
    preserve_row_norm: bool = True,
    fp8_export: bool = False,
    max_shard_size: str = "4GB",
    hf_namespace: str = "mr-andrey-isachenko",
    push_to_hub: bool = False,
    cleanup_cache: bool = True,
    keep_local_weights: bool = True,
    streaming_shards: bool | None = None,
    nano_config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Materialize a provider-ready Tenet checkpoint (ADR-006 Tier 3).

    Loads the base model, applies the H9 Lancet surgery in place, merges the
    subspace-safe LoRA adapters (unslop + optional SQL), optionally quantizes
    to FP8 e4m3, saves the HF-style checkpoint, validates provider readiness
    and recipe provenance, optionally pushes to the Hub and cleans up the
    local HF cache.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to materialize Tenet checkpoints")

    try:
        from sml.surgery.subspace import apply_lancet_recipe, load_lancet_recipe
    except ImportError:  # pragma: no cover
        from sml.adaptation.lora import apply_lancet_recipe  # type: ignore[no-redef]
        from sml.surgery.subspace import load_lancet_recipe

    resolved_recipe_dir = (
        Path(recipe_dir) if recipe_dir is not None else Path("results/tenet_releases") / release_id
    )
    target_dir = Path(output_dir) / release_id
    target_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_lancet_recipe(resolved_recipe_dir)
    resolved_base = base_model or recipe.base_model

    def _config_is_nano(candidate: str) -> bool:
        base_path = Path(candidate)
        config_path = base_path / "config.json"
        if not base_path.is_dir() or not config_path.is_file():
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(config, dict) and config.get("model_type") == "sml_nano"

    if backend != "auto":
        resolved_backend = backend
    elif (
        resolved_base == "sml-nano"
        or nano_config_path is not None
        or _config_is_nano(resolved_base)
    ):
        resolved_backend = "nano"
    else:
        resolved_backend = "llm"

    if _should_use_streaming_shards(resolved_base, release_id, resolved_backend, streaming_shards):
        return _materialize_streaming_shards(
            release_id=release_id,
            recipe=recipe,
            resolved_recipe_dir=resolved_recipe_dir,
            resolved_base=resolved_base,
            resolved_backend=resolved_backend,
            target_dir=target_dir,
            lora_path=lora_path,
            sql_lora_path=sql_lora_path,
            fp8_export=fp8_export,
            preserve_row_norm=preserve_row_norm,
            push_to_hub=push_to_hub,
            hf_repo_id=f"{hf_namespace}/{release_id}",
            keep_local_weights=keep_local_weights,
        )

    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    if resolved_backend == "nano":
        from sml.config import load_model_config
        from sml.model.transformer import Transformer

        cfg = load_model_config(
            Path(nano_config_path)
            if nano_config_path is not None
            else Path("configs/model/sml_nano.yaml")
        )
        torch.manual_seed(2026)
        model: Any = Transformer(cfg)
        base_path = Path(resolved_base)
        if base_path.is_dir() and (base_path / "model.safetensors").is_file():
            from safetensors.torch import load_file as _safetensors_load_file

            weights_state = _safetensors_load_file(str(base_path / "model.safetensors"))
            model.load_state_dict(weights_state, strict=False)
        tokenizer: Any = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype: Any = "auto"
        if dtype != "auto":
            torch_dtype = getattr(torch, str(dtype), None) or dtype
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=False)

    model = model.to(resolved_device)
    model.eval()

    # H9 Lancet surgery, in place.
    apply_lancet_recipe(model, recipe, in_place=True)

    lora_result: dict[str, Any] | None = None
    if lora_path is not None or sql_lora_path is not None:
        lora_result = _apply_lora_adapters_in_place(
            model,
            recipe,
            lora_path,
            sql_lora_path,
            preserve_row_norm=preserve_row_norm,
        )
        # Post-merge Lancet guard: re-assert the forbidden subspaces.
        apply_lancet_recipe(model, recipe, in_place=True)
    lora_applied = bool(lora_result and lora_result.get("applied"))

    if fp8_export:
        from sml.surgery.quantization import quantize_fp8_e4m3

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                quantized = quantize_fp8_e4m3(module.weight.data, per_channel=True)
                if isinstance(quantized, torch.Tensor):
                    module.weight.data.copy_(quantized.to(module.weight.dtype))

    if resolved_backend == "llm":
        if getattr(model.config, "auto_map", None):
            model.config.auto_map = None
        if getattr(model.config, "trust_remote_code", False):
            model.config.trust_remote_code = False
        model.save_pretrained(target_dir, max_shard_size=max_shard_size, safe_serialization=True)
        tokenizer.save_pretrained(target_dir)
        tokenizer_config_path = target_dir / "tokenizer_config.json"
        if tokenizer_config_path.exists():
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            if not tokenizer_config.get("chat_template"):
                tokenizer_config["chat_template"] = (
                    getattr(tokenizer, "chat_template", None) or _CHAT_TEMPLATE
                )
                tokenizer_config_path.write_text(
                    json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        else:
            _write_tokenizer_config_json(target_dir, None)
    else:
        from safetensors.torch import save_file as _safetensors_save_file_local

        state_dict = {
            name: tensor.detach().cpu().to(torch.bfloat16)
            for name, tensor in model.state_dict().items()
        }
        _safetensors_save_file_local(state_dict, str(target_dir / "model.safetensors"))
        nano_config = {
            "architectures": _nano_architectures(release_id),
            "model_type": "sml_nano",
            "torch_dtype": "bfloat16",
            "num_hidden_layers": cfg.n_layers,
            "hidden_size": cfg.d_model,
        }
        (target_dir / "config.json").write_text(
            json.dumps(nano_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_tokenizer_config_json(target_dir, None)

    if resolved_recipe_dir.resolve() != target_dir.resolve():
        for artifact_name in (
            "lancet_recipe.json",
            "lancet_basis.safetensors",
            "README.md",
            "cka_quant_recipe.json",
            "openrouter_onboarding.json",
            "OPENROUTER_ONBOARDING.md",
            "release_manifest.json",
            "Modelfile",
            "quantize_cka_gguf.sh",
        ):
            artifact_src = resolved_recipe_dir / artifact_name
            if artifact_src.is_file():
                shutil.copy2(artifact_src, target_dir / artifact_name)
        pareto_src = resolved_recipe_dir / "assets" / "lancet_vs_naive_pareto.svg"
        if pareto_src.is_file():
            (target_dir / "assets").mkdir(parents=True, exist_ok=True)
            shutil.copy2(pareto_src, target_dir / "assets" / "lancet_vs_naive_pareto.svg")

    readiness = verify_provider_readiness(target_dir)
    provenance = verify_recipe_provenance(target_dir)
    (target_dir / "provider_readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    hf_repo_id = f"{hf_namespace}/{release_id}"
    pushed = False
    if push_to_hub:
        if not readiness.get("ready", False):
            raise RuntimeError(f"Checkpoint failed provider readiness: {readiness['errors']}")
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=hf_repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=str(target_dir),
            repo_id=hf_repo_id,
            repo_type="model",
            commit_message=f"Materialize {release_id} (ADR-006 Tier 3)",
        )
        pushed = True
        if not keep_local_weights:
            for stale in list(target_dir.glob("model*.safetensors")) + list(
                target_dir.glob("model.safetensors.index.json")
            ):
                stale.unlink()

    cache_cleanup: dict[str, Any] | None = None
    if cleanup_cache and resolved_backend == "llm":
        cache_cleanup = cleanup_hf_model_cache(resolved_base)

    shards = len(list(target_dir.glob("*.safetensors")))

    return {
        "release_id": release_id,
        "release_dir": str(target_dir),
        "recipe_dir": str(resolved_recipe_dir),
        "hf_repo_id": hf_repo_id,
        "base_model": resolved_base,
        "backend": resolved_backend,
        "ready": bool(readiness.get("ready", False)),
        "provider_readiness": readiness,
        "recipe_provenance": provenance,
        "shards": shards,
        "lora_applied": lora_applied,
        "fp8_export": fp8_export,
        "pushed": pushed,
        "cache_cleanup": cache_cleanup,
        "streaming_shards": False,
    }
