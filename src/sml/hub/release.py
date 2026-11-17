"""Release bundling for Tenet checkpoints.

Runs the H9 Lancet surgery, H10 subspace-safe GRPO and H11 FP8 provider
audits, writes a provider-ready HF-style checkpoint, and emits the model
card, OpenRouter onboarding manifest and release manifest.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from sml.gateway.client import build_openrouter_onboarding_manifest
from sml.hub.model_card import generate_tenet_model_card
from sml.hub.validator import verify_provider_readiness

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

    tensor_args: list[str] = []
    for assignment in assignments:
        if assignment["quant"] == "Q4_K_M" or not assignment["tensors"]:
            continue
        token = "q8_0" if assignment["quant"] == "Q8_0" else "q6_k"
        for tensor in assignment["tensors"]:
            tensor_args.extend(
                ["--tensor-type", f"blk.{assignment['layer']}.{tensor}.weight={token}"]
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

    if n_layers > 0:
        total_bpw = n_layers * base_bpw
        total_bpw += len(q8_set) * (q8_layer_bpw - base_bpw)
        total_bpw += len(q6_set) * (q6_layer_bpw - base_bpw)
        estimated_avg_bpw = total_bpw / n_layers
    elif q8_set or q6_set:
        protected_count = len(q8_set) + len(q6_set)
        estimated_avg_bpw = (
            len(q8_set) * q8_layer_bpw + len(q6_set) * q6_layer_bpw
        ) / protected_count
    else:
        estimated_avg_bpw = base_bpw

    return {
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
        "llama_cpp_tensor_type_args": tensor_args,
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
        "estimated_avg_bpw": round(estimated_avg_bpw, 3),
        "scr_ranking_used": bool(scr_by_layer),
    }


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
) -> dict[str, Any]:
    """Build a full provider-ready Tenet release bundle.

    Runs the H9/H10/H11 benchmarks for ``backend`` (``"nano"`` or
    ``"llm"``), writes an HF-style checkpoint into
    ``Path(output_dir) / release_id``, validates it with
    :func:`verify_provider_readiness`, and writes the model card, the
    OpenRouter onboarding manifest and the release manifest.
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

    card_metrics = {**h9_res, **h10_res, **manifest_h11}
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
