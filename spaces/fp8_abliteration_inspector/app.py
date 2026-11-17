#!/usr/bin/env python3
"""FP8 Abliteration Inspector — a self-contained Hugging Face Space.

Audits abliterated checkpoints / Lancet recipe bundles for FP8 quantization
safety without loading a whole model into RAM:

* streams ``*.safetensors`` tensor-by-tensor with ``safetensors.safe_open``
* measures max |weight| and the per-tensor **row crest factor**
  ``max(|w|_row / rms(w_row))`` — naive abliteration rows that are dominated
  by one injected direction blow this up and make FP8 (per-tensor scales)
  clip the refusal direction back in ("re-leakage")
* verifies the provenance of ``lancet_recipe.json`` +
  ``lancet_basis.safetensors`` (schema v1.0, unit-norm basis rows)
* mirrors the 7 ``verify_provider_readiness`` checks from
  ``src/sml/hub/validator.py``

The module imports cleanly without gradio / torch / safetensors installed:
every pure function below stays usable and the UI simply degrades.
"""

from __future__ import annotations

import contextlib
import html
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

try:  # torch is optional: stats fall back to pure numpy
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised on lean Spaces images
    torch = None
    HAS_TORCH = False

try:  # safetensors is required only for real inspection work
    from safetensors import safe_open

    HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover
    safe_open = None
    HAS_SAFETENSORS = False

try:  # gradio is optional so pure functions work in plain CI runs
    import gradio as gr

    HAS_GRADIO = True
except ImportError:  # pragma: no cover
    gr = None
    HAS_GRADIO = False

# ---------------------------------------------------------------------------
# Constants (kept in sync with src/sml/hub/validator.py)
# ---------------------------------------------------------------------------
STANDARD_ARCHITECTURES = {"LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM"}
MAX_ABS_WEIGHT_THRESHOLD = 50.0
MAX_ROW_CREST_FACTOR_THRESHOLD = 25.0
MAX_BUNDLE_BYTES = 15 * 1024**3  # ADR-003 <15 GB limit
RECIPE_FILENAME = "lancet_recipe.json"
RECIPE_BASIS_FILENAME = "lancet_basis.safetensors"
RECIPE_SCHEMA_VERSION = "1.0"
RECIPE_BASIS_NORM_TOLERANCE = 1e-4

_CHECK_NAMES = (
    "standard_architecture",
    "no_trust_remote_code",
    "valid_chat_template",
    "no_nan_inf",
    "no_unmerged_lora",
    "weight_outliers_ok",
    "acceptable_row_crest_factor",
)
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


# ---------------------------------------------------------------------------
# Small helpers (torch-or-numpy dual backend)
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_torch_tensor(t: Any) -> bool:
    return HAS_TORCH and isinstance(t, torch.Tensor)


def _is_floating(t: Any) -> bool:
    if _is_torch_tensor(t):
        return bool(t.dtype.is_floating_point)
    return bool(np.issubdtype(np.asarray(t).dtype, np.floating))


def _ndim(t: Any) -> int:
    return len(t.shape)


def _all_finite(t: Any) -> bool:
    if _is_torch_tensor(t):
        return bool(torch.isfinite(t).all().item())
    return bool(np.isfinite(np.asarray(t)).all())


def _abs_max(t: Any) -> float:
    """max |t| via min/max (cheaper than materialising an abs() copy)."""
    if _is_torch_tensor(t):
        lo, hi = float(t.min().item()), float(t.max().item())
    else:
        a = np.asarray(t)
        lo, hi = float(np.min(a)), float(np.max(a))
    return max(abs(lo), abs(hi))


def _row_crest_factor(t: Any) -> float:
    """Row crest factor of a 2D float tensor.

    row_rms  = sqrt(mean(w^2, dim=1) + 1e-8)
    row_crest = max(abs(w).amax(dim=1) / row_rms)
    """
    if _is_torch_tensor(t):
        w = t.to(torch.float32)
        row_rms = torch.sqrt(torch.mean(w * w, dim=1) + 1e-8)
        return float((w.abs().amax(dim=1) / row_rms).max().item())
    w = np.asarray(t, dtype=np.float32)
    row_rms = np.sqrt(np.mean(w * w, axis=1) + 1e-8)
    return float(np.max(np.max(np.abs(w), axis=1) / row_rms))


def _row_norm_errors(t: Any) -> Any:
    """|‖row‖ - 1| for a 1D/2D tensor (array of per-row errors)."""
    if _is_torch_tensor(t):
        w = t.to(torch.float32)
        if _ndim(t) == 1:
            norms = torch.sqrt(torch.sum(w * w))
            return torch.abs(norms - 1.0)
        if _ndim(t) == 2:
            norms = torch.sqrt(torch.sum(w * w, dim=1))
            return torch.abs(norms - 1.0)
        return None
    w = np.asarray(t, dtype=np.float32)
    if w.ndim == 1:
        return np.abs(np.sqrt(np.sum(w * w)) - 1.0)[None]
    if w.ndim == 2:
        return np.abs(np.sqrt(np.sum(w * w, axis=1)) - 1.0)
    return None


def _layer_of(name: str) -> int:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else -1


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _iter_tensors(path: Path):
    """Yield (name, tensor) one tensor at a time — never the whole file."""
    if not HAS_SAFETENSORS:
        raise RuntimeError("safetensors is not installed")
    framework = "pt" if HAS_TORCH else "np"
    with safe_open(str(path), framework=framework, device="cpu") as handle:
        for key in list(handle.keys()):
            yield key, handle.get_tensor(key)


def _meta_lookup(meta: dict[str, Any], key: str) -> Any:
    for data in meta.values():
        if isinstance(data, dict) and key in data:
            return data[key]
        if isinstance(data, dict):
            for sub in data.values():
                if isinstance(sub, dict) and key in sub:
                    return sub[key]
    return None


def _recipe_lookup(recipe: dict[str, Any], key: str) -> Any:
    if key in recipe:
        return recipe[key]
    for container in ("config", "model", "base", "base_config", "target", "metadata", "meta"):
        sub = recipe.get(container)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    return None


# ---------------------------------------------------------------------------
# Recipe provenance verification (lancet_recipe.json + lancet_basis.safetensors)
# ---------------------------------------------------------------------------
def verify_recipe_provenance(checkpoint_dir: Path | str) -> dict[str, Any]:
    """Verify a Lancet recipe bundle.

    Checks that ``lancet_recipe.json`` declares schema version 1.0, that
    ``lancet_basis.safetensors`` holds finite tensors whose rows are unit
    norm within ``RECIPE_BASIS_NORM_TOLERANCE`` (1e-4), and that
    ``config.json`` ``hidden_size`` / ``num_hidden_layers`` agree with the
    recipe and with the basis tensor shapes.
    """
    d = Path(checkpoint_dir)
    res: dict[str, Any] = {
        "checkpoint_dir": str(d),
        "recipe_present": False,
        "recipe_schema_version": None,
        "recipe_schema_ok": False,
        "basis_present": False,
        "basis_bytes": None,
        "basis_tensors_checked": 0,
        "basis_max_norm_error": None,
        "basis_rows_unit_norm": False,
        "basis_finite": True,
        "basis_shapes": [],
        "config_present": False,
        "config_hidden_size": None,
        "config_num_hidden_layers": None,
        "hidden_size": None,
        "num_hidden_layers": None,
        "config_match": None,
        "ok": False,
        "errors": [],
    }
    errors = res["errors"]
    if not HAS_SAFETENSORS:
        errors.append("safetensors is not installed; cannot verify the recipe basis")
        return res
    if not d.is_dir():
        errors.append(f"checkpoint directory not found: {d}")
        return res

    # -- 1) recipe JSON schema ------------------------------------------------
    recipe_path = d / RECIPE_FILENAME
    if not recipe_path.is_file():
        errors.append(f"missing {RECIPE_FILENAME}")
    else:
        raw = _read_json(recipe_path)
        if not isinstance(raw, dict):
            errors.append(f"{RECIPE_FILENAME} must contain a JSON object")
        else:
            res["recipe_present"] = True
            version = str(raw.get("schema_version", raw.get("version", "")))
            res["recipe_schema_version"] = version
            if version == RECIPE_SCHEMA_VERSION:
                res["recipe_schema_ok"] = True
            else:
                errors.append(
                    f"unsupported {RECIPE_FILENAME} schema_version {version!r} "
                    f"(expected {RECIPE_SCHEMA_VERSION!r})"
                )
            res["hidden_size"] = _recipe_lookup(raw, "hidden_size")
            res["num_hidden_layers"] = _recipe_lookup(raw, "num_hidden_layers")

    # -- 2) basis safetensors: finite + unit-norm rows ------------------------
    basis_path = d / RECIPE_BASIS_FILENAME
    if not basis_path.is_file():
        errors.append(f"missing {RECIPE_BASIS_FILENAME}")
    else:
        res["basis_present"] = True
        res["basis_bytes"] = basis_path.stat().st_size
        max_err = 0.0
        finite = True
        framework = "pt" if HAS_TORCH else "np"
        try:
            with safe_open(str(basis_path), framework=framework, device="cpu") as handle:
                for key in list(handle.keys()):
                    t = handle.get_tensor(key)
                    res["basis_tensors_checked"] += 1
                    res["basis_shapes"].append(
                        {"name": str(key), "shape": [int(x) for x in t.shape]}
                    )
                    if not _all_finite(t):
                        finite = False
                        continue
                    errs = _row_norm_errors(t)
                    if errs is not None:
                        max_err = max(max_err, float(np.max(np.asarray(errs, dtype=np.float64))))
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"failed to read {RECIPE_BASIS_FILENAME}: {exc}")
            finite = False
        res["basis_finite"] = finite
        res["basis_max_norm_error"] = max_err
        res["basis_rows_unit_norm"] = (
            res["basis_tensors_checked"] > 0 and max_err <= RECIPE_BASIS_NORM_TOLERANCE
        )
        if not finite:
            errors.append(f"{RECIPE_BASIS_FILENAME} contains non-finite values")
        if res["basis_tensors_checked"] == 0:
            errors.append(f"{RECIPE_BASIS_FILENAME} contains no tensors")
        elif not res["basis_rows_unit_norm"]:
            errors.append(
                f"{RECIPE_BASIS_FILENAME} rows are not unit norm: max |‖row‖-1| = {max_err:.3e} "
                f"> tolerance {RECIPE_BASIS_NORM_TOLERANCE:.0e}"
            )

    # -- 3) config.json cross-check -------------------------------------------
    cfg = _read_json(d / "config.json")
    if not isinstance(cfg, dict):
        errors.append("missing config.json (cannot cross-check hidden_size / num_hidden_layers)")
    else:
        res["config_present"] = True
        hs, nl = cfg.get("hidden_size"), cfg.get("num_hidden_layers")
        res["config_hidden_size"] = hs
        res["config_num_hidden_layers"] = nl
        match = True
        for label, recipe_val, config_val in (
            ("hidden_size", res["hidden_size"], hs),
            ("num_hidden_layers", res["num_hidden_layers"], nl),
        ):
            if recipe_val is not None and config_val is not None:
                try:
                    if int(recipe_val) != int(config_val):
                        match = False
                        errors.append(
                            f"config.json {label}={config_val} disagrees with "
                            f"{RECIPE_FILENAME} {label}={recipe_val}"
                        )
                except (TypeError, ValueError):
                    match = False
                    errors.append(f"non-integer {label} in recipe/config")
        # basis directions must live in the model hidden space
        if res["config_hidden_size"] is not None:
            for entry in res["basis_shapes"]:
                shape = entry["shape"]
                if shape and int(shape[-1]) != int(res["config_hidden_size"]):
                    match = False
                    errors.append(
                        f"basis tensor {entry['name']} shape {shape} does not end in "
                        f"hidden_size={res['config_hidden_size']}"
                    )
        res["config_match"] = match

    res["ok"] = bool(
        res["recipe_present"]
        and res["recipe_schema_ok"]
        and res["basis_present"]
        and res["basis_rows_unit_norm"]
        and res["basis_finite"]
        and res["config_match"] is True
        and not errors
    )
    return res


# ---------------------------------------------------------------------------
# Streaming inspection of a checkpoint directory or single weight bundle
# ---------------------------------------------------------------------------
def inspect_directory_or_bundle(path: Path | str) -> dict[str, Any]:
    """Inspect a checkpoint dir / bundle for FP8 readiness without loading it.

    Streams every model ``*.safetensors`` tensor-by-tensor (the Lancet basis
    file is skipped from model weights), measures max |weight| and row crest
    factors, mirrors the 7 ``verify_provider_readiness`` checks plus recipe
    provenance, and falls back to metadata JSONs when no weight files exist.
    """
    p = Path(path)
    result: dict[str, Any] = {
        "path": str(p),
        "mode": "weights",
        "ready": False,
        "fp8_verdict": "N/A",
        "checks": {},
        "errors": [],
        "architectures": [],
        "max_abs_weight": None,
        "max_row_crest_factor": None,
        "row_crest_factor_p50": None,
        "row_crest_factor_p99": None,
        "top_3_outlier_layers": [],
        "layer_crest_profile": {},
        "scr_profile": {},
        "recipe_provenance": None,
        "total_weight_bytes": 0,
        "weight_files_checked": 0,
        "tensors_checked": 0,
        "svg_chart": None,
    }
    checks = result["checks"]
    errors = result["errors"]

    def _fail_all(reason: str, mode: str) -> dict[str, Any]:
        errors.append(reason)
        result["mode"] = mode
        for name in _CHECK_NAMES:
            checks[name] = False
        result["svg_chart"] = _render_inspection_svg(result)
        return result

    if not p.exists():
        return _fail_all(f"path does not exist: {p}", "missing")

    # -- metadata -------------------------------------------------------------
    config: dict[str, Any] = {}
    tokenizer_cfg: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    if p.is_dir():
        raw_cfg = _read_json(p / "config.json")
        if isinstance(raw_cfg, dict):
            config = raw_cfg
        raw_tok = _read_json(p / "tokenizer_config.json")
        if isinstance(raw_tok, dict):
            tokenizer_cfg = raw_tok
        for fname in (
            "provider_readiness.json",
            RECIPE_FILENAME,
            "final_metrics.json",
            "release_manifest.json",
            "openrouter_onboarding.json",
        ):
            data = _read_json(p / fname)
            if data is not None:
                meta[fname] = data
    readiness = meta.get("provider_readiness.json")
    if not isinstance(readiness, dict):
        readiness = {}

    # -- locate weight files ----------------------------------------------------
    if p.is_dir():
        all_st = sorted(p.glob("*.safetensors"))
    elif p.is_file() and p.suffix == ".safetensors":
        all_st = [p]
    else:
        all_st = []
    model_files = [f for f in all_st if f.name != RECIPE_BASIS_FILENAME]
    total_all = sum(f.stat().st_size for f in all_st)
    result["total_weight_bytes"] = sum(f.stat().st_size for f in model_files)
    result["weight_files_checked"] = len(model_files)

    if total_all >= MAX_BUNDLE_BYTES:
        return _fail_all(
            f"weight files total {total_all / 1024**3:.2f} GB which violates the ADR-003 "
            f"limit (bundle must stay < {MAX_BUNDLE_BYTES / 1024**3:.0f} GB)",
            "oversized",
        )

    # -- recipe provenance (when the recipe is present) ------------------------
    recipe_dir = p if p.is_dir() else p.parent
    if (recipe_dir / RECIPE_FILENAME).is_file():
        prov = verify_recipe_provenance(recipe_dir)
        result["recipe_provenance"] = prov
        checks["recipe_provenance"] = bool(prov.get("ok"))
        for err in prov.get("errors", []):
            errors.append(f"recipe_provenance: {err}")

    # -- streaming weight scan ---------------------------------------------------
    crest_values: list[float] = []
    crest_records: list[dict[str, Any]] = []
    nan_tensors: list[str] = []
    lora_tensors: list[str] = []
    layer_crest: dict[int, float] = {}
    max_abs: float | None = None
    max_crest: float | None = None

    if model_files and not HAS_SAFETENSORS:
        errors.append("safetensors is not installed; cannot stream weight tensors")
    elif model_files:
        for wf in model_files:
            try:
                for name, t in _iter_tensors(wf):
                    result["tensors_checked"] += 1
                    if ".lora_" in name.lower():
                        lora_tensors.append(name)
                    if not _is_floating(t):
                        continue
                    if not _all_finite(t):
                        nan_tensors.append(name)
                        continue
                    m = _abs_max(t)
                    if math.isfinite(m):
                        max_abs = m if max_abs is None else max(max_abs, m)
                    if _ndim(t) == 2 and int(t.shape[0]) > 0:
                        c = _row_crest_factor(t)
                        crest_values.append(c)
                        max_crest = c if max_crest is None else max(max_crest, c)
                        layer = _layer_of(name)
                        if layer >= 0:
                            layer_crest[layer] = max(layer_crest.get(layer, 0.0), c)
                        crest_records.append(
                            {
                                "name": str(name),
                                "layer": layer,
                                "row_crest_factor": round(c, 6),
                                "max_abs_weight": round(m, 6),
                                "shape": [int(x) for x in t.shape],
                            }
                        )
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"failed to read {wf.name}: {exc}")

    if crest_values:
        result["max_abs_weight"] = round(float(max_abs), 6) if max_abs is not None else None
        result["max_row_crest_factor"] = round(float(max_crest), 6)
        result["row_crest_factor_p50"] = round(float(np.percentile(crest_values, 50)), 6)
        result["row_crest_factor_p99"] = round(float(np.percentile(crest_values, 99)), 6)
        result["top_3_outlier_layers"] = sorted(
            crest_records, key=lambda r: r["row_crest_factor"], reverse=True
        )[:3]
        result["layer_crest_profile"] = {
            str(k): round(v, 6) for k, v in sorted(layer_crest.items())
        }
        # SCR profile: share of the FP8 crest budget consumed per layer
        # (1.0 == sitting exactly on the 25.0 threshold).
        result["scr_profile"] = {
            str(k): round(min(v / MAX_ROW_CREST_FACTOR_THRESHOLD, 1.999), 6)
            for k, v in sorted(layer_crest.items())
        }
    else:
        # -- metadata fallback (no measurable model weights) --------------------
        result["mode"] = "metadata_fallback"
        for key, cast in (
            ("max_abs_weight", float),
            ("max_row_crest_factor", float),
            ("row_crest_factor_p50", float),
            ("row_crest_factor_p99", float),
        ):
            val = _meta_lookup(meta, key)
            if val is not None:
                with contextlib.suppress(TypeError, ValueError):
                    result[key] = round(cast(val), 6)
        for key in ("top_3_outlier_layers", "layer_crest_profile", "scr_profile"):
            val = _meta_lookup(meta, key)
            if isinstance(val, (list, dict)):
                result[key] = val
        if result["max_row_crest_factor"] is None:
            errors.append(
                "no .safetensors model weights found (only "
                + ", ".join(sorted(meta))
                + ") and metadata carries no max_row_crest_factor"
            )

    # -- the 7 verify_provider_readiness checks ---------------------------------
    archs = list(config.get("architectures") or [])
    if not archs:
        md_archs = readiness.get("architectures") or _meta_lookup(meta, "architectures")
        if isinstance(md_archs, (list, tuple)):
            archs = list(md_archs)
    result["architectures"] = [str(a) for a in archs]
    checks["standard_architecture"] = any(
        a in STANDARD_ARCHITECTURES for a in result["architectures"]
    )
    if not checks["standard_architecture"]:
        errors.append(
            "no standard architecture in config.json (expected one of "
            + ", ".join(sorted(STANDARD_ARCHITECTURES))
            + ")"
        )

    trc = config.get("trust_remote_code")
    if trc is None:
        trc = readiness.get("trust_remote_code", _meta_lookup(meta, "trust_remote_code"))
    checks["no_trust_remote_code"] = trc is not True
    if trc is True:
        errors.append(
            "config.json sets trust_remote_code=true; arbitrary code execution is forbidden"
        )

    template = config.get("chat_template") or tokenizer_cfg.get("chat_template")
    has_jinja = p.is_dir() and any(
        (p / n).is_file() for n in ("chat_template.jinja", "chat_template.json")
    )
    checks["valid_chat_template"] = (
        bool(isinstance(template, str) and template.strip()) or has_jinja
    )
    if not checks["valid_chat_template"]:
        errors.append(
            "no valid chat_template found "
            "(config.json / tokenizer_config.json / chat_template.jinja)"
        )

    if nan_tensors:
        shown = ", ".join(nan_tensors[:5])
        extra = f" (+{len(nan_tensors) - 5} more)" if len(nan_tensors) > 5 else ""
        errors.append(f"non-finite (NaN/Inf) values in: {shown}{extra}")
    elif result["tensors_checked"] > 0:
        pass  # every floating tensor was finite
    else:
        flag = readiness.get("no_nan_inf", _meta_lookup(meta, "no_nan_inf"))
        if flag is None:
            flag = _meta_lookup(meta, "nan_inf_ok")
        checks["no_nan_inf"] = bool(flag)
        if not flag:
            errors.append("no weight tensors measured and metadata does not confirm no_nan_inf")
    if result["tensors_checked"] > 0 or nan_tensors:
        checks["no_nan_inf"] = not nan_tensors

    adapter_files = []
    if p.is_dir():
        adapter_files = sorted(f.name for f in p.iterdir() if f.name.lower().startswith("adapter_"))
    checks["no_unmerged_lora"] = not adapter_files and not lora_tensors
    if adapter_files or lora_tensors:
        errors.append(
            "unmerged LoRA adapters detected: " + ", ".join((adapter_files + lora_tensors)[:5])
        )

    if result["max_abs_weight"] is not None:
        checks["weight_outliers_ok"] = result["max_abs_weight"] < MAX_ABS_WEIGHT_THRESHOLD
        if not checks["weight_outliers_ok"]:
            errors.append(
                f"max_abs_weight {result['max_abs_weight']} exceeds {MAX_ABS_WEIGHT_THRESHOLD}"
            )
    else:
        checks["weight_outliers_ok"] = False
        errors.append("no weight magnitude measurement available for weight_outliers_ok")

    if result["max_row_crest_factor"] is not None:
        checks["acceptable_row_crest_factor"] = (
            result["max_row_crest_factor"] < MAX_ROW_CREST_FACTOR_THRESHOLD
        )
        if not checks["acceptable_row_crest_factor"]:
            errors.append(
                f"max_row_crest_factor {result['max_row_crest_factor']} exceeds "
                f"{MAX_ROW_CREST_FACTOR_THRESHOLD}; "
                "FP8 per-tensor scales would re-leak the refusal direction"
            )
    else:
        checks["acceptable_row_crest_factor"] = False
        errors.append("no row crest factor measurement available for acceptable_row_crest_factor")

    if result["max_row_crest_factor"] is not None:
        result["fp8_verdict"] = (
            "PASS" if result["max_row_crest_factor"] < MAX_ROW_CREST_FACTOR_THRESHOLD else "FAIL"
        )

    result["ready"] = bool(checks) and all(checks.values())
    result["svg_chart"] = _render_inspection_svg(result)
    return result


# ---------------------------------------------------------------------------
# Inline SVG rendering for live inspection results
# ---------------------------------------------------------------------------
def _render_inspection_svg(m: dict[str, Any]) -> str:
    """Inline SVG: row crest-factor distribution + per-layer SCR profile."""
    W, H = 920, 430
    verdict = str(m.get("fp8_verdict") or "N/A")
    verdict_color = {"PASS": "#10b981", "FAIL": "#ef4444"}.get(verdict, "#64748b")
    title_prefix = "FP8 Row Crest-Factor audit — verdict: "
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        'font-family="ui-sans-serif,system-ui,Segoe UI,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff" rx="10"/>',
        f'<text x="24" y="30" font-size="17" font-weight="700" fill="#0f172a">'
        f"{title_prefix}{_esc(verdict)}</text>",
        f'<circle cx="{24 + 7.5 * len(title_prefix) + 14}" cy="24" r="9" fill="{verdict_color}"/>',
    ]
    profile = {int(k): float(v) for k, v in (m.get("layer_crest_profile") or {}).items()}
    layers = sorted(layer for layer in profile if layer >= 0)

    if not layers:
        parts.append(
            '<text x="24" y="90" font-size="14" fill="#64748b">'
            "No row crest-factor measurements available.</text>"
        )
        for i, err in enumerate(list(m.get("errors", []))[:4]):
            parts.append(
                f'<text x="24" y="{118 + i * 20}" font-size="12" fill="#b91c1c">'
                f"{_esc('• ' + err[:120])}</text>"
            )
        parts.append("</svg>")
        return "".join(parts)

    crest_vals = [profile[layer] for layer in layers]
    # ---- left panel: crest-factor distribution -------------------------------
    Lx, Ly, Lw, Lh = 52, 70, 360, 280
    base_y = Ly + Lh
    ymax = max(max(crest_vals), MAX_ROW_CREST_FACTOR_THRESHOLD + 2.0)
    nbins = max(6, int(math.ceil(ymax / 2.0)))
    counts = [0] * nbins
    for v in crest_vals:
        counts[min(int(v / (ymax / nbins)), nbins - 1)] += 1
    peak = max(max(counts), 1)
    bin_w = Lw / nbins
    for i, c in enumerate(counts):
        if not c:
            continue
        bh = (c / peak) * (Lh - 20)
        parts.append(
            f'<rect x="{Lx + i * bin_w + 1:.1f}" y="{base_y - bh:.1f}" width="{bin_w - 2:.1f}" '
            f'height="{bh:.1f}" fill="#94a3b8" opacity="0.85"/>'
        )
    parts.append(f'<line x1="{Lx}" y1="{base_y}" x2="{Lx + Lw}" y2="{base_y}" stroke="#334155"/>')
    thr_x = Lx + (MAX_ROW_CREST_FACTOR_THRESHOLD / ymax) * Lw
    parts.append(
        f'<line x1="{thr_x:.1f}" y1="{Ly}" x2="{thr_x:.1f}" y2="{base_y}" stroke="#ef4444" '
        'stroke-width="2" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{thr_x + 6:.1f}" y="{Ly + 14}" font-size="12" fill="#ef4444" font-weight="600">'
        f"FP8 limit {MAX_ROW_CREST_FACTOR_THRESHOLD:.1f}</text>"
    )
    parts.append(
        f'<text x="{Lx}" y="{Ly + 14}" font-size="12" fill="#475569">'
        "per-tensor row crest factor (histogram)</text>"
    )
    parts.append(f'<text x="{Lx}" y="{base_y + 18}" font-size="11" fill="#64748b">0</text>')
    parts.append(
        f'<text x="{Lx + Lw - 30:.0f}" y="{base_y + 18}" font-size="11" '
        f'fill="#64748b">{ymax:.0f}</text>'
    )

    # ---- right panel: SCR layer profile ---------------------------------------
    Rx, Ry, Rw, Rh = 480, 70, 412, 280
    rbase = Ry + Rh
    scr = {int(k): float(v) for k, v in (m.get("scr_profile") or {}).items() if int(k) >= 0}
    top3 = {int(r.get("layer", -1)) for r in m.get("top_3_outlier_layers", [])}
    scr_max = max(max(scr.values(), default=1.0), 1.05)
    n = max(len(layers), 1)
    bar_w = Rw / n
    for i, layer in enumerate(layers):
        v = scr.get(layer, profile[layer] / MAX_ROW_CREST_FACTOR_THRESHOLD)
        bh = min(v / scr_max, 0.999) * Rh
        color = "#ef4444" if layer in top3 else "#0ea5e9"
        parts.append(
            f'<rect x="{Rx + i * bar_w + 1:.1f}" y="{rbase - bh:.1f}" '
            f'width="{max(bar_w - 2, 1):.1f}" height="{bh:.1f}" fill="{color}" opacity="0.9"/>'
        )
        if layer % 5 == 0:
            parts.append(
                f'<text x="{Rx + i * bar_w:.1f}" y="{rbase + 14}" '
                f'font-size="10" fill="#64748b">{layer}</text>'
            )
    thr_y = rbase - min(1.0 / scr_max, 0.999) * Rh
    parts.append(
        f'<line x1="{Rx}" y1="{thr_y:.1f}" x2="{Rx + Rw}" y2="{thr_y:.1f}" stroke="#ef4444" '
        'stroke-width="2" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{Rx + 4:.0f}" y="{thr_y - 6:.1f}" font-size="12" '
        'fill="#ef4444" font-weight="600">SCR = 1.0 (crest 25.0)</text>'
    )
    parts.append(
        f'<text x="{Rx}" y="{Ry + 14}" font-size="12" fill="#475569">'
        "SCR layer profile (red = top-3 outliers)</text>"
    )

    mx = m.get("max_row_crest_factor")
    parts.append(
        f'<text x="24" y="{H - 24}" font-size="12" fill="#334155">'
        f"max_row_crest_factor: {_esc(mx)} · p50: {_esc(m.get('row_crest_factor_p50'))} · "
        f"p99: {_esc(m.get('row_crest_factor_p99'))} · tensors: {m.get('tensors_checked', 0)} · "
        f"weight files: {m.get('weight_files_checked', 0)}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Live demo preset — real Qwen2.5-3B measurements (Naive vs Tenet Lancet)
# ---------------------------------------------------------------------------
DEMO_PRESET: dict[str, Any] = {
    "model": "Qwen2.5-3B · Qwen/Qwen2.5-3B-Instruct",
    "naive": {
        "label": "Naive Abliteration",
        "color": "#ef4444",
        "h9_kl": 2.6353,
        "h11_fp8_kl": 0.0975,
        "fp8_releakage": 0.5003,
        "fp8_retention": None,
        "max_row_crest_factor": 25.0,
        "max_row_crest_factor_display": "≥ 25.0",
        "fp8_verdict": "FAIL",
        "top_scr_layers": [21, 15, 19],
    },
    "tenet": {
        "label": "Tenet-3B-Qwen2.5-Unslop-v1 (Lancet)",
        "color": "#10b981",
        "h9_kl": 0.0244,
        "h11_fp8_kl": 0.0469,
        "fp8_releakage": 0.0270,
        "fp8_retention": 0.9541,
        "max_row_crest_factor": 17.27,
        "max_row_crest_factor_display": "17.27",
        "fp8_verdict": "PASS",
        "top_scr_layers": [21, 15, 19],
    },
}


def render_demo_svg() -> str:
    """Inline SVG comparing the two Qwen2.5-3B ablation methods."""
    W, H = 920, 580
    naive, tenet = DEMO_PRESET["naive"], DEMO_PRESET["tenet"]
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        'font-family="ui-sans-serif,system-ui,Segoe UI,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff" rx="10"/>',
        f'<text x="24" y="30" font-size="18" font-weight="700" fill="#0f172a">'
        f"Naive Abliteration vs Tenet Lancet — {_esc(DEMO_PRESET['model'])}</text>",
        '<text x="24" y="52" font-size="12" fill="#64748b">'
        "KL bars are √-scaled for visibility · lower is better except retention</text>",
    ]

    def pair_metric(x: int, y: int, w: int, h: int, title: str, vmax: float, fmt: str) -> None:
        parts.append(
            f'<text x="{x}" y="{y - 10}" font-size="13" font-weight="600" '
            f'fill="#334155">{_esc(title)}</text>'
        )
        for i, side in enumerate((naive, tenet)):
            val = float(side[fmt["key"]])
            frac = (val / vmax) ** 0.5
            bh = frac * h
            by = y + h - bh
            bw = w / 2 - 14
            bx = x + 4 + i * (w / 2)
            parts.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                f'fill="{side["color"]}" opacity="0.9" rx="4"/>'
            )
            label_text = fmt["naive_text"] if side["color"] == "#ef4444" else fmt["tenet_text"]
            parts.append(
                f'<text x="{bx + bw / 2:.1f}" y="{by - 6:.1f}" font-size="12" '
                f'font-weight="700" fill="#0f172a" text-anchor="middle">{label_text}</text>'
            )
        parts.append(f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#334155"/>')

    pair_metric(
        50,
        120,
        250,
        170,
        "H9 KL divergence (BF16)",
        3.0,
        {
            "key": "h9_kl",
            "naive_text": f"{naive['h9_kl']:.4f}",
            "tenet_text": f"{tenet['h9_kl']:.4f}",
        },
    )
    pair_metric(
        345,
        120,
        250,
        170,
        "H11 KL divergence (FP8)",
        0.12,
        {
            "key": "h11_fp8_kl",
            "naive_text": f"{naive['h11_fp8_kl']:.4f}",
            "tenet_text": f"{tenet['h11_fp8_kl']:.4f}",
        },
    )
    pair_metric(
        640,
        120,
        250,
        170,
        "FP8 re-leakage (↓)",
        0.6,
        {
            "key": "fp8_releakage",
            "naive_text": f"{naive['fp8_releakage']:.4f}",
            "tenet_text": f"{tenet['fp8_releakage']:.4f}",
        },
    )

    # max row crest factor panel with 25.0 threshold line
    cx, cy, cw, ch = 50, 400, 600, 130
    cbase = cy + ch
    crest_max_scale = 27.0
    parts.append(
        f'<text x="{cx}" y="{cy - 12}" font-size="13" font-weight="600" fill="#334155">'
        f"max_row_crest_factor (FP8 limit {MAX_ROW_CREST_FACTOR_THRESHOLD:.1f})</text>"
    )
    bw = 190
    # naive: sits on/over the threshold — draw a hatched bar up to the limit
    parts.append(
        f'<rect x="{cx + 20}" y="{cy + 8}" width="{bw}" height="{ch - 8}" '
        'fill="#ef4444" opacity="0.35" rx="4"/>'
    )
    parts.append(
        f'<text x="{cx + 20 + bw / 2}" y="{cy + 2}" font-size="12" font-weight="700" '
        'fill="#b91c1c" text-anchor="middle">≥ 25.0 (FAIL)</text>'
    )
    tbh = tenet["max_row_crest_factor"] / crest_max_scale * ch
    parts.append(
        f'<rect x="{cx + 320}" y="{cbase - tbh:.1f}" width="{bw}" height="{tbh:.1f}" '
        'fill="#10b981" opacity="0.9" rx="4"/>'
    )
    parts.append(
        f'<text x="{cx + 320 + bw / 2}" y="{cbase - tbh - 8:.1f}" font-size="12" '
        'font-weight="700" fill="#047857" text-anchor="middle">17.27 (PASS)</text>'
    )
    thr_y = cbase - MAX_ROW_CREST_FACTOR_THRESHOLD / crest_max_scale * ch
    parts.append(
        f'<line x1="{cx}" y1="{thr_y:.1f}" x2="{cx + cw}" y2="{thr_y:.1f}" '
        'stroke="#ef4444" stroke-width="2" stroke-dasharray="7 5"/>'
    )
    parts.append(
        f'<text x="{cx + cw + 8}" y="{thr_y + 4:.1f}" font-size="12" '
        'fill="#ef4444" font-weight="600">25.0</text>'
    )
    parts.append(f'<line x1="{cx}" y1="{cbase}" x2="{cx + cw}" y2="{cbase}" stroke="#334155"/>')
    parts.append(
        f'<text x="{cx + 20 + bw / 2}" y="{cbase + 18}" font-size="12" '
        'fill="#475569" text-anchor="middle">Naive</text>'
    )
    parts.append(
        f'<text x="{cx + 320 + bw / 2}" y="{cbase + 18}" font-size="12" '
        'fill="#475569" text-anchor="middle">Tenet Lancet</text>'
    )

    # right-side summary chips
    sx = 700
    parts.append(
        f'<text x="{sx}" y="{cy - 12}" font-size="13" font-weight="600" '
        'fill="#334155">Summary</text>'
    )
    parts.append(f'<rect x="{sx}" y="{cy + 4}" width="180" height="34" rx="8" fill="#d1fae5"/>')
    parts.append(
        f'<text x="{sx + 12}" y="{cy + 26}" font-size="12" fill="#065f46">'
        "Tenet FP8 retention: 0.9541</text>"
    )
    parts.append(f'<rect x="{sx}" y="{cy + 46}" width="180" height="34" rx="8" fill="#fee2e2"/>')
    parts.append(
        f'<text x="{sx + 12}" y="{cy + 68}" font-size="12" fill="#991b1b">'
        "Naive FP8 verdict: FAIL</text>"
    )
    parts.append(f'<rect x="{sx}" y="{cy + 88}" width="180" height="34" rx="8" fill="#dcfce7"/>')
    parts.append(
        f'<text x="{sx + 12}" y="{cy + 110}" font-size="12" fill="#166534">'
        "Tenet FP8 verdict: PASS</text>"
    )
    scr_text = " / ".join(f"L{layer}" for layer in tenet["top_scr_layers"])
    parts.append(f'<rect x="{sx}" y="{cy + 130}" width="180" height="34" rx="8" fill="#e0e7ff"/>')
    parts.append(
        f'<text x="{sx + 12}" y="{cy + 152}" font-size="12" fill="#3730a3">'
        f"Top-3 SCR layers: {scr_text}</text>"
    )

    parts.append(
        f'<text x="24" y="{H - 16}" font-size="12" fill="#64748b">'
        "Naive rows collapse onto the injected refusal direction → crest ≥ 25 → "
        "FP8 per-tensor scales re-leak it. Lancet subspace rejection keeps crest at 17.27.</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# CKA / Lancet-Safe GGUF & vLLM FP8 command generator
# ---------------------------------------------------------------------------
def generate_gguf_commands(
    model_id: str = "Qwen/Qwen2.5-3B-Instruct",
    bundle_dir: str = "",
    method: str = "lancet",
    gguf_quant: str = "Q4_K_M",
    include_vllm_fp8: bool = True,
    include_hf_upload: bool = False,
    repo_id: str = "",
) -> str:
    """Emit a ready-to-paste bash script for CKA / Lancet-Safe GGUF + vLLM FP8."""
    import shlex

    model_id = str(model_id or "Qwen/Qwen2.5-3B-Instruct").strip()
    bundle = str(bundle_dir or "").strip()
    method = (method or "lancet").lower()
    quant = str(gguf_quant or "Q4_K_M").strip()
    base_name = Path(model_id.rstrip("/")).name or "model"
    out_name = f"{base_name}-Tenet-Lancet" if method == "lancet" else f"{base_name}-CKA"
    q = shlex.quote

    method_desc = (
        "Lancet-Safe (recipe + orthonormal basis, crest-verified)"
        if method == "lancet"
        else "CKA (alignment-audited directions)"
    )
    lines: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# === {method.upper()} export pipeline for {model_id} ===",
        f"# Method: {method_desc}",
        f"MODEL_ID={q(model_id)}",
        f"OUT={q(out_name)}",
        "export MODEL_ID OUT",
    ]
    if bundle:
        lines.append(f"BUNDLE={q(bundle)}")
        lines.append("export BUNDLE")
    lines += [
        "",
        "# 0) Fetch the base checkpoint",
        'huggingface-cli download "$MODEL_ID" --local-dir "models/base" '
        "--local-dir-use-symlinks False",
        "",
    ]
    if method == "lancet":
        lines += [
            "# 1) Apply the Lancet recipe (schema 1.0): recipe JSON + orthonormal basis.",
            "#    Adjust to your lab's applier entry point (src/sml/...). The bundle must contain",
            "#    lancet_recipe.json + lancet_basis.safetensors; verify provenance first:",
            "python - <<'PY'",
            "import os",
            "from pathlib import Path",
            "from app import verify_recipe_provenance",
            "prov = verify_recipe_provenance(Path(os.environ.get('BUNDLE', 'models/base')))",
            "assert prov['ok'], prov['errors']",
            "print('recipe provenance OK')",
            "PY",
            "python -m sml.lancet apply \\",
            '    --base "models/base" \\',
            '    --recipe "${BUNDLE:-models/base}/lancet_recipe.json" \\',
            '    --basis "${BUNDLE:-models/base}/lancet_basis.safetensors" \\',
            '    --out "models/$OUT"',
        ]
    else:
        lines += [
            "# 1) Apply CKA-derived rejection directions (adjust to your lab's entry point):",
            'python -m sml.cka apply --base "models/base" --out "models/$OUT"',
            "",
            "# 1b) CKA similarity audit — per-layer CKA(base, candidate) must stay >= 0.95:",
            'python -m sml.cka audit --base "models/base" --candidate "models/$OUT" --min-cka 0.95',
        ]
    lines += [
        "",
        "# 2) FP8 crest gate — max_row_crest_factor must be < 25.0 before any FP8 export.",
        "#    Run from the directory containing this app.py (pure functions, no gradio needed):",
        "python - <<'PY'",
        "import os",
        "from pathlib import Path",
        "from app import inspect_directory_or_bundle",
        "r = inspect_directory_or_bundle(Path('models', os.environ.get('OUT', 'out')))",
        "print('fp8_verdict:', r['fp8_verdict'], '| max_row_crest_factor:', "
        "r['max_row_crest_factor'])",
        "assert r['fp8_verdict'] == 'PASS', 'row crest factor too high for FP8'",
        "PY",
        "",
        "# 3) GGUF export (llama.cpp)",
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp && "
        "pip install -r llama.cpp/requirements.txt",
        'python llama.cpp/convert_hf_to_gguf.py "models/$OUT" '
        '--outfile "gguf/$OUT.f16.gguf" --outtype f16',
        "",
        f"# 4) Quantize to {quant}",
        f'./llama.cpp/build/bin/llama-quantize "gguf/$OUT.f16.gguf" '
        f'"gguf/$OUT.{quant}.gguf" {quant}',
        "",
    ]
    if include_vllm_fp8:
        lines += [
            "# 5) vLLM FP8 (dynamic per-tensor scales; crest-safe because "
            "max_row_crest_factor < 25)",
            "pip install vllm",
            'vllm serve "models/$OUT" --quantization fp8 --max-model-len 8192',
            "",
        ]
    if include_hf_upload:
        target = repo_id.strip() or "<your-org>/" + out_name
        lines += [
            "# 6) Publish",
            f'huggingface-cli upload {q(target)} "gguf/$OUT.{quant}.gguf" '
            f'"{out_name}.{quant}.gguf"',
            "",
        ]
    lines.append(f'echo "Done: gguf/$OUT.{quant}.gguf"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI (optional at import time)
# ---------------------------------------------------------------------------
def _format_checks_summary(r: dict[str, Any]) -> str:
    icon = {"PASS": "✅", "FAIL": "❌"}.get(str(r.get("fp8_verdict")), "❔")
    lines = [
        f"**fp8_verdict: {icon} {r.get('fp8_verdict')}** · "
        f"ready: {'✅' if r.get('ready') else '❌'} · "
        f"mode: `{r.get('mode')}` · tensors checked: {r.get('tensors_checked', 0)} · "
        f"weight files: {r.get('weight_files_checked', 0)} "
        f"({r.get('total_weight_bytes', 0)} bytes)",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for name, ok in (r.get("checks") or {}).items():
        lines.append(f"| `{name}` | {'✅' if ok else '❌'} |")
    if r.get("architectures"):
        lines += ["", "architectures: " + ", ".join(f"`{a}`" for a in r["architectures"])]
    metric_keys = (
        "max_abs_weight",
        "max_row_crest_factor",
        "row_crest_factor_p50",
        "row_crest_factor_p99",
    )
    metric_bits = [f"{k}: `{r.get(k)}`" for k in metric_keys if r.get(k) is not None]
    if metric_bits:
        lines += ["", " · ".join(metric_bits)]
    top3 = r.get("top_3_outlier_layers") or []
    if top3:
        lines += ["", "**Top-3 outlier tensors**"]
        lines += [
            f"- `{t['name']}` (layer {t['layer']}, shape {t['shape']}): "
            f"crest {t['row_crest_factor']}, max|w| {t['max_abs_weight']}"
            for t in top3
        ]
    if r.get("recipe_provenance"):
        prov = r["recipe_provenance"]
        lines += ["", f"recipe provenance: {'✅ OK' if prov.get('ok') else '❌'}"]
    errors = r.get("errors") or []
    if errors:
        lines += ["", "**Blocking errors**"]
        lines += [f"- ⚠️ {e}" for e in errors]
    return "\n".join(lines)


def build_demo():
    """Build the 3-tab Gradio app; returns None when gradio is unavailable."""
    if not HAS_GRADIO:
        return None

    naive, tenet = DEMO_PRESET["naive"], DEMO_PRESET["tenet"]
    table = "\n".join(
        [
            "| metric | Naive Abliteration | Tenet-3B-Qwen2.5-Unslop-v1 |",
            "|---|---|---|",
            f"| H9 KL divergence (BF16) | {naive['h9_kl']} | {tenet['h9_kl']} |",
            f"| H11 KL divergence (FP8) | {naive['h11_fp8_kl']} | {tenet['h11_fp8_kl']} |",
            f"| FP8 re-leakage | {naive['fp8_releakage']} | {tenet['fp8_releakage']} |",
            f"| FP8 retention | — | {tenet['fp8_retention']} |",
            f"| max_row_crest_factor | ≥ 25.0 → **FAIL** | "
            f"{tenet['max_row_crest_factor']} → **PASS** |",
            "| Top-3 SCR layers | 21 / 15 / 19 | 21 / 15 / 19 |",
        ]
    )

    def _run_inspect(path: str):
        result = inspect_directory_or_bundle(path)
        metrics = {k: v for k, v in result.items() if k != "svg_chart"}
        return _format_checks_summary(result), result["svg_chart"], metrics

    with gr.Blocks(title="FP8 Abliteration Inspector") as demo:
        gr.Markdown(
            "# 🔬 FP8 Abliteration Inspector\n"
            "Audit abliterated checkpoints and Lancet recipe bundles for FP8 quantization "
            "safety — weights are streamed tensor-by-tensor, never loaded whole."
        )
        with gr.Tabs():
            with gr.Tab("1 · Live Demo Comparison"):
                gr.Markdown(
                    "Real measurements on **Qwen2.5-3B**: naive abliteration rows collapse "
                    "onto the injected refusal direction (crest ≥ 25) and FP8 per-tensor "
                    "scales re-leak it; the Tenet Lancet bundle keeps the crest at 17.27 and "
                    "stays FP8-safe."
                )
                gr.HTML(render_demo_svg())
                gr.Markdown(table)
            with gr.Tab("2 · Inspect Local Checkpoint / Recipe Bundle"):
                gr.Markdown(
                    "Give a checkpoint directory (or a single `*.safetensors` file). Enforces the "
                    "ADR-003 `< 15 GB` limit, runs the 7 `verify_provider_readiness` checks plus "
                    "recipe provenance, and falls back to metadata JSONs when no weights exist."
                )
                path_in = gr.Textbox(
                    label="Checkpoint / bundle path",
                    placeholder="/path/to/checkpoint-dir or /path/to/model.safetensors",
                )
                inspect_btn = gr.Button("Inspect", variant="primary")
                summary_md = gr.Markdown()
                chart_html = gr.HTML(label="Row crest-factor chart")
                metrics_json = gr.JSON(label="Full metrics")
                inspect_btn.click(_run_inspect, [path_in], [summary_md, chart_html, metrics_json])
            with gr.Tab("3 · CKA / Lancet-Safe GGUF Command Generator"):
                model_id_in = gr.Textbox(label="Base model id", value="Qwen/Qwen2.5-3B-Instruct")
                bundle_in = gr.Textbox(
                    label="Lancet bundle directory (optional)",
                    placeholder="dir containing lancet_recipe.json + lancet_basis.safetensors",
                )
                method_in = gr.Radio(["lancet", "cka"], value="lancet", label="Method")
                quant_in = gr.Dropdown(
                    ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M"],
                    value="Q4_K_M",
                    label="GGUF quant",
                )
                vllm_in = gr.Checkbox(value=True, label="Include vLLM FP8 serve commands")
                upload_in = gr.Checkbox(value=False, label="Include Hugging Face upload")
                repo_in = gr.Textbox(
                    label="HF repo id (for upload)",
                    placeholder="your-org/your-model",
                )
                gen_btn = gr.Button("Generate commands", variant="primary")
                cmd_out = gr.Code(label="bash script", language="bash")
                gen_btn.click(
                    generate_gguf_commands,
                    [model_id_in, bundle_in, method_in, quant_in, vllm_in, upload_in, repo_in],
                    cmd_out,
                )
    return demo


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--inspect":
        import pprint

        pprint.pp(inspect_directory_or_bundle(args[1] if len(args) > 1 else "."))
    elif HAS_GRADIO:
        demo = build_demo()
        if demo is not None:
            demo.launch(
                server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
            )
    else:
        print("gradio is not installed — pure functions remain available:")
        print("  inspect_directory_or_bundle(path), verify_recipe_provenance(dir),")
        print("  generate_gguf_commands(...), render_demo_svg()")
