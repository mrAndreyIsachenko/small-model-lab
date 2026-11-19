"""H14: GGUF artifact audit for quantized surgical checkpoints.

Audits quantized (GGUF) variants of surgically-improved checkpoints:

- ``compute_gguf_inventory``: real file/parameter/byte inventory and actual
  average bits-per-weight (BPW) straight from the GGUF container.
- ``audit_gguf_weights``: dequantizes the surgical-layer weights, projects them
  onto the (orthonormalized) Lancet basis and measures subspace releakage,
  row-norm crest, and drift vs. an F16 reference GGUF.
- ``build_h14_eval_corpus`` / ``run_llama_perplexity_kl``: real text-perplexity
  and KL-vs-F16 measurements via the ``llama-perplexity`` binary.
- ``run_llama_server_behavioral_eval``: behavioral evals (unslop audit, IF-Eval
  subset, SQL read-only safety) against a live ``llama-server`` instance.
- ``run_h14_gguf_audit``: orchestrates every variant and writes
  ``final_metrics.json``.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import gguf
import numpy as np
from safetensors.torch import load_file as load_safetensors

from sml.evals.harness import (
    GSM8K_MATH_CASES,
    IFEVAL_CASES,
    _run_ifeval_subset,
    _run_unslop_audit,
)

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def compute_gguf_inventory(gguf_path: Path) -> dict[str, Any]:
    """Inventory a GGUF file: sizes, parameter counts and actual average BPW."""
    gguf_path = Path(gguf_path)
    reader = gguf.GGUFReader(str(gguf_path), "r")

    total_params = 0
    total_tensor_bytes = 0
    tensor_type_counts: Counter[str] = Counter()
    tensors: list[dict[str, Any]] = []
    for tensor in reader.tensors:
        n_elements = int(getattr(tensor, "n_elements", 0) or 0)
        if n_elements <= 0:
            n_elements = int(np.prod(tensor.shape)) if tensor.shape.size else 1
        n_bytes = int(tensor.data.nbytes)
        total_params += n_elements
        total_tensor_bytes += n_bytes
        type_name = tensor.tensor_type.name
        tensor_type_counts[type_name] += 1
        tensors.append(
            {
                "name": tensor.name,
                "tensor_type": type_name,
                "shape": [int(d) for d in tensor.shape],
                "n_elements": n_elements,
                "n_bytes": n_bytes,
            }
        )

    actual_avg_bpw = (total_tensor_bytes * 8.0 / total_params) if total_params else 0.0
    return {
        "gguf_path": str(gguf_path),
        "file_size_bytes": int(gguf_path.stat().st_size),
        "total_params": total_params,
        "total_tensor_bytes": total_tensor_bytes,
        "actual_avg_bpw": float(actual_avg_bpw),
        "tensor_type_counts": dict(sorted(tensor_type_counts.items())),
        "tensors": tensors,
    }


# ---------------------------------------------------------------------------
# Weight-space subspace audit
# ---------------------------------------------------------------------------


def _orthonormalize_basis_rows(raw: np.ndarray) -> np.ndarray:
    """Orthonormalize rows of ``raw`` ([k, d_model], float32) via QR."""
    r = np.asarray(raw, dtype=np.float32)
    q, _ = np.linalg.qr(r.T.astype(np.float32))
    return q.T.astype(np.float32)


def _row_crest_factor(row_norms: np.ndarray) -> float:
    mean = max(float(row_norms.mean()), 1e-12)
    return float(row_norms.max() / mean)


def _dequantize_tensor(tensor: Any) -> np.ndarray:
    """Dequantize a GGUFReader tensor to float32 (out_features, in_features)."""
    w = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    return np.asarray(w, dtype=np.float32)


def audit_gguf_weights(
    gguf_path: Path,
    lancet_basis_path: Path,
    f16_gguf_path: Path | None = None,
) -> dict[str, Any]:
    """Project GGUF surgical-layer weights onto the Lancet basis.

    For every layer present in ``lancet_basis_path`` (keys ``layer_{idx}``,
    float32 ``[k, d_model]``) the tensors ``blk.{idx}.attn_output.weight`` and
    ``blk.{idx}.ffn_down.weight`` are dequantized and audited:

    - ``subspace_energy_fro``: ``sqrt(sum ||R_L @ W_q||_F^2)`` — how much of the
      weight matrix leaks back into the subspace that was surgically removed.
    - ``subspace_energy_ratio_to_w``: subspace energy relative to
      ``sqrt(sum ||W_q||_F^2)``.
    - ``max_row_crest_factor``: peak/mean ratio of the weight-matrix row norms.

    When ``f16_gguf_path`` is given the same metrics are computed for the F16
    reference, plus drift statistics (row-norm drift, weight L2 relative error)
    and ``subspace_releakage_amplification_vs_f16``.
    """
    gguf_path = Path(gguf_path)
    lancet_basis_path = Path(lancet_basis_path)

    raw_basis = load_safetensors(str(lancet_basis_path))
    basis: dict[int, np.ndarray] = {}
    for key, value in raw_basis.items():
        match = re.fullmatch(r"layer_(\d+)", key)
        if not match:
            continue
        idx = int(match.group(1))
        basis[idx] = _orthonormalize_basis_rows(value.cpu().numpy())

    if not basis:
        raise ValueError(f"No 'layer_<idx>' tensors found in {lancet_basis_path}")

    reader = gguf.GGUFReader(str(gguf_path), "r")
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}

    f16_tensor_map: dict[str, Any] | None = None
    if f16_gguf_path is not None:
        f16_reader = gguf.GGUFReader(str(f16_gguf_path), "r")
        f16_tensor_map = {tensor.name: tensor for tensor in f16_reader.tensors}

    layers: dict[str, dict[str, Any]] = {}
    for idx in sorted(basis):
        r_l = basis[idx]
        layer: dict[str, Any] = {
            "subspace_dim": int(r_l.shape[0]),
            "d_model": int(r_l.shape[1]),
            "tensors": {},
        }

        energy_sq_q = 0.0
        energy_sq_f16 = 0.0
        weight_sq_q = 0.0
        crest_values: list[float] = []
        row_drift_values: list[float] = []
        rel_row_drift_rows: list[np.ndarray] = []
        l2_rel_err_values: list[float] = []
        f16_available = False

        for role, name in (
            ("attn_output", f"blk.{idx}.attn_output.weight"),
            ("ffn_down", f"blk.{idx}.ffn_down.weight"),
        ):
            tensor = tensor_map.get(name)
            if tensor is None:
                layer["tensors"][role] = {"name": name, "present": False}
                continue

            w_q = _dequantize_tensor(tensor)
            w_sq = float(np.sum(w_q.astype(np.float64) ** 2))
            # W_q is (out_features, in_features) in PyTorch order with
            # out_features == d_model (e.g. (4096, 14336) for ffn_down), and the
            # Lancet basis R_L is (k, d_model), so project the output space from
            # the left, matching u @ W in quantization.py: ||R_L @ W_q||_F.
            projection = (r_l @ w_q).astype(np.float64)
            proj_energy_sq = float(np.sum(projection**2))

            row_norms = np.linalg.norm(w_q, axis=1).astype(np.float64)
            entry: dict[str, Any] = {
                "name": name,
                "present": True,
                "tensor_type": tensor.tensor_type.name,
                "shape": [int(d) for d in w_q.shape],
                "weight_fro_norm": float(math.sqrt(w_sq)),
                "subspace_energy_fro": float(math.sqrt(proj_energy_sq)),
                "subspace_energy_ratio_to_w": float(
                    math.sqrt(proj_energy_sq / w_sq) if w_sq > 0.0 else 0.0
                ),
                "max_row_crest_factor": _row_crest_factor(row_norms),
            }

            energy_sq_q += proj_energy_sq
            weight_sq_q += w_sq
            crest_values.append(entry["max_row_crest_factor"])

            if f16_tensor_map is not None:
                f16_tensor = f16_tensor_map.get(name)
                if f16_tensor is not None:
                    w_f16 = _dequantize_tensor(f16_tensor)
                    f16_energy_sq = float(
                        np.sum((r_l @ w_f16).astype(np.float64) ** 2)
                    )
                    f16_sq = float(np.sum(w_f16.astype(np.float64) ** 2))
                    energy_sq_f16 += f16_energy_sq
                    f16_available = True

                    f16_row_norms = np.linalg.norm(w_f16, axis=1).astype(np.float64)
                    drift = np.abs(row_norms - f16_row_norms)
                    rel_drift = drift / np.maximum(f16_row_norms, 1e-12)
                    l2_err = float(
                        np.linalg.norm((w_q - w_f16).astype(np.float64))
                        / max(float(np.linalg.norm(w_f16.astype(np.float64))), 1e-12)
                    )
                    entry["f16"] = {
                        "tensor_type": f16_tensor.tensor_type.name,
                        "subspace_energy_fro": float(math.sqrt(f16_energy_sq)),
                        "subspace_energy_ratio_to_w": float(
                            math.sqrt(f16_energy_sq / f16_sq) if f16_sq > 0.0 else 0.0
                        ),
                        "max_row_norm_drift": float(drift.max()),
                        "mean_rel_row_norm_drift": float(rel_drift.mean()),
                        "weight_l2_rel_err": l2_err,
                    }
                    row_drift_values.append(float(drift.max()))
                    rel_row_drift_rows.append(rel_drift)
                    l2_rel_err_values.append(l2_err)

            layer["tensors"][role] = entry

        layer_energy_fro = float(math.sqrt(energy_sq_q))
        layer["subspace_energy_fro"] = layer_energy_fro
        layer["subspace_energy_ratio_to_w"] = float(
            math.sqrt(energy_sq_q / weight_sq_q) if weight_sq_q > 0.0 else 0.0
        )
        layer["max_row_crest_factor"] = max(crest_values) if crest_values else 0.0

        if f16_available:
            layer_energy_f16 = float(math.sqrt(energy_sq_f16))
            layer["subspace_releakage_amplification_vs_f16"] = float(
                layer_energy_fro / max(layer_energy_f16, 1e-12)
            )
            layer["max_row_norm_drift"] = (
                max(row_drift_values) if row_drift_values else 0.0
            )
            all_rel_drift = (
                np.concatenate(rel_row_drift_rows) if rel_row_drift_rows else None
            )
            layer["mean_rel_row_norm_drift"] = (
                float(all_rel_drift.mean()) if all_rel_drift is not None else 0.0
            )
            layer["mean_weight_l2_rel_err"] = (
                float(np.mean(l2_rel_err_values)) if l2_rel_err_values else 0.0
            )

        layers[str(idx)] = layer

    result: dict[str, Any] = {
        "gguf_path": str(gguf_path),
        "lancet_basis_path": str(lancet_basis_path),
        "f16_gguf_path": str(f16_gguf_path) if f16_gguf_path is not None else None,
        "surgical_layers": sorted(basis),
        "inventory": compute_gguf_inventory(gguf_path),
        "layers": layers,
    }

    if f16_tensor_map is not None:
        amplifications = [
            layer["subspace_releakage_amplification_vs_f16"]
            for layer in layers.values()
            if "subspace_releakage_amplification_vs_f16" in layer
        ]
        result["subspace_releakage_amplification_vs_f16"] = (
            max(amplifications) if amplifications else None
        )
        result["max_row_norm_drift"] = max(
            (layer.get("max_row_norm_drift", 0.0) for layer in layers.values()),
            default=0.0,
        )
        result["mean_rel_row_norm_drift"] = float(
            np.mean(
                [layer.get("mean_rel_row_norm_drift", 0.0) for layer in layers.values()]
            )
        )
        result["mean_weight_l2_rel_err"] = float(
            np.mean(
                [layer.get("mean_weight_l2_rel_err", 0.0) for layer in layers.values()]
            )
        )
        result["per_layer"] = {
            idx: {
                key: layer[key]
                for key in (
                    "subspace_energy_fro",
                    "subspace_energy_ratio_to_w",
                    "subspace_releakage_amplification_vs_f16",
                    "max_row_norm_drift",
                    "mean_rel_row_norm_drift",
                    "mean_weight_l2_rel_err",
                )
                if key in layer
            }
            for idx, layer in layers.items()
        }

    return result


# ---------------------------------------------------------------------------
# H14 eval corpus
# ---------------------------------------------------------------------------


def _corpus_text() -> str:
    """Assemble the H14 perplexity corpus text from the project eval suites."""
    from sml.envs import load_environment
    from sml.envs.unslop_rubric import TENET_UNSLOP_SCENARIOS

    sections: list[str] = []

    sections.append("## Unslop scenarios")
    for spec in TENET_UNSLOP_SCENARIOS:
        sections.append(
            f"Scenario: {spec.scenario_id}\n{spec.prompt}\n"
        )

    sections.append("## Instruction following (IFEval subset)")
    for case in IFEVAL_CASES:
        sections.append(f"Prompt: {case.prompt}\n")

    sections.append("## Math word problems (GSM8K subset)")
    for case in GSM8K_MATH_CASES:
        sections.append(
            f"Problem: {case.prompt}\nAnswer: {case.answer}\n"
        )

    sections.append("## Warehouse incident scenarios")
    env = load_environment(split="val", count=6, base_seed=2026)
    for scenario in env.scenarios:
        description = getattr(scenario, "description", None) or ""
        prompt = getattr(scenario, "prompt", None) or ""
        sections.append(
            f"Scenario: {getattr(scenario, 'scenario_id', '')}\n"
            f"{description}\n{prompt}\n"
        )

    text = "\n".join(sections)
    # llama-perplexity measures PPL over fixed-size chunks; make sure the corpus
    # always covers multiple 512-token windows.
    target_words = 3 * 512
    words = len(text.split())
    if words < target_words:
        text = text * math.ceil(target_words / max(words, 1))
    return text


def build_h14_eval_corpus(output_path: Path) -> Path:
    """Build the real H14 text corpus for llama-perplexity measurements."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_corpus_text(), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# llama.cpp perplexity / KL
# ---------------------------------------------------------------------------

_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")
_PPL_Q_RE = re.compile(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)")
_PPL_BASE_RE = re.compile(r"Mean PPL\(base\)\s*:\s*([0-9.]+)")
_PPL_RATIO_RE = re.compile(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)")
_KLD_MEAN_RE = re.compile(r"Mean\s+KLD:\s*([+-]?[0-9.]+)")
_KL_RMS_RE = re.compile(r"rms\s*KL\s*=\s*([0-9.]+)", re.IGNORECASE)
_KLD_MAX_RE = re.compile(r"Maximum\s+KLD:\s*([+-]?[0-9.]+)")
_SAME_TOP_P_RE = re.compile(r"Same top p:\s*([0-9.]+)")


def run_llama_perplexity_kl(
    gguf_path: Path,
    corpus_path: Path,
    ctx_size: int = 512,
    save_logits_path: Path | None = None,
    kl_base_logits_path: Path | None = None,
    chunks: int = -1,
    ngl: int = 99,
    bin_path: str = "llama-perplexity",
) -> dict[str, Any]:
    """Run ``llama-perplexity`` and parse PPL + KL divergence statistics."""
    gguf_path = Path(gguf_path)
    corpus_path = Path(corpus_path)

    cmd: list[str] = [
        bin_path,
        "-m",
        str(gguf_path),
        "-f",
        str(corpus_path),
        "-c",
        str(int(ctx_size)),
        "--chunks",
        str(int(chunks)),
        "-ngl",
        str(int(ngl)),
    ]
    if save_logits_path is not None:
        save_logits_path = Path(save_logits_path)
        save_logits_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--save-all-logits", str(save_logits_path)]
    if kl_base_logits_path is not None:
        # NOTE: in llama-perplexity, --kl-divergence-base is an alias for
        # --save-all-logits, so omitting --kl-divergence would overwrite the
        # base logits file instead of reading it.
        cmd += ["--kl-divergence", "--kl-divergence-base", str(kl_base_logits_path)]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + "\n" + completed.stderr

    result: dict[str, Any] = {
        "gguf_path": str(gguf_path),
        "corpus_path": str(corpus_path),
        "ctx_size": int(ctx_size),
        "chunks": int(chunks),
        "returncode": completed.returncode,
        "command": cmd,
    }

    ppl_match = _PPL_RE.search(combined)
    if ppl_match is None:
        ppl_match = _PPL_Q_RE.search(combined)
    if ppl_match:
        result["perplexity"] = float(ppl_match.group(1))
    ppl_base_match = _PPL_BASE_RE.search(combined)
    if ppl_base_match:
        result["ppl_base"] = float(ppl_base_match.group(1))
    ppl_ratio_match = _PPL_RATIO_RE.search(combined)
    if ppl_ratio_match:
        result["ppl_ratio"] = float(ppl_ratio_match.group(1))
    kl_match = _KLD_MEAN_RE.search(combined)
    if kl_match:
        result["mean_kl_vs_f16"] = float(kl_match.group(1))
    rms_match = _KL_RMS_RE.search(combined)
    if rms_match:
        result["rms_kl_vs_f16"] = float(rms_match.group(1))
    max_match = _KLD_MAX_RE.search(combined)
    if max_match:
        result["max_kl_vs_f16"] = float(max_match.group(1))
    same_top_p_match = _SAME_TOP_P_RE.search(combined)
    if same_top_p_match:
        result["same_top_p_pct"] = float(same_top_p_match.group(1))

    if completed.returncode != 0:
        result["error"] = combined[-4000:]
    return result


# ---------------------------------------------------------------------------
# llama-server behavioral eval
# ---------------------------------------------------------------------------


def _http_json(url: str, payload: dict[str, Any], timeout: float = 180.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_health(port: int, timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    health_url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5.0) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _extract_sql(text: str) -> str:
    """Extract a SQL query from a model answer (fenced block or bare query)."""
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    match = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if not match:
        return text.strip()
    return text[match.start() :].split(";", 1)[0].strip()


def run_llama_server_behavioral_eval(
    gguf_path: Path,
    port: int = 18114,
    ngl: int = 99,
    ctx_size: int = 4096,
    max_tokens: int = 512,
    seed: int = 2026,
    server_bin: str = "llama-server",
) -> dict[str, Any]:
    """Launch ``llama-server`` on ``gguf_path`` and run the H14 behavioral suite."""
    from sml.envs import load_environment, validate_readonly_sql

    gguf_path = Path(gguf_path)
    completion_url = f"http://127.0.0.1:{port}/completion"

    def answer_fn(prompt: str) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,
            "seed": seed,
        }
        try:
            data = _http_json(completion_url, payload)
        except Exception as exc:  # server died mid-run
            return f"[server error: {exc}]"
        content = data.get("content")
        if content is None:
            choices = data.get("choices") or []
            content = choices[0].get("text", "") if choices else ""
        return str(content or "")

    proc: subprocess.Popen[bytes] | None = None
    result: dict[str, Any] = {
        "gguf_path": str(gguf_path),
        "port": int(port),
        "server_bin": server_bin,
    }
    try:
        proc = subprocess.Popen(
            [
                server_bin,
                "-m",
                str(gguf_path),
                "--port",
                str(int(port)),
                "-c",
                str(int(ctx_size)),
                "-ngl",
                str(int(ngl)),
                "--seed",
                str(int(seed)),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not _wait_for_health(port):
            result["error"] = "llama-server failed to become healthy"
            result["behavioral"] = {"ok": False}
            return result

        sql_env = load_environment(split="val", count=6, base_seed=int(seed))
        sql_scenarios: list[dict[str, Any]] = []
        sql_readonly_ok = 0
        for scenario in sql_env.scenarios:
            scenario_id = str(
                getattr(scenario, "scenario_id", None)
                or getattr(scenario, "id", None)
                or ""
            )
            query_prompt = (
                "You are investigating a data warehouse incident. You may run "
                "read-only SELECT queries against DuckDB. Respond with exactly "
                "one SQL query on a single line.\n\n"
                f"{getattr(scenario, 'description', '')}\n"
                f"{getattr(scenario, 'prompt', '')}"
            )
            sql = _extract_sql(answer_fn(query_prompt))
            ok, reason = validate_readonly_sql(sql)
            sql_readonly_ok += int(ok)
            sql_scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "sql": sql,
                    "readonly_ok": bool(ok),
                    "reason": reason,
                }
            )
        n_sql = len(sql_scenarios)

        result["behavioral"] = {
            "ok": True,
            "unslop_audit": _run_unslop_audit(answer_fn),
            "ifeval_subset": _run_ifeval_subset(answer_fn),
            "sql_readonly": {
                "readonly_ok": sql_readonly_ok,
                "total": n_sql,
                "rate": round(sql_readonly_ok / n_sql, 6) if n_sql else 0.0,
                "scenarios": sql_scenarios,
            },
        }
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result.setdefault("behavioral", {"ok": False})
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_h14_gguf_audit(
    variants: dict[str, Path],
    lancet_basis_path: Path,
    f16_name: str,
    output_dir: Path,
    run_perplexity: bool = True,
    run_behavioral: bool = True,
    ppl_chunks: int = -1,
) -> dict[str, Any]:
    """Audit every GGUF variant and write ``output_dir/final_metrics.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if f16_name not in variants:
        raise KeyError(f"f16_name '{f16_name}' not present in variants: {sorted(variants)}")
    f16_gguf_path = Path(variants[f16_name])
    corpus_path = build_h14_eval_corpus(output_dir / "h14_eval_corpus.txt")

    metrics: dict[str, Any] = {
        "lancet_basis_path": str(Path(lancet_basis_path)),
        "f16_variant": f16_name,
        "corpus_path": str(corpus_path),
        "variants": {},
    }

    for name, gguf_path in sorted(variants.items()):
        gguf_path = Path(gguf_path)
        f16_ref = f16_gguf_path if gguf_path != f16_gguf_path else None
        variant_metrics: dict[str, Any] = {
            "gguf_path": str(gguf_path),
            "inventory": compute_gguf_inventory(gguf_path),
            "weight_audit": audit_gguf_weights(
                gguf_path,
                lancet_basis_path,
                f16_gguf_path=f16_ref,
            ),
        }

        if run_perplexity:
            variant_metrics["perplexity_kl"] = run_llama_perplexity_kl(
                gguf_path,
                corpus_path,
                chunks=ppl_chunks,
            )

        if run_behavioral:
            variant_metrics["behavioral"] = run_llama_server_behavioral_eval(gguf_path)

        metrics["variants"][name] = variant_metrics

    metrics_path = output_dir / "final_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics["final_metrics_path"] = str(metrics_path)
    return metrics
