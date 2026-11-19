"""Unit tests for H14 GGUF audit (surgery/gguf_audit.py).

Uses real synthetic GGUF files written with ``gguf.GGUFWriter`` and a real
Lancet basis ``.safetensors`` file — no mocks.
"""

from __future__ import annotations

import math
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from sml.surgery.gguf_audit import (
    audit_gguf_weights,
    build_h14_eval_corpus,
    compute_gguf_inventory,
)

SEED = 2026
D_MODEL = 64
K = 8
N_LAYER = 2
ATTN_SHAPE = (D_MODEL, D_MODEL)  # blk.N.attn_output.weight: (d_model, d_model)
FFN_SHAPE = (D_MODEL, 2 * D_MODEL)  # blk.N.ffn_down.weight: (d_model, 2 * d_model)


def _quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """Manually pack a (out, in) float32 matrix into Q8_0 blocks.

    Block layout: float16 scale ``d`` followed by 32 int8 ``qs`` values.
    """
    x = np.asarray(x, dtype=np.float32)
    out_f, in_f = x.shape
    assert in_f % 32 == 0
    blocks = x.reshape(out_f, in_f // 32, 32)
    amax = np.abs(blocks).max(axis=-1)
    d = (amax / 127.0).astype(np.float16)
    d_safe = d.astype(np.float32)
    d_safe[d_safe == 0.0] = 1.0
    q = np.clip(np.round(blocks / d_safe[:, :, None]), -127, 127).astype(np.int8)
    packed = np.empty((out_f, in_f // 32, 34), dtype=np.uint8)
    packed[..., 0:2] = d.view(np.uint8).reshape(out_f, in_f // 32, 2)
    packed[..., 2:34] = q.view(np.uint8).reshape(out_f, in_f // 32, 32)
    # (out_f, blocks * 34) byte shape — GGUFWriter converts it to (out_f, in_f).
    return packed.reshape(out_f, (in_f // 32) * 34)


def _dequantize_q8_0(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Mirror of the Q8_0 math used to build expected values in tests."""
    out_f, in_f = shape
    blocks = packed.reshape(out_f, in_f // 32, 34)
    d = blocks[..., 0:2].copy().view(np.float16).astype(np.float32)
    q = blocks[..., 2:34].copy().view(np.int8).astype(np.float32)
    return (d * q).reshape(out_f, in_f)


def _write_gguf(
    path: Path,
    tensors: list[
        tuple[str, np.ndarray, gguf.GGMLQuantizationType | None, tuple[int, int] | None]
    ],
) -> None:
    writer = gguf.GGUFWriter(str(path), arch="llama")
    for name, data, raw_dtype, raw_shape in tensors:
        writer.add_tensor(name, data, raw_shape=raw_shape, raw_dtype=raw_dtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture(scope="module")
def synthetic_h14(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    tmp_path = tmp_path_factory.mktemp("h14_gguf")
    rng = np.random.default_rng(SEED)

    # Raw Lancet basis (not orthonormal yet) — the audit orthonormalizes it.
    raw_rows = rng.standard_normal((N_LAYER, K, D_MODEL)).astype(np.float32)
    basis_path = tmp_path / "lancet_basis.safetensors"
    save_safetensors(
        {f"layer_{idx}": torch.from_numpy(raw_rows[idx]) for idx in range(N_LAYER)},
        str(basis_path),
    )

    # Orthonormalize exactly like the implementation does (QR of R.T).
    orthonormal = {}
    for idx in range(N_LAYER):
        q, _ = np.linalg.qr(raw_rows[idx].T.astype(np.float32))
        orthonormal[idx] = q.T.astype(np.float32)  # (K, D_MODEL), orthonormal rows

    def orthogonal_weights(shape: tuple[int, int], idx: int) -> np.ndarray:
        """Random matrix whose columns (output space) live in the basis' orthogonal complement."""
        x = rng.standard_normal(shape).astype(np.float32)
        r = orthonormal[idx]
        return x - r.T @ (r @ x)  # (I - R^T R) X, so R @ X == 0

    f32_tensors: list[
        tuple[str, np.ndarray, gguf.GGMLQuantizationType | None, tuple[int, int] | None]
    ] = []
    f16_tensors: list[
        tuple[str, np.ndarray, gguf.GGMLQuantizationType | None, tuple[int, int] | None]
    ] = []
    q8_tensors: list[
        tuple[str, np.ndarray, gguf.GGMLQuantizationType | None, tuple[int, int] | None]
    ] = []
    q8_shapes: dict[str, tuple[int, int]] = {}
    expected: dict[str, dict[str, np.ndarray]] = {}
    for idx in range(N_LAYER):
        r = orthonormal[idx]
        for role, shape in (("attn_output", ATTN_SHAPE), ("ffn_down", FFN_SHAPE)):
            name = f"blk.{idx}.{role}.weight"
            w_f32 = orthogonal_weights(shape, idx)
            # Deliberate subspace noise: only the Q8_0 variant gets damaged.
            noise = rng.standard_normal((K, shape[1])).astype(np.float32)
            w_noisy = w_f32 + 0.05 * (r.T @ noise)
            f32_tensors.append((name, w_f32, gguf.GGMLQuantizationType.F32, None))
            f16_tensors.append(
                (name, w_f32.astype(np.float16), gguf.GGMLQuantizationType.F16, None)
            )
            packed = _quantize_q8_0(w_noisy)
            q8_tensors.append((name, packed, gguf.GGMLQuantizationType.Q8_0, None))
            q8_shapes[name] = shape
            expected[name] = {
                "f32": w_f32,
                "f16": w_f32.astype(np.float16).astype(np.float32),
                "q8": _dequantize_q8_0(packed, q8_shapes[name]),
            }

    paths = {
        "lancet_basis": basis_path,
        "f32": tmp_path / "h14_f32.gguf",
        "f16": tmp_path / "h14_f16.gguf",
        "q8_0": tmp_path / "h14_q8_0.gguf",
    }
    _write_gguf(paths["f32"], f32_tensors)
    _write_gguf(paths["f16"], f16_tensors)
    _write_gguf(paths["q8_0"], q8_tensors)
    paths["expected"] = expected  # type: ignore[assignment]
    paths["orthonormal"] = orthonormal  # type: ignore[assignment]
    return paths  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. Inventory
# ---------------------------------------------------------------------------


def test_compute_gguf_inventory_f32_exact_params_and_bpw(synthetic_h14) -> None:
    inv = compute_gguf_inventory(synthetic_h14["f32"])
    expected_params = N_LAYER * (ATTN_SHAPE[0] * ATTN_SHAPE[1] + FFN_SHAPE[0] * FFN_SHAPE[1])
    assert inv["total_params"] == expected_params
    assert inv["total_tensor_bytes"] == expected_params * 4
    assert inv["actual_avg_bpw"] == pytest.approx(32.0)
    assert inv["file_size_bytes"] == synthetic_h14["f32"].stat().st_size
    assert inv["tensor_type_counts"] == {"F32": N_LAYER * 2}
    assert inv["gguf_path"] == str(synthetic_h14["f32"])


def test_compute_gguf_inventory_f16_bpw(synthetic_h14) -> None:
    inv = compute_gguf_inventory(synthetic_h14["f16"])
    expected_params = N_LAYER * (ATTN_SHAPE[0] * ATTN_SHAPE[1] + FFN_SHAPE[0] * FFN_SHAPE[1])
    assert inv["total_params"] == expected_params
    assert inv["total_tensor_bytes"] == inv["total_params"] * 2
    assert inv["actual_avg_bpw"] == pytest.approx(16.0)
    assert inv["tensor_type_counts"] == {"F16": N_LAYER * 2}


def test_compute_gguf_inventory_q8_0_bpw(synthetic_h14) -> None:
    inv = compute_gguf_inventory(synthetic_h14["q8_0"])
    expected_params = N_LAYER * (ATTN_SHAPE[0] * ATTN_SHAPE[1] + FFN_SHAPE[0] * FFN_SHAPE[1])
    assert inv["total_params"] == expected_params
    # Q8_0: 34 bytes per 32 weights => 8.5 BPW.
    assert inv["actual_avg_bpw"] == pytest.approx(8.5)
    assert inv["tensor_type_counts"] == {"Q8_0": N_LAYER * 2}


# ---------------------------------------------------------------------------
# 2. Weight-space subspace audit
# ---------------------------------------------------------------------------


def test_audit_shapes_and_tensor_coverage(synthetic_h14) -> None:
    audit = audit_gguf_weights(synthetic_h14["f32"], synthetic_h14["lancet_basis"])
    assert audit["surgical_layers"] == [0, 1]
    for idx in ("0", "1"):
        layer = audit["layers"][idx]
        for role in ("attn_output", "ffn_down"):
            entry = layer["tensors"][role]
            assert entry["present"] is True
            shape = ATTN_SHAPE if role == "attn_output" else FFN_SHAPE
            assert entry["shape"] == list(shape)


def test_audit_f32_near_zero_subspace_energy(synthetic_h14) -> None:
    """Weights built orthogonal to the basis must have (near) zero leakage."""
    audit = audit_gguf_weights(synthetic_h14["f32"], synthetic_h14["lancet_basis"])
    for idx in ("0", "1"):
        layer = audit["layers"][idx]
        assert layer["subspace_energy_ratio_to_w"] < 1e-5
        for role in ("attn_output", "ffn_down"):
            entry = layer["tensors"][role]
            assert entry["subspace_energy_ratio_to_w"] < 1e-5
            # Expected value computed independently from the same basis.
            r = synthetic_h14["orthonormal"][int(idx)]
            w = synthetic_h14["expected"][f"blk.{idx}.{role}.weight"]["f32"]
            expected_energy = float(np.linalg.norm(r @ w) ** 2)
            assert entry["subspace_energy_fro"] == pytest.approx(
                math.sqrt(expected_energy), rel=1e-4, abs=1e-9
            )


def test_audit_f16_vs_itself_exact_identity(synthetic_h14) -> None:
    audit = audit_gguf_weights(
        synthetic_h14["f16"],
        synthetic_h14["lancet_basis"],
        f16_gguf_path=synthetic_h14["f16"],
    )
    assert audit["subspace_releakage_amplification_vs_f16"] == pytest.approx(1.0)
    assert audit["max_row_norm_drift"] == 0.0
    assert audit["mean_rel_row_norm_drift"] == 0.0
    assert audit["mean_weight_l2_rel_err"] == pytest.approx(0.0)
    for layer in audit["per_layer"].values():
        assert layer["max_row_norm_drift"] == 0.0
        assert layer["mean_weight_l2_rel_err"] == pytest.approx(0.0)


def test_audit_q8_amplification_and_exact_row_norm_drift(synthetic_h14) -> None:
    """Q8_0 variant (with subspace noise) must leak far more than the F16 ref."""
    audit = audit_gguf_weights(
        synthetic_h14["q8_0"],
        synthetic_h14["lancet_basis"],
        f16_gguf_path=synthetic_h14["f16"],
    )
    assert audit["subspace_releakage_amplification_vs_f16"] > 10.0
    assert audit["mean_weight_l2_rel_err"] > 0.0

    # Exact row-norm drift, recomputed independently from the known tensors.
    expected_max_drift = 0.0
    expected_rel_drifts: list[np.ndarray] = []
    for idx in range(N_LAYER):
        for role in ("attn_output", "ffn_down"):
            name = f"blk.{idx}.{role}.weight"
            w_q = synthetic_h14["expected"][name]["q8"]
            w_f16 = synthetic_h14["expected"][name]["f16"]
            rn_q = np.linalg.norm(w_q, axis=1).astype(np.float64)
            rn_f16 = np.linalg.norm(w_f16, axis=1).astype(np.float64)
            drift = np.abs(rn_q - rn_f16)
            expected_max_drift = max(expected_max_drift, float(drift.max()))
            expected_rel_drifts.append(drift / np.maximum(rn_f16, 1e-12))
    expected_rel = float(np.concatenate(expected_rel_drifts).mean())

    assert audit["max_row_norm_drift"] == pytest.approx(expected_max_drift, rel=1e-6)
    assert audit["mean_rel_row_norm_drift"] == pytest.approx(expected_rel, rel=1e-6)
    assert expected_max_drift > 0.0

    for idx in ("0", "1"):
        layer = audit["layers"][idx]
        assert layer["subspace_releakage_amplification_vs_f16"] > 1.0
        assert layer["mean_weight_l2_rel_err"] > 0.0


def test_audit_f32_vs_f16_small_drift(synthetic_h14) -> None:
    """F16 rounding must produce a small but non-zero drift vs F32 weights."""
    audit = audit_gguf_weights(
        synthetic_h14["f32"],
        synthetic_h14["lancet_basis"],
        f16_gguf_path=synthetic_h14["f16"],
    )
    assert audit["max_row_norm_drift"] > 0.0
    assert audit["max_row_norm_drift"] < 1.0
    assert audit["mean_weight_l2_rel_err"] < 1e-2
    # The F32 weights are exactly orthogonal, so they leak far less into the
    # subspace than the F16 reference (whose rounding re-leaks energy).
    assert 0.0 < audit["subspace_releakage_amplification_vs_f16"] < 1.0


def test_audit_reads_basis_from_real_safetensors(synthetic_h14) -> None:
    stored = load_safetensors(str(synthetic_h14["lancet_basis"]))
    assert set(stored) == {"layer_0", "layer_1"}
    for value in stored.values():
        assert value.dtype == torch.float32
        assert value.shape == (K, D_MODEL)


# ---------------------------------------------------------------------------
# 3. Eval corpus
# ---------------------------------------------------------------------------


def test_build_h14_eval_corpus_produces_non_empty_text(tmp_path: Path) -> None:
    corpus_path = build_h14_eval_corpus(tmp_path / "corpus" / "h14_eval_corpus.txt")
    assert corpus_path == tmp_path / "corpus" / "h14_eval_corpus.txt"
    assert corpus_path.exists()
    text = corpus_path.read_text(encoding="utf-8")
    assert len(text.strip()) > 0
    # Enough text for llama-perplexity to cover multiple 512-token chunks.
    assert len(text.split()) >= 3 * 512
