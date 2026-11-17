"""H12 subspace-safe merge: unit + acceptance tests."""

import pytest
import torch

from sml.adaptation.merge import (
    compute_forbidden_releakage,
    compute_subspace_interference_ratio,
    extract_top_singular_subspace,
    naive_linear_merge,
    run_h12_subspace_safe_merge_benchmark,
    run_llm_h12_subspace_safe_merge_benchmark,
    subspace_safe_ties_merge,
    ties_trim_and_elect,
)

TOL = 1e-5


def test_extract_top_singular_subspace_shape_and_orthogonality():
    torch.manual_seed(0)
    delta = torch.randn(16, 8)
    u = extract_top_singular_subspace(delta, rank_k=3)
    assert u.shape == (16, 3)
    assert u.dtype == torch.float32
    gram = u.T @ u
    assert torch.allclose(gram, torch.eye(3), atol=TOL)


def test_interference_and_releakage_metrics():
    torch.manual_seed(1)
    delta = torch.randn(16, 8)
    u = extract_top_singular_subspace(delta, rank_k=3)
    ratio = compute_subspace_interference_ratio(delta, u)
    assert 0.0 <= ratio <= 1.0 + TOL
    assert ratio > 0.5  # top singular directions carry most energy
    assert compute_subspace_interference_ratio(delta, torch.zeros(16, 0)) == 0.0
    # forbidden directions orthogonal to delta: no releakage
    q, _ = torch.linalg.qr(torch.randn(16, 2))
    proj = delta - q @ (q.T @ delta)
    leak = compute_forbidden_releakage(proj, [q[:, 0], q[:, 1]])
    assert leak < 1e-5
    # delta pushing along a forbidden direction does leak
    leaky = delta + 10.0 * q[:, 0:1]
    assert compute_forbidden_releakage(leaky, [q[:, 0]]) > 1.0


def test_ties_trim_and_elect_validation_and_disjoint_signs():
    d1 = torch.tensor([[2.0, -3.0], [0.5, 4.0]])
    d2 = torch.tensor([[-1.0, -1.0], [-2.0, 1.0]])
    merged = ties_trim_and_elect([d1, d2], density=1.0, weights=[1.0, 1.0])
    assert merged.shape == d1.shape
    # Entry (0,0): +2.0 vs -1.0 -> elected +1 -> keeps +2.0 only
    # Entry (0,1): -3.0 vs -1.0 -> elected -1 -> keeps -3.0 + -1.0 = -4.0
    # Entry (1,0): +0.5 vs -2.0 -> elected -1 -> keeps -2.0 only
    # Entry (1,1): +4.0 vs +1.0 -> elected +1 -> keeps +4.0 + +1.0 = +5.0
    expected = torch.tensor([[2.0, -4.0], [-2.0, 5.0]])
    assert torch.allclose(merged, expected, atol=TOL)


def test_ties_trim_and_elect_validation():
    deltas = [torch.randn(12, 6) for _ in range(3)]
    with pytest.raises(ValueError):
        ties_trim_and_elect([])
    with pytest.raises(ValueError):
        ties_trim_and_elect(deltas, density=0.5, weights=[1.0, 1.0])
    with pytest.raises(ValueError):
        ties_trim_and_elect(deltas, density=1.5)
    with pytest.raises(ValueError):
        ties_trim_and_elect(deltas, density=0.0)


def test_naive_vs_subspace_safe_merge_safety():
    torch.manual_seed(3)
    base = torch.randn(16, 12)
    delta_sql = torch.randn(16, 12) * 0.1
    delta_unslop = torch.randn(16, 12) * 0.1
    u_cols = extract_top_singular_subspace(delta_sql, rank_k=2)
    forbidden = [u_cols[:, i] for i in range(u_cols.shape[1])]

    naive = naive_linear_merge(base, delta_sql, delta_unslop)
    safe = subspace_safe_ties_merge(
        base, delta_sql, delta_unslop, forbidden_dirs=forbidden, sql_svd_rank=2
    )

    assert safe["subspace_safe_releakage"] < 1e-5
    assert safe["biprojected_sql_interference"] < 1e-5
    assert naive["merged_weight"].shape == base.shape
    # row norm preservation on the merged weight
    base_norms = base.norm(dim=-1)
    safe_norms = safe["merged_weight"].norm(dim=-1)
    assert torch.allclose(base_norms, safe_norms, rtol=0.25)


def _assert_acceptance(result: dict) -> None:
    assert result["h12_verified"] is True
    assert result["sql_incident_retention_ratio"] >= 0.95
    assert result["slop_and_refusal_rate"] <= 0.02
    assert result["subspace_safe_merge_releakage"] < 1e-5
    assert result["ties_biprojected_vs_naive_merge_gain"] >= 0.15
    assert result["provider_ready"] is True
    assert result["max_row_crest_factor"] < 25.0


def test_h12_benchmark_acceptance():
    result = run_h12_subspace_safe_merge_benchmark(seed=2026)
    _assert_acceptance(result)
    for arm in result["arms"].values():
        assert 0.0 <= arm["g_evidence"] <= 1.0
        assert 0.0 <= arm["r_correctness"] <= 1.0
        expected_joint = 0.5 * arm["sql_incident_score"] + 0.5 * arm["unslop_gated_score"]
        assert arm["joint_benchmark_score"] == pytest.approx(expected_joint, abs=1e-9)


def test_llm_h12_benchmark_acceptance():
    result = run_llm_h12_subspace_safe_merge_benchmark(
        model_name="sml-nonexistent/ci-fallback",
        seed=2026,
    )
    assert result["backend"] in {"nano", "llm"}
    checks = result.get("acceptance_checks", {})
    assert checks.get("llm_gate_slop_and_refusal_rate") is True
    assert checks.get("llm_gate_sql_incident_retention") is True
    assert checks.get("llm_gate_merge_gain") is True
    assert checks.get("provider_ready") is True
    _assert_acceptance(result)


def test_cli_verify_h12_end_to_end(tmp_path):
    from sml.cli import main

    rc = main(
        [
            "verify-h12",
            "--experiment",
            "experiments/h12_subspace_safe_merge.yaml",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "test_h12",
        ]
    )
    assert rc == 0
    assert (tmp_path / "test_h12" / "subspace_safe_merge.json").exists()
