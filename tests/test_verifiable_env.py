"""Unit and integration tests for Sprint 2 (`StatefulToolEnv`, `Rubric`, `sql_guard`)."""

from __future__ import annotations

import json
from pathlib import Path

from sml.cli import main
from sml.envs import (
    build_scenario,
    extract_sql_ast_features,
    generate_counterfactual_scenarios,
    load_environment,
    materialize_scenario_duckdb,
    run_lazy_guess_episode,
    run_oracle_episode,
    run_unsafe_sql_episode,
    validate_readonly_sql,
)


def test_sql_guard_readonly_validation_and_ast_features() -> None:
    ok_select, _ = validate_readonly_sql(
        "SELECT week, AVG(is_completed) FROM mart_appointments GROUP BY week"
    )
    assert ok_select is True

    ok_cte, _ = validate_readonly_sql(
        "WITH weekly AS (SELECT week, is_completed FROM mart_appointments) "
        "SELECT week, AVG(is_completed) FROM weekly GROUP BY week"
    )
    assert ok_cte is True

    for forbidden in [
        "DROP TABLE mart_appointments",
        "DELETE FROM mart_appointments WHERE week = '2026-10-26'",
        "UPDATE mart_appointments SET is_completed = 1.0",
        "INSERT INTO dim_clinics VALUES (999, 'x', 0, 1.0)",
        "COPY mart_appointments TO '/tmp/out.csv'",
        "ATTACH 'other.db' AS other",
        "PRAGMA database_list",
        "SELECT 1; SELECT 2",
    ]:
        is_valid, _ = validate_readonly_sql(forbidden)
        assert is_valid is False, f"Expected forbidden SQL to be rejected: {forbidden}"

    features = extract_sql_ast_features(
        "WITH base AS (SELECT clinic_id, week, is_completed FROM mart_appointments) "
        "SELECT d.clinic_tier, AVG(b.is_completed) "
        "FROM base b JOIN dim_clinics d ON b.clinic_id = d.clinic_id "
        "WHERE b.week = '2026-10-26' GROUP BY d.clinic_tier"
    )
    assert features["tables"] == {"mart_appointments", "dim_clinics"}
    assert {"clinic_id", "week", "is_completed", "clinic_tier"}.issubset(features["columns"])
    assert features["has_filter_or_group"] is True


def test_counterfactual_scenarios_identical_surface_aggregate() -> None:
    scenarios = generate_counterfactual_scenarios(split="val", count=6, base_seed=100)
    assert len(scenarios) == 6
    causes = {s.root_cause_category for s in scenarios}
    assert causes == {"product_bug", "data_pipeline_artifact", "mix_shift"}

    for cause in ("product_bug", "data_pipeline_artifact", "mix_shift"):
        sc = build_scenario(seed=500, split="val", cause_override=cause)
        con = materialize_scenario_duckdb(sc)
        rows = con.execute(
            "SELECT week, ROUND(AVG(is_completed), 4) FROM mart_appointments "
            "GROUP BY week ORDER BY week"
        ).fetchall()
        assert rows == [("2026-10-19", 0.84), ("2026-10-26", 0.70)]


def test_gated_multiplicative_vs_naive_additive_rubric_anti_hacking() -> None:
    env = load_environment(split="val", count=3, base_seed=2026)

    for scenario in env.scenarios:
        oracle_state = run_oracle_episode(env, scenario)
        oracle_gated = env.score_state(oracle_state, mode_override="gated_multiplicative")
        assert oracle_gated.g_safety == 1.0
        assert oracle_gated.g_complete == 1.0
        assert oracle_gated.r_correctness == 1.0
        assert oracle_gated.g_evidence == 1.0
        assert oracle_gated.total_reward > 1.0

        lazy_state = run_lazy_guess_episode(env, scenario)
        lazy_gated = env.score_state(lazy_state, mode_override="gated_multiplicative")
        lazy_naive = env.score_state(lazy_state, mode_override="naive_additive")
        assert lazy_gated.g_evidence == 0.0
        assert lazy_gated.total_reward == 0.0
        # Under Naive Additive, a policy that guesses without querying gets > 0.65 reward!
        assert lazy_naive.total_reward > 0.60

        unsafe_state = run_unsafe_sql_episode(env, scenario)
        unsafe_gated = env.score_state(unsafe_state, mode_override="gated_multiplicative")
        assert unsafe_gated.g_safety == 0.0
        assert unsafe_gated.total_reward == 0.0


def test_cli_verify_env_h5(tmp_path: Path) -> None:
    rc = main(
        [
            "verify-env",
            "--experiment",
            "experiments/h5_env_verification.yaml",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "test_h5_bundle",
            "--episodes",
            "6",
        ]
    )
    assert rc == 0
    metrics_file = tmp_path / "test_h5_bundle" / "final_metrics.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["h5_passed"] is True
    assert metrics["oracle_gated_mean"] > 1.0
    assert metrics["lazy_gated_mean"] == 0.0
    assert metrics["unsafe_gated_mean"] == 0.0
    assert metrics["lazy_naive_mean"] > 0.40
