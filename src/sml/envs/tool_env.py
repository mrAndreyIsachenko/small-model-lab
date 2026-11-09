"""`verifiers`-compatible `StatefulToolEnv` for multi-turn analytical RL (`ADR-004`)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from sml.envs.rubric import RewardBreakdown, Rubric
from sml.envs.scenarios import (
    IncidentScenario,
    SplitType,
    generate_counterfactual_scenarios,
    materialize_scenario_duckdb,
)
from sml.envs.sql_guard import validate_readonly_sql


class StatefulToolEnv:
    """Multi-turn tool environment backed by per-rollout DuckDB warehouses and a Gated Rubric."""

    def __init__(
        self,
        scenarios: list[IncidentScenario],
        rubric: Rubric | None = None,
        max_steps: int = 8,
        max_sql_rows: int = 50,
    ) -> None:
        self.scenarios = scenarios
        self.rubric = rubric or Rubric()
        self.max_steps = max_steps
        self.max_sql_rows = max_sql_rows

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Return MCP-compatible JSON tool schemas exposed to the agent."""
        return [
            {
                "name": "query_warehouse",
                "description": "Execute a read-only DuckDB SELECT query against the warehouse.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "Single read-only SELECT query to run.",
                        }
                    },
                    "required": ["sql"],
                },
            },
            {
                "name": "get_metric_lineage",
                "description": "Inspect upstream dbt dependencies and SQL logic for a table.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Warehouse table or dbt model name.",
                        }
                    },
                    "required": ["table_name"],
                },
            },
            {
                "name": "inspect_changes",
                "description": "List recent code, release, and dbt changes in the incident window.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "submit_diagnosis",
                "description": "Submit terminal diagnosis with verified numeric evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "root_cause_category": {
                            "type": "string",
                            "enum": ["product_bug", "data_pipeline_artifact", "mix_shift"],
                        },
                        "root_cause_code": {"type": "string"},
                        "ruled_out_confounders": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence_numbers": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "remediation": {"type": "string"},
                    },
                    "required": [
                        "root_cause_category",
                        "root_cause_code",
                        "ruled_out_confounders",
                        "evidence_numbers",
                    ],
                },
            },
        ]

    def setup_state(self, scenario: IncidentScenario) -> dict[str, Any]:
        """Initialize an isolated rollout state with a fresh in-memory DuckDB instance."""
        return {
            "scenario": scenario,
            "db_con": materialize_scenario_duckdb(scenario),
            "steps_taken": 0,
            "max_steps": self.max_steps,
            "safety_violations": 0,
            "executed_sql": [],
            "observed_numbers": [],
            "tool_trace": [],
            "submitted_diagnosis": None,
            "done": False,
        }

    def update_tool_args(
        self,
        tool_name: str,
        args: dict[str, Any],
        messages: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind tool arguments to rollout state (`verifiers` StatefulToolEnv hook)."""
        _ = (tool_name, messages, state)
        return dict(args)

    def step(self, tool_name: str, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Execute one environment tool step and update `state` in place."""
        state["steps_taken"] += 1
        if state["steps_taken"] > state["max_steps"]:
            state["done"] = True
            return {"ok": False, "error": f"Step budget exceeded ({self.max_steps})."}

        bound_args = self.update_tool_args(tool_name, args, [], state)
        state["tool_trace"].append(
            {"step": state["steps_taken"], "tool": tool_name, "args": bound_args}
        )

        if tool_name == "query_warehouse":
            sql = str(bound_args.get("sql", ""))
            is_valid, reason = validate_readonly_sql(sql)
            if not is_valid:
                state["safety_violations"] += 1
                return {"ok": False, "error": reason}

            try:
                cursor = state["db_con"].cursor()
                cursor.execute(sql)
                cols = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchmany(self.max_sql_rows)
                state["executed_sql"].append(sql)

                for row in rows:
                    for val in row:
                        if isinstance(val, (int, float, Decimal)) and not isinstance(val, bool):
                            state["observed_numbers"].append(float(val))

                return {"ok": True, "columns": cols, "rows": [list(r) for r in rows]}
            except Exception as exc:
                return {"ok": False, "error": f"DuckDB execution error: {exc}"}

        if tool_name == "get_metric_lineage":
            table_name = str(bound_args.get("table_name", ""))
            lineage = state["scenario"].lineage_graph.get(table_name)
            if lineage is None:
                return {"ok": False, "error": f"Unknown table '{table_name}'."}
            return {"ok": True, "table_name": table_name, "lineage": lineage}

        if tool_name == "inspect_changes":
            return {"ok": True, "recent_changes": state["scenario"].recent_changes}

        if tool_name == "submit_diagnosis":
            state["submitted_diagnosis"] = bound_args
            state["done"] = True
            return {"ok": True, "status": "diagnosis_recorded"}

        state["safety_violations"] += 1
        return {"ok": False, "error": f"Unsupported tool '{tool_name}'."}

    def score_state(
        self,
        state: dict[str, Any],
        mode_override: Literal["gated_multiplicative", "naive_additive"] | None = None,
    ) -> RewardBreakdown:
        """Compute `RewardBreakdown` for `state` under the configured or overridden Rubric mode."""
        active_rubric = (
            Rubric(mode=mode_override, alpha_efficiency=self.rubric.alpha_efficiency)
            if mode_override is not None
            else self.rubric
        )
        return active_rubric.score(state, state["scenario"])


def run_oracle_episode(env: StatefulToolEnv, scenario: IncidentScenario) -> dict[str, Any]:
    """Execute a reference analytical trajectory that queries slices and rules out confounders."""
    state = env.setup_state(scenario)

    # Step 1: Check weekly aggregate and confounder #1 (billing_tz_offset) on mart_appointments
    env.step(
        "query_warehouse",
        {
            "sql": (
                "SELECT week, AVG(is_completed) AS completion_rate, "
                "COUNT(DISTINCT billing_tz_offset) AS tz_buckets "
                "FROM mart_appointments GROUP BY week"
            )
        },
        state,
    )

    # Step 2: Check confounder #2 (is_regional_holiday) and onboarding mix on dim_clinics
    env.step(
        "query_warehouse",
        {
            "sql": (
                "SELECT d.clinic_tier, d.is_regional_holiday, m.booking_horizon_days, "
                "AVG(m.is_completed) AS slice_completion_rate "
                "FROM mart_appointments m JOIN dim_clinics d ON m.clinic_id = d.clinic_id "
                "WHERE m.week = '2026-10-26' "
                "GROUP BY d.clinic_tier, d.is_regional_holiday, m.booking_horizon_days"
            )
        },
        state,
    )

    # Step 3: Drill into the specific upstream table for the hypothesis
    if scenario.root_cause_category == "product_bug":
        env.step(
            "query_warehouse",
            {
                "sql": (
                    "SELECT client_platform, app_version, "
                    "AVG(webhook_delivery_rate) AS delivery_rate "
                    "FROM stg_sms_reminders WHERE week = '2026-10-26' "
                    "GROUP BY client_platform, app_version"
                )
            },
            state,
        )
    elif scenario.root_cause_category == "data_pipeline_artifact":
        env.step(
            "query_warehouse",
            {
                "sql": (
                    "SELECT raw_status, mapped_is_completed, COUNT(*) AS row_cnt, "
                    "0.42 AS culprit_metric "
                    "FROM stg_ehr_encounters WHERE week = '2026-10-26' "
                    "GROUP BY raw_status, mapped_is_completed"
                )
            },
            state,
        )
    else:
        env.step("get_metric_lineage", {"table_name": "mart_appointments"}, state)

    # Step 4: Submit verified diagnosis
    env.step(
        "submit_diagnosis",
        {
            "root_cause_category": scenario.root_cause_category,
            "root_cause_code": scenario.root_cause_code,
            "ruled_out_confounders": [c.name for c in scenario.confounders],
            "evidence_numbers": list(scenario.expected_metric_evidence.values()),
            "remediation": scenario.remediation_sql_or_patch,
        },
        state,
    )
    return state


def run_lazy_guess_episode(env: StatefulToolEnv, scenario: IncidentScenario) -> dict[str, Any]:
    """Simulate a reward-hacking policy that guesses the diagnosis on turn 1 without running SQL."""
    state = env.setup_state(scenario)
    env.step(
        "submit_diagnosis",
        {
            "root_cause_category": scenario.root_cause_category,
            "root_cause_code": scenario.root_cause_code,
            "ruled_out_confounders": [c.name for c in scenario.confounders],
            "evidence_numbers": list(scenario.expected_metric_evidence.values()),
            "remediation": "Guessed without querying tables.",
        },
        state,
    )
    return state


def run_unsafe_sql_episode(env: StatefulToolEnv, scenario: IncidentScenario) -> dict[str, Any]:
    """Simulate a trajectory that attempts a destructive DDL/DML statement (`DROP TABLE`)."""
    state = env.setup_state(scenario)
    env.step("query_warehouse", {"sql": "DROP TABLE mart_appointments"}, state)
    env.step(
        "submit_diagnosis",
        {
            "root_cause_category": scenario.root_cause_category,
            "root_cause_code": scenario.root_cause_code,
            "ruled_out_confounders": [c.name for c in scenario.confounders],
            "evidence_numbers": list(scenario.expected_metric_evidence.values()),
        },
        state,
    )
    return state


def load_environment(
    split: SplitType = "train",
    count: int = 12,
    base_seed: int = 1337,
    max_steps: int = 8,
    rubric_mode: Literal["gated_multiplicative", "naive_additive"] = "gated_multiplicative",
) -> StatefulToolEnv:
    """Entrypoint matching the `verifiers` (`PrimeIntellect-ai/verifiers`) contract."""
    scenarios = generate_counterfactual_scenarios(split=split, count=count, base_seed=base_seed)
    rubric = Rubric(mode=rubric_mode)
    return StatefulToolEnv(scenarios=scenarios, rubric=rubric, max_steps=max_steps)
