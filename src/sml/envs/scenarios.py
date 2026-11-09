"""Multi-seed counterfactual analytical & data-incident scenario generator on DuckDB (`ADR-004`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import duckdb

RootCauseCategory = Literal["product_bug", "data_pipeline_artifact", "mix_shift"]
SplitType = Literal["train", "val", "test"]

CAUSE_CATEGORIES: tuple[RootCauseCategory, ...] = (
    "product_bug",
    "data_pipeline_artifact",
    "mix_shift",
)


@dataclass(frozen=True)
class PlantedConfounder:
    """A plausible-sounding distractor change that is statistically flat when queried in SQL."""

    name: str
    description: str
    table_to_check: str
    column_to_check: str


@dataclass(frozen=True)
class IncidentScenario:
    """Deterministic counterfactual episode specification for `StatefulToolEnv`."""

    scenario_id: str
    seed: int
    split: SplitType
    question: str
    root_cause_category: RootCauseCategory
    root_cause_code: str
    required_tables: tuple[str, ...]
    required_slice_columns: tuple[str, ...]
    confounders: tuple[PlantedConfounder, ...]
    expected_metric_evidence: dict[str, float]
    lineage_graph: dict[str, dict[str, Any]]
    recent_changes: list[dict[str, str]]
    remediation_sql_or_patch: str


def build_scenario(
    seed: int,
    split: SplitType = "train",
    cause_override: RootCauseCategory | None = None,
) -> IncidentScenario:
    """Construct a counterfactual scenario with identical surface symptoms across causes."""
    root_cause_category: RootCauseCategory = (
        cause_override if cause_override is not None else CAUSE_CATEGORIES[seed % 3]
    )
    scenario_id = f"{split}_{root_cause_category}_seed{seed}"

    question = (
        "Weekly mart_appointments completion_rate dropped from 0.8400 (week '2026-10-19') to "
        "0.7000 (week '2026-10-26'). Investigate the warehouse tables, dbt lineage, and recent "
        "changes to identify the true root cause, cite exact metric evidence, and rule out "
        "the planted confounders."
    )

    confounders = (
        PlantedConfounder(
            name="billing_tz_backfill",
            description=(
                "Backfilled billing_tz_offset (-5 vs -6) on mart_appointments; completion_rate "
                "is invariant (0.7000 across both offsets in week '2026-10-26')."
            ),
            table_to_check="mart_appointments",
            column_to_check="billing_tz_offset",
        ),
        PlantedConfounder(
            name="holiday_flag_update",
            description=(
                "Updated is_regional_holiday on dim_clinics; completion_rate does not differ "
                "between holiday=0 and holiday=1 clinics."
            ),
            table_to_check="dim_clinics",
            column_to_check="is_regional_holiday",
        ),
    )

    recent_changes = [
        {
            "id": "CHG-101",
            "component": "mobile_app_release",
            "summary": "Rolled out Android client v4.18.2 touching SMS reminder webhook handler.",
        },
        {
            "id": "CHG-102",
            "component": "dbt_stg_ehr_encounters",
            "summary": (
                "Upstream EHR vendor introduced status code 'FINISHED' alongside 'COMPLETED'."
            ),
        },
        {
            "id": "CHG-103",
            "component": "clinic_onboarding",
            "summary": "Onboarded 'onboarding_tier_c' clinics with >=30-day booking horizons.",
        },
        {
            "id": "CHG-104",
            "component": "billing_tz_backfill",
            "summary": "Backfilled billing_tz_offset column in mart_appointments.",
        },
        {
            "id": "CHG-105",
            "component": "holiday_flag_update",
            "summary": "Refreshed is_regional_holiday calendar table in dim_clinics.",
        },
    ]

    lineage_graph: dict[str, dict[str, Any]] = {
        "mart_appointments": {
            "upstream": ["stg_ehr_encounters", "stg_sms_reminders", "dim_clinics"],
            "sql": (
                "SELECT a.appointment_id, a.clinic_id, a.week, "
                "e.mapped_is_completed AS is_completed, "
                "s.client_platform, s.app_version, a.booking_horizon_days, a.billing_tz_offset "
                "FROM raw_appointments a "
                "LEFT JOIN stg_ehr_encounters e USING (appointment_id, week) "
                "LEFT JOIN stg_sms_reminders s USING (appointment_id, week)"
            ),
        },
        "stg_ehr_encounters": {
            "upstream": ["raw_ehr_encounters"],
            "sql": (
                "SELECT encounter_id, appointment_id, week, raw_status, "
                "CASE WHEN raw_status = 'COMPLETED' THEN 1 ELSE 0 END AS mapped_is_completed "
                "FROM raw_ehr_encounters"
            ),
        },
        "stg_sms_reminders": {
            "upstream": ["raw_sms_webhooks"],
            "sql": (
                "SELECT reminder_id, appointment_id, week, client_platform, app_version, "
                "webhook_delivery_rate FROM raw_sms_webhooks"
            ),
        },
        "dim_clinics": {
            "upstream": ["raw_crm_clinics"],
            "sql": (
                "SELECT clinic_id, clinic_tier, is_regional_holiday, baseline_completion_rate "
                "FROM raw_crm_clinics"
            ),
        },
    }

    expected_metric_evidence = {
        "baseline_completion_rate": 0.84,
        "incident_completion_rate": 0.70,
        "culprit_slice_metric": 0.42,
    }

    if root_cause_category == "product_bug":
        root_cause_code = "android_v4_18_2_sms_webhook_regression"
        required_tables = ("mart_appointments", "stg_sms_reminders")
        required_slice_columns = ("client_platform", "app_version")
        remediation_sql_or_patch = (
            "Rollback Android v4.18.2 SMS webhook handler and replay failed reminder webhooks "
            "where client_platform = 'android' AND app_version = '4.18.2'."
        )
    elif root_cause_category == "data_pipeline_artifact":
        root_cause_code = "ehr_status_finished_unmapped_in_dbt"
        required_tables = ("mart_appointments", "stg_ehr_encounters")
        required_slice_columns = ("raw_status", "mapped_is_completed")
        remediation_sql_or_patch = (
            "CASE WHEN raw_status IN ('COMPLETED', 'FINISHED') "
            "THEN 1 ELSE 0 END AS mapped_is_completed"
        )
    else:
        root_cause_code = "onboarding_tier_c_long_horizon_mix_shift"
        required_tables = ("mart_appointments", "dim_clinics")
        required_slice_columns = ("clinic_tier", "booking_horizon_days")
        remediation_sql_or_patch = (
            "Stratify weekly completion reporting by clinic_tier and booking_horizon_days >= 30 "
            "to isolate Simpson's paradox from onboarding_tier_c volume shift."
        )

    return IncidentScenario(
        scenario_id=scenario_id,
        seed=seed,
        split=split,
        question=question,
        root_cause_category=root_cause_category,
        root_cause_code=root_cause_code,
        required_tables=required_tables,
        required_slice_columns=required_slice_columns,
        confounders=confounders,
        expected_metric_evidence=expected_metric_evidence,
        lineage_graph=lineage_graph,
        recent_changes=recent_changes,
        remediation_sql_or_patch=remediation_sql_or_patch,
    )


def materialize_scenario_duckdb(scenario: IncidentScenario) -> duckdb.DuckDBPyConnection:
    """Create an isolated in-memory DuckDB warehouse populated for `scenario`."""
    con = duckdb.connect(":memory:")

    con.execute(
        """
        CREATE TABLE mart_appointments (
            appointment_id INT,
            clinic_id INT,
            week VARCHAR,
            is_completed DOUBLE,
            client_platform VARCHAR,
            app_version VARCHAR,
            booking_horizon_days INT,
            billing_tz_offset INT
        );
        CREATE TABLE stg_ehr_encounters (
            encounter_id INT,
            appointment_id INT,
            week VARCHAR,
            raw_status VARCHAR,
            mapped_is_completed INT
        );
        CREATE TABLE stg_sms_reminders (
            reminder_id INT,
            appointment_id INT,
            week VARCHAR,
            client_platform VARCHAR,
            app_version VARCHAR,
            webhook_delivery_rate DOUBLE
        );
        CREATE TABLE dim_clinics (
            clinic_id INT,
            clinic_tier VARCHAR,
            is_regional_holiday INT,
            baseline_completion_rate DOUBLE
        );
        """
    )

    con.execute(
        """
        INSERT INTO dim_clinics VALUES
        (101, 'core_tier_a', 0, 0.84),
        (102, 'core_tier_a', 1, 0.84),
        (103, 'onboarding_tier_c', 0, 0.42);
        """
    )

    # Week '2026-10-19': 4 rows across clinics 101 & 102 with exact mean 0.8400
    con.execute(
        """
        INSERT INTO mart_appointments VALUES
        (1, 101, '2026-10-19', 0.84, 'ios', '4.18.1', 7, -5),
        (2, 101, '2026-10-19', 0.84, 'android', '4.18.0', 7, -5),
        (3, 102, '2026-10-19', 0.84, 'ios', '4.18.1', 10, -6),
        (4, 102, '2026-10-19', 0.84, 'android', '4.18.0', 10, -6);
        """
    )

    if scenario.root_cause_category == "product_bug":
        # Week '2026-10-26': overall mean = (0.98 + 0.42 + 0.98 + 0.42) / 4 = 0.7000
        # Android v4.18.2 drops to 0.42 completion and 0.42 webhook_delivery_rate
        con.execute(
            """
            INSERT INTO mart_appointments VALUES
            (5, 101, '2026-10-26', 0.98, 'ios', '4.18.1', 7, -5),
            (6, 101, '2026-10-26', 0.42, 'android', '4.18.2', 7, -5),
            (7, 102, '2026-10-26', 0.98, 'ios', '4.18.1', 10, -6),
            (8, 102, '2026-10-26', 0.42, 'android', '4.18.2', 10, -6);

            INSERT INTO stg_sms_reminders VALUES
            (1, 1, '2026-10-19', 'ios', '4.18.1', 0.96),
            (2, 2, '2026-10-19', 'android', '4.18.0', 0.96),
            (3, 5, '2026-10-26', 'ios', '4.18.1', 0.96),
            (4, 6, '2026-10-26', 'android', '4.18.2', 0.42),
            (5, 7, '2026-10-26', 'ios', '4.18.1', 0.96),
            (6, 8, '2026-10-26', 'android', '4.18.2', 0.42);

            INSERT INTO stg_ehr_encounters VALUES
            (1, 5, '2026-10-26', 'COMPLETED', 1),
            (2, 6, '2026-10-26', 'CANCELLED', 0),
            (3, 7, '2026-10-26', 'COMPLETED', 1),
            (4, 8, '2026-10-26', 'CANCELLED', 0);
            """
        )
    elif scenario.root_cause_category == "data_pipeline_artifact":
        # Week '2026-10-26': overall mean = 0.7000; SMS reminders healthy (0.96),
        # but stg_ehr_encounters has unmapped raw_status = 'FINISHED' (mapped_is_completed = 0)
        con.execute(
            """
            INSERT INTO mart_appointments VALUES
            (5, 101, '2026-10-26', 0.70, 'ios', '4.18.1', 7, -5),
            (6, 101, '2026-10-26', 0.70, 'android', '4.18.2', 7, -5),
            (7, 102, '2026-10-26', 0.70, 'ios', '4.18.1', 10, -6),
            (8, 102, '2026-10-26', 0.70, 'android', '4.18.2', 10, -6);

            INSERT INTO stg_sms_reminders VALUES
            (1, 5, '2026-10-26', 'ios', '4.18.1', 0.96),
            (2, 6, '2026-10-26', 'android', '4.18.2', 0.96),
            (3, 7, '2026-10-26', 'ios', '4.18.1', 0.96),
            (4, 8, '2026-10-26', 'android', '4.18.2', 0.96);

            INSERT INTO stg_ehr_encounters VALUES
            (1, 5, '2026-10-26', 'COMPLETED', 1),
            (2, 6, '2026-10-26', 'FINISHED', 0),
            (3, 7, '2026-10-26', 'COMPLETED', 1),
            (4, 8, '2026-10-26', 'FINISHED', 0);
            """
        )
    else:
        # mix_shift: onboarding_tier_c (clinic_id=103) with horizon=45 has 0.42 completion
        # Overall mean across 6 rows = (0.84*4 + 0.42*2) / 6 = 4.20 / 6 = 0.7000
        con.execute(
            """
            INSERT INTO mart_appointments VALUES
            (5, 101, '2026-10-26', 0.84, 'ios', '4.18.1', 7, -5),
            (6, 101, '2026-10-26', 0.84, 'android', '4.18.2', 7, -5),
            (7, 102, '2026-10-26', 0.84, 'ios', '4.18.1', 10, -6),
            (8, 102, '2026-10-26', 0.84, 'android', '4.18.2', 10, -6),
            (9, 103, '2026-10-26', 0.42, 'ios', '4.18.1', 45, -5),
            (10, 103, '2026-10-26', 0.42, 'android', '4.18.2', 45, -6);

            INSERT INTO stg_sms_reminders VALUES
            (1, 5, '2026-10-26', 'ios', '4.18.1', 0.96),
            (2, 6, '2026-10-26', 'android', '4.18.2', 0.96),
            (3, 9, '2026-10-26', 'ios', '4.18.1', 0.96),
            (4, 10, '2026-10-26', 'android', '4.18.2', 0.96);

            INSERT INTO stg_ehr_encounters VALUES
            (1, 5, '2026-10-26', 'COMPLETED', 1),
            (2, 6, '2026-10-26', 'COMPLETED', 1),
            (3, 9, '2026-10-26', 'COMPLETED', 1),
            (4, 10, '2026-10-26', 'CANCELLED', 0);
            """
        )

    return con


def generate_counterfactual_scenarios(
    split: SplitType = "train",
    count: int = 12,
    base_seed: int = 1337,
) -> list[IncidentScenario]:
    """Generate `count` deterministic counterfactual scenarios across all 3 root causes."""
    return [
        build_scenario(
            seed=base_seed + idx,
            split=split,
            cause_override=CAUSE_CATEGORIES[idx % len(CAUSE_CATEGORIES)],
        )
        for idx in range(count)
    ]
