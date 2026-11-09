from dataclasses import dataclass
from typing import Any, Literal

from sml.envs.scenarios import IncidentScenario
from sml.envs.sql_guard import extract_sql_ast_features


@dataclass(frozen=True)
class RewardBreakdown:
    total_reward: float
    g_safety: float
    g_complete: float
    r_correctness: float
    g_evidence: float
    r_efficiency: float
    mode: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "g_safety": self.g_safety,
            "g_complete": self.g_complete,
            "r_correctness": self.r_correctness,
            "g_evidence": self.g_evidence,
            "r_efficiency": self.r_efficiency,
            "mode": self.mode,
            "diagnostics": self.diagnostics,
        }


class Rubric:
    def __init__(
        self,
        mode: Literal["gated_multiplicative", "naive_additive"] = "gated_multiplicative",
        alpha_efficiency: float = 0.15,
    ):
        self.mode = mode
        self.alpha_efficiency = alpha_efficiency

    def score(self, state: dict[str, Any], scenario: IncidentScenario) -> RewardBreakdown:
        # Safety gate
        g_safety = 1.0 if state.get("safety_violations", 0) == 0 else 0.0

        # Completion gate
        submitted = state.get("submitted_diagnosis")
        steps_taken = int(state.get("steps_taken", 0))
        max_steps = int(state.get("max_steps", 8))
        g_complete = 1.0 if (submitted is not None and 0 < steps_taken <= max_steps) else 0.0

        # Correctness reward
        if submitted is None:
            r_correctness = 0.0
        else:
            # Root cause category score
            cat_score = (
                0.60
                if submitted.get("root_cause_category") == scenario.root_cause_category
                else 0.0
            )

            # Root cause code score
            code_score = (
                0.20 if submitted.get("root_cause_code") == scenario.root_cause_code else 0.0
            )

            # Confounder exclusion score
            ruled_out = set(submitted.get("ruled_out_confounders") or [])
            expected_confounders = {c.name for c in scenario.confounders}
            conf_score = 0.20 * (
                len(ruled_out & expected_confounders) / max(1, len(expected_confounders))
            )

            r_correctness = round(cat_score + code_score + conf_score, 6)

        # Evidence gate
        executed_sql = state.get("executed_sql", [])
        queried_tables = set()
        queried_columns = set()

        # Extract features from all executed SQL
        for sql in executed_sql:
            try:
                features = extract_sql_ast_features(sql)
                queried_tables.update(features.get("tables", []))
                queried_columns.update(features.get("columns", []))
            except Exception:
                # If parsing fails, we simply don't count this query
                pass

        # Convert to lowercase for comparison
        req_tables = {t.lower() for t in scenario.required_tables}

        # Hard anti-guessing gate: must query all required tables
        if not req_tables.issubset({t.lower() for t in queried_tables}):
            g_evidence = 0.0
        else:
            # Slice coverage
            req_cols = {c.lower() for c in scenario.required_slice_columns}
            slice_cov = len(req_cols & {c.lower() for c in queried_columns}) / max(1, len(req_cols))

            # Confounder column coverage
            conf_cols = {c.column_to_check.lower() for c in scenario.confounders}
            conf_cov = len(conf_cols & {c.lower() for c in queried_columns}) / max(
                1, len(conf_cols)
            )

            # Citation verification
            cited_nums = (
                [float(x) for x in (submitted.get("evidence_numbers") or [])] if submitted else []
            )
            observed_nums = [float(x) for x in state.get("observed_numbers", [])]
            expected_nums = list(scenario.expected_metric_evidence.values())

            verified_count = 0
            for exp_val in expected_nums:
                # Check if this expected value is both cited and observed
                cited_match = any(abs(c - exp_val) <= 1e-2 for c in cited_nums)
                observed_match = any(abs(o - exp_val) <= 1e-2 for o in observed_nums)
                if cited_match and observed_match:
                    verified_count += 1

            citation_cov = verified_count / max(1, len(expected_nums))

            # Combine evidence components
            g_evidence = round(0.40 * slice_cov + 0.30 * conf_cov + 0.30 * citation_cov, 6)

        # Efficiency reward
        if steps_taken > 0:
            # Efficiency is maximized when taking fewer steps beyond minimum required (3)
            r_efficiency = max(0.0, min(1.0, (max_steps - steps_taken) / max(1, max_steps - 3)))
        else:
            r_efficiency = 0.0

        # Compute total reward based on mode
        if self.mode == "gated_multiplicative":
            total = (
                g_safety
                * g_complete
                * (r_correctness * g_evidence * (1.0 + self.alpha_efficiency * r_efficiency))
            )
        else:  # naive_additive
            total = (
                0.35 * r_correctness
                + 0.25 * g_evidence
                + 0.20 * g_safety
                + 0.10 * g_complete
                + 0.10 * r_efficiency
            )

        return RewardBreakdown(
            total_reward=round(total, 6),
            g_safety=g_safety,
            g_complete=g_complete,
            r_correctness=r_correctness,
            g_evidence=g_evidence,
            r_efficiency=r_efficiency,
            mode=self.mode,
            diagnostics={
                "steps_taken": steps_taken,
                "max_steps": max_steps,
                "safety_violations": state.get("safety_violations", 0),
                "slice_cov": round(slice_cov, 6) if "slice_cov" in locals() else 0.0,
                "conf_cov": round(conf_cov, 6) if "conf_cov" in locals() else 0.0,
                "citation_cov": round(citation_cov, 6) if "citation_cov" in locals() else 0.0,
            },
        )
