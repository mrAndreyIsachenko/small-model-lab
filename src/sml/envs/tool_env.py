"""`verifiers`-compatible `StatefulToolEnv` for multi-turn analytical RL (`ADR-004`)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    """Execute a reference analytical trajectory that queries slices and rules out confounders.

    Every category takes exactly 4 steps: two shared `query_warehouse` probes
    (Steps 1-2), one category-specific verification (Step 3), and the terminal
    `submit_diagnosis` (Step 4). Only Step 3 differs by `root_cause_category`.
    """
    state = env.setup_state(scenario)

    # Step 1 (shared): Check weekly aggregate and confounder #1 (billing_tz_offset)
    # on mart_appointments
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

    # Step 2 (shared): Check confounder #2 (is_regional_holiday) and onboarding mix
    # (clinic_tier, booking_horizon_days) via the mart_appointments JOIN dim_clinics
    # slice query — this already covers the mix_shift slices on every category
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

    # Step 3 (category-specific verification): all categories use query_warehouse.
    #   - product_bug / data_pipeline_artifact: query_warehouse on the planted
    #     upstream table (stg_sms_reminders / stg_ehr_encounters).
    #   - mix_shift: query_warehouse on the mart_appointments JOIN dim_clinics
    #     slice for the onboarding_tier_c cohort, surfacing the exact 0.42
    #     culprit metric right before submit_diagnosis.
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
    else:  # mix_shift
        env.step(
            "query_warehouse",
            {
                "sql": (
                    "SELECT d.clinic_tier, m.booking_horizon_days, "
                    "AVG(m.is_completed) AS culprit_metric "
                    "FROM mart_appointments m JOIN dim_clinics d ON m.clinic_id = d.clinic_id "
                    "WHERE m.week = '2026-10-26' AND d.clinic_tier = 'onboarding_tier_c' "
                    "GROUP BY d.clinic_tier, m.booking_horizon_days"
                )
            },
            state,
        )

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


@dataclass
class RolloutTrajectory:
    """A scored multi-turn LLM rollout with token-level action masking."""

    scenario_id: str
    messages: list[dict[str, str]]
    input_ids: list[int]
    assistant_action_mask: list[int]
    state: dict[str, Any]
    reward: float
    breakdown: RewardBreakdown | None = None


_TOOL_CALL_ALIASES = {
    "run_sql": "query_warehouse",
    "inspect_schema": "get_metric_lineage",
    "submit_root_cause": "submit_diagnosis",
}

_TOOL_CALL_PROMPT_SUFFIX = (
    "Emit exactly one tool call per turn, formatted as:\n"
    '<tool_call>{"name": "...", "arguments": {...}}</tool_call>\n'
    "Wait for the tool observation before deciding your next action."
)


def _canonical_tool_name(name: str) -> str:
    """Map legacy tool aliases onto the canonical `tool_schemas` names."""
    stripped = name.strip()
    return _TOOL_CALL_ALIASES.get(stripped, stripped)


def _tool_call_from_payload(payload: Any) -> tuple[str, dict[str, Any]] | None:
    """Interpret a decoded JSON object as a `(tool_name, args)` call, or return None."""
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("tool")
    args = payload.get("arguments")
    if args is None:
        args = payload.get("args")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(args, dict):
        args = {}
    return _canonical_tool_name(name), dict(args)


def _json_payload_candidates(text: str) -> list[Any]:
    """Collect JSON payloads from `<tool_call>` tags, fenced blocks, and bare objects."""
    decoder = json.JSONDecoder()
    payloads: list[Any] = []

    segments: list[str] = []
    segments.extend(
        match.strip()
        for match in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL)
    )
    segments.extend(
        match.strip()
        for match in re.findall(r"```(?:json)?[ \t]*\r?\n?(.*?)```", text, flags=re.DOTALL)
    )
    for segment in segments:
        if not segment:
            continue
        try:
            payloads.append(json.loads(segment))
        except json.JSONDecodeError:
            continue

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        payloads.append(payload)
    return payloads


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Extract `(tool_name, arguments)` from an assistant response, or None if absent."""
    if not isinstance(text, str):
        return None
    for payload in _json_payload_candidates(text):
        parsed = _tool_call_from_payload(payload)
        if parsed is not None:
            return parsed
    return None


def format_incident_prompt(
    env: StatefulToolEnv,
    scenario: IncidentScenario,
    extra_context: str = "",
) -> list[dict[str, str]]:
    """Build the `[system_msg, user_msg]` opening prompt for an incident rollout."""
    tool_lines = "\n".join(
        f"- {json.dumps(schema, sort_keys=True)}" for schema in env.tool_schemas()
    )
    system_msg = (
        "You are an incident response agent investigating a data-quality incident in a "
        "DuckDB warehouse. Investigate systematically: query slices, rule out confounders, "
        "and gather verified numeric evidence before submitting a diagnosis. "
        f"Step budget: {env.max_steps} steps max. Always call submit_diagnosis on step "
        f"{env.max_steps} (or earlier) before the budget runs out. "
        f"Available tools:\n{tool_lines}\n\n{_TOOL_CALL_PROMPT_SUFFIX}"
    )
    incident_fields = (
        ("Symptom", getattr(scenario, "symptom", None)),
        ("Affected metric", getattr(scenario, "affected_metric", None)),
        ("Alert summary", getattr(scenario, "alert_summary", None)),
        ("Window start", getattr(scenario, "window_start", None)),
        ("Window end", getattr(scenario, "window_end", None)),
        ("Incident description", scenario.question),
    )
    user_msg = "Incident report:\n" + "".join(
        f"- {label}: {value}\n" for label, value in incident_fields if value is not None
    )
    if extra_context:
        user_msg += f"\nAdditional context:\n{extra_context}\n"
    user_msg += f"\n{_TOOL_CALL_PROMPT_SUFFIX}"
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def format_tool_observation(
    tool_name: str,
    observation: dict[str, Any],
    state: dict[str, Any],
) -> str:
    """Serialize a tool observation and append a live step-budget counter line."""
    payload = json.dumps(observation, default=str)
    message = f"Tool observation ({tool_name}): {payload}"
    max_steps = int(state.get("max_steps", 0))
    if max_steps > 0 and not state.get("done"):
        steps_taken = int(state.get("steps_taken", 0))
        remaining = max(0, max_steps - steps_taken)
        if remaining <= 1:
            counter = (
                f"[Step {steps_taken}/{max_steps} — {remaining} step left; "
                f"call submit_diagnosis NOW on step {max_steps}]"
            )
        else:
            counter = (
                f"[Step {steps_taken}/{max_steps} — {remaining} steps remaining; "
                f"call submit_diagnosis on step {max_steps}]"
            )
        message = f"{message}\n{counter}"
    return message


def build_masked_trajectory_tokens(
    messages: list[dict[str, str]],
    tokenizer: Any,
) -> tuple[list[int], list[int]]:
    """Tokenize the full chat and mask 1 only on assistant-turn token spans."""
    full_ids = _to_plain_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    mask = [0] * len(full_ids)
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prefix_ids = _to_plain_ids(
            tokenizer.apply_chat_template(
                messages[:index], tokenize=True, add_generation_prompt=True
            )
        )
        with_assistant_ids = _to_plain_ids(
            tokenizer.apply_chat_template(
                messages[: index + 1], tokenize=True, add_generation_prompt=False
            )
        )
        for position in range(len(prefix_ids), min(len(with_assistant_ids), len(full_ids))):
            mask[position] = 1
    return full_ids, mask


def _to_plain_ids(value: Any) -> list[int]:
    """Normalize tensors / nested sequences into a flat `list[int]`."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    flat = list(value)
    if flat and isinstance(flat[0], (list, tuple)):
        flat = list(flat[0])
    return [int(token) for token in flat]


def _resolve_model_device(model: Any) -> Any:
    """Best-effort device lookup for HF / PyTorch models without failing on dummies."""
    device = getattr(model, "device", None)
    if device is not None:
        return device
    params_fn = getattr(model, "parameters", None)
    if callable(params_fn):
        first = next(iter(params_fn()), None)
        if first is not None:
            return first.device
    return None


def _coerce_model_inputs(prompt_ids: list[int], model: Any = None) -> dict[str, Any]:
    """Wrap prompt ids in the batched tensor shape expected by `model.generate`."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for dummy models
        return {"input_ids": [list(prompt_ids)]}
    device = _resolve_model_device(model)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def run_llm_episode(
    env: StatefulToolEnv,
    scenario: IncidentScenario,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    do_sample: bool = True,
    extra_context: str = "",
    rubric_mode: Literal["gated_multiplicative", "naive_additive"] | None = None,
) -> RolloutTrajectory:
    """Roll out an HF `AutoModelForCausalLM` policy against `env` for one scenario."""
    state = env.setup_state(scenario)
    messages = format_incident_prompt(env, scenario, extra_context=extra_context)

    for _step in range(env.max_steps):
        if state["done"]:
            break

        prompt_ids = _to_plain_ids(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
        inputs = _coerce_model_inputs(prompt_ids, model=model)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        output = model.generate(**inputs, **gen_kwargs)
        generated_ids = _to_plain_ids(output[0] if isinstance(output, (list, tuple)) else output)
        new_token_ids = generated_ids[len(prompt_ids):]
        response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)

        messages.append({"role": "assistant", "content": response_text})

        parsed = parse_tool_call(response_text)
        if parsed is None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Format error: no valid tool call found. "
                        f"{_TOOL_CALL_PROMPT_SUFFIX}"
                    ),
                }
            )
            continue

        tool_name, tool_args = parsed
        observation = env.step(tool_name, tool_args, state)
        messages.append(
            {
                "role": "user",
                "content": format_tool_observation(tool_name, observation, state),
            }
        )
        if state["done"]:
            break

    breakdown = env.score_state(state, mode_override=rubric_mode)
    input_ids, assistant_action_mask = build_masked_trajectory_tokens(messages, tokenizer)
    scenario_id = str(
        getattr(scenario, "scenario_id", None) or getattr(scenario, "id", None) or ""
    )
    return RolloutTrajectory(
        scenario_id=scenario_id,
        messages=messages,
        input_ids=input_ids,
        assistant_action_mask=assistant_action_mask,
        state=state,
        reward=breakdown.total_reward,
        breakdown=breakdown,
    )


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
