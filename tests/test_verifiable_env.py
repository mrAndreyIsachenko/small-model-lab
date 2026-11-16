"""Unit and integration tests for Sprint 2 (`StatefulToolEnv`, `Rubric`, `sql_guard`)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

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
from sml.envs.tool_env import (
    build_masked_trajectory_tokens,
    format_incident_prompt,
    parse_tool_call,
    run_llm_episode,
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


class _DummyTokenizer:
    """Lightweight stand-in for `AutoTokenizer` with an HF-style chat template."""

    _CHUNK_RE = re.compile(r"<\|[^>]*\|>|<[^>]*>|\s+|[^\s]+")

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._tokens: list[str] = []

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._CHUNK_RE.findall(text):
            if chunk not in self._vocab:
                self._vocab[chunk] = len(self._tokens)
                self._tokens.append(chunk)
            ids.append(self._vocab[chunk])
        return ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **_kwargs: Any,
    ) -> list[int] | str:
        parts = [
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        ]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        return self.encode(text) if tokenize else text

    def decode(self, ids: list[int], skip_special_tokens: bool = False, **_kwargs: Any) -> str:
        tokens = [self._tokens[int(token)] for token in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if not (t.startswith("<|") and t.endswith("|>"))]
        return "".join(tokens)


class _DummyModel:
    """Scripted stand-in for `AutoModelForCausalLM` that replays fixed assistant turns."""

    def __init__(self, tokenizer: _DummyTokenizer, scripted_turns: list[str]) -> None:
        self.tokenizer = tokenizer
        self.scripted_turns = list(scripted_turns)
        self.generate_calls = 0

    def _flatten_ids(self, value: Any) -> list[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        flat = list(value)
        if flat and isinstance(flat[0], (list, tuple)):
            flat = list(flat[0])
        return [int(token) for token in flat]

    def generate(self, input_ids: Any = None, **_kwargs: Any) -> list[list[int]]:
        self.generate_calls += 1
        turn = self.scripted_turns[min(self.generate_calls - 1, len(self.scripted_turns) - 1)]
        prompt_ids = self._flatten_ids(input_ids)
        response_ids = self.tokenizer.encode(turn)
        eos_id = self.tokenizer.encode("<|im_end|>")[0]
        return [prompt_ids + response_ids + [eos_id]]


def test_parse_tool_call_variants_and_aliases() -> None:
    name, args = parse_tool_call(
        '<tool_call>{"name": "query_warehouse", "arguments": {"sql": "SELECT 1"}}</tool_call>'
    )
    assert name == "query_warehouse"
    assert args == {"sql": "SELECT 1"}

    # Fenced ```json block with `tool`/`args` keys plus an alias.
    name, args = parse_tool_call(
        'I will inspect the lineage.\n```json\n'
        '{"tool": "inspect_schema", "args": {"table_name": "mart_appointments"}}\n```'
    )
    assert name == "get_metric_lineage"
    assert args == {"table_name": "mart_appointments"}

    # Fenced block without a language tag.
    name, args = parse_tool_call(
        '```\n{"name": "inspect_changes", "arguments": {}}\n```'
    )
    assert name == "inspect_changes"
    assert args == {}

    # Bare JSON object with an alias.
    name, args = parse_tool_call(
        '{"name": "submit_root_cause", "arguments": {"root_cause_category": "mix_shift"}}'
    )
    assert name == "submit_diagnosis"
    assert args == {"root_cause_category": "mix_shift"}

    # Unterminated <tool_call> tag falls back to the bare JSON scan.
    name, args = parse_tool_call('<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT 1"}}')
    assert name == "query_warehouse"
    assert args == {"sql": "SELECT 1"}

    assert parse_tool_call("Let me think about the symptom first.") is None
    assert parse_tool_call("") is None
    assert parse_tool_call('{"sql": "SELECT 1"}') is None


def test_format_incident_prompt_lists_tools_and_incident_fields() -> None:
    env = load_environment(split="val", count=1, base_seed=7)
    scenario = env.scenarios[0]
    messages = format_incident_prompt(env, scenario, extra_context="Dashboard link attached.")

    assert [message["role"] for message in messages] == ["system", "user"]
    system = messages[0]["content"]
    for tool in ("query_warehouse", "get_metric_lineage", "inspect_changes", "submit_diagnosis"):
        assert tool in system
    assert "<tool_call>" in system

    user = messages[1]["content"]
    assert "Incident report:" in user
    assert scenario.question in user
    assert "Dashboard link attached." in user


def test_step_budget_in_prompt_and_tool_observation() -> None:
    env = load_environment(split="val", count=1, base_seed=7, max_steps=3)
    scenario = env.scenarios[0]

    system_msg = format_incident_prompt(env, scenario)[0]["content"]
    assert f"Step budget: {env.max_steps} steps max" in system_msg
    assert f"call submit_diagnosis on step {env.max_steps}" in system_msg

    tokenizer = _DummyTokenizer()
    model = _DummyModel(
        tokenizer,
        [
            '<tool_call>{"name": "inspect_changes", "arguments": {}}</tool_call>',
            "Still gathering context.",
        ],
    )
    traj = run_llm_episode(env, scenario, model, tokenizer)

    observations = [
        m["content"] for m in traj.messages if m["content"].startswith("Tool observation")
    ]
    assert len(observations) == 1
    assert "[Step 1/" in observations[0]
    assert "submit_diagnosis" in observations[0]


def test_build_masked_trajectory_tokens_masks_assistant_spans_only() -> None:
    tokenizer = _DummyTokenizer()
    messages = [
        {"role": "system", "content": "You investigate incidents."},
        {"role": "user", "content": "Metric dropped."},
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "inspect_changes", "arguments": {}}</tool_call>',
        },
        {"role": "user", "content": "Tool observation: no recent changes."},
        {
            "role": "assistant",
            "content": '<tool_call>{"name": "submit_root_cause", "arguments": {}}</tool_call>',
        },
    ]

    input_ids, mask = build_masked_trajectory_tokens(messages, tokenizer)

    assert len(input_ids) == len(mask)
    assert all(value in (0, 1) for value in mask)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix_len = len(
            tokenizer.apply_chat_template(
                messages[:index], tokenize=True, add_generation_prompt=True
            )
        )
        span_end = len(
            tokenizer.apply_chat_template(
                messages[: index + 1], tokenize=True, add_generation_prompt=False
            )
        )
        span = mask[prefix_len:span_end]
        assert span and all(value == 1 for value in span)
        assert mask[prefix_len - 1] == 0  # token just before each span belongs to a user turn

    masked_text = tokenizer.decode(
        [token for token, flag in zip(input_ids, mask, strict=True) if flag == 1],
        skip_special_tokens=True,
    )
    assert "submit_root_cause" in masked_text
    assert "Tool observation" not in masked_text
    assert "You investigate incidents." not in masked_text


def test_run_llm_episode_dummy_policy_matches_oracle_reward() -> None:
    env = load_environment(split="val", count=1, base_seed=2026, max_steps=6)
    scenario = env.scenarios[0]
    tokenizer = _DummyTokenizer()
    diagnosis = {
        "root_cause_category": scenario.root_cause_category,
        "root_cause_code": scenario.root_cause_code,
        "ruled_out_confounders": [c.name for c in scenario.confounders],
        "evidence_numbers": list(scenario.expected_metric_evidence.values()),
        "remediation": scenario.remediation_sql_or_patch,
    }
    scripted_turns = [
        'Checking the weekly aggregate.\n<tool_call>{"name": "run_sql", "arguments": '
        '{"sql": "SELECT week, AVG(is_completed) AS completion_rate, '
        'COUNT(DISTINCT billing_tz_offset) AS tz_buckets '
        'FROM mart_appointments GROUP BY week"}}</tool_call>',
        '<tool_call>{"tool": "run_sql", "args": {"sql": "SELECT d.clinic_tier, '
        "d.is_regional_holiday, m.booking_horizon_days, "
        "AVG(m.is_completed) AS slice_completion_rate "
        "FROM mart_appointments m JOIN dim_clinics d ON m.clinic_id = d.clinic_id "
        "WHERE m.week = '2026-10-26' "
        'GROUP BY d.clinic_tier, d.is_regional_holiday, m.booking_horizon_days"}}</tool_call>',
    ]
    if scenario.root_cause_category == "product_bug":
        scripted_turns.append(
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT client_platform, '
            "app_version, AVG(webhook_delivery_rate) AS delivery_rate "
            "FROM stg_sms_reminders WHERE week = '2026-10-26' "
            'GROUP BY client_platform, app_version"}}</tool_call>'
        )
    elif scenario.root_cause_category == "data_pipeline_artifact":
        scripted_turns.append(
            '<tool_call>{"name": "run_sql", "arguments": {"sql": "SELECT raw_status, '
            "mapped_is_completed, COUNT(*) AS row_cnt, 0.42 AS culprit_metric "
            "FROM stg_ehr_encounters WHERE week = '2026-10-26' "
            'GROUP BY raw_status, mapped_is_completed"}}</tool_call>'
        )
    else:
        scripted_turns.append(
            '<tool_call>{"name": "inspect_schema", '
            '"arguments": {"table_name": "mart_appointments"}}</tool_call>'
        )
    scripted_turns.append(
        "Submitting the verified diagnosis.\n<tool_call>"
        + json.dumps(
            {"name": "submit_root_cause", "arguments": diagnosis}, sort_keys=True
        )
        + "</tool_call>"
    )
    model = _DummyModel(tokenizer, scripted_turns)

    traj = run_llm_episode(env, scenario, model, tokenizer)

    assert model.generate_calls == 4
    assert traj.state["done"] is True
    assert traj.state["steps_taken"] == 4
    assert traj.state["safety_violations"] == 0
    assert [entry["tool"] for entry in traj.state["tool_trace"]] == [
        "query_warehouse",
        "query_warehouse",
        (
            "get_metric_lineage"
            if scenario.root_cause_category == "mix_shift"
            else "query_warehouse"
        ),
        "submit_diagnosis",
    ]

    # The scripted policy replays run_oracle_episode's exact actions, so rewards must match.
    oracle_state = run_oracle_episode(env, scenario)
    assert traj.breakdown is not None
    assert traj.reward == traj.breakdown.total_reward
    assert traj.reward == env.score_state(oracle_state).total_reward
    assert traj.reward > 1.0
    assert traj.breakdown.g_safety == 1.0
    assert traj.breakdown.g_evidence == 1.0

    assert traj.messages[0]["role"] == "system"
    assert traj.messages[1]["role"] == "user"
    assistant_contents = [m["content"] for m in traj.messages if m["role"] == "assistant"]
    assert len(assistant_contents) == 4
    observations = [
        m["content"] for m in traj.messages if m["content"].startswith("Tool observation")
    ]
    assert len(observations) == 4
    assert '"status": "diagnosis_recorded"' in observations[-1]

    first_assistant_index = next(
        i for i, m in enumerate(traj.messages) if m["role"] == "assistant"
    )
    span_start = len(
        tokenizer.apply_chat_template(
            traj.messages[:first_assistant_index], tokenize=True, add_generation_prompt=True
        )
    )
    assert len(traj.input_ids) == len(traj.assistant_action_mask)
    assert traj.assistant_action_mask[:span_start] == [0] * span_start
    assert sum(traj.assistant_action_mask) > 0
    masked_text = tokenizer.decode(
        [
            token
            for token, flag in zip(traj.input_ids, traj.assistant_action_mask, strict=True)
            if flag == 1
        ],
        skip_special_tokens=True,
    )
    assert "submit_root_cause" in masked_text
    assert "Tool observation" not in masked_text
    assert "incident response agent" not in masked_text


def test_run_llm_episode_format_error_feedback_and_step_budget() -> None:
    env = load_environment(split="val", count=1, base_seed=7, max_steps=3)
    scenario = env.scenarios[0]
    tokenizer = _DummyTokenizer()
    model = _DummyModel(tokenizer, ["I should think carefully before acting."])

    traj = run_llm_episode(env, scenario, model, tokenizer)

    assert model.generate_calls == 3  # loop stops at env.max_steps
    assert traj.state["steps_taken"] == 0
    assert traj.state["done"] is False
    format_errors = [
        m for m in traj.messages if m["role"] == "user" and m["content"].startswith("Format error")
    ]
    assert len(format_errors) == 3
    assert traj.breakdown is not None
    assert traj.reward == traj.breakdown.total_reward
    assert traj.reward == 0.0
