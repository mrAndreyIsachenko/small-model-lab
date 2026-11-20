"""Unit tests for H17: artifact verification, the Tenet arena-house series,
the domain imatrix corpus builder, and the peer-eval release gate.

Uses real synthetic think-tagged responses (no mocks) covering every house
artifact type: exact numeric, read-only SQL, Python AST, JSON config, and
unslop technical prose.
"""

from __future__ import annotations

from collections import Counter

import pytest

import sml.surgery.gguf_audit as gguf_audit_module
from sml.evals.artifact_verifier import (
    TENET_ARENA_HOUSE_SERIES,
    ArenaHousePrompt,
    _contains_slop_or_refusal,
    build_tenet_domain_imatrix_corpus,
    evaluate_arena_house_series,
    verify_house_artifact,
)
from sml.evals.peer_eval import run_peer_eval_gate

_EXPECTED_CATEGORIES = (
    "think_verification",
    "sql_readonly",
    "code_ast",
    "json_config",
    "unslop_technical",
    "edge_boundary",
)

_SQL_BY_ID = {
    "house-sql-dept-share": (
        "WITH ranked AS (\n"
        "    SELECT employee_id, department, salary,\n"
        "           ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn\n"
        "    FROM employees\n"
        ")\n"
        "SELECT employee_id, department, salary, salary * 1.0 AS salary_share\n"
        "FROM ranked WHERE rn <= 3;\n"
    ),
    "house-sql-latest-order": (
        "WITH latest AS (\n"
        "    SELECT customer_id, order_date,\n"
        "           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn\n"
        "    FROM orders\n"
        ")\n"
        "SELECT customer_id, order_date FROM latest WHERE rn = 1 ORDER BY customer_id;\n"
    ),
    "house-sql-top-revenue": (
        "WITH totals AS (\n"
        "    SELECT product_id, SUM(quantity * unit_price) AS revenue\n"
        "    FROM order_items\n"
        "    GROUP BY product_id\n"
        ")\n"
        "SELECT products.product_name, totals.revenue\n"
        "FROM totals JOIN products ON products.product_id = totals.product_id\n"
        "ORDER BY totals.revenue DESC LIMIT 5;\n"
    ),
    "house-sql-rolling-avg": (
        "SELECT visit_date,\n"
        "       AVG(visitors) OVER (\n"
        "           ORDER BY visit_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW\n"
        "       ) AS rolling_7d\n"
        "FROM daily_visits\n"
        "ORDER BY visit_date;\n"
    ),
}

_PYTHON_BY_ID = {
    "house-code-fib-memo": (
        "def fib(n, memo=None):\n"
        "    if memo is None:\n"
        "        memo = {}\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    if n not in memo:\n"
        "        memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n"
        "    return memo[n]\n"
    ),
    "house-code-flatten": (
        "def flatten(nested):\n"
        "    flat = []\n"
        "    for item in nested:\n"
        "        if isinstance(item, list):\n"
        "            flat.extend(flatten(item))\n"
        "        else:\n"
        "            flat.append(item)\n"
        "    return flat\n"
    ),
    "house-code-binary-search": (
        "def binary_search(items, target):\n"
        "    lo, hi = 0, len(items) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    ),
    "house-code-rate-limiter": (
        "class TokenBucket:\n"
        "    def __init__(self, rate, capacity):\n"
        "        self.rate = rate\n"
        "        self.capacity = capacity\n"
        "        self.credits = float(capacity)\n"
        "        self.stamp = 0.0\n"
        "\n"
        "    def allow(self, tokens=1):\n"
        "        now = self.stamp + 1.0\n"
        "        self.credits = min(self.capacity, self.credits + (now - self.stamp) * self.rate)\n"
        "        self.stamp = now\n"
        "        if self.credits < tokens:\n"
        "            return False\n"
        "        self.credits -= tokens\n"
        "        return True\n"
    ),
}

_JSON_BY_ID = {
    "house-json-cache-config": '{"max_entries": 512, "ttl_seconds": 300, "eviction": "lru"}',
    "house-json-ci-config": (
        '{"name": "ci", "on": {"push": {"branches": ["main"]}}, '
        '"jobs": {"test": {"runs-on": "ubuntu-latest"}}}'
    ),
    "house-json-feature-flags": (
        '{"service": "search", "flags": {"reranker_v2": true, "spell_correct": false}}'
    ),
    "house-json-retry-config": (
        '{"max_attempts": 3, "backoff": "exponential", "timeout_seconds": 30}'
    ),
}

_PROSE_BY_ID = {
    "house-unslop-btree-lsm": (
        "B-tree pages keep keys sorted with a bounded branching factor, so lookups cost "
        "O(log n) page reads and leaf scans stay cache friendly. Write-heavy workloads "
        "amplify differently: LSM trees batch inserts into memtables and flush sorted "
        "runs, then compaction merges them, trading slower point reads for much higher "
        "ingest throughput and predictable write stalls."
    ),
    "house-unslop-tcp-loss": (
        "TCP treats packet loss as a congestion signal. Timeout-driven recovery collapses "
        "the window to one segment and re-enters slow start, while duplicate-ACKs trigger "
        "fast retransmit with fast recovery that halves the window instead. Slow start "
        "grows the window exponentially until loss or the ssthresh threshold is reached, "
        "and congestion avoidance then raises it additively per round trip."
    ),
    "house-unslop-gc-strategies": (
        "Reference counting (RC) frees objects the moment their count hits zero, but it "
        "pays a mutation overhead on every assignment and leaks cycles unless a cycle "
        "collector intervenes. Generational collectors trace roots periodically, promote "
        "surviving objects, and collect the young generation far more often than the old "
        "one, trading longer pauses for complete cycle reclamation and amortized cost."
    ),
    "house-unslop-wal-fsync": (
        "Write-ahead logging appends a commit record to a sequential log before data "
        "pages change, so batching fsync calls groups many commits into a single flush. "
        "Throughput rises and mean commit latency falls, but the crash window widens "
        "because acknowledged commits can sit in the OS page cache until the next flush "
        "completes and durability is only guaranteed at the barrier."
    ),
}

_NUMERIC_SURFACE_BY_ID = {
    "house-think-verify-sum": "The correct three-day total is 4,326 units.",
    "house-think-verify-remainder": "The remainder is 16.",
    "house-think-verify-primes": "The range contains 7 primes.",
    "house-think-verify-celsius": "The Celsius value is 20.0.",
    "house-edge-int8-wrap": "The counter wraps to -128.",
    "house-edge-float-exact": "The largest such integer is 9007199254740992.",
    "house-edge-leap-years": "The range contains 25 leap years.",
    "house-edge-python-modulo": "The result is 2.",
}


def _clean_surface(spec: ArenaHousePrompt) -> str:
    """Build a mechanically valid post-think surface answer for ``spec``."""
    if spec.artifact_type == "exact_numeric":
        return _NUMERIC_SURFACE_BY_ID[spec.id]
    if spec.artifact_type == "sql_readonly":
        return f"```sql\n{_SQL_BY_ID[spec.id]}```"
    if spec.artifact_type == "code_ast":
        return f"```python\n{_PYTHON_BY_ID[spec.id]}```"
    if spec.artifact_type == "json_config":
        return f"```json\n{_JSON_BY_ID[spec.id]}\n```"
    assert spec.artifact_type == "unslop_technical"
    return _PROSE_BY_ID[spec.id]


def _clean_response(spec: ArenaHousePrompt) -> str:
    """Build a full clean response: think-tagged deliberation + valid artifact."""
    deliberation = (
        f"Task {spec.id}: work through the request, verify each intermediate step, "
        "then produce the requested artifact in the required format."
    )
    return f"<think>\n{deliberation}\n</think>\n{_clean_surface(spec)}"


_SPEC_BY_ID = {spec.id: spec for spec in TENET_ARENA_HOUSE_SERIES}
_CLEAN_RESPONSES = {spec.id: _clean_response(spec) for spec in TENET_ARENA_HOUSE_SERIES}
_CLEAN_RESPONSE_BY_PROMPT = {
    spec.prompt: _CLEAN_RESPONSES[spec.id] for spec in TENET_ARENA_HOUSE_SERIES
}


# ---------------------------------------------------------------------------
# Series shape.
# ---------------------------------------------------------------------------


def test_house_series_shape_24_prompts_6_categories() -> None:
    assert len(TENET_ARENA_HOUSE_SERIES) == 24
    categories = [spec.category for spec in TENET_ARENA_HOUSE_SERIES]
    assert set(categories) == set(_EXPECTED_CATEGORIES)
    assert len(set(categories)) == 6
    assert Counter(categories) == {category: 4 for category in _EXPECTED_CATEGORIES}
    assert len({spec.id for spec in TENET_ARENA_HOUSE_SERIES}) == 24
    assert len({spec.prompt for spec in TENET_ARENA_HOUSE_SERIES}) == 24


# ---------------------------------------------------------------------------
# verify_house_artifact: pass cases.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_id",
    [
        "house-sql-dept-share",
        "house-sql-latest-order",
        "house-sql-top-revenue",
        "house-sql-rolling-avg",
    ],
)
def test_verify_house_artifact_sql_readonly_pass(spec_id: str) -> None:
    result = verify_house_artifact(_SPEC_BY_ID[spec_id], _CLEAN_RESPONSES[spec_id])
    assert result.passed
    assert result.artifact_valid
    assert result.cot_tag_valid
    assert result.unslop_passed
    assert not result.length_collapsed
    assert result.hp_score == 100.0
    assert result.reason == "ok"


@pytest.mark.parametrize(
    "spec_id",
    [
        "house-code-fib-memo",
        "house-code-flatten",
        "house-code-binary-search",
        "house-code-rate-limiter",
    ],
)
def test_verify_house_artifact_valid_code_ast_pass(spec_id: str) -> None:
    result = verify_house_artifact(_SPEC_BY_ID[spec_id], _CLEAN_RESPONSES[spec_id])
    assert result.passed
    assert result.artifact_valid
    assert result.cot_tag_valid
    assert result.hp_score == 100.0


@pytest.mark.parametrize(
    "spec_id",
    [
        "house-think-verify-sum",
        "house-edge-int8-wrap",
        "house-edge-float-exact",
        "house-unslop-wal-fsync",
        "house-json-cache-config",
    ],
)
def test_verify_house_artifact_mixed_types_pass(spec_id: str) -> None:
    result = verify_house_artifact(_SPEC_BY_ID[spec_id], _CLEAN_RESPONSES[spec_id])
    assert result.passed
    assert result.artifact_valid
    assert result.cot_tag_valid
    assert result.unslop_passed
    assert result.hp_score == 100.0


# ---------------------------------------------------------------------------
# verify_house_artifact: fail cases.
# ---------------------------------------------------------------------------


def test_verify_house_artifact_missing_think_close_fails() -> None:
    spec = _SPEC_BY_ID["house-code-fib-memo"]
    response = "<think>\nPlanning the implementation, then writing the code block."
    result = verify_house_artifact(spec, response)
    assert not result.passed
    assert not result.cot_tag_valid
    assert result.artifact_valid is False  # surface is empty: no fenced block
    assert result.unslop_passed is False  # empty surface fails the unslop check
    assert result.hp_score == 0.0
    assert "Malformed/unbalanced" in result.reason


def test_verify_house_artifact_tag_leakage_inside_sql_fails() -> None:
    spec = _SPEC_BY_ID["house-sql-dept-share"]
    # A </think> tag bleeds into the SQL artifact: the segmenter excises the
    # stray close, leaving "SELECT  FROM employees;" which no longer parses,
    # and the unpaired close marks the CoT span malformed.
    response = (
        "<think>\nPlan the read-only query, then emit the SQL block.\n"
        "</think>\n```sql\nSELECT </think> FROM employees;\n```"
    )
    result = verify_house_artifact(spec, response)
    assert not result.passed
    assert not result.cot_tag_valid


def test_verify_house_artifact_tag_leakage_inside_json_fails() -> None:
    spec = _SPEC_BY_ID["house-json-feature-flags"]
    # A <think> tag bleeds into the JSON artifact: the segmenter treats it as
    # an unclosed deliberation open, so the JSON block is never extractable
    # as a complete object and the CoT span is malformed.
    response = (
        "<think>\nPlan the flag file, then emit the JSON block.\n"
        "</think>\n```json\n"
        '{"service": "search", "flags": {"v1": "<think>"}}\n```'
    )
    result = verify_house_artifact(spec, response)
    assert not result.passed
    assert not result.cot_tag_valid


def test_verify_house_artifact_sql_ddl_rejected() -> None:
    spec = _SPEC_BY_ID["house-sql-rolling-avg"]
    response = (
        "<think>\nPlan: drop and recreate the table to reset the daily data.\n"
        "</think>\n```sql\nDROP TABLE daily_visits;\n```"
    )
    result = verify_house_artifact(spec, response)
    assert not result.passed
    assert not result.artifact_valid
    assert result.cot_tag_valid  # tags themselves are fine; the SQL is not
    assert "SQL guard rejected" in result.reason


def test_verify_house_artifact_length_collapse_tracked() -> None:
    spec = _SPEC_BY_ID["house-code-flatten"]
    response = (
        "<think>\nOutline the recursion, then emit a minimal code block.\n"
        "</think>\n```python\ndef flatten(nested):\n    return []\n```"
    )
    result = verify_house_artifact(spec, response)
    assert result.passed  # collapse is tracked but not gating
    assert result.length_collapsed
    assert result.artifact_valid
    assert result.hp_score == 100.0


# ---------------------------------------------------------------------------
# evaluate_arena_house_series: clean synthetic responses score 100%.
# ---------------------------------------------------------------------------


def test_evaluate_arena_house_series_clean_responses_perfect() -> None:
    result = evaluate_arena_house_series(_CLEAN_RESPONSE_BY_PROMPT.__getitem__)

    assert result["total_prompts"] == 24
    assert result["valid_artifacts"] == 24
    assert result["house_artifact_classification_rate"] == 1.0
    assert result["cot_tag_integrity_rate"] == 1.0
    assert result["unslop_pass_rate"] == 1.0
    assert result["length_collapse_rate"] == 0.0
    assert result["simulated_arena_hp"] == 100.0

    assert len(result["results"]) == 24
    assert all(entry["hp_score"] == 100.0 for entry in result["results"])
    assert all(entry["passed"] for entry in result["results"])

    per_category = result["per_category"]
    assert set(per_category) == set(_EXPECTED_CATEGORIES)
    for category in _EXPECTED_CATEGORIES:
        breakdown = per_category[category]
        assert breakdown["prompts"] == 4
        assert breakdown["artifact_rate"] == 1.0
        assert breakdown["cot_tag_integrity_rate"] == 1.0
        assert breakdown["unslop_pass_rate"] == 1.0
        assert breakdown["mean_hp_score"] == 100.0


def test_evaluate_arena_house_series_collapse_shows_in_breakdown() -> None:
    def responder(prompt: str) -> str:
        return "<think>\nBrief plan, then a minimal artifact.\n</think>\nOK."

    result = evaluate_arena_house_series(responder)
    # exact_numeric prompts are never collapse-eligible.
    assert result["length_collapse_rate"] == pytest.approx(16 / 24)
    assert result["house_artifact_classification_rate"] < 1.0
    assert result["simulated_arena_hp"] < 100.0


# ---------------------------------------------------------------------------
# build_tenet_domain_imatrix_corpus: zero slop + real file output.
# ---------------------------------------------------------------------------


def test_build_tenet_domain_imatrix_corpus_zero_slop_and_writes_file(tmp_path) -> None:
    output_path = tmp_path / "corpus" / "imatrix_corpus.txt"
    result = build_tenet_domain_imatrix_corpus(output_path)

    assert result["zero_slop_verified"] is True
    assert result["path"] == str(output_path)
    assert result["num_blocks"] == 96  # 24 base blocks x 4 repeats
    assert result["num_chars"] > 0

    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert text == result["text"]
    assert not _contains_slop_or_refusal(text)
    assert "<think>" in text and "</think>" in text
    assert "```sql" in text and "```python" in text and "```json" in text


def test_build_tenet_domain_imatrix_corpus_no_path_returns_text_only() -> None:
    result = build_tenet_domain_imatrix_corpus()
    assert result["path"] is None
    assert result["zero_slop_verified"] is True
    assert result["num_blocks"] == 96


def test_build_tenet_domain_imatrix_corpus_rejects_bad_repeats(tmp_path) -> None:
    with pytest.raises(ValueError):
        build_tenet_domain_imatrix_corpus(tmp_path / "corpus.txt", repeats=0)


# ---------------------------------------------------------------------------
# run_peer_eval_gate.
# ---------------------------------------------------------------------------


def test_run_peer_eval_gate_passes_on_clean_responses() -> None:
    result = run_peer_eval_gate(_CLEAN_RESPONSE_BY_PROMPT.__getitem__)

    assert result["total_evaluated"] == 24
    assert result["cot_tag_integrity_rate"] == 1.0
    assert result["length_collapse_rate"] == 0.0
    assert result["slop_and_refusal_rate"] == 0.0
    assert result["gate_passed"] is True
    assert all(detail["passed"] for detail in result["gate_details"].values())


def test_run_peer_eval_gate_fails_on_length_collapse() -> None:
    result = run_peer_eval_gate(lambda _prompt: "<think>\nShort plan.\n</think>\nOK")

    assert result["gate_passed"] is False
    assert result["gate_details"]["length_collapse"]["passed"] is False
    assert result["gate_details"]["cot_tag_integrity"]["passed"] is True
    assert result["gate_details"]["slop_and_refusal"]["passed"] is True
    assert result["length_collapse_rate"] == pytest.approx(1.0)


def test_run_peer_eval_gate_fails_on_slop() -> None:
    surface = (
        "Let me delve into the details of this problem and explain the "
        "reasoning carefully with enough words to avoid any collapse."
    )
    result = run_peer_eval_gate(lambda _prompt: f"<think>\nPlan the answer.\n</think>\n{surface}")

    assert result["gate_passed"] is False
    assert result["gate_details"]["slop_and_refusal"]["passed"] is False
    assert result["gate_details"]["cot_tag_integrity"]["passed"] is True
    assert result["gate_details"]["length_collapse"]["passed"] is True
    assert result["slop_and_refusal_rate"] == pytest.approx(1.0)


def test_run_peer_eval_gate_fails_on_broken_tags() -> None:
    result = run_peer_eval_gate(
        lambda _prompt: "<think>\nDeliberation that never closes and drifts on"
    )

    assert result["gate_passed"] is False
    assert result["gate_details"]["cot_tag_integrity"]["passed"] is False
    assert result["cot_tag_integrity_rate"] == pytest.approx(0.0)


def test_run_peer_eval_gate_bare_numeric_surface_is_not_collapse() -> None:
    # No prompt specs are supplied, so responses cannot be classified; a bare
    # numeric surface must still be treated as exact-numeric, not collapse.
    result = run_peer_eval_gate(responses=["4326"], require_think=False)

    assert result["total_evaluated"] == 1
    assert result["cot_tag_integrity_rate"] == 1.0
    assert result["length_collapse_rate"] == 0.0
    assert result["gate_passed"] is True

    contrast = run_peer_eval_gate(responses=["roughly four thousand"], require_think=False)
    assert contrast["gate_passed"] is False
    assert contrast["gate_details"]["length_collapse"]["passed"] is False


# ---------------------------------------------------------------------------
# gguf_audit re-export.
# ---------------------------------------------------------------------------


def test_gguf_audit_reexports_build_tenet_domain_imatrix_corpus() -> None:
    assert gguf_audit_module.build_tenet_domain_imatrix_corpus is build_tenet_domain_imatrix_corpus
