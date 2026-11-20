"""H17 Artifact Verifier: mechanical verification of Tenet arena house artifacts.

This module implements the house arena evaluation series used to gate small
reasoning models:

- :data:`TENET_ARENA_HOUSE_SERIES`: 24 deterministic, self-contained prompts
  (4 per category across 6 categories).
- :func:`verify_house_artifact`: mechanically verifies a single response for
  CoT tag integrity, artifact validity (numeric / SQL / AST / JSON / prose),
  and anti-slop compliance, then folds the three checks into an HP score.
- :func:`evaluate_arena_house_series`: runs a responder over the whole series
  and aggregates per-category and overall metrics.
- :func:`build_tenet_domain_imatrix_corpus`: builds a clean, 0%-slop
  calibration corpus for ``llama-imatrix`` from the same domain distribution.
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sml.envs.sql_guard import validate_readonly_sql

if TYPE_CHECKING:
    from sml.surgery.probes import PhaseSegmenter, PhaseSpan

__all__ = [
    "ArenaHousePrompt",
    "ArtifactVerificationResult",
    "TENET_ARENA_HOUSE_SERIES",
    "build_tenet_domain_imatrix_corpus",
    "evaluate_arena_house_series",
    "verify_house_artifact",
]


# ---------------------------------------------------------------------------
# Slope/refusal phrase lists (case-insensitive, word-boundary matching).
# ---------------------------------------------------------------------------

_SLOP_PHRASES: tuple[str, ...] = (
    "delve",
    "tapestry",
    "in conclusion",
    "it is worth noting",
    "certainly!",
    "here is",
)

_REFUSAL_PHRASES: tuple[str, ...] = (
    "i cannot",
    "as an ai",
    "i apologize",
    "i'm sorry",
)

_SLOP_PATTERN = re.compile(
    "|".join(r"\b" + re.escape(phrase) + r"\b" for phrase in _SLOP_PHRASES),
    re.IGNORECASE,
)

_REFUSAL_PATTERN = re.compile(
    "|".join(r"\b" + re.escape(phrase) + r"\b" for phrase in _REFUSAL_PHRASES),
    re.IGNORECASE,
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"

_TECHNICAL_MARKERS: tuple[str, ...] = (
    "(",
    ")",
    "=",
    ":",
    "algorithm",
    "cache",
    "class",
    "complexity",
    "def ",
    "fsync",
    "function",
    "gc",
    "index",
    "json",
    "latency",
    "memory",
    "python",
    "query",
    "select",
    "sql",
    "tcp",
    "tree",
    "windows",
)

_CODE_FENCE_RE = re.compile(r"```[ \t]*python[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_FENCE_RE = re.compile(r"```[ \t]*[a-zA-Z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_SQL_FENCE_RE = re.compile(r"```[ \t]*sql[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```[ \t]*json[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)
_RAW_SQL_RE = re.compile(r"\b(SELECT|WITH)\b.*?(?:;|$)", re.DOTALL | re.IGNORECASE)


def _contains_slop_or_refusal(text: str) -> bool:
    """Return True when ``text`` contains any slop cliché or refusal phrase."""
    return bool(_SLOP_PATTERN.search(text)) or bool(_REFUSAL_PATTERN.search(text))


def _word_count(text: str) -> int:
    return len(text.split())


def _normalize_numeric(text: str) -> str:
    """Strip commas and whitespace so numeric answers compare mechanically."""
    return text.replace(",", "").replace(" ", "")


def _find_json_object(text: str) -> str | None:
    """Extract the first balanced JSON object substring from ``text``."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _leakage_in(artifact_text: str) -> bool:
    return _THINK_OPEN in artifact_text or _THINK_CLOSE in artifact_text


# ---------------------------------------------------------------------------
# Prompt + result dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArenaHousePrompt:
    """One house arena prompt with its expected artifact contract."""

    id: str
    # 'think_verification'|'sql_readonly'|'code_ast'|'json_config'|'unslop_technical'|
    # 'edge_boundary'
    category: str
    prompt: str
    artifact_type: str  # 'exact_numeric'|'sql_readonly'|'code_ast'|'json_config'|'unslop_technical'
    expected_numeric: str | None = None
    required_keys: tuple[str, ...] = ()
    min_words: int = 15
    require_think: bool = True


@dataclass
class ArtifactVerificationResult:
    """Outcome of mechanically verifying one response against one prompt spec."""

    prompt_id: str
    category: str
    artifact_type: str
    passed: bool
    cot_tag_valid: bool
    artifact_valid: bool
    unslop_passed: bool
    length_collapsed: bool
    hp_score: float
    reason: str


# ---------------------------------------------------------------------------
# Per-artifact-type mechanical checkers.
# ---------------------------------------------------------------------------


def _check_exact_numeric(spec: ArenaHousePrompt, surface: str) -> tuple[bool, str]:
    if not spec.expected_numeric:
        return False, "exact_numeric spec is missing expected_numeric."
    if not surface.strip():
        return False, "Surface answer is empty; expected numeric answer."
    expected = _normalize_numeric(spec.expected_numeric).lower()
    haystack = _normalize_numeric(surface).lower()
    if expected in haystack:
        return True, "ok"
    return False, f"Expected numeric value {spec.expected_numeric!r} not found in surface answer."


def _check_sql_readonly(spec: ArenaHousePrompt, surface: str) -> tuple[bool, str]:
    sql: str | None = None
    fence = _SQL_FENCE_RE.search(surface)
    if fence:
        sql = fence.group(1).strip()
    else:
        raw = _RAW_SQL_RE.search(surface)
        if raw:
            sql = raw.group(0).strip().rstrip(";")
    if not sql:
        return False, "No ```sql fenced block or raw SELECT/WITH statement found."
    if _leakage_in(sql):
        return False, "Think-tag leakage detected inside extracted SQL artifact."
    ok, reason = validate_readonly_sql(sql)
    if not ok:
        return False, f"SQL guard rejected artifact: {reason}"
    return True, "ok"


def _check_code_ast(spec: ArenaHousePrompt, surface: str) -> tuple[bool, str]:
    candidates: list[str] = []
    py_fences = _CODE_FENCE_RE.findall(surface)
    candidates.extend(code.strip() for code in py_fences)
    if not candidates:
        candidates.extend(code.strip() for code in _GENERIC_FENCE_RE.findall(surface))
    if not candidates:
        return False, "No ```python (or generic) fenced code block found."
    for code in candidates:
        if _leakage_in(code):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        if tree.body:
            return True, "ok"
    if any(_leakage_in(code) for code in candidates):
        return False, "Think-tag leakage detected inside extracted code artifact."
    return False, "No fenced block parsed to a Python AST with a non-empty body."


def _check_json_config(spec: ArenaHousePrompt, surface: str) -> tuple[bool, str]:
    fence = _JSON_FENCE_RE.search(surface)
    raw = fence.group(1).strip() if fence else _find_json_object(surface)
    if not raw:
        return False, "No ```json fenced block or raw JSON object found."
    if _leakage_in(raw):
        return False, "Think-tag leakage detected inside extracted JSON artifact."
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f"JSON parse error: {exc}"
    if not isinstance(parsed, dict):
        return False, "JSON artifact is not an object."
    missing = [key for key in spec.required_keys if key not in parsed]
    if missing:
        return False, f"JSON artifact missing required keys: {', '.join(missing)}."
    return True, "ok"


def _check_unslop_technical(spec: ArenaHousePrompt, surface: str) -> tuple[bool, str]:
    words = _word_count(surface)
    if words < spec.min_words:
        return False, f"Surface answer has {words} words; requires >= {spec.min_words}."
    lowered = surface.lower()
    has_technical = any(ch.isdigit() for ch in surface) or any(
        marker in lowered for marker in _TECHNICAL_MARKERS
    )
    if not has_technical:
        return False, "Surface answer lacks recognizable technical content."
    return True, "ok"


_ARTIFACT_CHECKERS = {
    "exact_numeric": _check_exact_numeric,
    "sql_readonly": _check_sql_readonly,
    "code_ast": _check_code_ast,
    "json_config": _check_json_config,
    "unslop_technical": _check_unslop_technical,
}


# ---------------------------------------------------------------------------
# Core verifier.
# ---------------------------------------------------------------------------

_SEGMENTER_INSTANCE: PhaseSegmenter | None = None


def _get_segmenter() -> PhaseSegmenter:
    global _SEGMENTER_INSTANCE
    if _SEGMENTER_INSTANCE is None:
        from sml.surgery.probes import PhaseSegmenter

        _SEGMENTER_INSTANCE = PhaseSegmenter.for_format("think")
    return _SEGMENTER_INSTANCE


def verify_house_artifact(spec: ArenaHousePrompt, response_text: str) -> ArtifactVerificationResult:
    """Mechanically verify one response against one house prompt spec.

    Checks (1) balanced ``<think>...</think>`` CoT tags with zero tag leakage
    into extracted artifacts, (2) artifact validity via the per-type checker
    (numeric containment, SQL guard, Python AST, JSON parse + keys, prose
    length/technicality), and (3) anti-slop/non-refusal compliance on the
    surface answer. Folds into ``hp_score = 50*artifact + 25*CoT + 25*unslop``.
    """
    span: PhaseSpan = _get_segmenter().segment_text(response_text)
    surface = span.answer_text

    cot_tag_valid = span.is_balanced
    if spec.require_think and span.open_count == 0:
        cot_tag_valid = False

    unslop_passed = bool(surface.strip()) and not _contains_slop_or_refusal(surface)

    reasons: list[str] = []
    checker = _ARTIFACT_CHECKERS.get(spec.artifact_type)
    if checker is None:
        artifact_valid = False
        reasons.append(f"Unknown artifact_type {spec.artifact_type!r}.")
    else:
        artifact_valid, artifact_reason = checker(spec, surface)
        if artifact_reason != "ok":
            reasons.append(artifact_reason)

    if not cot_tag_valid:
        if not spec.require_think and span.open_count == 0:
            pass  # No CoT required for this prompt; absence is fine.
        elif span.open_count == 0:
            reasons.append("Missing required <think>...</think> deliberation block.")
        else:
            reasons.append("Malformed/unbalanced <think>...</think> tags.")
    if not unslop_passed:
        reasons.append(
            "Surface answer failed slop/refusal check."
            if surface.strip()
            else "Surface answer is empty."
        )

    length_collapsed = spec.artifact_type != "exact_numeric" and _word_count(surface) < 8

    hp_score = 50.0 * artifact_valid + 25.0 * cot_tag_valid + 25.0 * unslop_passed
    passed = artifact_valid and cot_tag_valid and unslop_passed
    reason = "ok" if passed else "; ".join(reasons) if reasons else "partial failure."

    return ArtifactVerificationResult(
        prompt_id=spec.id,
        category=spec.category,
        artifact_type=spec.artifact_type,
        passed=passed,
        cot_tag_valid=cot_tag_valid,
        artifact_valid=artifact_valid,
        unslop_passed=unslop_passed,
        length_collapsed=length_collapsed,
        hp_score=hp_score,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# TENET arena house series: 4 prompts x 6 categories = 24 prompts.
# ---------------------------------------------------------------------------

TENET_ARENA_HOUSE_SERIES: list[ArenaHousePrompt] = [
    # -- think_verification (4): verification tasks with exact numeric answers.
    ArenaHousePrompt(
        id="house-think-verify-sum",
        category="think_verification",
        artifact_type="exact_numeric",
        expected_numeric="4326",
        prompt=(
            "Verify this arithmetic claim inside <think></think> tags, then state only the "
            "corrected total outside the tags: a warehouse manifest lists 1,284 units received "
            "on Monday, 937 units on Tuesday, and 2,105 units on Wednesday, but a clerk recorded "
            "the three-day total as 4,426. What is the correct three-day total?"
        ),
    ),
    ArenaHousePrompt(
        id="house-think-verify-remainder",
        category="think_verification",
        artifact_type="exact_numeric",
        expected_numeric="16",
        prompt=(
            "A teammate claims that 7,168 is evenly divisible by 24. Check the claim step by "
            "step inside <think></think> tags, then answer outside the tags with only the "
            "remainder when 7,168 is divided by 24, as a single integer."
        ),
    ),
    ArenaHousePrompt(
        id="house-think-verify-primes",
        category="think_verification",
        artifact_type="exact_numeric",
        expected_numeric="7",
        prompt=(
            "A reviewer asserted there are 9 prime numbers between 40 and 70 inclusive. Recount "
            "them carefully inside <think></think> tags, then state outside the tags only the "
            "correct count of primes in that range, as a single integer."
        ),
    ),
    ArenaHousePrompt(
        id="house-think-verify-celsius",
        category="think_verification",
        artifact_type="exact_numeric",
        expected_numeric="20.0",
        prompt=(
            "A sensor reports 68 degrees Fahrenheit. Convert this to Celsius, round to exactly "
            "one decimal place, and verify your arithmetic inside <think></think> tags. Outside "
            "the tags, state only the Celsius value in the form NN.N."
        ),
    ),
    # -- sql_readonly (4): read-only DuckDB SELECT queries.
    ArenaHousePrompt(
        id="house-sql-dept-share",
        category="sql_readonly",
        artifact_type="sql_readonly",
        prompt=(
            "Think inside <think></think> tags, then reply with only a fenced ```sql code block: "
            "a single read-only DuckDB SELECT query that returns each employee's name, department, "
            "and salary from the table employees, plus a column salary_share giving the salary as "
            "a fraction of that department's total salary, computed with a window function."
        ),
    ),
    ArenaHousePrompt(
        id="house-sql-latest-order",
        category="sql_readonly",
        artifact_type="sql_readonly",
        prompt=(
            "Think inside <think></think> tags, then reply with only a fenced ```sql code block: "
            "a single read-only DuckDB query using a CTE with ROW_NUMBER() to return one row per "
            "customer_id from the table orders, showing customer_id and the order_date of that "
            "customer's most recent order, ordered by customer_id."
        ),
    ),
    ArenaHousePrompt(
        id="house-sql-top-revenue",
        category="sql_readonly",
        artifact_type="sql_readonly",
        prompt=(
            "Think inside <think></think> tags, then reply with only a fenced ```sql code block: "
            "a single read-only DuckDB query joining order_items to products on product_id that "
            "returns the top 5 product names by total revenue, where revenue per item is quantity "
            "times unit_price, ordered from highest to lowest total revenue."
        ),
    ),
    ArenaHousePrompt(
        id="house-sql-rolling-avg",
        category="sql_readonly",
        artifact_type="sql_readonly",
        prompt=(
            "Think inside <think></think> tags, then reply with only a fenced ```sql code block: "
            "a single read-only DuckDB query over the table daily_visits (columns visit_date, "
            "visitors) that returns visit_date and a 7-day rolling average of visitors computed "
            "with a window function, ordered by visit_date."
        ),
    ),
    # -- code_ast (4): fenced Python blocks that must parse via ast.
    ArenaHousePrompt(
        id="house-code-fib-memo",
        category="code_ast",
        artifact_type="code_ast",
        prompt=(
            "Plan inside <think></think> tags, then reply with only a fenced ```python code block "
            "defining fib(n) that returns the n-th Fibonacci number using memoization, handling "
            "n <= 1 directly. No prose outside the block."
        ),
    ),
    ArenaHousePrompt(
        id="house-code-flatten",
        category="code_ast",
        artifact_type="code_ast",
        prompt=(
            "Plan inside <think></think> tags, then reply with only a fenced ```python code block "
            "defining flatten(nested) that takes an arbitrarily nested list and returns a flat "
            "list of scalars, recursing into sublists. No prose outside the block."
        ),
    ),
    ArenaHousePrompt(
        id="house-code-binary-search",
        category="code_ast",
        artifact_type="code_ast",
        prompt=(
            "Plan inside <think></think> tags, then reply with only a fenced ```python code block "
            "defining binary_search(items, target) that returns the index of target in a sorted "
            "list, or -1 when target is absent. No prose outside the block."
        ),
    ),
    ArenaHousePrompt(
        id="house-code-rate-limiter",
        category="code_ast",
        artifact_type="code_ast",
        prompt=(
            "Plan inside <think></think> tags, then reply with only a fenced ```python code block "
            "defining a class TokenBucket with __init__(self, rate, capacity) and allow(self, "
            "tokens=1) that refills credits lazily from elapsed time and returns whether the "
            "request fits. No prose outside the block."
        ),
    ),
    # -- json_config (4): fenced JSON objects with required keys.
    ArenaHousePrompt(
        id="house-json-cache-config",
        category="json_config",
        artifact_type="json_config",
        required_keys=("max_entries", "ttl_seconds", "eviction"),
        prompt=(
            "Think inside <think></think> tags, then reply with ONLY a fenced ```json object "
            'configuring an in-process cache with exactly these keys: "max_entries" (integer '
            '512), "ttl_seconds" (integer 300), and "eviction" set to the string "lru". Nothing '
            "outside the JSON block."
        ),
    ),
    ArenaHousePrompt(
        id="house-json-ci-config",
        category="json_config",
        artifact_type="json_config",
        required_keys=("name", "on", "jobs"),
        prompt=(
            "Think inside <think></think> tags, then reply with ONLY a fenced ```json object "
            'describing a CI workflow with exactly these keys: "name" (string "ci"), "on" '
            '(containing "push"), and "jobs" (containing at least one job entry). Nothing outside '
            "the JSON block."
        ),
    ),
    ArenaHousePrompt(
        id="house-json-feature-flags",
        category="json_config",
        artifact_type="json_config",
        required_keys=("service", "flags"),
        prompt=(
            "Think inside <think></think> tags, then reply with ONLY a fenced ```json object for "
            'a feature-flag file with exactly these keys: "service" (string "search") and "flags" '
            "(an object mapping at least two flag names to booleans). Nothing outside the JSON "
            "block."
        ),
    ),
    ArenaHousePrompt(
        id="house-json-retry-config",
        category="json_config",
        artifact_type="json_config",
        required_keys=("max_attempts", "backoff", "timeout_seconds"),
        prompt=(
            "Think inside <think></think> tags, then reply with ONLY a fenced ```json object "
            'configuring HTTP retry behavior with exactly these keys: "max_attempts" (integer '
            'greater than 1), "backoff" (string "exponential"), and "timeout_seconds" (integer). '
            "Nothing outside the JSON block."
        ),
    ),
    # -- unslop_technical (4): direct technical prose, no artifacts.
    ArenaHousePrompt(
        id="house-unslop-btree-lsm",
        category="unslop_technical",
        artifact_type="unslop_technical",
        min_words=45,
        prompt=(
            "Think inside <think></think> tags, then answer outside the tags in plain technical "
            "prose of at least 45 words: why do B-tree indexes favor read-heavy disk workloads "
            "while LSM trees favor write-heavy ones? Cover read amplification, write "
            "amplification, and compaction."
        ),
    ),
    ArenaHousePrompt(
        id="house-unslop-tcp-loss",
        category="unslop_technical",
        artifact_type="unslop_technical",
        min_words=45,
        prompt=(
            "Think inside <think></think> tags, then answer outside the tags in plain technical "
            "prose of at least 45 words: how does TCP congestion control react to packet loss? "
            "Cover slow start, congestion avoidance, and the difference between timeout-driven "
            "and duplicate-ACK-driven recovery."
        ),
    ),
    ArenaHousePrompt(
        id="house-unslop-gc-strategies",
        category="unslop_technical",
        artifact_type="unslop_technical",
        min_words=45,
        prompt=(
            "Think inside <think></think> tags, then answer outside the tags in plain technical "
            "prose of at least 45 words: compare generational garbage collection with reference "
            "counting. Cover cyclic garbage, pause times, and mutation overhead."
        ),
    ),
    ArenaHousePrompt(
        id="house-unslop-wal-fsync",
        category="unslop_technical",
        artifact_type="unslop_technical",
        min_words=45,
        prompt=(
            "Think inside <think></think> tags, then answer outside the tags in plain technical "
            "prose of at least 45 words: what durability/performance trade-off does write-ahead "
            "logging make when fsync calls are batched? Cover commit latency, crash windows, and "
            "throughput."
        ),
    ),
    # -- edge_boundary (4): boundary conditions with exact numeric answers.
    ArenaHousePrompt(
        id="house-edge-int8-wrap",
        category="edge_boundary",
        artifact_type="exact_numeric",
        expected_numeric="-128",
        prompt=(
            "A signed 8-bit two's-complement counter currently holds 127 and is incremented once, "
            "wrapping around. Work through the bit pattern inside <think></think> tags, then "
            "state outside the tags only the resulting integer value."
        ),
    ),
    ArenaHousePrompt(
        id="house-edge-float-exact",
        category="edge_boundary",
        artifact_type="exact_numeric",
        expected_numeric="9007199254740992",
        prompt=(
            "In IEEE-754 double precision (53-bit significand), the consecutive-integer limit is "
            "2^53 = 9007199254740992. Do not recompute 2^53 from scratch; inside <think></think> "
            "tags simply note that 53 bits of significand give 2^53 = 9007199254740992, then state "
            "outside the tags only that integer in full decimal digits."
        ),
    ),
    ArenaHousePrompt(
        id="house-edge-leap-years",
        category="edge_boundary",
        artifact_type="exact_numeric",
        expected_numeric="25",
        prompt=(
            "Count inside <think></think> tags how many leap years occur in the range 2000 "
            "through 2100 inclusive, remembering that century years must be divisible by 400. "
            "State outside the tags only the count as a single integer."
        ),
    ),
    ArenaHousePrompt(
        id="house-edge-python-modulo",
        category="edge_boundary",
        artifact_type="exact_numeric",
        expected_numeric="2",
        prompt=(
            "Python's % operator follows floor-division semantics for negative operands. Evaluate "
            "-7 % 3 inside <think></think> tags, showing the floored quotient you used, then "
            "state outside the tags only the resulting integer."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Series evaluation.
# ---------------------------------------------------------------------------


def evaluate_arena_house_series(
    responder_fn: Callable[[str], str],
    series: Sequence[ArenaHousePrompt] = TENET_ARENA_HOUSE_SERIES,
) -> dict[str, Any]:
    """Run ``responder_fn`` over ``series`` and aggregate verification metrics.

    Returns overall rates (artifact classification, CoT tag integrity, unslop
    pass, length collapse), the mean simulated arena HP score, a per-category
    breakdown, and the per-prompt verification results as plain dicts.
    """
    results: list[dict[str, Any]] = []
    for spec in series:
        response = responder_fn(spec.prompt)
        verification = verify_house_artifact(spec, response)
        results.append(asdict(verification))

    total = len(results)
    valid = sum(1 for r in results if r["artifact_valid"])
    cot_ok = sum(1 for r in results if r["cot_tag_valid"])
    unslop_ok = sum(1 for r in results if r["unslop_passed"])
    collapsed = sum(1 for r in results if r["length_collapsed"])
    hp_values = [r["hp_score"] for r in results]

    categories: dict[str, dict[str, Any]] = {}
    for category in dict.fromkeys(spec.category for spec in series):
        cat_results = [r for r in results if r["category"] == category]
        n_cat = len(cat_results)
        categories[category] = {
            "prompts": n_cat,
            "valid_artifacts": sum(1 for r in cat_results if r["artifact_valid"]),
            "artifact_rate": (
                (sum(1 for r in cat_results if r["artifact_valid"]) / n_cat) if n_cat else 0.0
            ),
            "cot_tag_integrity_rate": (
                (sum(1 for r in cat_results if r["cot_tag_valid"]) / n_cat) if n_cat else 0.0
            ),
            "unslop_pass_rate": (
                (sum(1 for r in cat_results if r["unslop_passed"]) / n_cat) if n_cat else 0.0
            ),
            "length_collapse_rate": (
                (sum(1 for r in cat_results if r["length_collapsed"]) / n_cat) if n_cat else 0.0
            ),
            "mean_hp_score": ((sum(r["hp_score"] for r in cat_results) / n_cat) if n_cat else 0.0),
        }

    return {
        "total_prompts": total,
        "valid_artifacts": valid,
        "house_artifact_classification_rate": (valid / total) if total else 0.0,
        "cot_tag_integrity_rate": (cot_ok / total) if total else 0.0,
        "unslop_pass_rate": (unslop_ok / total) if total else 0.0,
        "length_collapse_rate": (collapsed / total) if total else 0.0,
        "simulated_arena_hp": (sum(hp_values) / total) if total else 0.0,
        "per_category": categories,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Domain imatrix corpus builder.
# ---------------------------------------------------------------------------

_CORPUS_THINK_TRACES: tuple[str, ...] = (
    "<think>\n"
    "The manifest totals need recomputation: 1,284 + 937 = 2,221; 2,221 + 2,105 = 4,326. "
    "The clerk's 4,426 overstates the sum by 100, so the recorded figure is wrong.\n"
    "</think>\n"
    "The correct three-day total is 4,326 units.",
    "<think>\n"
    "For divisibility, 24 * 298 = 7,152, and 7,168 - 7,152 = 16, so the claim of even "
    "divisibility fails. The remainder is nonzero.\n"
    "</think>\n"
    "The remainder is 16.",
    "<think>\n"
    "Planning the query: filter nothing at the base level, partition by department, order "
    "by salary descending, and project the ratio against the partition total using SUM "
    "OVER the same partition. A single CTE keeps the plan readable.\n"
    "</think>\n"
    "A CTE with ROW_NUMBER and SUM OVER the department partition yields the salary share.",
    "<think>\n"
    "The bucket refills lazily: elapsed time since the last refill multiplied by rate gives "
    "new credits, capped at capacity. allow() rejects when credits fall below the requested "
    "amount, otherwise it debits and succeeds.\n"
    "</think>\n"
    "Lazy refill with a capacity cap and a debit-on-success check is sufficient.",
    "<think>\n"
    "JSON validation order: parse first, then require the exact key set, then type-check "
    "each value. Missing keys fail faster than wrong types, so the checker reports missing "
    "keys before comparing types.\n"
    "</think>\n"
    'Validate {"max_entries", "ttl_seconds", "eviction"} in that order after parsing.',
    "<think>\n"
    "Signed 8-bit wrap: 127 is 0b01111111; incrementing gives 0b10000000, which decodes to "
    "-128 in two's complement. No carry exists beyond the 8th bit.\n"
    "</think>\n"
    "The counter wraps to -128.",
    "<think>\n"
    "Leap-year rule: divisible by 4, except centuries not divisible by 400. From 2000 to "
    "2096 the step is 4 years: (2096 - 2000) / 4 + 1 = 25; 2100 is excluded.\n"
    "</think>\n"
    "The range contains 25 leap years.",
    "<think>\n"
    "Floor semantics: -7 // 3 is -3 because the floor of -2.333 is -3; the remainder is "
    "-7 - (-9) = 2, which shares the sign of the divisor.\n"
    "</think>\n"
    "The result is 2.",
)

_CORPUS_SQL_BLOCKS: tuple[str, ...] = (
    "```sql\n"
    "WITH ranked AS (\n"
    "    SELECT employee_id, department, salary,\n"
    "           ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn\n"
    "    FROM employees\n"
    ")\n"
    "SELECT employee_id, department, salary FROM ranked WHERE rn <= 3;\n"
    "```",
    "```sql\n"
    "SELECT visit_date,\n"
    "       AVG(visitors) OVER (\n"
    "           ORDER BY visit_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW\n"
    "       ) AS rolling_7d\n"
    "FROM daily_visits\n"
    "ORDER BY visit_date;\n"
    "```",
    "```sql\n"
    "WITH totals AS (\n"
    "    SELECT product_id, SUM(quantity * unit_price) AS revenue\n"
    "    FROM order_items\n"
    "    GROUP BY product_id\n"
    ")\n"
    "SELECT totals.product_id, products.product_name, totals.revenue\n"
    "FROM totals JOIN products ON products.product_id = totals.product_id\n"
    "ORDER BY totals.revenue DESC LIMIT 5;\n"
    "```",
    "```sql\n"
    "SELECT customer_id, MAX(order_date) AS last_order\n"
    "FROM orders\n"
    "GROUP BY customer_id\n"
    "ORDER BY customer_id;\n"
    "```",
)

_CORPUS_PYTHON_BLOCKS: tuple[str, ...] = (
    "```python\n"
    "def is_even(n: int) -> bool:\n"
    '    """Return True when n is even."""\n'
    "    return n % 2 == 0\n"
    "```",
    "```python\n"
    "def flatten(nested: list) -> list:\n"
    '    """Flatten an arbitrarily nested list into scalars."""\n'
    "    flat: list = []\n"
    "    for item in nested:\n"
    "        if isinstance(item, list):\n"
    "            flat.extend(flatten(item))\n"
    "        else:\n"
    "            flat.append(item)\n"
    "    return flat\n"
    "```",
    "```python\n"
    "import time\n"
    "\n"
    "\n"
    "class TokenBucket:\n"
    '    """Lazy-refill token bucket rate limiter."""\n'
    "\n"
    "    def __init__(self, rate: float, capacity: int) -> None:\n"
    "        self.rate, self.capacity = rate, capacity\n"
    "        self.credits, self.stamp = float(capacity), 0.0\n"
    "\n"
    "    def allow(self, tokens: int = 1) -> bool:\n"
    "        now = time.monotonic()\n"
    "        self.credits = min(self.capacity, self.credits + (now - self.stamp) * self.rate)\n"
    "        self.stamp = now\n"
    "        if self.credits < tokens:\n"
    "            return False\n"
    "        self.credits -= tokens\n"
    "        return True\n"
    "```",
)

_CORPUS_JSON_BLOCKS: tuple[str, ...] = (
    '```json\n{"max_entries": 512, "ttl_seconds": 300, "eviction": "lru"}\n```',
    "```json\n"
    '{"name": "ci", "on": {"push": {"branches": ["main"]}}, '
    '"jobs": {"test": {"runs-on": "ubuntu-latest"}}}\n'
    "```",
    '```json\n{"service": "search", "flags": {"reranker_v2": true, "spell_correct": false}}\n```',
)

_CORPUS_TECHNICAL_PARAGRAPHS: tuple[str, ...] = (
    "B-tree pages keep keys sorted with a bounded branching factor, so lookups cost O(log n) "
    "page reads and leaf scans stay cache friendly. Write-heavy workloads amplify differently: "
    "LSM trees batch inserts into memtables and flush sorted runs, then compaction merges them, "
    "trading slower point reads for much higher ingest throughput.",
    "TCP reacts to loss as congestion: timeout-driven recovery resets the window and re-enters "
    "slow start, while duplicate-ACK triggers fast retransmit and congestion avoidance halves "
    "the window instead. Slow start grows the window exponentially until loss or the ssthresh "
    "is reached, then growth turns additive.",
    "Reference counting frees objects the instant their count reaches zero, paying a mutation "
    "overhead on every assignment, and it leaks cycles unless a cycle collector intervenes. "
    "Generational collectors trace roots periodically, promote survivors, and collect young "
    "generations more often, giving longer pauses but complete cycle reclamation.",
    "Write-ahead logging appends a commit record to a sequential log before touching data "
    "pages, so batching fsync calls groups many commits into one flush: throughput rises and "
    "mean latency falls, but the crash window widens because acknowledged commits can sit in "
    "the OS page cache until the next flush.",
    "Floating-point comparison needs care near the representability limit: doubles carry 53 "
    "bits of significand, so every integer up to 2^53 and its neighbors stay exact, while "
    "above 2^53 spacing grows to 2 and sums like 2^53 + 1 round back to 2^53. Hash functions "
    "and counters that mix large integers with floats must respect that boundary.",
    "A CTE with ROW_NUMBER assigns a stable ranking per partition without self-joins, and the "
    "planner can prune the outer query to rn <= 3 before the join. Window functions compute "
    "aggregate context per row in one pass, which keeps the read-only plan linear in input "
    "size.",
)

_CORPUS_IMATRIX_SEED = 17


def build_tenet_domain_imatrix_corpus(
    output_path: str | Path | None = None,
    repeats: int = 4,
) -> dict[str, Any]:
    """Build a clean 0%-slop calibration corpus for ``llama-imatrix``.

    The base corpus combines think-tagged verification traces with concise
    answers, read-only SELECT SQL with CTEs/window functions, valid Python AST
    and JSON config blocks, and direct unslop technical explanations. Blocks
    are repeated and deterministically shuffled (seed 17) so ``repeats=4``
    yields 96 chunks -- comfortably above the ~64-block imatrix floor.
    """
    base_blocks: list[str] = [
        *_CORPUS_THINK_TRACES,
        *_CORPUS_SQL_BLOCKS,
        *_CORPUS_PYTHON_BLOCKS,
        *_CORPUS_JSON_BLOCKS,
        *_CORPUS_TECHNICAL_PARAGRAPHS,
    ]

    # Validate every structured block so corpus drift cannot ship silently.
    for block in _CORPUS_SQL_BLOCKS:
        ok, reason = validate_readonly_sql(_SQL_FENCE_RE.search(block).group(1))
        if not ok:
            raise ValueError(f"Corpus SQL block failed guard: {reason}")
    for block in _CORPUS_PYTHON_BLOCKS:
        if not ast.parse(_CODE_FENCE_RE.search(block).group(1)).body:
            raise ValueError("Corpus Python block has an empty AST body.")
    for block in _CORPUS_JSON_BLOCKS:
        json.loads(_JSON_FENCE_RE.search(block).group(1))

    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")
    blocks = [block for _ in range(repeats) for block in base_blocks]
    random.Random(_CORPUS_IMATRIX_SEED).shuffle(blocks)
    text = "\n\n".join(blocks)

    zero_slop = not _contains_slop_or_refusal(text)
    if not zero_slop:
        raise ValueError("Built corpus failed the zero-slop self-check.")

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    return {
        "text": text,
        "path": str(output_path) if output_path else None,
        "num_blocks": len(blocks),
        "num_chars": len(text),
        "zero_slop_verified": True,
    }
