"""H17 Peer Eval Gate: Tenet v2 release gates over peer responses.

Evaluates either fresh responses (via a responder function over the house
series or custom prompts) or a pre-generated response batch, computes CoT tag
integrity / length-collapse / slop-and-refusal metrics, and enforces the Tenet
v2 gates:

- ``cot_tag_integrity_rate >= 0.995``
- ``length_collapse_rate < 0.01``
- ``slop_and_refusal_rate <= 0.02``
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from sml.evals.artifact_verifier import (
    TENET_ARENA_HOUSE_SERIES,
    ArenaHousePrompt,
    _contains_slop_or_refusal,
    _word_count,
)

if TYPE_CHECKING:
    from sml.surgery.probes import PhaseSegmenter, PhaseSpan

__all__ = ["run_peer_eval_gate"]

_MIN_WORDS_BEFORE_COLLAPSE = 8

# A surface answer that is itself a bare number ("4326", "-12.5", "78%") is an
# exact-numeric surface even when no prompt spec is available to classify it.
_BARE_NUMERIC_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?%?$")

# Tenet v2 release gates.
COT_TAG_INTEGRITY_MIN = 0.995
LENGTH_COLLAPSE_MAX = 0.01
SLOP_REFUSAL_MAX = 0.02

_SEGMENTER_INSTANCE: PhaseSegmenter | None = None


def _get_segmenter() -> PhaseSegmenter:
    global _SEGMENTER_INSTANCE
    if _SEGMENTER_INSTANCE is None:
        from sml.surgery.probes import PhaseSegmenter

        _SEGMENTER_INSTANCE = PhaseSegmenter.for_format("think")
    return _SEGMENTER_INSTANCE


def _resolve_jobs(
    responder_fn: Callable[[str], str] | None,
    responses: Sequence[str] | None,
    prompts: Sequence[str] | Sequence[ArenaHousePrompt] | None,
) -> list[tuple[ArenaHousePrompt | None, str]]:
    """Pair each response with its prompt spec (when one is known)."""
    if responses is None:
        if responder_fn is None:
            raise ValueError("Provide responder_fn or responses.")
        if prompts is None:
            prompt_items: Sequence[ArenaHousePrompt] = TENET_ARENA_HOUSE_SERIES
        else:
            prompt_items = list(prompts)
        jobs: list[tuple[ArenaHousePrompt | None, str]] = []
        for item in prompt_items:
            prompt_text = item.prompt if isinstance(item, ArenaHousePrompt) else item
            spec = item if isinstance(item, ArenaHousePrompt) else None
            jobs.append((spec, responder_fn(prompt_text)))
        return jobs

    if prompts is not None and len(prompts) != len(responses):
        raise ValueError(
            f"prompts ({len(prompts)}) and responses ({len(responses)}) lengths differ."
        )
    jobs = []
    for i, response in enumerate(responses):
        spec = None
        if prompts is not None:
            item = prompts[i]
            if isinstance(item, ArenaHousePrompt):
                spec = item
        elif len(responses) == len(TENET_ARENA_HOUSE_SERIES):
            spec = TENET_ARENA_HOUSE_SERIES[i]
        jobs.append((spec, response))
    return jobs


def run_peer_eval_gate(
    responder_fn: Callable[[str], str] | None = None,
    responses: Sequence[str] | None = None,
    require_think: bool = True,
    prompts: Sequence[str] | Sequence[ArenaHousePrompt] | None = None,
) -> dict[str, Any]:
    """Evaluate responses against the Tenet v2 peer-eval gates.

    ``responder_fn`` + ``prompts`` (defaulting to the house series) generates
    responses; alternatively pass ``responses`` directly. Returns metrics, a
    boolean ``gate_passed``, and a per-gate ``gate_details`` breakdown.
    """
    jobs = _resolve_jobs(responder_fn, responses, prompts)
    if not jobs:
        raise ValueError("No responses to evaluate.")

    total = len(jobs)
    cot_valid = 0
    collapsed = 0
    slop_refusal_hits = 0
    collapse_eligible = 0
    response_word_counts: list[int] = []
    cot_word_counts: list[int] = []

    for spec, response in jobs:
        span: PhaseSpan = _get_segmenter().segment_text(response)
        surface = span.answer_text
        full_text = response

        is_cot_valid = span.is_balanced
        if require_think and span.open_count == 0:
            is_cot_valid = False
        cot_valid += int(is_cot_valid)

        artifact_type = spec.artifact_type if spec is not None else None
        is_exact_numeric = artifact_type == "exact_numeric"
        if not is_exact_numeric and spec is None and _BARE_NUMERIC_RE.match(
            surface.strip()
        ):
            # No prompt spec to classify against, but a bare numeric surface
            # ("4326", "-12.5", "78%") is an exact-numeric answer, not a
            # length collapse.
            is_exact_numeric = True
        if not is_exact_numeric:
            collapse_eligible += 1
            if _word_count(surface) < _MIN_WORDS_BEFORE_COLLAPSE:
                collapsed += 1

        slop_target = surface if surface.strip() else full_text
        slop_refusal_hits += int(_contains_slop_or_refusal(slop_target))

        response_word_counts.append(_word_count(full_text))
        cot_word_counts.append(_word_count(span.deliberation_text))

    cot_tag_integrity_rate = cot_valid / total
    length_collapse_rate = (collapsed / collapse_eligible) if collapse_eligible else 0.0
    slop_and_refusal_rate = slop_refusal_hits / total
    mean_response_words = sum(response_word_counts) / total
    mean_cot_words = sum(cot_word_counts) / total

    metrics = {
        "total_evaluated": total,
        "cot_tag_integrity_rate": cot_tag_integrity_rate,
        "length_collapse_rate": length_collapse_rate,
        "slop_and_refusal_rate": slop_and_refusal_rate,
        "mean_response_words": mean_response_words,
        "mean_cot_words": mean_cot_words,
    }

    gate_details = {
        "cot_tag_integrity": {
            "value": cot_tag_integrity_rate,
            "threshold_min": COT_TAG_INTEGRITY_MIN,
            "passed": cot_tag_integrity_rate >= COT_TAG_INTEGRITY_MIN,
        },
        "length_collapse": {
            "value": length_collapse_rate,
            "threshold_max": LENGTH_COLLAPSE_MAX,
            "passed": length_collapse_rate < LENGTH_COLLAPSE_MAX,
        },
        "slop_and_refusal": {
            "value": slop_and_refusal_rate,
            "threshold_max": SLOP_REFUSAL_MAX,
            "passed": slop_and_refusal_rate <= SLOP_REFUSAL_MAX,
        },
    }

    return {
        **metrics,
        "gate_passed": all(detail["passed"] for detail in gate_details.values()),
        "gate_details": gate_details,
    }