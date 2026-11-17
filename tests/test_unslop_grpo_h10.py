"""Sprint 6 Batch 3: H10 unslop GRPO rubric, LoRA merge, and benchmark tests."""

import copy

import pytest
import torch

from sml.adaptation.grpo import TenetRehabPolicy, run_h10_unslop_grpo_benchmark
from sml.adaptation.lora import (
    LoRALinear,
    apply_lora_to_transformer,
    merge_lora_weights_in_place,
)
from sml.envs.unslop_rubric import (
    SLOP_PHRASES,
    TENET_UNSLOP_SCENARIOS,
    TenetUnslopRubric,
    UnslopPromptSpec,
    score_unslop_response,
)

_PLAIN_SPEC = UnslopPromptSpec(
    scenario_id="test-plain",
    prompt="Write two sentences about a lighthouse keeper.",
)

_JSON_SPEC = UnslopPromptSpec(
    scenario_id="test-json",
    prompt="Reply with a JSON object.",
    require_json=True,
    required_keys=["city", "temperature_c"],
    min_words=1,
    max_words=60,
)

_CLEAN_TEXT = "Rain taps the tin roof while a quiet dog dreams beside the warm stove."

_REFUSAL_TEXTS = (
    "As an AI assistant, I am glad to help with anything at all.",
    "It is important to remember that accuracy matters most here.",
    "I cannot fulfill that request under any circumstances.",
)

_SLOP_TEXT = (
    "The garden was a tapestry of summer colors, a testament to decades of "
    "patient care, and shivers down the spine came with every cold gust."
)


def test_slop_phrases_and_scenarios_counts() -> None:
    """SLOP_PHRASES is large enough and the scenario tuple has 24 entries."""
    assert len(SLOP_PHRASES) >= 40
    assert len(TENET_UNSLOP_SCENARIOS) == 24
    for phrase in ("tapestry", "testament to", "shivers down"):
        assert phrase in SLOP_PHRASES


def test_g_compliance_gate_blocks_refusals_and_empty() -> None:
    """Refusals and empty responses zero the gate; additive mode still pays."""
    for text in (*_REFUSAL_TEXTS, ""):
        gated = score_unslop_response(text, _PLAIN_SPEC, mode="gated")
        assert gated.g_compliance == 0.0
        assert gated.total_reward == 0.0
        additive = score_unslop_response(text, _PLAIN_SPEC, mode="additive")
        assert additive.total_reward > 0.0
    for text in _REFUSAL_TEXTS:
        assert score_unslop_response(text, _PLAIN_SPEC).refusal_hits >= 1
    clean = score_unslop_response(_CLEAN_TEXT, _PLAIN_SPEC, mode="gated")
    assert clean.g_compliance == 1.0
    assert clean.total_reward > 0.0


def test_g_antislop_penalizes_cliches_and_low_distinct3() -> None:
    """Slop clichés cut g_antislop/total; repetitive text cuts distinct_3."""
    clean = score_unslop_response(_CLEAN_TEXT, _PLAIN_SPEC, mode="gated")
    slop = score_unslop_response(_SLOP_TEXT, _PLAIN_SPEC, mode="gated")
    assert slop.slop_hits >= 3
    assert set(slop.diagnostics["matched_slop_phrases"]) >= {
        "tapestry",
        "testament to",
        "shivers down",
    }
    assert slop.g_antislop < clean.g_antislop
    assert slop.total_reward < clean.total_reward

    repetitive = ("the cat sat on the mat. " * 4).strip()
    repeat = score_unslop_response(repetitive, _PLAIN_SPEC, mode="gated")
    assert repeat.distinct_3 < clean.distinct_3
    assert repeat.g_antislop < clean.g_antislop


def test_r_ifeval_constraints() -> None:
    """JSON schema, word limits, forbidden words, and keywords all score."""
    unfenced = '{"city": "Oslo", "temperature_c": 3, "condition": "snow"}'
    fenced = '```json\n{"city": "Oslo", "temperature_c": 3}\n```'
    missing_key = '{"city": "Oslo"}'
    not_json = "The city of Oslo reports a temperature near three degrees."

    scores = {
        text: score_unslop_response(text, _JSON_SPEC)
        for text in (unfenced, fenced, missing_key, not_json)
    }
    assert scores[unfenced].r_ifeval == 1.0
    assert scores[fenced].r_ifeval == scores[unfenced].r_ifeval
    assert scores[missing_key].r_ifeval == pytest.approx(0.75)
    assert scores[not_json].r_ifeval == pytest.approx(0.5)
    json_scores = [
        score.diagnostics["constraint_scores"]["json_schema"] for score in scores.values()
    ]
    assert json_scores == [1.0, 1.0, 0.5, 0.0]

    limit_spec = UnslopPromptSpec(
        scenario_id="test-limit",
        prompt="Answer in 10 to 12 words.",
        min_words=10,
        max_words=12,
    )
    within = "Our team ships the new release every second Friday without fail."
    short = "Ships the release."
    long = (
        "Our team ships the new release every second Friday without fail and "
        "we celebrate with cake each time."
    )
    assert score_unslop_response(within, limit_spec).r_ifeval == 1.0
    assert score_unslop_response(short, limit_spec).r_ifeval == 0.0
    assert score_unslop_response(long, limit_spec).r_ifeval == 0.0

    forbidden_spec = UnslopPromptSpec(
        scenario_id="test-forbidden",
        prompt="Describe the market.",
        forbidden_words=["tapestry", "testament to"],
    )
    forbidden_scores = {
        text: score_unslop_response(text, forbidden_spec)
        for text in (
            "The harbor market sells fresh herring every single morning.",
            "The harbor market tapestry sells fresh herring.",
            "The harbor tapestry stood as a testament to the market.",
        )
    }
    forbidden_sub = [
        score.diagnostics["constraint_scores"]["forbidden_words"]
        for score in forbidden_scores.values()
    ]
    assert forbidden_sub == [1.0, 0.5, 0.0]
    assert [score.r_ifeval for score in forbidden_scores.values()] == [
        1.0,
        pytest.approx(0.75),
        pytest.approx(0.5),
    ]

    keyword_spec = UnslopPromptSpec(
        scenario_id="test-keywords",
        prompt="Describe the figurine.",
        required_keywords=["bronze", "owl"],
    )
    keyword_scores = {
        text: score_unslop_response(text, keyword_spec)
        for text in (
            "The bronze owl figurine stares across the quiet gallery.",
            "The bronze figurine stares across the quiet gallery.",
            "The figurine stares across the quiet gallery.",
        )
    }
    keyword_sub = [
        score.diagnostics["constraint_scores"]["required_keywords"]
        for score in keyword_scores.values()
    ]
    assert keyword_sub == [1.0, 0.5, 0.0]
    assert [score.r_ifeval for score in keyword_scores.values()] == [
        1.0,
        pytest.approx(0.75),
        pytest.approx(0.5),
    ]


def test_r_coherence_penalizes_repetition_and_truncation() -> None:
    """Degenerate repetition and missing end punctuation lower r_coherence."""
    clean_text = "The garden gate squeaks whenever the autumn wind pushes through it."
    clean = score_unslop_response(clean_text, _PLAIN_SPEC)
    assert clean.r_coherence == 1.0
    assert clean.diagnostics["coherence"]["degenerate"] is False
    assert clean.diagnostics["coherence"]["ends_cleanly"] is True

    degenerate_text = "yes yes yes yes yes yes yes yes."
    degenerate = score_unslop_response(degenerate_text, _PLAIN_SPEC)
    assert degenerate.r_coherence < clean.r_coherence
    assert degenerate.diagnostics["coherence"]["degenerate"] is True

    truncated = score_unslop_response(clean_text.rstrip("."), _PLAIN_SPEC)
    assert truncated.r_coherence < clean.r_coherence
    assert truncated.diagnostics["coherence"]["ends_cleanly"] is False


def test_merge_lora_weights_in_place_logit_equivalence() -> None:
    """Merging non-zero LoRA deltas in place preserves logits exactly."""
    torch.manual_seed(1234)
    base = TenetRehabPolicy()
    policy = copy.deepcopy(base)
    wrapped = apply_lora_to_transformer(policy, rank=4, alpha=8.0)
    assert wrapped is policy
    lora_modules = [m for m in policy.modules() if isinstance(m, LoRALinear)]
    assert len(lora_modules) > 0

    generator = torch.Generator().manual_seed(7)
    for module in lora_modules:
        with torch.no_grad():
            module.lora_b.normal_(mean=0.0, std=0.05, generator=generator)
        assert bool(torch.count_nonzero(module.lora_b))

    scenario_ids = torch.arange(len(TENET_UNSLOP_SCENARIOS))
    with torch.no_grad():
        adapted_logits = policy(scenario_ids).detach().clone()
        base_logits = base(scenario_ids)
    # The non-zero adapters must actually change the outputs being merged.
    assert float((adapted_logits - base_logits).abs().max()) > 1e-3

    merged = merge_lora_weights_in_place(policy)
    assert merged is policy
    with torch.no_grad():
        merged_logits = policy(scenario_ids)
    max_abs_logit_diff = float((adapted_logits - merged_logits).abs().max())
    assert max_abs_logit_diff < 1e-4
    assert sum(isinstance(m, LoRALinear) for m in policy.modules()) == 0


def _wrap_with_seeded_lora(base_model, seed: int):
    """Deep-copy `base_model`, attach LoRA, and fill lora_b with non-zero values."""
    policy = copy.deepcopy(base_model)
    apply_lora_to_transformer(policy, rank=4, alpha=8.0)
    modules = [m for m in policy.modules() if isinstance(m, LoRALinear)]
    assert modules
    generator = torch.Generator().manual_seed(seed)
    for module in modules:
        with torch.no_grad():
            module.lora_b.normal_(mean=0.0, std=0.05, generator=generator)
        assert bool(torch.count_nonzero(module.lora_b))
    return policy, modules


def test_subspace_safe_lora_merge_and_row_norm() -> None:
    """Safe merges remove delta leakage into forbidden dirs; row norms survive."""
    torch.manual_seed(2026)
    base = TenetRehabPolicy()
    pristine = copy.deepcopy(base)
    policy_safe, _ = _wrap_with_seeded_lora(base, seed=7)
    policy_raw, _ = _wrap_with_seeded_lora(base, seed=7)
    policy_norm, _ = _wrap_with_seeded_lora(base, seed=7)

    base_all = {
        name: module.weight.detach().clone()
        for name, module in pristine.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    def resolve(policy: torch.nn.Module, path: str):
        parts = path.split(".")
        parent = policy
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def weight_of(policy: torch.nn.Module, path: str) -> torch.Tensor:
        parent, attr = resolve(policy, path)
        module = getattr(parent, attr)
        if isinstance(module, LoRALinear):
            for inner_name in ("linear", "wrapped", "base_layer", "layer", "base"):
                inner = getattr(module, inner_name, None)
                if isinstance(inner, torch.nn.Linear):
                    return inner.weight.detach().clone()
            raise AssertionError(f"LoRALinear base module not found at {path}")
        return module.weight.detach().clone()

    paths = [name for name, m in policy_safe.named_modules() if isinstance(m, LoRALinear)]
    assert paths
    base_weights = {path: base_all[path] for path in paths}
    out_dim = base_weights[paths[0]].shape[0]
    assert all(base_weights[path].shape[0] == out_dim for path in paths)
    shared_dirs = torch.linalg.qr(torch.randn(out_dim, 3))[0].t().contiguous()
    per_dirs = {
        path: torch.linalg.qr(torch.randn(out_dim, 3))[0].t().contiguous() for path in paths
    }

    def merged_delta(policy: torch.nn.Module, path: str) -> torch.Tensor:
        return weight_of(policy, path).float() - base_weights[path].float()

    # Unsafe control: merging without projection leaks > 1e-3 into the forbidden span.
    for path in paths:
        parent, attr = resolve(policy_raw, path)
        merged_layer = getattr(parent, attr).merge(forbidden_dirs=None, preserve_row_norm=False)
        setattr(parent, attr, merged_layer)
    for path in paths:
        raw_leakage = float(torch.linalg.norm(merged_delta(policy_raw, path).T @ per_dirs[path].T))
        assert raw_leakage > 1e-3

    # Safe merge: column-space leakage into the forbidden span is < 1e-5.
    for path in paths:
        parent, attr = resolve(policy_safe, path)
        merged_layer = getattr(parent, attr).merge(
            forbidden_dirs=per_dirs[path], preserve_row_norm=False
        )
        setattr(parent, attr, merged_layer)
    for path in paths:
        leakage = float(torch.linalg.norm(merged_delta(policy_safe, path).T @ per_dirs[path].T))
        assert leakage < 1e-5

    # Row-norm-preserving merge via the in-place helper.
    merged = merge_lora_weights_in_place(
        policy_norm, forbidden_dirs=shared_dirs, preserve_row_norm=True
    )
    assert merged is policy_norm
    max_row_norm_drift = 0.0
    for path in paths:
        weight_after = weight_of(policy_norm, path).float()
        drift = (base_weights[path].float().norm(dim=1) - weight_after.norm(dim=1)).abs().max()
        max_row_norm_drift = max(max_row_norm_drift, float(drift.item()))
    assert max_row_norm_drift < 1e-5
    assert sum(isinstance(m, LoRALinear) for m in policy_norm.modules()) == 0


def test_length_collapse_protection_in_rubric() -> None:
    """Length collapses are punished sharply; valid 15-word answers keep full credit."""
    spec = UnslopPromptSpec(
        scenario_id="len_test",
        prompt="Explain.",
        min_words=10,
        max_words=60,
    )
    rubric = TenetUnslopRubric(mode="gated")
    valid_text = "Rain taps the tin roof while a quiet dog dreams beside the warm stove tonight."
    assert len(valid_text.split()) == 15

    collapsed = rubric.score("Done.", spec)
    valid = rubric.score(valid_text, spec)

    assert collapsed.length_retention_score < 0.2
    assert valid.length_retention_score == 1.0
    assert collapsed.r_coherence < valid.r_coherence
    assert collapsed.total_reward < 0.5 * valid.total_reward


def test_run_h10_unslop_grpo_benchmark_acceptance() -> None:
    """The H10 benchmark passes every acceptance threshold with 3 arms."""
    result = run_h10_unslop_grpo_benchmark()
    assert result["h10_verified"] is True
    assert result["slop_and_refusal_rate"] <= 0.02
    assert result["h10_gated_vs_naive_gain"] >= 0.15
    assert result["h10_post_surgery_gain"] >= 0.15
    assert result["max_abs_logit_diff"] < 1e-4
    assert result["merged_lora_count"] == 0
    assert result["subspace_safe_lora_releakage"] < 1e-5
    assert result["post_merge_suppression_retention"] >= 0.90
    assert result["length_collapse_rate"] == 0.0
    assert result["mean_response_words"] > 0
    assert set(result["arms"]) == {
        "1_lancet_surgical_only",
        "2_tenet_grpo_additive",
        "3_tenet_grpo_gated",
    }
