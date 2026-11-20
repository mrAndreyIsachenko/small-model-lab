"""Unit and integration tests for Tenet v2 (H15 & H16)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from sml.adaptation.grpo import run_h16_length_calibrated_grpo_benchmark
from sml.config import load_model_config
from sml.envs.unslop_rubric import (
    TENET_UNSLOP_SCENARIOS,
    TENET_V2_SCENARIOS,
    TenetUnslopRubric,
    UnslopPromptSpec,
    score_unslop_response,
)
from sml.model import Transformer
from sml.surgery.pareto import run_h15_cot_tri_anchor_lancet_benchmark
from sml.surgery.probes import COT_REASONING_ANCHOR_PAIRS, PhaseSegmenter
from sml.surgery.subspace import (
    biproject_direction,
    build_tri_anchor_basis,
    compute_layer_scr_profile,
    extract_contrastive_direction,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# H15: PhaseSegmenter & Tri-Anchor Lancet
# ---------------------------------------------------------------------------


def test_phase_segmenter_formats_and_validation() -> None:
    think = PhaseSegmenter.for_format("think")
    assert (think.deliberation_start, think.deliberation_end) == ("<think>", "</think>")

    reasoning = PhaseSegmenter.for_format("reasoning")
    assert (reasoning.deliberation_start, reasoning.deliberation_end) == (
        "<reasoning>",
        "</reasoning>",
    )

    harmony = PhaseSegmenter.for_format("harmony")
    assert (harmony.deliberation_start, harmony.deliberation_end) == (
        "<|channel|>analysis<|message|>",
        "<|end|>",
    )

    with pytest.raises(ValueError, match="Unsupported phase format"):
        PhaseSegmenter.for_format("unknown_format")


def test_phase_segmenter_segment_text_cases() -> None:
    seg = PhaseSegmenter.for_format("think")

    balanced = seg.segment_text("<think>step 1\nstep 2</think>\nFinal answer.")
    assert balanced.has_deliberation is True
    assert balanced.is_balanced is True
    assert balanced.malformed is False
    assert balanced.open_count == 1
    assert balanced.close_count == 1
    assert "step 1" in balanced.deliberation_text
    assert balanced.answer_text.strip() == "Final answer."

    unclosed = seg.segment_text("<think>truncated reasoning without end")
    assert unclosed.has_deliberation is True
    assert unclosed.is_balanced is False
    assert unclosed.malformed is True
    assert unclosed.open_count == 1
    assert unclosed.close_count == 0

    stray_close = seg.segment_text("Answer prefix </think> trailing text")
    assert stray_close.is_balanced is False
    assert stray_close.malformed is True
    assert stray_close.close_count == 1

    plain = seg.segment_text("Direct answer with no tags.")
    assert plain.has_deliberation is False
    assert plain.is_balanced is True
    assert plain.malformed is False
    assert plain.answer_text == "Direct answer with no tags."


def test_phase_segmenter_masks_token_ids_tokenizer_and_text_fallback() -> None:
    # 1. Explicit token IDs (e.g. sml-nano synthetic tags 3 and 4)
    seg_ids = PhaseSegmenter(deliberation_start_id=3, deliberation_end_id=4)
    ids = torch.tensor([[10, 3, 11, 12, 4, 13, 14], [10, 11, 12, 13, 14, 15, 16]])
    delib_mask, ans_mask = seg_ids.compute_phase_masks(ids)
    assert delib_mask[0].tolist() == [False, True, True, True, True, False, False]
    assert ans_mask[0].tolist() == [True, False, False, False, False, True, True]
    assert delib_mask[1].sum().item() == 0
    assert bool(ans_mask[1].all())

    # 2. Tokenizer subsequence matching
    class _MockTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return {"<think>": [9, 9], "</think>": [5, 5]}[text]

    seg_tok = PhaseSegmenter.for_format("think")
    tok_ids = torch.tensor([[1, 9, 9, 20, 5, 5, 21]])
    d_tok, a_tok = seg_tok.compute_phase_masks(tok_ids, tokenizer=_MockTokenizer())
    assert d_tok[0].tolist() == [False, True, True, True, True, True, False]
    assert a_tok[0].tolist() == [True, False, False, False, False, False, True]

    # 3. Char-ratio fallback from texts
    d_txt, a_txt = seg_tok.compute_phase_masks(
        torch.zeros((2, 10), dtype=torch.long),
        texts=["<think>12345</think>67890", "no tags at all"],
    )
    assert int(d_txt[0].sum().item()) == 5
    assert int(d_txt[1].sum().item()) == 0
    assert bool(a_txt[1].all())


def test_tri_anchor_basis_orthogonality_and_biprojection() -> None:
    torch.manual_seed(42)
    a_cot = [torch.randn(32)]
    a_math = [torch.randn(32)]
    a_sql = [torch.randn(32)]
    # Add a linearly dependent vector that MGS must drop.
    dependent = a_cot[0] * 3.5

    basis = build_tri_anchor_basis(a_cot + [dependent], a_math, a_sql)
    assert len(basis) == 3
    assert all(b.dtype == torch.float32 for b in basis)

    stack = torch.stack(basis, dim=0)
    gram = stack @ stack.T
    eye = torch.eye(len(basis), dtype=torch.float32)
    orth_err = float((gram - eye).abs().max().item())
    assert orth_err < 1e-6

    target = torch.randn(32)
    projected = biproject_direction(target, basis)
    assert abs(float(projected.norm(p=2).item()) - 1.0) < 1e-6
    max_dot = max(abs(float((projected @ b).item())) for b in basis)
    assert max_dot < 1e-6


def test_phase_segmented_extraction_and_cot_scr_on_sml_nano() -> None:
    torch.manual_seed(2026)
    cfg = load_model_config(str(REPO_ROOT / "configs/model/sml_nano.yaml"))
    model = Transformer(cfg)
    model.eval()

    seg = PhaseSegmenter(deliberation_start_id=3, deliberation_end_id=4)
    pos_ids = torch.tensor([[10, 3, 20, 21, 22, 4, 30, 31], [11, 3, 23, 24, 25, 4, 32, 33]])
    neg_ids = torch.tensor([[10, 3, 40, 41, 42, 4, 50, 51], [11, 3, 43, 44, 45, 4, 52, 53]])

    d_delib = extract_contrastive_direction(
        model, pos_ids, neg_ids, layer_idx=0, pool_mode="deliberation_only", phase_segmenter=seg
    )
    d_ans = extract_contrastive_direction(
        model, pos_ids, neg_ids, layer_idx=0, pool_mode="answer_only", phase_segmenter=seg
    )
    d_all = extract_contrastive_direction(model, pos_ids, neg_ids, layer_idx=0, pool_mode="all")

    assert d_delib.shape == (cfg.d_model,)
    assert d_ans.shape == (cfg.d_model,)
    assert abs(float(d_delib.norm().item()) - 1.0) < 1e-5
    assert abs(float(d_ans.norm().item()) - 1.0) < 1e-5
    # Distinct pooling windows produce distinct directions.
    assert float((d_delib - d_ans).norm().item()) > 1e-3
    assert float((d_all - d_ans).norm().item()) > 1e-3

    target_dirs = {idx: [d_ans] for idx in range(cfg.n_layers)}
    profile = compute_layer_scr_profile(
        model,
        target_dirs,
        anchor_input_ids=neg_ids,
        cot_input_ids=pos_ids,
        cot_tag_token_ids=[3, 4],
        lambda_cot=0.5,
    )
    assert len(profile) == cfg.n_layers
    for row in profile:
        assert {"layer_idx", "signal", "kl_anchor", "kl_cot", "kl_collateral", "scr"} <= set(row)
        assert row["kl_anchor"] >= 0.0
        assert row["kl_cot"] >= 0.0
        assert row["kl_collateral"] >= 0.0


def test_h15_cot_tri_anchor_lancet_benchmark() -> None:
    assert len(COT_REASONING_ANCHOR_PAIRS) >= 8
    result = run_h15_cot_tri_anchor_lancet_benchmark(seed=2026, top_k_layers=3, lambda_cot=0.5)
    assert result["h15_verified"] is True
    assert result["tri_anchor_orthogonality_error"] < 1e-6
    assert result["biprojected_orthogonality_error"] < 1e-6
    assert result["max_row_crest_factor"] < 25.0
    assert result["max_row_norm_drift"] < 1e-5


# ---------------------------------------------------------------------------
# H16: Dual-Window Length-Calibrated & Artifact-Completeness Gated GRPO
# ---------------------------------------------------------------------------


def test_v2_rubric_cot_integrity_gate() -> None:
    rubric = TenetUnslopRubric(mode="v2")
    spec = UnslopPromptSpec(
        scenario_id="test-cot",
        prompt="Return JSON with status.",
        require_json=True,
        required_keys=["status"],
        artifact_type="json",
        require_cot=True,
        target_ans_words=4,
        min_words=2,
        max_words=100,
    )

    good = rubric.score(
        '<think>Checking schema keys: status is required.</think> {"status": "ok"}', spec
    )
    assert good.g_cot_integrity == 1.0
    assert good.artifact_valid is True
    assert good.total_reward > 0.0

    unclosed = rubric.score('<think>Forgot to close {"status": "ok"}', spec)
    assert unclosed.g_cot_integrity == 0.0
    assert unclosed.total_reward == 0.0

    leaked = rubric.score('<think>ok</think> </think> {"status": "ok"}', spec)
    assert leaked.g_cot_integrity == 0.0
    assert leaked.total_reward == 0.0

    looping = rubric.score(
        "<think>" + "check step next " * 8 + '</think> {"status": "ok"}',
        spec,
    )
    assert looping.g_cot_integrity == 0.0
    assert looping.total_reward == 0.0


def test_v2_rubric_dual_window_length_calibration() -> None:
    rubric = TenetUnslopRubric(mode="v2")
    spec = UnslopPromptSpec(
        scenario_id="test-len",
        prompt="Explain backpressure.",
        task_type="concise_factual",
        target_ans_words=40,
        target_cot_words=30,
        require_cot=True,
        min_words=5,
        max_words=200,
    )

    def _make_resp(cot_n: int, ans_n: int) -> str:
        cot = " ".join(f"c{i}" for i in range(cot_n))
        ans = " ".join(f"w{i}" for i in range(ans_n))
        return f"<think> {cot} </think> {ans}"

    r_collapsed = rubric.score(_make_resp(10, 4), spec)
    r_mid = rubric.score(_make_resp(10, 20), spec)
    r_target = rubric.score(_make_resp(10, 40), spec)
    r_overthink = rubric.score(_make_resp(120, 40), spec)

    # Smooth positive gradient fighting length_collapse.
    assert r_collapsed.g_length_cal < r_mid.g_length_cal < r_target.g_length_cal
    assert r_target.g_length_cal > 0.99
    # Overthinking beyond target_cot_words is penalized.
    assert r_overthink.g_length_cal < r_target.g_length_cal


def test_v2_rubric_antislop_density_and_artifacts() -> None:
    rubric = TenetUnslopRubric(mode="v2")

    sql_spec = UnslopPromptSpec(
        scenario_id="test-sql",
        prompt="Write a read-only SQL query.",
        task_type="analytical_sql",
        artifact_type="sql_readonly",
        require_cot=True,
        min_words=5,
        max_words=150,
    )
    good_sql = (
        "<think> Need a SELECT query filtering active users. </think> "
        "```sql\nSELECT id, email FROM users WHERE active = true;\n```"
    )
    bad_sql = "<think> Dropping table. </think> ```sql\nDROP TABLE users;\n```"
    assert rubric.score(good_sql, sql_spec).artifact_valid is True
    assert rubric.score(bad_sql, sql_spec).artifact_valid is False

    code_spec = UnslopPromptSpec(
        scenario_id="test-code",
        prompt="Write is_even in Python.",
        artifact_type="code",
        require_cot=True,
        min_words=3,
        max_words=150,
    )
    assert (
        rubric.score(
            "<think> Check modulo 2. </think> ```python\n"
            "def is_even(n: int) -> bool:\n    return n % 2 == 0\n```",
            code_spec,
        ).artifact_valid
        is True
    )
    assert (
        rubric.score(
            "<think> Broken. </think> ```python\ndef is_even(:\n```",
            code_spec,
        ).artifact_valid
        is False
    )

    num_spec = UnslopPromptSpec(
        scenario_id="test-num",
        prompt="Compute 480 * 3.",
        artifact_type="exact_numeric",
        exact_answer="1440",
        require_cot=True,
        min_words=1,
        max_words=100,
    )
    assert rubric.score("<think> 480 * 3 = 1440. </think> 1440", num_spec).artifact_valid is True
    assert rubric.score("<think> Guessing. </think> 1400", num_spec).artifact_valid is False

    # Intro/outro paragraph slop receives hard penalty; refusal gets 0.0.
    prose_spec = TENET_V2_SCENARIOS[0]
    slop_intro = (
        "<think> Standard guitar has six strings. </think> "
        "It is a testament to design.\n\nA standard guitar has six strings tuned EADGBE."
    )
    clean_prose = (
        "<think> Standard guitar has six strings. </think> "
        "A standard guitar has six strings tuned EADGBE."
    )
    assert (
        rubric.score(slop_intro, prose_spec).g_antislop_density
        < rubric.score(clean_prose, prose_spec).g_antislop_density
    )
    assert rubric.score("I cannot assist with that request.", prose_spec).total_reward == 0.0

    # Backwards compatibility with v1 gated mode.
    v1_res = score_unslop_response(
        "A direct, concrete explanation without clichés.",
        TENET_UNSLOP_SCENARIOS[0],
        mode="gated",
    )
    assert v1_res.mode == "gated"
    assert v1_res.g_compliance == 1.0


def test_h16_length_calibrated_grpo_benchmark() -> None:
    result = run_h16_length_calibrated_grpo_benchmark(seed=2026)
    assert result["h16_verified"] is True
    assert result["cot_tag_integrity_rate"] >= 0.995
    assert result["length_collapse_rate"] < 0.01
    assert result["slop_and_refusal_rate"] <= 0.02
    assert result["artifact_completeness_rate"] >= 0.99
    assert result["subspace_safe_lora_releakage"] < 1e-5
    assert result["merged_lora_count"] == 0
    assert result["max_abs_logit_diff"] < 1e-4
    assert result["h16_v2_vs_v1_gain"] > 0.2


def test_tenet_v2_experiment_configs_parse() -> None:
    for rel_path, expected_id in (
        ("experiments/h15_cot_tri_anchor_lancet.yaml", "H15"),
        ("experiments/h16_length_calibrated_grpo.yaml", "H16"),
    ):
        data = yaml.safe_load((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
        assert data["hypothesis_id"] == expected_id


def test_phase_span_aliases_and_fact_density() -> None:
    # H15: PhaseSpan exposes span aliases equal to the extracted phase texts.
    seg = PhaseSegmenter.for_format("think")
    span = seg.segment_text("<think>step 1\nstep 2</think>\nFinal answer with detail.")
    assert span.deliberation_span == span.deliberation_text
    assert span.surface_answer_span == span.answer_text

    # v2: fact-dense answers (numbers/code/entities/SQL) outscore vague filler
    # of comparable word count on information density.
    rubric = TenetUnslopRubric(mode="v2")
    spec = TENET_V2_SCENARIOS[0]
    dense = (
        "<think> Standard guitar has six strings. </think> "
        "A standard guitar has six strings, i.e. 6, tuned EADGBE via "
        "`standard_tuning`; GuitarPro lists model_id 6. SQL: select strings, "
        "join specs where tuning = EADGBE group by model_id."
    )
    filler = (
        "<think> Standard guitar has six strings. </think> "
        "There are six broad aspects here that could matter in this particular "
        "situation, and it honestly just depends on various unrelated factors "
        "rather than anything specific."
    )
    dense_score = rubric.score(dense, spec)
    filler_score = rubric.score(filler, spec)
    assert dense_score.r_info_density > filler_score.r_info_density
    assert dense_score.total_reward > filler_score.total_reward
    assert dense_score.diagnostics["fact_density"] > filler_score.diagnostics["fact_density"]
