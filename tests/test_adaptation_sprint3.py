"""Tests for Sprint 3 adaptation: Hybrid RAG, LoRA, Distillation, GRPO, and H5/H6 Comparison."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sml.adaptation import (
    GRPOConfig,
    HybridRunbookRetriever,
    apply_lora_to_transformer,
    collect_rejection_sampled_trajectories,
    compute_distillation_kl_loss,
    compute_group_normalized_advantages,
    compute_grpo_loss,
    format_trajectory_for_sft,
    lora_parameter_summary,
    merge_lora_weights,
    run_adaptation_comparison,
    train_grpo_action_policy,
)
from sml.cli import main
from sml.config import ModelConfig
from sml.envs import load_environment
from sml.model import Transformer


def test_hybrid_retriever_top1_recall() -> None:
    retriever = HybridRunbookRetriever()
    q_bug = "Android app_version 4.18.2 SMS reminder webhook_delivery_rate"
    q_pipe = "dbt stg_ehr_encounters raw_status FINISHED mapped_is_completed"
    q_mix = "dim_clinics onboarding_tier_c booking_horizon_days Simpson paradox"

    assert retriever.retrieve(q_bug, top_k=1)[0][0].category == "product_bug"
    assert retriever.retrieve(q_pipe, top_k=1)[0][0].category == "data_pipeline_artifact"
    assert retriever.retrieve(q_mix, top_k=1)[0][0].category == "mix_shift"


def test_lora_application_and_exact_merge_equivalence() -> None:
    torch.manual_seed(42)
    cfg = ModelConfig(
        name="tiny-lora-test",
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
    )
    model = Transformer(cfg)
    apply_lora_to_transformer(model, rank=4, alpha=8.0)

    summary = lora_parameter_summary(model)
    assert summary["trainable_params"] > 0
    assert 0.0 < summary["trainable_pct"] < 25.0

    # Initialize non-zero lora_b so the LoRA delta is non-trivial
    for name, param in model.named_parameters():
        if "lora_b" in name:
            torch.nn.init.normal_(param, mean=0.0, std=0.05)

    model.eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        logits_unmerged = model(x).logits

    merge_lora_weights(model)
    after_summary = lora_parameter_summary(model)
    assert after_summary["trainable_params"] == 0

    with torch.no_grad():
        logits_merged = model(x).logits

    max_diff = float((logits_unmerged - logits_merged).abs().max().item())
    assert max_diff < 1e-5


def test_rejection_sampling_and_distillation_kl_loss() -> None:
    env = load_environment(split="train", count=6, base_seed=42)
    accepted, stats = collect_rejection_sampled_trajectories(env)

    # 6 scenarios * 3 candidates (oracle, lazy, unsafe) = 18 candidates; only 6 oracle accepted
    assert stats["total_candidates"] == 18
    assert stats["accepted_count"] == 6
    assert len(accepted) == 6
    assert all(t.accepted for t in accepted)

    sft_text = format_trajectory_for_sft(accepted[0])
    assert "<|user|>" in sft_text
    assert "<|tool_call|>" in sft_text
    assert "<|eos|>" in sft_text

    student_logits = torch.randn(2, 8, 64, requires_grad=True)
    teacher_logits = torch.randn(2, 8, 64)
    targets = torch.randint(0, 64, (2, 8))
    loss = compute_distillation_kl_loss(student_logits, teacher_logits, targets)
    loss.backward()
    assert loss.item() > 0.0
    assert student_logits.grad is not None


def test_grpo_advantages_loss_and_policy_convergence() -> None:
    rewards = torch.tensor([0.0, 0.5, 1.12, 0.0, 0.5, 1.12])
    adv = compute_group_normalized_advantages(rewards)
    assert abs(float(adv.mean().item())) < 1e-5

    logp = torch.log_softmax(torch.randn(6, 3), dim=-1)[:, 0].requires_grad_(True)
    old_lp = logp.detach().clone()
    ref_lp = logp.detach().clone() - 0.05
    loss_dict = compute_grpo_loss(logp, old_lp, ref_lp, adv)
    loss_dict["loss"].backward()
    assert "kl_div" in loss_dict

    env = load_environment(split="train", count=3, base_seed=100)
    gated_res = train_grpo_action_policy(
        env, rubric_mode="gated_multiplicative", config=GRPOConfig(steps=16)
    )
    naive_res = train_grpo_action_policy(
        env, rubric_mode="naive_additive", config=GRPOConfig(steps=16)
    )

    assert gated_res["final_verify_prob"] > 0.85
    assert naive_res["final_guess_prob"] > gated_res["final_guess_prob"]


def test_run_adaptation_comparison_and_cli(tmp_path: Path) -> None:
    comp = run_adaptation_comparison(split="test", count=6, seed=2026)
    assert len(comp["results"]) == 7
    assert comp["rag_retrieval_recall_at_1"] == 1.0
    assert comp["hypotheses"]["H5_verified"] is True
    assert comp["hypotheses"]["H6_verified"] is True

    rc = main(
        [
            "compare-adaptation",
            "--experiment",
            "experiments/h6_adaptation_comparison.yaml",
            "--output-dir",
            str(tmp_path),
            "--run-id",
            "test_h6_run",
            "--episodes",
            "6",
        ]
    )
    assert rc == 0
    comp_json = json.loads(
        (tmp_path / "test_h6_run" / "adaptation_comparison.json").read_text(encoding="utf-8")
    )
    assert comp_json["hypotheses"]["H5_verified"] is True
    assert comp_json["hypotheses"]["H6_verified"] is True
