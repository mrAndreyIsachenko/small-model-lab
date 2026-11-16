"""Tests for Sprint 3 adaptation: Hybrid RAG, LoRA, Distillation, GRPO, and H5/H6 Comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from sml.adaptation import (
    GRPOConfig,
    HybridRunbookRetriever,
    LLMGRPOConfig,
    LoRALinear,
    apply_lora_to_transformer,
    collect_oracle_sft_trajectories,
    collect_rejection_sampled_trajectories,
    collect_teacher_gateway_trajectories,
    compute_distillation_kl_loss,
    compute_group_normalized_advantages,
    compute_grpo_loss,
    format_trajectory_for_sft,
    lora_parameter_summary,
    merge_lora_weights,
    run_adaptation_comparison,
    run_llm_adaptation_comparison,
    run_teacher_gateway_episode,
    train_grpo_action_policy,
    train_grpo_llm_policy,
    train_lora_sft_llm,
)
from sml.adaptation.grpo import compute_token_level_grpo_loss, compute_token_logprobs
from sml.cli import main
from sml.config import ModelConfig
from sml.envs import load_environment
from sml.envs.tool_env import run_oracle_episode
from sml.gateway import GatewayConfig, OpenRouterGatewayClient
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


def _build_tiny_qwen() -> Qwen2ForCausalLM:
    """Tiny in-memory Qwen2 causal LM (fast enough for CPU rollout/training tests)."""
    torch.manual_seed(1234)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    return Qwen2ForCausalLM(config)


class TinyChatTokenizer:
    """Deterministic character-level dummy tokenizer with ChatML-style chat template."""

    eos_token = "<|im_end|>"
    pad_token = "<|im_end|>"
    eos_token_id = 0
    pad_token_id = 0

    _special_ids = {
        "<|im_end|>": 0,
        "<|im_start|>": 1,
        "<|user|>": 2,
        "<|assistant|>": 3,
        "<|tool_call|>": 4,
    }

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: object) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for token, token_id in sorted(
                self._special_ids.items(), key=lambda item: -len(item[0])
            ):
                if text.startswith(token, i):
                    ids.append(token_id)
                    i += len(token)
                    break
            else:
                ids.append(5 + (ord(text[i]) % 250))
                i += 1
        return ids

    def decode(
        self, token_ids: list[int], skip_special_tokens: bool = False, **kwargs: object
    ) -> str:
        id_to_token = {tid: tok for tok, tid in self._special_ids.items()}
        chars = []
        for tid in token_ids:
            if tid in id_to_token:
                if not skip_special_tokens:
                    chars.append(id_to_token[tid])
            else:
                chars.append(chr(97 + (tid % 26)))
        return "".join(chars)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs: object,
    ) -> str | list[int]:
        text = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


def _assert_all_metrics_finite(metrics: object, path: str = "metrics") -> None:
    """Recursively assert every numeric leaf in a metrics mapping is finite."""
    if metrics is None or isinstance(metrics, (bool, str)):
        return
    if isinstance(metrics, (int, float)):
        assert math.isfinite(float(metrics)), f"non-finite value at {path}: {metrics}"
    elif isinstance(metrics, dict):
        for key, value in metrics.items():
            _assert_all_metrics_finite(value, f"{path}.{key}")
    elif isinstance(metrics, (list, tuple)):
        for idx, value in enumerate(metrics):
            _assert_all_metrics_finite(value, f"{path}[{idx}]")


def test_collect_oracle_sft_trajectories_and_lora_sft_loss_decreases() -> None:
    env = load_environment(split="train", count=2, base_seed=7)
    tokenizer = TinyChatTokenizer()
    trajectories = collect_oracle_sft_trajectories(env, tokenizer)

    assert len(trajectories) == 2
    for traj in trajectories:
        assert traj.reward > 1.0
        assert len(traj.input_ids) == len(traj.assistant_action_mask)
        assert set(traj.assistant_action_mask) == {0, 1}

    model = _build_tiny_qwen()
    sft = train_lora_sft_llm(
        model, tokenizer, trajectories, rank=4, alpha=8.0, lr=1e-2, steps=8
    )

    wrapped = [name for name, mod in model.named_modules() if isinstance(mod, LoRALinear)]
    assert len(wrapped) == 4  # one layer x (q_proj, k_proj, v_proj, o_proj)
    assert {name.rsplit(".", 1)[-1] for name in wrapped} == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }

    assert sft["steps"] == 8
    assert sft["num_trajectories"] == 2
    assert sft["lora_parameter_summary"]["trainable_params"] > 0
    assert math.isfinite(sft["initial_loss"])
    assert math.isfinite(sft["final_loss"])
    assert sft["final_loss"] < sft["initial_loss"]


def test_token_level_grpo_loss_and_train_grpo_llm_policy() -> None:
    model = _build_tiny_qwen()
    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 20, 256
    input_ids = torch.randint(0, vocab, (batch, seq_len))
    assistant_action_mask = torch.zeros(batch, seq_len, dtype=torch.long)
    assistant_action_mask[:, 8:16] = 1  # assistant span with both 0s and 1s

    logprobs, shifted_mask = compute_token_logprobs(model, input_ids, assistant_action_mask)
    assert logprobs.shape == (batch, seq_len - 1)
    assert shifted_mask.shape == (batch, seq_len - 1)
    assert torch.equal(shifted_mask, assistant_action_mask[:, 1:])
    assert bool(torch.isfinite(logprobs).all())

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        manual = (
            torch.log_softmax(logits[:, :-1].float(), dim=-1)
            .gather(-1, input_ids[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )
    assert torch.allclose(logprobs, manual, atol=1e-5)

    old_logprobs = logprobs.detach().clone()
    ref_logprobs = logprobs.detach() - 0.05
    advantages = torch.tensor([1.0, 0.0])
    loss_dict = compute_token_level_grpo_loss(
        logprobs, old_logprobs, ref_logprobs, shifted_mask, advantages
    )
    assert {"loss", "policy_obj", "kl_div", "mean_ratio"} <= set(loss_dict)
    assert all(math.isfinite(float(v.detach())) for v in loss_dict.values())
    assert float(loss_dict["policy_obj"].detach()) == pytest.approx(0.5, abs=1e-5)
    assert float(loss_dict["kl_div"].detach()) > 0.0

    single_mask = torch.zeros_like(shifted_mask)
    single_mask[0, 10] = 1  # restrict the action mask to one assistant position
    single_dict = compute_token_level_grpo_loss(
        logprobs, old_logprobs, ref_logprobs, single_mask, advantages
    )
    assert float(single_dict["policy_obj"].detach()) == pytest.approx(1.0, abs=1e-5)

    loss_dict["loss"].backward()
    assert any(
        param.grad is not None and float(param.grad.abs().sum()) > 0.0
        for param in model.parameters()
    )

    env = load_environment(split="train", count=1, base_seed=11)
    grpo_model = _build_tiny_qwen()
    grpo_metrics = train_grpo_llm_policy(
        env,
        grpo_model,
        TinyChatTokenizer(),
        rubric_mode="gated_multiplicative",
        config=LLMGRPOConfig(steps=2, group_size=2, max_new_tokens=16, temperature=0.7),
        seed=7,
    )
    assert grpo_metrics["rubric_mode"] == "gated_multiplicative"
    assert grpo_metrics["steps"] == 2
    assert len(grpo_metrics["history"]) == 2
    assert grpo_metrics["tokens_processed"] > 0
    _assert_all_metrics_finite(grpo_metrics)


def test_run_llm_adaptation_comparison_with_tiny_model() -> None:
    model = _build_tiny_qwen()
    tokenizer = TinyChatTokenizer()
    comp = run_llm_adaptation_comparison(
        split="val",
        count=1,
        sft_steps=2,
        grpo_steps=1,
        group_size=2,
        max_new_tokens=16,
        model=model,
        tokenizer=tokenizer,
    )

    strategies = [result["strategy_name"] for result in comp["results"]]
    assert len(comp["results"]) == 4
    assert set(strategies) == {
        "1_base_zero_shot",
        "3_hybrid_rag",
        "4_lora_sft",
        "6_grpo_gated_rl",
    }

    backend = comp["llm_backend"]
    assert isinstance(backend, dict)
    assert {"model_name", "device", "dtype", "tokens_processed"} <= set(backend)
    assert backend["tokens_processed"] > 0

    assert comp["sft_diagnostics"]["steps"] == 2
    assert "gated_multiplicative" in comp["grpo_policy_diagnostics"]
    _assert_all_metrics_finite(comp)


def _gateway_response(content: str) -> dict[str, object]:
    """OpenAI-shaped chat-completions response body for the mocked transport."""
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def test_collect_teacher_gateway_trajectories_filters_by_gated_rubric() -> None:
    env = load_environment(split="train", count=2, base_seed=7)
    assert len(env.scenarios) == 2
    tokenizer = TinyChatTokenizer()

    good_scenario = env.scenarios[0]
    bad_scenario = env.scenarios[1]

    # Scenario 0 replays the oracle trace verbatim -> reward >= 0.95.
    oracle_state = run_oracle_episode(env, good_scenario)
    good_responses: list[str] = []
    for trace in oracle_state["tool_trace"]:
        payload = json.dumps(
            {"name": str(trace["tool"]), "arguments": dict(trace["args"])}, default=str
        )
        good_responses.append(f"<tool_call>{payload}</tool_call>")
    assert str(oracle_state["tool_trace"][-1]["tool"]) == "submit_diagnosis"

    # Scenario 1 submits an ungrounded, wrong diagnosis with zero verification
    # queries (string diagnosis fields replaced by a fabricated guess).
    bad_trace = run_oracle_episode(env, bad_scenario)["tool_trace"]
    submit_args = dict(bad_trace[-1]["args"])
    assert str(bad_trace[-1]["tool"]) == "submit_diagnosis"
    wrong_args = {
        key: ("ungrounded_guess_no_evidence" if isinstance(value, str) else value)
        for key, value in submit_args.items()
    }
    bad_payload = json.dumps(
        {"name": "submit_diagnosis", "arguments": wrong_args}, default=str
    )
    bad_responses = [f"<tool_call>{bad_payload}</tool_call>"]

    def make_client(responses: list[str]) -> OpenRouterGatewayClient:
        queue = list(responses)

        def transport(
            payload: dict[str, object], *args: object, **kwargs: object
        ) -> dict[str, object]:
            assert queue, f"unexpected gateway call #{len(responses) - len(queue) + 1}"
            return _gateway_response(queue.pop(0))

        return OpenRouterGatewayClient(
            GatewayConfig(api_key="test-key"), transport=transport
        )

    # Premise check: the ungrounded episode scores below the gate on its own.
    bad_traj = run_teacher_gateway_episode(
        env, bad_scenario, make_client(bad_responses), tokenizer=tokenizer
    )
    assert bad_traj.reward < 0.95

    trajectories = collect_teacher_gateway_trajectories(
        env,
        tokenizer=tokenizer,
        gateway_client=make_client(good_responses + bad_responses),
    )

    assert len(trajectories) == 1
    kept = trajectories[0]
    good_id = str(
        getattr(good_scenario, "scenario_id", None)
        or getattr(good_scenario, "id", None)
        or ""
    )
    assert kept.scenario_id == good_id
    assert kept.reward >= 0.95
    assert kept.state["tool_trace"]  # real executed tool calls, not just text
    assert kept.breakdown.diagnostics  # gated-rubric audit of the episode
    assert len(kept.input_ids) == len(kept.assistant_action_mask)
    assert len(kept.messages) > 2
