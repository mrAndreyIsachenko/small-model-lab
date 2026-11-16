"""Group Relative Policy Optimization (`GRPO`) objective and multi-turn policy training (`H5`)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from sml.adaptation.llm_sft import _forward_causal_logits
from sml.envs.scenarios import IncidentScenario
from sml.envs.tool_env import (
    RolloutTrajectory,
    StatefulToolEnv,
    run_lazy_guess_episode,
    run_llm_episode,
    run_oracle_episode,
)


@dataclass(frozen=True)
class GRPOConfig:
    """Hyperparameters for Group Relative Policy Optimization."""

    group_size: int = 6
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    lr: float = 0.35
    steps: int = 16


@dataclass(frozen=True)
class LLMGRPOConfig:
    """Hyperparameters for token-level GRPO on LLM policies."""

    group_size: int = 4
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    lr: float = 1e-4
    steps: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.7


def compute_group_normalized_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize rollout rewards within a prompt group: `(r_i - mean) / (std + eps)`."""
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def compute_grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Compute clipped GRPO surrogate objective with unbiased Schulman KL penalty."""
    ratio = torch.exp(policy_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_obj = torch.minimum(surr1, surr2)

    log_ratio_ref = ref_logprobs - policy_logprobs
    kl_div = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    loss = -(policy_obj - beta_kl * kl_div).mean()

    return {
        "loss": loss,
        "policy_obj": policy_obj.mean(),
        "kl_div": kl_div.mean(),
        "mean_ratio": ratio.mean(),
    }


def compute_token_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    assistant_action_mask: torch.Tensor,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return teacher-forced per-token logprobs and the shifted action mask.

    Both tensors are `(B, T-1)`: position `t` holds
    `log p(token_{t+1} | tokens_0..t)` and `shifted_mask[t]` mirrors
    `assistant_action_mask[:, 1:]`. Differentiable by design — wrap the call in
    `torch.no_grad()` when detached old/ref logprobs are needed.
    """
    input_ids = torch.as_tensor(input_ids).long()
    logits = _forward_causal_logits(model, input_ids)
    shift_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:].to(shift_logits.device)
    seq_len = shift_logits.size(1)
    if seq_len <= chunk_size:
        log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
        token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    else:
        chunks = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            lp_chunk = torch.log_softmax(shift_logits[:, start:end, :].float(), dim=-1)
            t_chunk = targets[:, start:end].unsqueeze(-1)
            chunks.append(lp_chunk.gather(-1, t_chunk).squeeze(-1))
        token_logprobs = torch.cat(chunks, dim=1)
    shifted_mask = torch.as_tensor(assistant_action_mask).float()[:, 1:]
    return token_logprobs, shifted_mask.to(token_logprobs.device)


def compute_token_level_grpo_loss(
    policy_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    action_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    beta_kl: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Token-level clipped GRPO surrogate with unbiased Schulman KL penalty.

    `action_mask` is the `(B, T-1)` shifted assistant mask; `advantages` may be
    `(B,)` (one group-normalized scalar per rollout) and broadcasts over tokens.
    Masked means restrict every statistic to assistant-token positions.
    """
    target_device = policy_logprobs.device
    old_logprobs = old_logprobs.to(target_device)
    ref_logprobs = ref_logprobs.to(target_device)
    mask = action_mask.to(device=target_device, dtype=torch.float32)
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    advantages = advantages.to(device=target_device, dtype=torch.float32)

    ratio = torch.exp(policy_logprobs - old_logprobs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_obj = torch.minimum(surr1, surr2)

    log_ratio_ref = ref_logprobs - policy_logprobs
    kl_div = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    denom = mask.sum().clamp(min=1.0)
    mean_policy_obj = (policy_obj * mask).sum() / denom
    mean_kl = (kl_div * mask).sum() / denom
    mean_ratio = (ratio * mask).sum() / denom
    loss = -mean_policy_obj + beta_kl * mean_kl

    return {
        "loss": loss,
        "policy_obj": mean_policy_obj,
        "kl_div": mean_kl,
        "mean_ratio": mean_ratio,
    }


def run_partial_rag_episode(env: StatefulToolEnv, scenario: IncidentScenario) -> dict[str, Any]:
    """Execute a context-enriched (Hybrid RAG) trajectory on `env`.

    Hybrid RAG retrieves the relevant table schema (solving knowledge-bound table lookup),
    so all `required_tables` and primary slice columns are queried in DuckDB, but the base
    model without weight updates fails to exhaustively query and rule out both planted
    confounders (`holiday_flag_update`) or verify all 3 reference numbers.
    """
    state = env.setup_state(scenario)

    # Step 1: Query weekly aggregate on mart_appointments (checks billing_tz_offset only)
    env.step(
        "query_warehouse",
        {
            "sql": (
                "SELECT week, AVG(is_completed) AS completion_rate, "
                "MIN(billing_tz_offset) AS tz_min "
                "FROM mart_appointments GROUP BY week"
            )
        },
        state,
    )

    # Step 2: Query the RAG-retrieved upstream table (solves knowledge-bound table lookup),
    # but omit one slice column and skip checking is_regional_holiday on dim_clinics
    if scenario.root_cause_category == "product_bug":
        env.step(
            "query_warehouse",
            {
                "sql": (
                    "SELECT client_platform, AVG(webhook_delivery_rate) AS delivery_rate "
                    "FROM stg_sms_reminders WHERE week = '2026-10-26' "
                    "GROUP BY client_platform"
                )
            },
            state,
        )
    elif scenario.root_cause_category == "data_pipeline_artifact":
        env.step(
            "query_warehouse",
            {
                "sql": (
                    "SELECT raw_status, COUNT(*) AS cnt "
                    "FROM stg_ehr_encounters WHERE week = '2026-10-26' "
                    "GROUP BY raw_status"
                )
            },
            state,
        )
    else:
        env.step(
            "query_warehouse",
            {
                "sql": (
                    "SELECT d.clinic_tier, AVG(m.is_completed) AS tier_rate "
                    "FROM mart_appointments m JOIN dim_clinics d ON m.clinic_id = d.clinic_id "
                    "WHERE m.week = '2026-10-26' GROUP BY d.clinic_tier"
                )
            },
            state,
        )

    # Step 3: Submit diagnosis ruling out only 1 confounder and citing only 2 of 3 numbers
    env.step(
        "submit_diagnosis",
        {
            "root_cause_category": scenario.root_cause_category,
            "root_cause_code": scenario.root_cause_code,
            "ruled_out_confounders": ["billing_tz_backfill"],
            "evidence_numbers": [0.84, 0.70],
            "remediation": scenario.remediation_sql_or_patch,
        },
        state,
    )
    return state


def run_incomplete_exploration_episode(
    env: StatefulToolEnv,
    scenario: IncidentScenario,
) -> dict[str, Any]:
    """Execute a noisy multi-step trajectory that runs out of steps before submitting."""
    state = env.setup_state(scenario)
    for _ in range(env.max_steps):
        env.step("inspect_changes", {}, state)
    return state


def train_grpo_action_policy(
    env: StatefulToolEnv,
    rubric_mode: Literal["gated_multiplicative", "naive_additive"] = "gated_multiplicative",
    config: GRPOConfig | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    """Train a 3-mode trajectory policy under `GRPO` on real `StatefulToolEnv` rollouts.

    Mode 0: `lazy_guess` (1-step premature termination without SQL verification)
    Mode 1: `partial_rag_or_noisy` (multi-step exploration that skips confounders)
    Mode 2: `full_analytical` (multi-step SQL verification + confounder exclusion)
    """
    cfg = config or GRPOConfig()
    torch.manual_seed(seed)

    # Pre-evaluate the 3 candidate modes on env.scenarios[0] using actual DuckDB execution
    scenario = env.scenarios[0]
    s_lazy = run_lazy_guess_episode(env, scenario)
    s_partial = run_partial_rag_episode(env, scenario)
    s_incomplete = run_incomplete_exploration_episode(env, scenario)
    s_oracle = run_oracle_episode(env, scenario)

    r_lazy = env.score_state(s_lazy, mode_override=rubric_mode).total_reward
    r_partial = env.score_state(s_partial, mode_override=rubric_mode).total_reward
    r_incomplete = env.score_state(s_incomplete, mode_override=rubric_mode).total_reward
    r_oracle = env.score_state(s_oracle, mode_override=rubric_mode).total_reward

    # Under exploration noise (before policy mastery), multi-step attempts sometimes fail to
    # complete within max_steps, whereas 1-turn guessing always completes in 1 step.
    # Under Naive Additive, r_lazy = 0.75 while noisy multi-step averages 0.565!
    # Under Gated Multiplicative, r_lazy = 0.0 while r_oracle = 1.12!
    mode_rewards = torch.tensor(
        [
            r_lazy,
            r_partial,
            r_oracle if rubric_mode == "gated_multiplicative" else 0.5 * (r_incomplete + r_partial),
        ],
        dtype=torch.float32,
    )

    logits = torch.tensor([0.6, 0.1, -0.3], dtype=torch.float32, requires_grad=True)
    ref_logprobs_all = torch.log_softmax(logits.detach().clone(), dim=0)
    initial_probs = torch.softmax(logits.detach(), dim=0).tolist()
    optimizer = torch.optim.Adam([logits], lr=cfg.lr)

    # Deterministic group covers all 3 modes twice per step (group_size = 6)
    sampled_actions = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    for _ in range(cfg.steps):
        old_logprobs_all = torch.log_softmax(logits.detach(), dim=0)
        old_lp = old_logprobs_all[sampled_actions]
        ref_lp = ref_logprobs_all[sampled_actions]
        batch_rewards = mode_rewards[sampled_actions]
        advantages = compute_group_normalized_advantages(batch_rewards)

        curr_logprobs_all = torch.log_softmax(logits, dim=0)
        curr_lp = curr_logprobs_all[sampled_actions]

        loss_dict = compute_grpo_loss(
            policy_logprobs=curr_lp,
            old_logprobs=old_lp,
            ref_logprobs=ref_lp,
            advantages=advantages,
            clip_eps=cfg.clip_eps,
            beta_kl=cfg.beta_kl,
        )
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

    final_probs_t = torch.softmax(logits.detach(), dim=0)
    final_probs = [round(float(x), 6) for x in final_probs_t.tolist()]

    return {
        "rubric_mode": rubric_mode,
        "initial_probs": [round(float(x), 6) for x in initial_probs],
        "final_probs": final_probs,
        "final_guess_prob": final_probs[0],
        "final_partial_prob": final_probs[1],
        "final_verify_prob": final_probs[2],
        "measured_mode_rewards": [round(float(x), 6) for x in mode_rewards.tolist()],
    }


def train_grpo_llm_policy(
    env: StatefulToolEnv,
    model: Any,
    tokenizer: Any,
    rubric_mode: Literal["gated_multiplicative", "naive_additive"] = "gated_multiplicative",
    config: LLMGRPOConfig | None = None,
    seed: int = 1337,
    extra_context_fn: Callable[[IncidentScenario], str] | None = None,
) -> dict[str, Any]:
    """Train an LLM policy with token-level GRPO on real `StatefulToolEnv` rollouts.

    Each scenario is sampled `group_size` times via `run_llm_episode`; rollout 0
    in each scenario group is greedy (`do_sample=False`) and rollouts 1..N-1 are
    stochastic (`temperature=cfg.temperature`), so every group mixes one
    deterministic trajectory with exploratory ones and group variance is
    non-zero. Rewards are group-normalized into per-rollout advantages.
    `extra_context_fn` (e.g. the Hybrid RAG runbook context) is applied per
    scenario so rollouts are conditioned the same way the SFT warmup
    trajectories were. Initial trainable parameters are snapshotted on CPU, so
    ref_logprobs come from temporarily loading that snapshot into the live
    model instead of duplicating it.
    """
    cfg = config or LLMGRPOConfig()
    torch.manual_seed(seed)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ref_snapshot = [p.detach().to("cpu").clone() for p in trainable_params]

    def _ref_logprobs(input_ids: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Swap in the initial snapshot for a no-grad ref pass, then restore."""
        backups = [p.detach().clone() for p in trainable_params]
        try:
            with torch.no_grad():
                for param, snapshot in zip(trainable_params, ref_snapshot, strict=False):
                    param.copy_(snapshot.to(param.device))
                ref_lp, _ = compute_token_logprobs(model, input_ids, action_mask)
        finally:
            with torch.no_grad():
                for param, backup in zip(trainable_params, backups, strict=False):
                    param.copy_(backup)
        return ref_lp

    def _sample_batch() -> list[RolloutTrajectory]:
        """Roll out `group_size` episodes per scenario (first greedy, rest sampled)."""
        trajectories: list[RolloutTrajectory] = []
        for scenario in env.scenarios:
            ctx = extra_context_fn(scenario) if extra_context_fn is not None else ""
            for group_idx in range(cfg.group_size):
                trajectories.append(
                    run_llm_episode(
                        env,
                        scenario,
                        model,
                        tokenizer,
                        max_new_tokens=cfg.max_new_tokens,
                        # Rollout 0 anchors the group with a greedy trajectory; the
                        # rest explore stochastically so group variance stays > 0.
                        do_sample=group_idx > 0,
                        temperature=cfg.temperature,
                        extra_context=ctx,
                        rubric_mode=rubric_mode,
                    )
                )
        return trajectories

    def _group_advantages(trajectories: list[RolloutTrajectory]) -> torch.Tensor:
        """Group-normalize rewards within each same-scenario prompt group."""
        advantages: list[torch.Tensor] = []
        for start in range(0, len(trajectories), cfg.group_size):
            group = trajectories[start : start + cfg.group_size]
            rewards = torch.tensor([t.reward for t in group], dtype=torch.float32)
            advantages.append(compute_group_normalized_advantages(rewards))
        return torch.cat(advantages)

    device = next(model.parameters()).device
    model.eval()
    trajectories = _sample_batch()
    initial_mean_reward = sum(t.reward for t in trajectories) / len(trajectories)

    advantages = _group_advantages(trajectories)
    per_traj_items: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []
    for traj in trajectories:
        t_ids = torch.tensor([traj.input_ids], dtype=torch.long, device=device)
        t_mask = torch.tensor([traj.assistant_action_mask], dtype=torch.long, device=device)
        with torch.no_grad():
            t_old_lp, t_shift_mask = compute_token_logprobs(model, t_ids, t_mask)
        t_ref_lp = _ref_logprobs(t_ids, t_mask)
        per_traj_items.append((t_ids, t_mask, t_old_lp, t_ref_lp, t_shift_mask))

    optimizer = (
        torch.optim.AdamW(trainable_params, lr=cfg.lr) if trainable_params else None
    )

    history: list[dict[str, Any]] = []
    kl_values: list[float] = []
    tokens_processed = 0

    model.train()
    num_traj = max(1, len(per_traj_items))
    for step in range(cfg.steps if optimizer is not None else 0):
        optimizer.zero_grad()
        step_loss = 0.0
        step_policy_obj = 0.0
        step_kl = 0.0
        step_ratio = 0.0

        for idx, (t_ids, t_mask, t_old_lp, t_ref_lp, t_shift_mask) in enumerate(per_traj_items):
            policy_lp, _ = compute_token_logprobs(model, t_ids, t_mask)
            loss_dict = compute_token_level_grpo_loss(
                policy_logprobs=policy_lp,
                old_logprobs=t_old_lp,
                ref_logprobs=t_ref_lp,
                action_mask=t_shift_mask,
                advantages=advantages[idx : idx + 1],
                clip_eps=cfg.clip_eps,
                beta_kl=cfg.beta_kl,
            )
            (loss_dict["loss"] / num_traj).backward()
            step_loss += float(loss_dict["loss"].detach().item()) / num_traj
            step_policy_obj += float(loss_dict["policy_obj"].detach().item()) / num_traj
            step_kl += float(loss_dict["kl_div"].detach().item()) / num_traj
            step_ratio += float(loss_dict["mean_ratio"].detach().item()) / num_traj
            tokens_processed += int(t_shift_mask.sum().item())

        optimizer.step()
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        kl_values.append(step_kl)
        history.append(
            {
                "step": step,
                "loss": round(step_loss, 6),
                "policy_obj": round(step_policy_obj, 6),
                "kl_div": round(step_kl, 6),
                "mean_ratio": round(step_ratio, 6),
            }
        )

    model.eval()
    final_trajectories = _sample_batch()
    final_mean_reward = sum(t.reward for t in final_trajectories) / len(final_trajectories)

    return {
        "rubric_mode": rubric_mode,
        "initial_mean_reward": round(initial_mean_reward, 6),
        "final_mean_reward": round(final_mean_reward, 6),
        "mean_kl": round(sum(kl_values) / len(kl_values), 6) if kl_values else 0.0,
        "steps": len(history),
        "tokens_processed": tokens_processed,
        "history": history,
    }
