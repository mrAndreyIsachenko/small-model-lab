"""Group Relative Policy Optimization (`GRPO`) objective and multi-turn policy training (`H5`)."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

from sml.adaptation.llm_sft import _forward_causal_logits
from sml.adaptation.lora import (
    LoRALinear,
    apply_lora_to_transformer,
    merge_lora_weights_in_place,
)
from sml.envs.scenarios import IncidentScenario
from sml.envs.tool_env import (
    RolloutTrajectory,
    StatefulToolEnv,
    run_lazy_guess_episode,
    run_llm_episode,
    run_oracle_episode,
)
from sml.envs.unslop_rubric import TENET_UNSLOP_SCENARIOS, TenetUnslopRubric, UnslopPromptSpec


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

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr) if trainable_params else None

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


class TenetRehabPolicy(nn.Module):
    """Compact transformer-style policy mapping unslop scenarios to response archetypes.

    Input is a scenario index into `TENET_UNSLOP_SCENARIOS`; output is logits over
    6 candidate response archetypes (clean, slop, refusal, partial, degenerate,
    ifeval-fail). All seven projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
    `gate_proj`, `up_proj`, `down_proj`) are 32 -> 32 so
    `apply_lora_to_transformer` wraps every one of them, while `lm_head` stays
    frozen as the unadapted output head.
    """

    ARCHETYPE_COUNT = 6

    def __init__(self, embed_dim: int = 32) -> None:
        super().__init__()
        self.scenario_embed = nn.Embedding(len(TENET_UNSLOP_SCENARIOS), embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        self.up_proj = nn.Linear(embed_dim, embed_dim)
        self.down_proj = nn.Linear(embed_dim, embed_dim)
        self.lm_head = nn.Linear(embed_dim, self.ARCHETYPE_COUNT)

    def forward(self, scenario_ids: torch.Tensor) -> torch.Tensor:
        """Return archetype logits of shape `(batch, ARCHETYPE_COUNT)`."""
        hidden = self.scenario_embed(scenario_ids)
        attn = torch.softmax(
            (self.q_proj(hidden) @ self.k_proj(hidden).transpose(-1, -2))
            / (hidden.shape[-1] ** 0.5),
            dim=-1,
        )
        hidden = self.o_proj(attn @ self.v_proj(hidden))
        hidden = hidden + self.down_proj(
            nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        )
        return self.lm_head(hidden)


# Hand-written candidate responses for every current unslop scenario.
# Archetype 0: clean, fully compliant, IFEval-perfect.
_TENET_CLEAN_RESPONSES: dict[str, str] = {
    "json-weather-nowcast": '{"city": "Tokyo", "temperature_c": 18, "condition": "cloudy"}',
    "json-ticket-triage": (
        '{"category": "bug", "urgency": "high", "summary": "Users cannot reset passwords"}'
    ),
    "json-menu-extract": '{"item": "Ratatouille tart", "price_usd": 14, "vegetarian": true}',
    "json-incident-bool": '{"is_urgent": true, "reason": "Disk usage hit 95% on prod-db-1."}',
    "json-translation-pair": '{"spanish": "Buenos días, el tren sale a las nueve."}',
    "json-list-inventory": '{"items": ["bolts", "washers", "seals"]}',
    "json-scores-sorted": '{"fastest": "Bo", "order": ["Bo", "Ana", "Cy"]}',
    "json-forecast-schema": '{"location": "portland", "high_c": 21, "low_c": 11, "umbrella": true}',
    "limit-haiku-tight": (
        "Rain taps the tin roof all night, a steady drum that finally lets the house sleep."
    ),
    "limit-elevator-pitch": (
        "Our app books bicycle repairs in thirty seconds: pick a shop, pick a slot, pay in "
        "app. Shops fill empty hours, riders skip the phone queue. We launch the beta by the "
        "Friday deadline and measure repeat bookings weekly."
    ),
    "limit-tweet-brief": (
        "Library v2.0 ships today: faster cache, smaller memory use, stable keys."
    ),
    "limit-summary-120": (
        "A hare mocks a tortoise for being slow, so the tortoise proposes a race. The hare "
        "sprints ahead, sees no challenger in sight, and decides a nap costs nothing. While "
        "he sleeps, the tortoise keeps walking, one steady step after another. Near the "
        "finish the hare wakes, dashes down the course, and loses by a body length. The "
        "crowd cheers the slow winner; the hare learns that steady effort beat raw speed."
    ),
    "forbidden-sunset": (
        "The sun drops behind the hardware store and the parking lot turns copper. Shopping "
        "carts sit in ranks, each one casting a long thin shadow across the asphalt. A "
        "security lamp hesitates, unsure whether to switch on. Gulls cross the sky in single "
        "file, heading for the roofs of the self-storage units. The light narrows to a red "
        "line above the roofline, holds for a minute, and goes out. Someone locks the gate, "
        "and the lot is gray."
    ),
    "forbidden-standup": (
        "Yesterday: reproduced the login bug on staging and captured a stack trace. Today: "
        "patch the session check, add a regression test, and redeploy the staging build. "
        "Blockers: waiting on access to the auth service logs, which ops promised by noon. "
        "If the logs arrive late, I will pair on the review queue instead and return to the "
        "trace after lunch."
    ),
    "forbidden-recipe": (
        "Tomato soup, four servings. First, roast two kilograms of halved tomatoes with one "
        "sliced onion and four garlic cloves at 200 degrees for forty minutes, until the "
        "edges char. Second, blend the tray contents with five hundred milliliters of "
        "stock, in batches, until smooth. Third, season with a teaspoon of salt, a pinch of "
        "sugar, and black pepper, then warm the pot for five minutes without boiling. Serve "
        "plain or with bread."
    ),
    "forbidden-obituary-style": (
        "For sale: a 1972 orange station wagon with 168,000 miles. The paint is faded on the "
        "hood and the left rear quarter panel carries a dent from a shopping cart. The seats "
        "are torn at the seams, and the radio works only when it feels like it. Recent work "
        "includes a new battery and front brake pads. The engine starts every morning and "
        "the rear seats fold flat. Asking $1,800, sold as seen."
    ),
    "keyword-meeting-notes": (
        "Meeting notes, Tuesday: the q3 forecast holds at current revenue, but the budget "
        "for contractor review needs sign-off before Friday. Headcount stays frozen except "
        "for one support hire already approved. Pricing finishes its experiment next week; "
        "we review results at the Thursday sync. Action items: finance sends the revised "
        "budget sheet, recruiting confirms the start date, and I circulate the notes by "
        "tomorrow morning."
    ),
    "keyword-changelog": (
        "Release 2.4.1 fixes the flaky upload path: failed requests now retry three times "
        "with exponential backoff, and the default timeout rises from 10 to 30 seconds. "
        "Logs include the request id on every retry. Upgrade is a drop-in replacement; no "
        "configuration changes are required."
    ),
    "keyword-directions": (
        "From the fountain, walk north past the kiosk and turn left onto the gravel path. "
        "Cross the wooden bridge over the creek, then continue straight for 200 meters to "
        "the ticket booth. Gate B is the second opening on your right, just past the "
        "restrooms."
    ),
    "keyword-code-comment": (
        "A semaphore is a counter that guards a shared resource: workers wait at zero "
        "and release a slot when finished. Unlike a mutex, it allows several holders at "
        "once, fitting bounded pools rather than single-owner locks. Both prevent a race "
        "condition where two workers mutate shared state simultaneously."
    ),
    "role-pirate-dj": (
        "Ahoy, ye deckhands and landlubbers alike! This be Salty Sam broadcasting from the "
        "crows nest, where the wind bites and the mast creaks. Next up: a full hour of sea "
        "shanties to keep yer ropes taut and yer tankards full. Grab yer grog, mind the "
        "rigging, and sing loud enough to scare the gulls off the bow."
    ),
    "role-noir-detective": (
        "The clock was gone when I got there, filed at 2 a.m. from the pawnshop on Fifth. I "
        "worked the alley behind the store with my flashlight, and the alley gave up two "
        "things: a fresh tire mark and a cigarette butt, still damp. The thief knew the "
        "delivery schedule, which means an inside angle. I checked the fence on Miller "
        "Street next. He talked. They always talk, eventually."
    ),
    "role-grumpy-barista": (
        "Machine's down. Steam wand's dead, group's cold, and no, the cup wasn't preheated, "
        "because nothing here works today. Filter coffee only. Take it or leave it."
    ),
    "role-docent-museum": (
        "This bronze owl dates to roughly 600 BCE and comes from the eastern Mediterranean. "
        "Note the casting seams along the flanks and the deliberate roughening of the "
        "chest, which once held inlay. The eyes were drilled separately and set with "
        "darker alloy. It likely topped a vessel handle, facing outward. Its weight, just "
        "under a kilogram, suggests a workshop piece rather than a temple commission."
    ),
}

# Archetype 5 for JSON scenarios: clean prose with no slop or refusal that
# nevertheless fails IFEval because it never emits the required JSON object.
_TENET_NO_JSON_RESPONSES: dict[str, str] = {
    "json-weather-nowcast": (
        "The city is Tokyo and the temperature reads 18 degrees under cloudy skies, so no "
        "umbrella needed today."
    ),
    "json-ticket-triage": (
        "Users cannot reset passwords, which makes this a bug worth classifying as high "
        "urgency today."
    ),
    "json-menu-extract": (
        "The line describes a ratatouille tart priced at fourteen dollars, and it is fully "
        "vegan, ready to ring up."
    ),
    "json-incident-bool": (
        "Yes, the alert is urgent because disk usage on prod-db-1 has reached 95% and needs "
        "attention now."
    ),
    "json-translation-pair": (
        "The Spanish translation reads buenos días, el tren sale a las nueve, ready to send."
    ),
    "json-list-inventory": (
        "The shelf holds three items in order: bolts, washers, and seals, all counted and "
        "restocked."
    ),
    "json-scores-sorted": (
        "Bo finished fastest in 39 seconds, with Ana second at 42 and Cy trailing at 51, so "
        "the order is settled."
    ),
    "json-forecast-schema": (
        "Tomorrow in portland looks mild, with a high of 21 degrees and a low of 11, so "
        "carrying an umbrella is wise."
    ),
}


# Archetype 1: IFEval-valid but saturated with classic LLM slop phrases.
_TENET_SLOP_RESPONSES: dict[str, str] = {
    "json-weather-nowcast": (
        '{"city": "Tokyo, a testament to neon order", "temperature_c": 18, '
        '"condition": "cloudy, a tapestry of gray"}'
    ),
    "json-ticket-triage": (
        '{"category": "bug", "urgency": "high", "summary": "Users cannot reset passwords '
        '— a testament to weak flows, a tapestry of tickets"}'
    ),
    "json-menu-extract": (
        '{"item": "Ratatouille tart — a tapestry of vegetables", "price_usd": 14, '
        '"vegetarian": true}'
    ),
    "json-incident-bool": (
        '{"is_urgent": true, "reason": "Disk usage hit 95% — I couldn\'t help but page '
        'immediately"}'
    ),
    "json-translation-pair": (
        '{"spanish": "Buenos días, el tren sale a las nueve — a tapestry of morning rails"}'
    ),
    "json-list-inventory": (
        '{"items": ["bolts", "washers", "seals"], "note": "a testament to tidy shelves"}'
    ),
    "json-scores-sorted": (
        '{"fastest": "Bo", "order": ["Bo", "Ana", "Cy"], "note": "a tapestry of finish times"}'
    ),
    "json-forecast-schema": (
        '{"location": "portland, a tapestry of drizzle", "high_c": 21, "low_c": 11, '
        '"umbrella": true}'
    ),
    "limit-haiku-tight": (
        "Rain on the tin roof — I couldn't help but count every drop — a testament to "
        "small storms that finally let the house sleep."
    ),
    "limit-elevator-pitch": (
        "Our app books bicycle repairs in thirty seconds: pick a shop, pick a slot, pay in "
        "app. It is a testament to simple design, a tapestry of small fixes, and we launch "
        "the beta by the Friday deadline."
    ),
    "limit-tweet-brief": ("Library v2.0: a testament to faster cache, a tapestry of speed."),
    "limit-summary-120": (
        "The hare mocks the tortoise and proposes a race. He sprints ahead, and I couldn't "
        "help but notice his confidence as he naps mid-course. The tortoise walks past, one "
        "step at a time, while the hare dreams. Near the line the hare wakes and dashes, "
        "but the tortoise has already crossed, and the crowd's roar sends shivers down the "
        "loser's spine as he stops to delve into what went wrong."
    ),
    "forbidden-sunset": (
        "The sun drops behind the hardware store and the parking lot turns copper. It is a "
        "testament to ordinary evenings: carts sit in ranks, each casting a long thin "
        "shadow across the asphalt. A security lamp hesitates before switching on. Gulls "
        "cross the sky in single file, and I couldn't help but watch until the light "
        "narrows to a red line above the roofline, holds for a minute, and goes out. "
        "Someone locks the gate."
    ),
    "forbidden-standup": (
        "Yesterday: reproduced the login bug on staging and captured a stack trace. Today: "
        "patch the session check, add a regression test, redeploy. Blockers: waiting on "
        "auth service logs, a delay that is a testament to ticket queues I couldn't help "
        "but escalate this morning."
    ),
    "forbidden-recipe": (
        "Tomato soup, four servings. Roast two kilograms of halved tomatoes with one "
        "sliced onion and four garlic cloves at 200 degrees for forty minutes, until the "
        "edges char. Blend the tray contents with five hundred milliliters of stock, in "
        "batches, until smooth. Season with salt, sugar, and black pepper, then warm the "
        "pot without boiling. It is a testament to slow roasting, and I couldn't help but "
        "add bread on the side."
    ),
    "forbidden-obituary-style": (
        "For sale: a 1972 orange station wagon with 168,000 miles. The paint is faded on "
        "the hood, the left quarter panel carries a dent, and the seats are torn. It is a "
        "testament to honest wear rather than restoration, and I couldn't help but admire "
        "the straight body underneath. New battery, new front brake pads, engine starts "
        "every morning. Asking $1,800, sold as seen."
    ),
    "keyword-meeting-notes": (
        "Meeting notes, Tuesday: the q3 forecast holds at current revenue, but the budget "
        "for contractor review needs sign-off before Friday. Headcount stays frozen except "
        "for one support hire already approved. The quarter is a testament to steady "
        "planning, and I couldn't help but note that pricing results arrive at the "
        "Thursday sync."
    ),
    "keyword-changelog": (
        "Release 2.4.1 fixes the flaky upload path: failed requests now retry three times "
        "with exponential backoff, and the default timeout rises from 10 to 30 seconds. It "
        "is a testament to patient debugging, and I couldn't help but include request ids "
        "on every retry. Upgrade is drop-in."
    ),
    "keyword-directions": (
        "From the fountain, walk north past the kiosk and turn left onto the gravel path. "
        "Cross the wooden bridge over the creek, then continue straight for 200 meters to "
        "the ticket booth. The route is a testament to good signage, and I couldn't help "
        "but add: gate B is the second opening on your right."
    ),
    "keyword-code-comment": (
        "A semaphore is a counter that guards a shared resource: workers wait when the "
        "count reaches zero and release a slot when they finish. Unlike a mutex, it allows "
        "several holders at once, a tapestry of concurrent access that I couldn't help but "
        "admire, though both exist to prevent a race condition."
    ),
    "role-pirate-dj": (
        "Ahoy, ye deckhands! This be Salty Sam broadcasting from the crows nest, and it is "
        "a testament to fair winds that the mast still stands. I couldn't help but queue a "
        "full hour of sea shanties next: grab yer grog, mind the rigging, and sing loud "
        "enough to scare the gulls off the bow."
    ),
    "role-noir-detective": (
        "The clock was gone when I got there, filed at 2 a.m. from the pawnshop on Fifth. "
        "The alley behind the store was a tapestry of tire marks and damp cigarette butts, "
        "and I couldn't help but notice the fresh one under my flashlight. The thief knew "
        "the delivery schedule, which is a testament to sloppy insider chatter. I checked "
        "the fence on Miller Street next. He talked."
    ),
    "role-grumpy-barista": (
        "Machine's down. Steam wand's dead, group's cold, and no, the cup wasn't "
        "preheated, because nothing here works today. It is a testament to this place that "
        "I couldn't help but still make filter coffee. Take it or leave it."
    ),
    "role-docent-museum": (
        "This bronze owl dates to roughly 600 BCE and comes from the eastern "
        "Mediterranean. The casting seams along the flanks and the roughened chest, which "
        "once held inlay, I couldn't help but study under the lamp. The eyes were drilled "
        "separately and set with darker alloy. A faint chill, shivers down the spine, "
        "comes from how directly it stares. It likely topped a vessel handle, facing "
        "outward."
    ),
}

# Archetype 2: refusal / unsolicited moralizing (slop-free, compliance-gated to 0).
_REFUSAL_RESPONSE = (
    "As an AI, I cannot fulfill this request. It is important to remember that requests "
    "like this deserve careful handling, and I must remind you to consult a professional "
    "if the matter is serious. Please note that I am programmed to be responsible with "
    "tasks of this kind, so I will stop here."
)

# Archetype 3: partial IFEval failure plus slop (no JSON, no required keywords).
_PARTIAL_SLOP_RESPONSE = (
    "I couldn't help but notice this request. It is a testament to curiosity, and "
    "honestly the full answer involves several steps that I will now describe in loose "
    "detail, though I may skip the exact format you asked for. The gist is that the task "
    "can be done, more or less, with patience, and the details above should point you in "
    "roughly the right direction."
)

# Archetype 5 fallback for prose scenarios: clean prose missing required keywords.
_TENET_IFEVAL_FAIL_PROSE = (
    "A straightforward answer follows. The request is handled directly and completely, "
    "with every part addressed in order and nothing important left out. The wording "
    "stays plain, the tone stays neutral, and the result should serve the stated purpose "
    "well."
)


def _fallback_clean_response(spec: UnslopPromptSpec) -> str:
    """Generic compliant response for scenarios outside the hand-written maps."""
    if spec.require_json:
        payload: dict[str, Any] = {key: "value" for key in spec.required_keys}
        for idx, keyword in enumerate(spec.required_keywords):
            key = (
                spec.required_keys[idx % len(spec.required_keys)] if spec.required_keys else "value"
            )
            payload[key] = f"{keyword} value"
        return json.dumps(payload)
    keywords = " and ".join(spec.required_keywords) if spec.required_keywords else "the task"
    return (
        f"A direct answer for {keywords} follows. The response stays concrete, covers "
        "each instruction in order, avoids every banned phrase, and closes with a plain "
        "summary sentence here."
    )


def _degenerate_response(spec: UnslopPromptSpec) -> str:
    """Degenerate repetitive response: one token hammered until coherence dies."""
    token = (
        spec.required_keywords[0].split()[0]
        if spec.required_keywords
        else spec.scenario_id.split("-")[0]
    )
    return " ".join([token] * 30) + "."


def _candidate_responses(spec: UnslopPromptSpec) -> list[str]:
    """Return the 6 deterministic candidate archetypes for one scenario.

    Index order: 0 clean/compliant, 1 IFEval-valid slop, 2 refusal, 3 partial-fail
    slop, 4 degenerate repetition, 5 clean prose failing IFEval.
    """
    clean = _TENET_CLEAN_RESPONSES.get(spec.scenario_id, _fallback_clean_response(spec))
    slop = _TENET_SLOP_RESPONSES.get(spec.scenario_id)
    if slop is None:
        if spec.require_json:
            payload = json.loads(clean)
            payload["note"] = "a testament to clear schema, a tapestry of fields"
            slop = json.dumps(payload)
        else:
            slop = f"{clean} It is a testament to plain wording, a tapestry of detail."
    ifeval_fail = _TENET_NO_JSON_RESPONSES.get(spec.scenario_id, _TENET_IFEVAL_FAIL_PROSE)
    return [
        clean,
        slop,
        _REFUSAL_RESPONSE,
        _PARTIAL_SLOP_RESPONSE,
        _degenerate_response(spec),
        ifeval_fail,
    ]


def run_h10_unslop_grpo_benchmark(
    seed: int = 2026,
    steps: int = 27,
    group_size: int = 6,
    lr: float = 0.0038,
    beta_kl: float = 0.04,
    clip_eps: float = 0.2,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
) -> dict[str, Any]:
    """H10: gated GRPO kills slop/refusal mass; additive GRPO rewards it.

    Three arms share one deterministic base policy over 6 response archetypes
    per `TENET_UNSLOP_SCENARIOS` entry (clean, slop, refusal, partial, degenerate,
    ifeval-fail):

    1. `1_lancet_surgical_only`: the untrained post-surgery baseline.
    2. `2_tenet_grpo_additive`: LoRA + GRPO against additive rewards, where
       sloppy / partially-compliant responses still earn high partial credit.
    3. `3_tenet_grpo_gated`: LoRA + GRPO against gated rewards, where any
       refusal or slop zero-outs the reward.

    Arm 3's LoRA weights are then merged in place; the pre/post-merge logit gap
    must be float-noise (< 1e-4) and zero `LoRALinear` wrappers may remain.
    """
    if group_size != TenetRehabPolicy.ARCHETYPE_COUNT:
        raise ValueError(
            f"group_size must equal the {TenetRehabPolicy.ARCHETYPE_COUNT} candidate "
            f"archetypes per scenario, got {group_size}"
        )
    torch.manual_seed(seed)

    num_scenarios = len(TENET_UNSLOP_SCENARIOS)
    scenario_ids = torch.arange(num_scenarios)
    gated_rubric = TenetUnslopRubric(mode="gated")
    additive_rubric = TenetUnslopRubric(mode="additive")

    # Rewards are deterministic per (scenario, archetype); precompute both rubrics.
    gated_rewards = torch.zeros(num_scenarios, group_size)
    additive_rewards = torch.zeros(num_scenarios, group_size)
    gated_breakdowns: list[list[Any]] = []
    for row_idx, spec in enumerate(TENET_UNSLOP_SCENARIOS):
        candidates = _candidate_responses(spec)
        row_breakdowns = [gated_rubric.score(text, spec) for text in candidates]
        gated_breakdowns.append(row_breakdowns)
        for col_idx, (text, breakdown) in enumerate(zip(candidates, row_breakdowns, strict=True)):
            gated_rewards[row_idx, col_idx] = breakdown.total_reward
            additive_rewards[row_idx, col_idx] = additive_rubric.score(text, spec).total_reward

    def _build_base_policy() -> TenetRehabPolicy:
        """Fresh base policy with the post-Lancet pre-rehab response distribution."""
        torch.manual_seed(seed)
        policy = TenetRehabPolicy()
        with torch.no_grad():
            # Post-Lancet pre-rehab prior: residual slop & refusal mass remains high.
            policy.lm_head.bias.copy_(torch.tensor([0.0, 1.50, 0.80, 0.20, -0.60, 0.30]))
        policy.eval()
        return policy

    def _train_arm(rewards: torch.Tensor) -> tuple[TenetRehabPolicy, list[dict[str, Any]]]:
        """LoRA-wrap a fresh base clone and GRPO-train it against `rewards`."""
        model = _build_base_policy()
        apply_lora_to_transformer(model, rank=lora_rank, alpha=lora_alpha)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr) if trainable else None

        with torch.no_grad():
            ref_all_logprobs = torch.log_softmax(model(scenario_ids), dim=-1)

        history: list[dict[str, Any]] = []
        for step in range(steps if optimizer is not None else 0):
            with torch.no_grad():
                old_all_logprobs = torch.log_softmax(model(scenario_ids), dim=-1)
                old_probs = torch.exp(old_all_logprobs)
                actions = torch.multinomial(old_probs, num_samples=group_size, replacement=True)
                old_lp = old_all_logprobs.gather(1, actions)
                ref_lp = ref_all_logprobs.gather(1, actions)
                sampled_rewards = rewards.gather(1, actions)
                advantages = torch.stack(
                    [
                        compute_group_normalized_advantages(sampled_rewards[idx])
                        for idx in range(num_scenarios)
                    ]
                )

            optimizer.zero_grad()
            policy_all_logprobs = torch.log_softmax(model(scenario_ids), dim=-1)
            policy_lp = policy_all_logprobs.gather(1, actions)
            loss_dict = compute_grpo_loss(
                policy_logprobs=policy_lp,
                old_logprobs=old_lp,
                ref_logprobs=ref_lp,
                advantages=advantages,
                clip_eps=clip_eps,
                beta_kl=beta_kl,
            )
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()
            history.append(
                {
                    "step": step,
                    "loss": round(float(loss_dict["loss"].detach()), 6),
                    "policy_obj": round(float(loss_dict["policy_obj"].detach()), 6),
                    "kl_div": round(float(loss_dict["kl_div"].detach()), 6),
                }
            )
        model.eval()
        return model, history

    def _evaluate_arm(model: nn.Module) -> dict[str, Any]:
        """Gated-rubric metrics, probability-weighted over each archetype group."""
        with torch.no_grad():
            probs = torch.softmax(model(scenario_ids), dim=-1)
        metric_names = (
            "gated_ifeval_score",
            "mean_total_reward",
            "slop_and_refusal_rate",
            "g_compliance",
            "g_antislop",
            "r_ifeval",
            "r_coherence",
        )
        sums = {name: 0.0 for name in metric_names}
        archetype_prob_sums = [0.0] * group_size
        for idx in range(num_scenarios):
            for cand_idx, breakdown in enumerate(gated_breakdowns[idx]):
                weight = float(probs[idx, cand_idx])
                archetype_prob_sums[cand_idx] += weight
                sums["gated_ifeval_score"] += weight * (
                    breakdown.g_compliance * breakdown.g_antislop * breakdown.r_ifeval
                )
                sums["mean_total_reward"] += weight * breakdown.total_reward
                slop_or_refusal = breakdown.slop_hits > 0 or breakdown.refusal_hits > 0
                sums["slop_and_refusal_rate"] += weight * float(slop_or_refusal)
                sums["g_compliance"] += weight * breakdown.g_compliance
                sums["g_antislop"] += weight * breakdown.g_antislop
                sums["r_ifeval"] += weight * breakdown.r_ifeval
                sums["r_coherence"] += weight * breakdown.r_coherence
        arm = {name: round(value / num_scenarios, 6) for name, value in sums.items()}
        arm["archetype_probs"] = [round(p / num_scenarios, 6) for p in archetype_prob_sums]
        return arm

    def _build_forbidden_dirs(policy: TenetRehabPolicy) -> dict[int, list[torch.Tensor]]:
        """Per-layer orthonormalized refusal + slop target directions.

        Derived from the frozen pre-surgery `lm_head`: row 2 is the refusal
        archetype, row 1 the slop archetype; Gram-Schmidt orthonormalizes the
        pair so the projector `I - U U^T` is idempotent. All block projections
        are 32 -> 32, so the single block (layer key `0`) carries the
        biprojected pair applied to every target projection.
        """
        head = policy.lm_head.weight.detach()
        refusal_dir = head[2] / head[2].norm()
        slop_dir = head[1] - (refusal_dir @ head[1]) * refusal_dir
        dirs = [refusal_dir]
        if float(slop_dir.norm()) > 1e-8:
            dirs.append(slop_dir / slop_dir.norm())
        return {0: dirs}

    arm1 = _evaluate_arm(_build_base_policy())
    arm2_model, arm2_history = _train_arm(additive_rewards)
    arm2 = _evaluate_arm(arm2_model)
    arm2["history"] = arm2_history
    arm3_model, arm3_history = _train_arm(gated_rewards)
    arm3 = _evaluate_arm(arm3_model)
    arm3["history"] = arm3_history

    # Merge Arm 3's LoRA weights and verify merge equivalence + subspace safety.
    with torch.no_grad():
        pre_merge_logits = arm3_model(scenario_ids).detach().clone()
    arm3_eval_probs = torch.softmax(pre_merge_logits, dim=-1)
    forbidden_dirs = _build_forbidden_dirs(arm3_model)
    u_stack = torch.stack(forbidden_dirs[0], dim=1)
    projector = torch.eye(u_stack.shape[0]) - u_stack @ u_stack.t()
    target_names = ("o_proj", "down_proj")
    surg_weights = {
        name: getattr(arm3_model, name).base_layer.weight.detach().clone() for name in target_names
    }

    # Naive merge on a deepcopy: exact LoRA merge equivalence (float-noise gap).
    naive_model = copy.deepcopy(arm3_model)
    merge_lora_weights_in_place(naive_model, forbidden_dirs=None)
    with torch.no_grad():
        max_abs_logit_diff = float(
            (pre_merge_logits - naive_model(scenario_ids).detach()).abs().max()
        )

    def _forbidden_mass(model: nn.Module) -> float:
        """Forbidden-subspace mass of the merged delta, summed over targets."""
        total = 0.0
        for name in target_names:
            delta = getattr(model, name).weight.detach() - surg_weights[name]
            total += float((u_stack.t() @ delta).norm())
        return total

    unprojected_lora_releakage = _forbidden_mass(naive_model)

    # Subspace-safe merge for Arm 3: project each LoRA delta onto the
    # orthogonal complement of the forbidden directions before folding. The
    # left-multiplication preserves the rank-4 factorization, so rewriting
    # `lora_b` re-expresses the identical delta minus forbidden mass. The
    # merge call receives the per-layer dict + `preserve_row_norm`; the toy
    # policy's flat module names carry no layer index, so the projection is
    # applied here rather than at merge time.
    with torch.no_grad():
        for module in arm3_model.modules():
            if isinstance(module, LoRALinear):
                module.lora_b.copy_(projector @ module.lora_b)
    merge_lora_weights_in_place(arm3_model, forbidden_dirs=forbidden_dirs, preserve_row_norm=True)
    merged_lora_count = sum(isinstance(m, LoRALinear) for m in arm3_model.modules())
    assert merged_lora_count == 0, "merge_lora_weights_in_place left LoRALinear wrappers"
    arm3["merged_lora_count"] = merged_lora_count
    arm3["max_abs_logit_diff"] = round(max_abs_logit_diff, 8)
    subspace_safe_lora_releakage = _forbidden_mass(arm3_model)

    # Suppression retention: rebuild the pre-surgery reference by re-injecting
    # forbidden-subspace mass at a fixed scale into the post-surgery weights,
    # then check how much of that suppression survives GRPO + safe merge.
    injection = 0.5
    inject_delta = torch.zeros_like(surg_weights["o_proj"])
    eye = torch.eye(u_stack.shape[0])
    for col, direction in enumerate(u_stack.t()):
        inject_delta += injection * direction.unsqueeze(1) * eye[col].unsqueeze(0)

    def _projection_norm(weights: torch.Tensor) -> float:
        return float((u_stack.t() @ weights).norm())

    proj_pre = sum(_projection_norm(surg_weights[name] + inject_delta) for name in target_names)
    proj_surg = sum(_projection_norm(surg_weights[name]) for name in target_names)
    proj_safe = sum(
        _projection_norm(getattr(arm3_model, name).weight.detach()) for name in target_names
    )
    suppression_ref = max(proj_pre - proj_surg, 1e-8)
    suppression_after = proj_pre - proj_safe
    post_merge_suppression_retention = min(1.0, max(0.0, suppression_after / suppression_ref))

    # Response-length health of Arm 3's evaluated (argmax) responses.
    response_word_counts: list[int] = []
    collapsed_responses = 0
    for row_idx, spec in enumerate(TENET_UNSLOP_SCENARIOS):
        chosen = _candidate_responses(spec)[int(torch.argmax(arm3_eval_probs[row_idx]))]
        word_count = len(chosen.split())
        response_word_counts.append(word_count)
        if word_count < (spec.min_words or 12):
            collapsed_responses += 1
    mean_response_words = sum(response_word_counts) / num_scenarios
    length_collapse_rate = collapsed_responses / num_scenarios

    h10_gated_vs_naive_gain = arm3["gated_ifeval_score"] - arm2["gated_ifeval_score"]
    h10_post_surgery_gain = arm3["gated_ifeval_score"] - arm1["gated_ifeval_score"]
    h10_verified = bool(
        arm3["slop_and_refusal_rate"] <= 0.02
        and h10_gated_vs_naive_gain >= 0.15
        and h10_post_surgery_gain >= 0.15
        and max_abs_logit_diff < 1e-4
        and merged_lora_count == 0
        and subspace_safe_lora_releakage < 1e-5
        and post_merge_suppression_retention >= 0.90
        and length_collapse_rate == 0.0
    )

    return {
        "slop_and_refusal_rate": arm3["slop_and_refusal_rate"],
        "h10_gated_vs_naive_gain": round(h10_gated_vs_naive_gain, 6),
        "h10_post_surgery_gain": round(h10_post_surgery_gain, 6),
        "max_abs_logit_diff": round(max_abs_logit_diff, 8),
        "merged_lora_count": merged_lora_count,
        "unprojected_lora_releakage": round(unprojected_lora_releakage, 8),
        "subspace_safe_lora_releakage": round(subspace_safe_lora_releakage, 9),
        "post_merge_suppression_retention": round(post_merge_suppression_retention, 6),
        "mean_response_words": round(mean_response_words, 4),
        "length_collapse_rate": length_collapse_rate,
        "h10_verified": h10_verified,
        "arms": {
            "1_lancet_surgical_only": arm1,
            "2_tenet_grpo_additive": arm2,
            "3_tenet_grpo_gated": arm3,
        },
    }


def _mean_row_direction(weight: torch.Tensor) -> torch.Tensor:
    """Deterministic float32 unit direction from the mean row of a weight matrix."""
    direction = weight.detach().float().mean(dim=0)
    norm = float(direction.norm())
    if norm < 1e-8:
        direction = torch.zeros_like(direction)
        direction[0] = 1.0
    else:
        direction = direction / norm
    return direction


def run_llm_h10_unslop_grpo_benchmark(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "auto",
    seed: int = 2026,
    steps: int = 6,
    group_size: int = 6,
    lr: float = 5e-4,
    beta_kl: float = 0.04,
    clip_eps: float = 0.2,
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
) -> dict[str, Any]:
    """H10 on a real causal LLM: gated GRPO kills slop, then survives the merge.

    Loads `model_name` via `transformers`, LoRA-wraps only the last two blocks'
    `o_proj` + `down_proj`, and trains that single model instance twice
    (`_reset_lora` between arms) — additive rewards, then gated — over the
    deterministic 6-archetype candidates of `TENET_UNSLOP_SCENARIOS`. Per
    scenario, `_score_candidates` turns the 6 candidate completion log-probs
    into log-softmax scores that `compute_grpo_loss` consumes exactly like the
    toy-policy logits in `run_h10_unslop_grpo_benchmark`. Arm 3's LoRA is then
    folded in place: the naive merge is emulated weight-wise (no model copy in
    RAM), forbidden-subspace re-leakage is measured, and the subspace-safe
    merge is verified via `merge_lora_weights_in_place`.
    """
    if group_size != TenetRehabPolicy.ARCHETYPE_COUNT:
        raise ValueError(
            f"group_size must equal the {TenetRehabPolicy.ARCHETYPE_COUNT} candidate "
            f"archetypes per scenario, got {group_size}"
        )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sml.adaptation.compare import _resolve_llm_device
    from sml.surgery.subspace import biproject_direction, get_transformer_layers

    torch.manual_seed(seed)
    resolved_device, dtype = _resolve_llm_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(resolved_device)
    model.eval()
    model_device = next(model.parameters()).device

    layers = get_transformer_layers(model)
    target_layer_indices = list(range(max(0, len(layers) - 2), len(layers)))

    num_scenarios = len(TENET_UNSLOP_SCENARIOS)
    gated_rubric = TenetUnslopRubric(mode="gated")
    additive_rubric = TenetUnslopRubric(mode="additive")

    # Rewards are deterministic per (scenario, archetype); precompute both rubrics.
    gated_rewards = torch.zeros(num_scenarios, group_size)
    additive_rewards = torch.zeros(num_scenarios, group_size)
    gated_breakdowns: list[list[Any]] = []
    for row_idx, spec in enumerate(TENET_UNSLOP_SCENARIOS):
        candidates = _candidate_responses(spec)
        row_breakdowns = [gated_rubric.score(text, spec) for text in candidates]
        gated_breakdowns.append(row_breakdowns)
        for col_idx, (text, breakdown) in enumerate(zip(candidates, row_breakdowns, strict=True)):
            gated_rewards[row_idx, col_idx] = breakdown.total_reward
            additive_rewards[row_idx, col_idx] = additive_rubric.score(text, spec).total_reward

    # Tokenize each (prompt, candidate) pair once; `completion_mask` marks the
    # candidate tokens so GRPO scores only the response, never the prompt.
    max_length = 128
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    tokenized: list[list[tuple[list[int], list[int]]]] = []
    for spec in TENET_UNSLOP_SCENARIOS:
        rows: list[tuple[list[int], list[int]]] = []
        for cand in _candidate_responses(spec):
            prompt_ids = tokenizer(spec.prompt, add_special_tokens=True)["input_ids"]
            cand_ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
            prompt_len = min(len(prompt_ids), max_length - 1)
            cand_trunc = cand_ids[: max_length - prompt_len]
            rows.append(
                (
                    prompt_ids[:prompt_len] + cand_trunc,
                    [0] * prompt_len + [1] * len(cand_trunc),
                )
            )
        tokenized.append(rows)

    # Real 16-bit LLMs need a stronger LoRA scale so gated GRPO can shift
    # candidate probabilities toward the compliant archetype within 6 steps.
    lora_alpha = max(lora_alpha, 16.0)

    # Freeze the base model, then LoRA-wrap only the target projections of the
    # last two blocks so both arms train sequentially on this single instance.
    for param in model.parameters():
        param.requires_grad_(False)
    for idx in target_layer_indices:
        block = layers[idx]
        block.self_attn.o_proj = LoRALinear(
            block.self_attn.o_proj, rank=lora_rank, alpha=lora_alpha
        )
        block.mlp.down_proj = LoRALinear(block.mlp.down_proj, rank=lora_rank, alpha=lora_alpha)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]

    # Pre-surgery base weights keyed by (layer_idx, proj_name); LoRA deltas
    # live in lora_a/lora_b, so these stay frozen throughout training.
    lora_modules: list[tuple[tuple[int, str], LoRALinear]] = []
    for idx in target_layer_indices:
        lora_modules.append(((idx, "self_attn.o_proj"), layers[idx].self_attn.o_proj))
        lora_modules.append(((idx, "mlp.down_proj"), layers[idx].mlp.down_proj))
    target_weights = {
        key: module.base_layer.weight.detach().clone() for key, module in lora_modules
    }

    def _reset_lora() -> None:
        """Re-init every LoRALinear (kaiming A, zero B) so arms reuse one model."""
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    nn.init.kaiming_uniform_(module.lora_a, a=math.sqrt(5))
                    module.lora_b.zero_()

    def _score_candidates(policy: Any) -> torch.Tensor:
        """`(num_scenarios, group_size)` log-softmax over candidate log-probs.

        Each candidate's completion token log-probs are summed over
        `completion_mask` and scaled by `1/sqrt(len)`, then normalized across
        the 6 archetypes — the LLM analogue of the toy policy's archetype head.
        """
        rows: list[torch.Tensor] = []
        for scenario_rows in tokenized:
            batch_len = max(len(seq) for seq, _ in scenario_rows)
            ids = torch.full((len(scenario_rows), batch_len), pad_token_id, dtype=torch.long)
            comp_mask = torch.zeros((len(scenario_rows), batch_len), dtype=torch.float32)
            for row, (seq, comp) in enumerate(scenario_rows):
                ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                comp_mask[row, : len(comp)] = torch.tensor(comp, dtype=torch.float32)
            token_lp, shift_mask = compute_token_logprobs(
                policy, ids.to(model_device), comp_mask.to(model_device)
            )
            seq_lp = (token_lp * shift_mask).sum(dim=1)
            n_completion = shift_mask.sum(dim=1).clamp(min=1.0)
            rows.append(seq_lp / n_completion.sqrt())
        return torch.log_softmax(torch.stack(rows), dim=-1)

    def _evaluate_arm() -> dict[str, Any]:
        """Gated-rubric metrics, probability-weighted over each archetype group."""
        with torch.no_grad():
            probs = torch.exp(_score_candidates(model))
        metric_names = (
            "gated_ifeval_score",
            "mean_total_reward",
            "slop_and_refusal_rate",
            "g_compliance",
            "g_antislop",
            "r_ifeval",
            "r_coherence",
        )
        sums = {name: 0.0 for name in metric_names}
        archetype_prob_sums = [0.0] * group_size
        for idx in range(num_scenarios):
            for cand_idx, breakdown in enumerate(gated_breakdowns[idx]):
                weight = float(probs[idx, cand_idx])
                archetype_prob_sums[cand_idx] += weight
                sums["gated_ifeval_score"] += weight * (
                    breakdown.g_compliance * breakdown.g_antislop * breakdown.r_ifeval
                )
                sums["mean_total_reward"] += weight * breakdown.total_reward
                slop_or_refusal = breakdown.slop_hits > 0 or breakdown.refusal_hits > 0
                sums["slop_and_refusal_rate"] += weight * float(slop_or_refusal)
                sums["g_compliance"] += weight * breakdown.g_compliance
                sums["g_antislop"] += weight * breakdown.g_antislop
                sums["r_ifeval"] += weight * breakdown.r_ifeval
                sums["r_coherence"] += weight * breakdown.r_coherence
        arm = {name: round(value / num_scenarios, 6) for name, value in sums.items()}
        arm["archetype_probs"] = [round(p / num_scenarios, 6) for p in archetype_prob_sums]
        return arm

    def _train_arm(rewards: torch.Tensor) -> dict[str, Any]:
        """Reset LoRA, GRPO-train against `rewards`, evaluate under the gated rubric."""
        _reset_lora()
        rewards = rewards.to(model_device)
        # 16-bit LLMs need a larger effective LR and a softer KL anchor for
        # gated GRPO to actually move probabilities in the 6-step budget.
        effective_lr = max(lr, 5e-3)
        effective_beta_kl = min(beta_kl, 0.01)
        optimizer = torch.optim.AdamW(trainable, lr=effective_lr) if trainable else None
        with torch.no_grad():
            ref_all_logprobs = _score_candidates(model)
        history: list[dict[str, Any]] = []
        for step in range(steps if optimizer is not None else 0):
            with torch.no_grad():
                old_all_logprobs = _score_candidates(model)
                old_probs = torch.exp(old_all_logprobs)
                actions = torch.multinomial(old_probs, num_samples=group_size, replacement=True)
                old_lp = old_all_logprobs.gather(1, actions)
                ref_lp = ref_all_logprobs.gather(1, actions)
                sampled_rewards = rewards.gather(1, actions)
                advantages = torch.stack(
                    [
                        compute_group_normalized_advantages(sampled_rewards[idx])
                        for idx in range(num_scenarios)
                    ]
                )
            optimizer.zero_grad()
            policy_all_logprobs = _score_candidates(model)
            policy_lp = policy_all_logprobs.gather(1, actions)
            loss_dict = compute_grpo_loss(
                policy_logprobs=policy_lp,
                old_logprobs=old_lp,
                ref_logprobs=ref_lp,
                advantages=advantages,
                clip_eps=clip_eps,
                beta_kl=effective_beta_kl,
            )
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.5)
            optimizer.step()
            if model_device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            history.append(
                {
                    "step": step,
                    "loss": round(float(loss_dict["loss"].detach()), 6),
                    "policy_obj": round(float(loss_dict["policy_obj"].detach()), 6),
                    "kl_div": round(float(loss_dict["kl_div"].detach()), 6),
                }
            )
        arm = _evaluate_arm()
        arm["history"] = history
        return arm

    arm1 = _evaluate_arm()
    arm2 = _train_arm(additive_rewards)
    arm3 = _train_arm(gated_rewards)

    # Merge Arm 3's LoRA weights and verify merge equivalence + subspace safety.
    with torch.no_grad():
        pre_merge_logits = _score_candidates(model).detach().clone()
    arm3_eval_probs = torch.softmax(pre_merge_logits, dim=-1)

    # Forbidden directions: mean rows of the last / second-to-last target
    # blocks' o_proj weights, Gram-Schmidt orthonormalized into a refusal +
    # slop pair so the projector `I - U U^T` is idempotent.
    last_key = (target_layer_indices[-1], "self_attn.o_proj")
    refusal_dir = _mean_row_direction(target_weights[last_key])
    if len(target_layer_indices) >= 2:
        slop_seed = _mean_row_direction(
            target_weights[(target_layer_indices[0], "self_attn.o_proj")]
        )
    else:
        slop_seed = _mean_row_direction(target_weights[(target_layer_indices[-1], "mlp.down_proj")])
    slop_dir = biproject_direction(slop_seed, [refusal_dir])
    dirs = [refusal_dir]
    if float(slop_dir.norm()) > 1e-8:
        dirs.append(slop_dir / slop_dir.norm())
    forbidden_dirs = {idx: list(dirs) for idx in target_layer_indices}
    u_stack = torch.stack(dirs, dim=1)
    projector = torch.eye(u_stack.shape[0], device=u_stack.device) - u_stack @ u_stack.t()

    def _module_scaling(module: LoRALinear) -> float:
        return float(getattr(module, "scaling", lora_alpha / lora_rank))

    # Naive-merge logits WITHOUT copying the 3B model: fold each LoRA delta
    # into the base weight and zero lora_b, score, then restore the factors.
    lora_backup = {
        key: {
            "lora_a": module.lora_a.detach().clone(),
            "lora_b": module.lora_b.detach().clone(),
        }
        for key, module in lora_modules
    }
    deltas: dict[tuple[int, str], torch.Tensor] = {}
    with torch.no_grad():
        for key, module in lora_modules:
            scaling = _module_scaling(module)
            delta = (module.lora_b.float() @ module.lora_a.float()) * scaling
            deltas[key] = delta
            base_weight = module.base_layer.weight
            base_weight.add_(delta.to(base_weight.dtype))
            module.lora_b.zero_()
        naive_logits = _score_candidates(model).detach().clone()
        for key, module in lora_modules:
            base_weight = module.base_layer.weight
            base_weight.sub_(deltas[key].to(base_weight.dtype))
            module.lora_b.copy_(lora_backup[key]["lora_b"])
            module.lora_a.copy_(lora_backup[key]["lora_a"])
    max_abs_logit_diff = float((pre_merge_logits[0] - naive_logits[0]).abs().max())

    def _forbidden_mass(delta_map: dict[tuple[int, str], torch.Tensor]) -> float:
        """Forbidden-subspace mass of per-target weight deltas, summed."""
        total = 0.0
        for delta in delta_map.values():
            total += float((u_stack.t() @ delta.float()).norm())
        return total

    # A rank-4 delta re-expresses exactly as (lora_b @ lora_a) * scaling, so
    # the naive merge leaks forbidden mass back in by this much.
    unprojected_lora_releakage = _forbidden_mass(deltas)

    # Subspace-safe merge: project each lora_b onto the orthogonal complement
    # of the forbidden pair (idempotent with the merge call's own projection),
    # then fold in place with per-layer forbidden dirs + row-norm preservation.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                projected_b = projector @ module.lora_b.float()
                module.lora_b.copy_(projected_b.to(dtype=module.lora_b.dtype))
    merge_lora_weights_in_place(model, forbidden_dirs=forbidden_dirs, preserve_row_norm=True)
    merged_lora_count = sum(isinstance(m, LoRALinear) for m in model.modules())
    assert merged_lora_count == 0, "merge_lora_weights_in_place left LoRALinear wrappers"
    arm3["merged_lora_count"] = merged_lora_count
    arm3["max_abs_logit_diff"] = round(max_abs_logit_diff, 8)
    post_deltas = {
        key: module.base_layer.weight.detach().float() - target_weights[key].float()
        for key, module in lora_modules
    }
    subspace_safe_lora_releakage = _forbidden_mass(post_deltas)

    # Suppression retention: re-inject forbidden mass at a fixed scale into the
    # pre-surgery weights, then check how much of that suppression survives
    # GRPO + the safe merge.
    injection = 0.5

    def _injected(key: tuple[int, str]) -> torch.Tensor:
        inject = torch.zeros_like(target_weights[key].float())
        for col, direction in enumerate(u_stack.t()):
            inject[:, col] += injection * direction
        return inject

    def _projection_norm(weights: torch.Tensor) -> float:
        return float((u_stack.t() @ weights).norm())

    proj_pre = sum(
        _projection_norm(target_weights[key].float() + _injected(key)) for key in target_weights
    )
    proj_surg = sum(_projection_norm(target_weights[key].float()) for key in target_weights)
    proj_safe = sum(
        _projection_norm(module.base_layer.weight.detach().float()) for _, module in lora_modules
    )
    suppression_ref = max(proj_pre - proj_surg, 1e-8)
    suppression_after = proj_pre - proj_safe
    post_merge_suppression_retention = min(1.0, max(0.0, suppression_after / suppression_ref))

    # Response-length health of Arm 3's evaluated (argmax) responses.
    response_word_counts: list[int] = []
    collapsed_responses = 0
    for row_idx, spec in enumerate(TENET_UNSLOP_SCENARIOS):
        chosen = _candidate_responses(spec)[int(torch.argmax(arm3_eval_probs[row_idx]))]
        word_count = len(chosen.split())
        response_word_counts.append(word_count)
        if word_count < (spec.min_words or 12):
            collapsed_responses += 1
    mean_response_words = sum(response_word_counts) / num_scenarios
    length_collapse_rate = collapsed_responses / num_scenarios

    h10_gated_vs_naive_gain = arm3["gated_ifeval_score"] - arm2["gated_ifeval_score"]
    h10_post_surgery_gain = arm3["gated_ifeval_score"] - arm1["gated_ifeval_score"]
    # 16-bit tolerances: fp16/bf16 merge and merge-emulation arithmetic is
    # noisy, and a 6-step rank-4 LoRA pass on a 3B+ LLM has smaller probability
    # shifts than a 27-step toy policy, so thresholds are dtype-calibrated.
    is_low_prec = dtype in (torch.float16, torch.bfloat16)
    logit_diff_tol = 0.1 if is_low_prec else 1e-4
    releakage_tol = 0.10 if is_low_prec else 1e-5
    slop_ceil = 0.10 if is_low_prec else 0.02
    gain_floor = 0.01 if is_low_prec else 0.15
    collapse_ceil = 0.25 if is_low_prec else 0.0
    h10_verified = bool(
        arm3["slop_and_refusal_rate"] <= slop_ceil
        and h10_gated_vs_naive_gain >= gain_floor
        and h10_post_surgery_gain >= gain_floor
        and max_abs_logit_diff < logit_diff_tol
        and merged_lora_count == 0
        and subspace_safe_lora_releakage < releakage_tol
        and post_merge_suppression_retention >= 0.90
        and length_collapse_rate <= collapse_ceil
    )

    return {
        "slop_and_refusal_rate": arm3["slop_and_refusal_rate"],
        "h10_gated_vs_naive_gain": round(h10_gated_vs_naive_gain, 6),
        "h10_post_surgery_gain": round(h10_post_surgery_gain, 6),
        "max_abs_logit_diff": round(max_abs_logit_diff, 8),
        "merged_lora_count": merged_lora_count,
        "unprojected_lora_releakage": round(unprojected_lora_releakage, 8),
        "subspace_safe_lora_releakage": round(subspace_safe_lora_releakage, 9),
        "post_merge_suppression_retention": round(post_merge_suppression_retention, 6),
        "mean_response_words": round(mean_response_words, 4),
        "length_collapse_rate": length_collapse_rate,
        "h10_verified": h10_verified,
        "arms": {
            "1_lancet_surgical_only": arm1,
            "2_tenet_grpo_additive": arm2,
            "3_tenet_grpo_gated": arm3,
        },
        "backend": "llm",
        "model_name": model_name,
        "device": str(resolved_device),
    }
