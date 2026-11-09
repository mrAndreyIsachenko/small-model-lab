"""Teacher trajectory collection, Gated Rubric rejection sampling, and KL distillation loss."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from sml.envs.rubric import RewardBreakdown
from sml.envs.tool_env import (
    StatefulToolEnv,
    run_lazy_guess_episode,
    run_oracle_episode,
    run_unsafe_sql_episode,
)


@dataclass(frozen=True)
class EpisodeTrajectory:
    """Multi-turn tool-use trajectory scored by `StatefulToolEnv` for SFT / Distillation."""

    scenario_id: str
    root_cause_category: str
    question: str
    tool_trace: list[dict[str, Any]]
    submitted_diagnosis: dict[str, Any] | None
    reward: RewardBreakdown
    accepted: bool


def collect_rejection_sampled_trajectories(
    env: StatefulToolEnv,
    min_reward: float = 0.95,
    min_evidence: float = 0.90,
) -> tuple[list[EpisodeTrajectory], dict[str, int | float]]:
    """Collect candidate multi-turn rollouts and filter them via Gated Rubric rejection sampling."""
    accepted_trajectories: list[EpisodeTrajectory] = []
    total_candidates = 0

    for scenario in env.scenarios:
        candidate_states = [
            run_oracle_episode(env, scenario),
            run_lazy_guess_episode(env, scenario),
            run_unsafe_sql_episode(env, scenario),
        ]
        for state in candidate_states:
            total_candidates += 1
            reward = env.score_state(state, mode_override="gated_multiplicative")
            is_accepted = (
                reward.total_reward >= min_reward
                and reward.g_evidence >= min_evidence
                and reward.g_safety == 1.0
                and reward.g_complete == 1.0
            )
            traj = EpisodeTrajectory(
                scenario_id=scenario.scenario_id,
                root_cause_category=scenario.root_cause_category,
                question=scenario.question,
                tool_trace=list(state["tool_trace"]),
                submitted_diagnosis=state["submitted_diagnosis"],
                reward=reward,
                accepted=is_accepted,
            )
            if is_accepted:
                accepted_trajectories.append(traj)

    accepted_count = len(accepted_trajectories)
    acceptance_rate = accepted_count / max(1, total_candidates)
    mean_accepted_reward = (
        sum(t.reward.total_reward for t in accepted_trajectories) / accepted_count
        if accepted_count > 0
        else 0.0
    )

    stats = {
        "total_candidates": total_candidates,
        "accepted_count": accepted_count,
        "acceptance_rate": round(acceptance_rate, 4),
        "mean_accepted_reward": round(mean_accepted_reward, 4),
    }
    return accepted_trajectories, stats


def format_trajectory_for_sft(traj: EpisodeTrajectory) -> str:
    """Serialize a verified `EpisodeTrajectory` into structured chat/tool tokens for SFT."""
    parts = [f"<|user|>\n{traj.question}"]
    for step in traj.tool_trace:
        payload = json.dumps(
            {"tool": step["tool"], "args": step["args"]},
            sort_keys=True,
        )
        parts.append(f"<|assistant|>\n<|tool_call|>{payload}")
    parts.append("<|eos|>")
    return "\n".join(parts)


def compute_distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 2.0,
    alpha_kl: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute combined Cross-Entropy + Temperature-Scaled Forward KL distillation loss."""
    vocab_size = student_logits.size(-1)
    ce_loss = F.cross_entropy(
        student_logits.float().view(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )

    t = max(temperature, 1e-3)
    student_log_probs = F.log_softmax(student_logits.float() / t, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach().float() / t, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t * t)

    return (1.0 - alpha_kl) * ce_loss + alpha_kl * kl_loss
