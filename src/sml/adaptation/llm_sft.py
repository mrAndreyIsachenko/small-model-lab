"""Supervised fine-tuning (`SFT`) of LoRA-adapted LLM policies on oracle trajectories (`H6`)."""

from __future__ import annotations

import json
from typing import Any

import torch
import torch.nn.functional as F

from sml.adaptation.lora import (
    LoRALinear,
    apply_lora_to_transformer,
    lora_parameter_summary,
)
from sml.adaptation.rag import HybridRunbookRetriever
from sml.envs.scenarios import IncidentScenario
from sml.envs.tool_env import (
    _TOOL_CALL_PROMPT_SUFFIX,
    RolloutTrajectory,
    StatefulToolEnv,
    build_masked_trajectory_tokens,
    format_incident_prompt,
    format_tool_observation,
    parse_tool_call,
    run_oracle_episode,
)
from sml.gateway import GatewayConfig, OpenRouterGatewayClient


def _forward_causal_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a causal-LM forward pass and return raw logits of shape `(B, T, V)`.

    Supports HF `AutoModelForCausalLM` outputs (`.logits`), dict outputs, and
    from-scratch `Transformer` modules that return logits directly.
    """
    device = getattr(model, "device", None)
    if device is None and callable(getattr(model, "parameters", None)):
        first = next(iter(model.parameters()), None)
        if first is not None:
            device = first.device
    if device is not None and input_ids.device != device:
        input_ids = input_ids.to(device)
    try:
        output = model(input_ids=input_ids)
    except TypeError:
        output = model(input_ids)
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        logits = output
    return logits


def _scenario_id(scenario: IncidentScenario) -> str:
    """Resolve a stable scenario identifier for trajectory bookkeeping."""
    return str(getattr(scenario, "scenario_id", None) or getattr(scenario, "id", None) or "")


def _rag_context_for_scenario(retriever: HybridRunbookRetriever, scenario: IncidentScenario) -> str:
    """Retrieve the top-1 runbook document and format it as grounded prompt context."""
    query = f"{scenario.question} {scenario.root_cause_category}"
    doc, _score = retriever.retrieve(query, top_k=1)[0]
    return (
        f"Runbook [{doc.title} | category={doc.category} | table={doc.target_table}]: "
        f"{doc.content}"
        "\nFollow the exact 3-step runbook sequence above: (1) query_warehouse on "
        "mart_appointments, (2) query_warehouse on the target upstream table, "
        "(3) submit_diagnosis with the exact fields from the runbook."
    )


def collect_oracle_sft_trajectories(
    env: StatefulToolEnv,
    tokenizer: Any,
    min_reward: float = 0.95,
) -> list[RolloutTrajectory]:
    """Replay oracle tool traces as chat trajectories and keep high-reward ones.

    For each scenario the oracle tool trace is re-executed on a fresh rollout
    state; each replayed call is recorded as an assistant `<tool_call>` message
    followed by a user `Tool observation (...): ...` message. Token spans are
    masked to assistant turns only via `build_masked_trajectory_tokens`. The
    scenario prompt is grounded with the top-1 retrieved runbook document so the
    policy learns to condition on the scenario category. With `max_steps=4` the
    oracle reward is exactly 1.0 (zero efficiency penalty), so the default
    `min_reward=0.95` keeps every oracle trajectory.
    """
    retriever = HybridRunbookRetriever()
    trajectories: list[RolloutTrajectory] = []

    for scenario in env.scenarios:
        oracle_state = run_oracle_episode(env, scenario)

        # Fresh state: oracle steps are replayed so scoring reflects real execution.
        state = env.setup_state(scenario)
        messages = format_incident_prompt(
            env, scenario, extra_context=_rag_context_for_scenario(retriever, scenario)
        )

        for trace in oracle_state["tool_trace"]:
            tool_name = str(trace["tool"])
            tool_args = dict(trace["args"])

            assistant_payload = json.dumps({"name": tool_name, "arguments": tool_args}, default=str)
            messages.append(
                {"role": "assistant", "content": f"<tool_call>{assistant_payload}</tool_call>"}
            )

            observation = env.step(tool_name, tool_args, state)
            messages.append(
                {
                    "role": "user",
                    "content": format_tool_observation(tool_name, observation, state),
                }
            )
            if state["done"]:
                break

        breakdown = env.score_state(state)
        input_ids, assistant_action_mask = build_masked_trajectory_tokens(messages, tokenizer)
        trajectory = RolloutTrajectory(
            scenario_id=_scenario_id(scenario),
            messages=messages,
            input_ids=input_ids,
            assistant_action_mask=assistant_action_mask,
            state=state,
            reward=breakdown.total_reward,
            breakdown=breakdown,
        )
        if trajectory.reward > min_reward:
            trajectories.append(trajectory)

    return trajectories


def collect_partial_sft_trajectories(
    env: StatefulToolEnv,
    tokenizer: Any,
) -> list[RolloutTrajectory]:
    """Replay mixed oracle/partial-RAG episode traces as chat SFT trajectories.

    For scenarios with `root_cause_category != "mix_shift"` (i.e. `product_bug`
    and `data_pipeline_artifact`, 2 of 3 causes) the full oracle tool trace is
    replayed via `run_oracle_episode`; the `mix_shift` category replays
    `run_partial_rag_episode`, so the SFT mixture mirrors realistic imperfect
    demonstrations (accurate required tables and slice columns on 2 of 3
    causes, 1 of 2 confounders checked on the 3rd cause, matching the
    `4_lora_sft` baseline docstring in `compare.py`). Each
    replayed call is recorded as an assistant `<tool_call>` message followed by
    a user `Tool observation (...): ...` message, every trajectory is returned
    regardless of reward (no `min_reward > 1.0` filter), and the scenario prompt
    is grounded with the top-1 retrieved runbook document so the policy learns
    to condition on the scenario category.
    """
    from sml.adaptation.grpo import run_partial_rag_episode

    retriever = HybridRunbookRetriever()
    trajectories: list[RolloutTrajectory] = []

    for scenario in env.scenarios:
        if scenario.root_cause_category != "mix_shift":
            episode_state = run_oracle_episode(env, scenario)
        else:
            episode_state = run_partial_rag_episode(env, scenario)

        # Fresh state: episode steps are replayed so scoring reflects real execution.
        state = env.setup_state(scenario)
        messages = format_incident_prompt(
            env, scenario, extra_context=_rag_context_for_scenario(retriever, scenario)
        )

        for trace in episode_state["tool_trace"]:
            tool_name = str(trace["tool"])
            tool_args = dict(trace["args"])

            assistant_payload = json.dumps({"name": tool_name, "arguments": tool_args}, default=str)
            messages.append(
                {"role": "assistant", "content": f"<tool_call>{assistant_payload}</tool_call>"}
            )

            observation = env.step(tool_name, tool_args, state)
            messages.append(
                {
                    "role": "user",
                    "content": format_tool_observation(tool_name, observation, state),
                }
            )
            if state["done"]:
                break

        breakdown = env.score_state(state)
        input_ids, assistant_action_mask = build_masked_trajectory_tokens(messages, tokenizer)
        trajectories.append(
            RolloutTrajectory(
                scenario_id=_scenario_id(scenario),
                messages=messages,
                input_ids=input_ids,
                assistant_action_mask=assistant_action_mask,
                state=state,
                reward=breakdown.total_reward,
                breakdown=breakdown,
            )
        )

    return trajectories


def run_teacher_gateway_episode(
    env: StatefulToolEnv,
    scenario: IncidentScenario,
    gateway_client: OpenRouterGatewayClient,
    teacher_model: str = "qwen/qwen-2.5-72b-instruct",
    tokenizer: Any = None,
    extra_context: str = "",
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> RolloutTrajectory:
    """Run one teacher-driven tool episode through the OpenRouter gateway.

    The scenario prompt is seeded with `extra_context` (typically the retrieved
    runbook), the teacher model is queried once per step via
    `gateway_client.chat_completions`, and its reply is parsed with
    `parse_tool_call`. Unparseable replies get a `Format error` user message
    plus the tool-call prompt suffix and the step is retried on the next loop
    iteration; valid replies are executed with `env.step` and the formatted
    observation is appended. The episode is scored with the
    `gated_multiplicative` rubric so ungrounded teacher traces score low.
    """
    state = env.setup_state(scenario)
    messages = format_incident_prompt(env, scenario, extra_context=extra_context)

    for step_idx in range(env.max_steps):
        if state["done"]:
            break
        response = gateway_client.chat_completions(
            {
                "model": teacher_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            request_id=f"{_scenario_id(scenario)}-step-{step_idx + 1}",
        )
        response_text = str(response["choices"][0]["message"]["content"])
        messages.append({"role": "assistant", "content": response_text})

        parsed = parse_tool_call(response_text)
        if parsed is None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Format error: could not parse a <tool_call> payload from "
                        "your reply. " + _TOOL_CALL_PROMPT_SUFFIX
                    ),
                }
            )
            continue

        tool_name, tool_args = parsed
        observation = env.step(tool_name, tool_args, state)
        messages.append(
            {
                "role": "user",
                "content": format_tool_observation(tool_name, observation, state),
            }
        )
        if state["done"]:
            break

    breakdown = env.score_state(state, mode_override="gated_multiplicative")
    if tokenizer is not None:
        input_ids, assistant_action_mask = build_masked_trajectory_tokens(messages, tokenizer)
    else:
        input_ids, assistant_action_mask = [], []

    return RolloutTrajectory(
        scenario_id=_scenario_id(scenario),
        messages=messages,
        input_ids=input_ids,
        assistant_action_mask=assistant_action_mask,
        state=state,
        reward=breakdown.total_reward,
        breakdown=breakdown,
    )


def collect_teacher_gateway_trajectories(
    env: StatefulToolEnv,
    tokenizer: Any = None,
    gateway_client: OpenRouterGatewayClient | None = None,
    teacher_model: str = "qwen/qwen-2.5-72b-instruct",
    min_reward: float = 0.95,
    use_rag_context: bool = True,
) -> list[RolloutTrajectory]:
    """Collect gated-rubric-filtered SFT trajectories from a teacher LLM.

    Each scenario runs one `run_teacher_gateway_episode` through the gateway
    (defaulting to `OpenRouterGatewayClient(GatewayConfig.from_env())`), and
    only trajectories scoring `>= min_reward` under the `gated_multiplicative`
    rubric are kept. When `use_rag_context` is True the scenario prompt is
    grounded with the top-1 retrieved runbook document, mirroring the oracle
    SFT collector so the policy conditions on the scenario category.
    """
    if gateway_client is None:
        gateway_client = OpenRouterGatewayClient(GatewayConfig.from_env())
    retriever = HybridRunbookRetriever() if use_rag_context else None

    trajectories: list[RolloutTrajectory] = []
    for scenario in env.scenarios:
        extra_context = (
            _rag_context_for_scenario(retriever, scenario) if retriever is not None else ""
        )
        trajectory = run_teacher_gateway_episode(
            env,
            scenario,
            gateway_client,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            extra_context=extra_context,
        )
        if trajectory.reward >= min_reward:
            trajectories.append(trajectory)

    return trajectories


def compute_masked_sft_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    assistant_action_mask: torch.Tensor,
) -> torch.Tensor:
    """Next-token cross entropy averaged over assistant-token positions only.

    `logits` is `(B, T, V)`, `input_ids` is `(B, T)`, and `assistant_action_mask`
    is `(B, T)` with 1 on assistant-produced tokens. The shift drops the final
    logit and the first target; the mask is aligned to the targets via
    `assistant_action_mask[:, 1:]`.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_targets = input_ids[:, 1:].to(logits.device)
    shift_mask = assistant_action_mask[:, 1:].to(device=logits.device, dtype=torch.float32)

    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_targets = shift_targets.reshape(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    per_token_loss = per_token_loss.view(shift_targets.shape)

    denom = shift_mask.sum().clamp(min=1.0)
    return (per_token_loss * shift_mask).sum() / denom


def _pad_token_id(tokenizer: Any) -> int:
    """Pick a padding token id from the tokenizer, falling back to 0."""
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    return int(pad_id) if pad_id is not None else 0


def _batch_trajectories(
    trajectories: list[RolloutTrajectory],
    tokenizer: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad trajectories into `(B, T)` token and assistant-mask tensors."""
    pad_id = _pad_token_id(tokenizer)
    max_len = max(len(t.input_ids) for t in trajectories)

    input_ids = torch.full((len(trajectories), max_len), pad_id, dtype=torch.long)
    action_mask = torch.zeros((len(trajectories), max_len), dtype=torch.float32)
    for row, trajectory in enumerate(trajectories):
        seq_len = len(trajectory.input_ids)
        input_ids[row, :seq_len] = torch.tensor(trajectory.input_ids, dtype=torch.long)
        action_mask[row, :seq_len] = torch.tensor(
            trajectory.assistant_action_mask, dtype=torch.float32
        )
    return input_ids.to(device), action_mask.to(device)


def train_lora_sft_llm(
    model: Any,
    tokenizer: Any,
    trajectories: list[RolloutTrajectory],
    rank: int = 16,
    alpha: float = 32.0,
    lr: float = 1e-3,
    steps: int = 8,
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
) -> dict[str, Any]:
    """LoRA fine-tune an LLM policy on masked oracle SFT trajectories.

    Applies `apply_lora_to_transformer` when the model carries no `LoRALinear`
    yet, then trains only the trainable parameters with AdamW on the masked
    next-token cross entropy from `compute_masked_sft_loss`.
    """
    if not any(isinstance(module, LoRALinear) for module in model.modules()):
        model = apply_lora_to_transformer(
            model, rank=rank, alpha=alpha, target_modules=target_modules
        )

    summary = lora_parameter_summary(model)
    if not trajectories:
        return {
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "steps": 0,
            "num_trajectories": 0,
            "tokens_processed": 0,
            "lora_parameter_summary": summary,
        }

    device = next(model.parameters()).device
    input_ids, action_mask = _batch_trajectories(trajectories, tokenizer, device)
    supervised_tokens = int(action_mask.sum().item())

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def _eval_loss() -> float:
        """Average masked SFT loss across individual rows (no batched forward pass)."""
        was_training = model.training
        model.eval()
        row_losses: list[float] = []
        with torch.no_grad():
            for row in range(input_ids.size(0)):
                row_ids = input_ids[row : row + 1]
                row_mask = action_mask[row : row + 1]
                logits = _forward_causal_logits(model, row_ids)
                loss = compute_masked_sft_loss(logits, row_ids, row_mask)
                row_losses.append(float(loss.item()))
        if was_training:
            model.train()
        return sum(row_losses) / max(1, len(row_losses))

    model.train()
    initial_loss = _eval_loss()

    # Row-by-row gradient accumulation: forwarding the full padded batch at once
    # triggers Apple MPS unified-memory swap on 3B models, so each trajectory is
    # processed individually and gradients are accumulated before optimizer.step().
    num_traj = max(1, input_ids.size(0))
    for _step in range(steps):
        optimizer.zero_grad()
        for row in range(input_ids.size(0)):
            row_ids = input_ids[row : row + 1]
            row_mask = action_mask[row : row + 1]
            logits = _forward_causal_logits(model, row_ids)
            loss = compute_masked_sft_loss(logits, row_ids, row_mask)
            (loss / num_traj).backward()
        optimizer.step()
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    final_loss = _eval_loss()

    return {
        "initial_loss": round(initial_loss, 6),
        "final_loss": round(final_loss, 6),
        "steps": steps,
        "num_trajectories": len(trajectories),
        "tokens_processed": supervised_tokens * steps,
        "lora_parameter_summary": lora_parameter_summary(model),
    }
