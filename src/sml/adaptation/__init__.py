"""Sprint 3 adaptation suite: Hybrid RAG, LoRA, Distillation, and Multi-Turn GRPO."""

from sml.adaptation.compare import (
    StrategyEvaluationResult,
    run_adaptation_comparison,
    run_llm_adaptation_comparison,
)
from sml.adaptation.distill import (
    EpisodeTrajectory,
    collect_rejection_sampled_trajectories,
    compute_distillation_kl_loss,
    format_trajectory_for_sft,
)
from sml.adaptation.grpo import (
    GRPOConfig,
    LLMGRPOConfig,
    compute_group_normalized_advantages,
    compute_grpo_loss,
    run_incomplete_exploration_episode,
    run_partial_rag_episode,
    train_grpo_action_policy,
    train_grpo_llm_policy,
)
from sml.adaptation.llm_sft import (
    collect_oracle_sft_trajectories,
    collect_teacher_gateway_trajectories,
    compute_masked_sft_loss,
    run_teacher_gateway_episode,
    train_lora_sft_llm,
)
from sml.adaptation.lora import (
    LoRALinear,
    apply_lora_to_transformer,
    lora_parameter_summary,
    merge_lora_weights,
)
from sml.adaptation.rag import (
    HybridRunbookRetriever,
    RunbookDocument,
    build_incident_knowledge_base,
)

__all__ = [
    "EpisodeTrajectory",
    "GRPOConfig",
    "HybridRunbookRetriever",
    "LLMGRPOConfig",
    "LoRALinear",
    "RunbookDocument",
    "StrategyEvaluationResult",
    "apply_lora_to_transformer",
    "build_incident_knowledge_base",
    "collect_oracle_sft_trajectories",
    "collect_rejection_sampled_trajectories",
    "collect_teacher_gateway_trajectories",
    "compute_distillation_kl_loss",
    "compute_group_normalized_advantages",
    "compute_grpo_loss",
    "compute_masked_sft_loss",
    "format_trajectory_for_sft",
    "lora_parameter_summary",
    "merge_lora_weights",
    "run_adaptation_comparison",
    "run_incomplete_exploration_episode",
    "run_llm_adaptation_comparison",
    "run_partial_rag_episode",
    "run_teacher_gateway_episode",
    "train_grpo_action_policy",
    "train_grpo_llm_policy",
    "train_lora_sft_llm",
]
