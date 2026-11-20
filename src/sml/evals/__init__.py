"""Bits-per-byte (bpb), zero-shot multiple-choice harness, induction metrics, bootstrap CIs.

Also includes the H17 artifact verifier / peer eval gate.
"""

from sml.evals.artifact_verifier import (
    TENET_ARENA_HOUSE_SERIES,
    ArenaHousePrompt,
    ArtifactVerificationResult,
    build_tenet_domain_imatrix_corpus,
    evaluate_arena_house_series,
    verify_house_artifact,
)
from sml.evals.harness import (
    bootstrap_confidence_interval,
    compute_bpb,
    evaluate_induction_metrics,
    evaluate_validation_bpb,
    score_multiple_choice_example,
)
from sml.evals.peer_eval import run_peer_eval_gate

__all__ = [
    "ArenaHousePrompt",
    "ArtifactVerificationResult",
    "TENET_ARENA_HOUSE_SERIES",
    "bootstrap_confidence_interval",
    "build_tenet_domain_imatrix_corpus",
    "compute_bpb",
    "evaluate_arena_house_series",
    "evaluate_induction_metrics",
    "evaluate_validation_bpb",
    "run_peer_eval_gate",
    "score_multiple_choice_example",
    "verify_house_artifact",
]
