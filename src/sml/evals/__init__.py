"""Bits-per-byte (bpb), zero-shot multiple-choice harness, induction metrics, and bootstrap CIs."""

from sml.evals.harness import (
    bootstrap_confidence_interval,
    compute_bpb,
    evaluate_induction_metrics,
    evaluate_validation_bpb,
    score_multiple_choice_example,
)

__all__ = [
    "bootstrap_confidence_interval",
    "compute_bpb",
    "evaluate_induction_metrics",
    "evaluate_validation_bpb",
    "score_multiple_choice_example",
]
