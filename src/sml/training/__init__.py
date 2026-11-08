"""Pretraining loop, WSD schedules, cooldown branching, and MFU waterfall profiling."""

from sml.training.profiler import (
    WaterfallStageResult,
    resolve_device,
    run_mfu_waterfall,
    synchronize_device,
)
from sml.training.runner import run_pretraining_experiment
from sml.training.trainer import (
    Trainer,
    build_log_spaced_steps,
    compute_branched_cooldown_lr,
    compute_lr,
)

__all__ = [
    "Trainer",
    "WaterfallStageResult",
    "build_log_spaced_steps",
    "compute_branched_cooldown_lr",
    "compute_lr",
    "resolve_device",
    "run_mfu_waterfall",
    "run_pretraining_experiment",
    "synchronize_device",
]
