"""Compute and cloud GPU cost estimation and accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sml.config import ExperimentSpec, HardwareProfile


@dataclass(frozen=True)
class CostRecord:
    """Measured or estimated compute cost record written to cost.json."""

    hardware_name: str
    hourly_rate_usd: float
    price_as_of: str
    num_gpus: int
    active_training_seconds: float
    billed_instance_seconds: float
    active_gpu_hours: float
    billed_gpu_hours: float
    infrastructure_overhead_fraction: float
    active_cost_usd: float
    billed_cost_usd: float
    estimated_pre_run_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_experiment_compute(
    spec: ExperimentSpec,
    assumed_mfu: float = 0.30,
    num_gpus: int = 1,
) -> dict[str, float | int]:
    """Compute pre-run FLOP, GPU-hour, and USD cost estimates before provisioning hardware."""
    tokens = spec.train.total_planned_tokens
    if spec.train.branch_cooldown_tokens is not None:
        cooldown_tokens = int(spec.train.total_planned_tokens * spec.train.decay_fraction)
        tokens += cooldown_tokens

    flops_per_tok = spec.model.flops_per_token(spec.train.seq_len)
    total_flops = tokens * flops_per_tok

    effective_tflops = spec.hardware.peak_bf16_tflops * assumed_mfu * num_gpus
    effective_flops_per_sec = effective_tflops * 1e12
    clean_hours = total_flops / (effective_flops_per_sec * 3600.0)
    clean_gpu_hours = clean_hours * num_gpus
    estimated_cost_usd = clean_gpu_hours * spec.hardware.hourly_rate_usd

    return {
        "planned_tokens": tokens,
        "flops_per_token": flops_per_tok,
        "total_flops": total_flops,
        "assumed_mfu": assumed_mfu,
        "estimated_clean_gpu_hours": round(clean_gpu_hours, 4),
        "estimated_cost_usd": round(estimated_cost_usd, 4),
    }


def compute_run_cost(
    hardware: HardwareProfile,
    active_seconds: float,
    billed_seconds: float | None = None,
    num_gpus: int = 1,
    estimated_pre_run_cost_usd: float = 0.0,
) -> CostRecord:
    """Calculate active vs. billed GPU-hours and USD cost."""
    raw_billed = billed_seconds if billed_seconds is not None else active_seconds
    billed_sec = max(raw_billed, active_seconds)
    active_gpu_hours = (active_seconds / 3600.0) * num_gpus
    billed_gpu_hours = (billed_sec / 3600.0) * num_gpus

    overhead_fraction = (
        (billed_gpu_hours - active_gpu_hours) / billed_gpu_hours if billed_gpu_hours > 0 else 0.0
    )
    active_cost = active_gpu_hours * hardware.hourly_rate_usd
    billed_cost = billed_gpu_hours * hardware.hourly_rate_usd

    return CostRecord(
        hardware_name=hardware.name,
        hourly_rate_usd=hardware.hourly_rate_usd,
        price_as_of=hardware.price_as_of,
        num_gpus=num_gpus,
        active_training_seconds=round(active_seconds, 3),
        billed_instance_seconds=round(billed_sec, 3),
        active_gpu_hours=round(active_gpu_hours, 6),
        billed_gpu_hours=round(billed_gpu_hours, 6),
        infrastructure_overhead_fraction=round(overhead_fraction, 4),
        active_cost_usd=round(active_cost, 4),
        billed_cost_usd=round(billed_cost, 4),
        estimated_pre_run_cost_usd=round(estimated_pre_run_cost_usd, 4),
    )
