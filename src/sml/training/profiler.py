"""MFU and throughput waterfall profiler across precision, attention, and fusion (`H2`)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from sml.config import ExperimentSpec
from sml.model.attention import AttnBackend
from sml.model.transformer import Transformer


@dataclass(frozen=True)
class WaterfallStageResult:
    """Measured throughput and MFU metrics for one optimization stage in the H2 waterfall."""

    stage_name: str
    precision: str
    attn_backend: str
    fused_kernels: bool
    compiled: bool
    tokens_per_sec: float
    step_time_ms: float
    mfu: float
    peak_memory_mb: float
    speedup_vs_baseline: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve device string (`auto`, `cuda`, `mps`, `cpu`) to a `torch.device`."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def synchronize_device(dev: torch.device) -> None:
    """Synchronize accelerator execution before/after wall-clock timing."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def _configure_kernel_backend(model: torch.nn.Module, fused_kernels: bool) -> None:
    backend = "auto" if fused_kernels else "pytorch"
    for module in model.modules():
        if hasattr(module, "kernel_backend"):
            module.kernel_backend = backend


def _execute_stage_step(
    active_model: Any,
    base_model: Transformer,
    compiled_active: bool,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    attn_backend: AttnBackend,
    autocast_device: str,
    use_autocast: bool,
) -> tuple[Any, bool]:
    optimizer.zero_grad(set_to_none=True)
    try:
        with torch.autocast(
            device_type=autocast_device,
            dtype=torch.bfloat16,
            enabled=use_autocast,
        ):
            out = active_model(input_ids, targets=targets, attn_backend=attn_backend)
            assert out.loss is not None
        out.loss.backward()
    except Exception:
        if compiled_active:
            active_model = base_model
            compiled_active = False
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=autocast_device,
                dtype=torch.bfloat16,
                enabled=use_autocast,
            ):
                out = active_model(input_ids, targets=targets, attn_backend=attn_backend)
                assert out.loss is not None
            out.loss.backward()
        else:
            raise
    optimizer.step()
    return active_model, compiled_active


def run_mfu_waterfall(
    spec: ExperimentSpec,
    device: str = "auto",
    warmup_steps: int = 1,
    measure_steps: int = 3,
    seq_len_override: int | None = None,
    micro_batch_override: int | None = None,
) -> list[WaterfallStageResult]:
    """Run the 5-stage MFU throughput waterfall (`H2`) on `Transformer(spec.model)`."""
    dev = resolve_device(device)
    seq_len = min(seq_len_override or spec.train.seq_len, spec.model.max_seq_len)
    bsz = micro_batch_override or spec.train.micro_batch_size

    stages: list[tuple[str, str, AttnBackend, bool, bool]] = [
        ("1_fp32_naive_eager", "fp32", "naive", False, False),
        ("2_fp32_sdpa_eager", "fp32", "sdpa", False, False),
        ("3_bf16_sdpa_eager", "bf16", "sdpa", False, False),
        ("4_bf16_sdpa_fused", "bf16", "sdpa", True, False),
        ("5_bf16_sdpa_compiled", "bf16", "sdpa", True, True),
    ]

    results: list[WaterfallStageResult] = []
    baseline_tok_per_sec: float | None = None

    for stage_name, precision, attn_backend, fused_kernels, compiled in stages:
        torch.manual_seed(spec.seed)
        base_model = Transformer(spec.model).to(dev)
        _configure_kernel_backend(base_model, fused_kernels=fused_kernels)
        base_model.train()

        active_model: Any = base_model
        compiled_active = False
        if compiled:
            try:
                active_model = torch.compile(base_model)
                compiled_active = True
            except Exception:
                active_model = base_model
                compiled_active = False

        optimizer = torch.optim.AdamW(base_model.parameters(), lr=spec.train.lr)
        input_ids = torch.randint(0, spec.model.vocab_size, (bsz, seq_len), device=dev)
        targets = torch.randint(0, spec.model.vocab_size, (bsz, seq_len), device=dev)

        use_autocast = precision == "bf16" and dev.type in {"cuda", "cpu"}
        autocast_device = dev.type if dev.type in {"cuda", "cpu"} else "cpu"

        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

        for _ in range(max(1, warmup_steps)):
            active_model, compiled_active = _execute_stage_step(
                active_model=active_model,
                base_model=base_model,
                compiled_active=compiled_active,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                attn_backend=attn_backend,
                autocast_device=autocast_device,
                use_autocast=use_autocast,
            )

        synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(max(1, measure_steps)):
            active_model, compiled_active = _execute_stage_step(
                active_model=active_model,
                base_model=base_model,
                compiled_active=compiled_active,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                attn_backend=attn_backend,
                autocast_device=autocast_device,
                use_autocast=use_autocast,
            )
        synchronize_device(dev)
        elapsed_sec = max(time.perf_counter() - t0, 1e-6)

        steps_run = max(1, measure_steps)
        total_tokens = steps_run * bsz * seq_len
        tokens_per_sec = total_tokens / elapsed_sec
        step_time_ms = (elapsed_sec * 1000.0) / steps_run
        flops_per_tok = spec.model.flops_per_token(seq_len=seq_len)
        mfu = (tokens_per_sec * flops_per_tok) / (spec.hardware.peak_bf16_tflops * 1e12)

        if baseline_tok_per_sec is None:
            baseline_tok_per_sec = tokens_per_sec
        speedup = tokens_per_sec / max(baseline_tok_per_sec, 1e-6)

        if dev.type == "cuda":
            peak_memory_mb = float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0)
        elif dev.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            peak_memory_mb = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
        else:
            param_bytes = sum(p.numel() * p.element_size() for p in base_model.parameters())
            peak_memory_mb = float(param_bytes) / (1024.0 * 1024.0)

        results.append(
            WaterfallStageResult(
                stage_name=stage_name,
                precision=precision,
                attn_backend=attn_backend,
                fused_kernels=fused_kernels,
                compiled=compiled_active,
                tokens_per_sec=round(tokens_per_sec, 2),
                step_time_ms=round(step_time_ms, 3),
                mfu=round(mfu, 6),
                peak_memory_mb=round(peak_memory_mb, 2),
                speedup_vs_baseline=round(speedup, 4),
            )
        )

        del active_model, base_model, optimizer

    return results
