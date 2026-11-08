"""Custom GPU kernels (@triton.jit) and hardware-aware dispatchers for SML."""

from sml.kernels.triton_ops import (
    HAS_TRITON,
    KernelBackend,
    TritonRMSNormFunction,
    TritonSwiGLUFunction,
    compute_rmsnorm_hbm_bytes,
    compute_swiglu_hbm_bytes,
    fused_rmsnorm,
    fused_swiglu,
    is_triton_available,
    ref_rmsnorm,
    ref_swiglu,
    simulate_tiled_rmsnorm_fwd_bwd,
    simulate_tiled_swiglu_fwd_bwd,
)

__all__ = [
    "HAS_TRITON",
    "KernelBackend",
    "TritonRMSNormFunction",
    "TritonSwiGLUFunction",
    "compute_rmsnorm_hbm_bytes",
    "compute_swiglu_hbm_bytes",
    "fused_rmsnorm",
    "fused_swiglu",
    "is_triton_available",
    "ref_rmsnorm",
    "ref_swiglu",
    "simulate_tiled_rmsnorm_fwd_bwd",
    "simulate_tiled_swiglu_fwd_bwd",
]
