"""Custom @triton.jit GPU kernels for Fused RMSNorm and Fused SwiGLU with autograd support.

In decoder-only LLMs, elementwise and reduction ops between linear projections (`RMSNorm` and
`SwiGLU`) are memory-bandwidth bound on HBM:
- Eager `RMSNorm` launches multiple small kernels (`to(fp32)`, `pow(2)`, `mean`, `rsqrt`,
  `mul`, `to(dtype)`, `mul(weight)`), reading/writing activations to HBM repeatedly.
- `@triton.jit` `FusedRMSNorm` loads each token row `[1, d_model]` into SRAM once, computes the
  fp32 inverse RMS in registers, applies the weight, and writes the normalized row back to HBM
  in a single pass.
- `@triton.jit` `FusedSwiGLU` fuses `silu(gate) * up = (gate * sigmoid(gate)) * up` and its
  analytical backward pass without materializing intermediate `silu(gate)` tensors in HBM.

When running on macOS (`mps`/`cpu`) where CUDA Triton is unavailable, `fused_rmsnorm` and
`fused_swiglu` cleanly dispatch to the exact PyTorch reference, while `simulate_tiled_*`
validates the exact `BLOCK_SIZE` power-of-2 tiling, boundary masking (`cols < N`), and
analytical backward equations on CPU/MPS in unit tests.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F

KernelBackend = Literal["auto", "torch", "triton"]

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]

    HAS_TRITON = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


def is_triton_available(device: torch.device | str | None = None) -> bool:
    """Return True if Triton is installed and the target device is CUDA."""
    if not HAS_TRITON or not torch.cuda.is_available():
        return False
    if device is None:
        return True
    dev = torch.device(device)
    return dev.type == "cuda"


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n (for Triton BLOCK_SIZE)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# 1. Reference PyTorch Implementations
# ---------------------------------------------------------------------------


def ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference PyTorch RMSNorm computed in fp32."""
    x_fp32 = x.float()
    rrms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rrms).to(dtype=x.dtype) * weight


def ref_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference PyTorch SwiGLU: silu(gate) * up."""
    return F.silu(gate) * up


# ---------------------------------------------------------------------------
# 2. Custom @triton.jit Kernels (CUDA)
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr: Any,
        W_ptr: Any,
        Y_ptr: Any,
        RRMS_ptr: Any,
        stride_x_row: int,
        stride_y_row: int,
        N: int,
        eps: float,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        """Single-pass SRAM row-wise RMSNorm forward kernel."""
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x_ptrs = X_ptr + row_idx * stride_x_row + cols
        w_ptrs = W_ptr + cols
        y_ptrs = Y_ptr + row_idx * stride_y_row + cols

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

        mean_sq = tl.sum(x * x, axis=0) / N
        rrms = tl.rsqrt(mean_sq + eps)
        tl.store(RRMS_ptr + row_idx, rrms)

        y = (x * rrms) * w
        tl.store(y_ptrs, y.to(Y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_bwd_dx_kernel(
        DY_ptr: Any,
        X_ptr: Any,
        W_ptr: Any,
        RRMS_ptr: Any,
        DX_ptr: Any,
        DW_partial_ptr: Any,
        stride_row: int,
        N: int,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        """Analytical backward pass for RMSNorm computing dx and per-row dw contributions."""
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        offset = row_idx * stride_row + cols
        dy = tl.load(DY_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.load(RRMS_ptr + row_idx).to(tl.float32)

        x_hat = x * rrms
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / N
        dx = rrms * (wdy - x_hat * c1)
        dw_row = dy * x_hat

        tl.store(DX_ptr + offset, dx.to(DX_ptr.dtype.element_ty), mask=mask)
        tl.store(DW_partial_ptr + offset, dw_row, mask=mask)

    @triton.jit
    def _swiglu_fwd_kernel(
        Gate_ptr: Any,
        Up_ptr: Any,
        Out_ptr: Any,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        """Fused 1D elementwise SwiGLU forward kernel: (gate * sigmoid(gate)) * up."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        g = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig_g = tl.sigmoid(g)
        silu_g = g * sig_g
        out = silu_g * u

        tl.store(Out_ptr + offsets, out.to(Out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        DOut_ptr: Any,
        Gate_ptr: Any,
        Up_ptr: Any,
        DGate_ptr: Any,
        DUp_ptr: Any,
        n_elements: int,
        BLOCK_SIZE: tl.constexpr,
    ) -> None:
        """Fused 1D elementwise SwiGLU backward kernel computing d_gate and d_up in one pass."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        dout = tl.load(DOut_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(Gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(Up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sig_g = tl.sigmoid(g)
        silu_g = g * sig_g
        d_up = dout * silu_g
        d_silu = sig_g * (1.0 + g * (1.0 - sig_g))
        d_gate = dout * u * d_silu

        tl.store(DGate_ptr + offsets, d_gate.to(DGate_ptr.dtype.element_ty), mask=mask)
        tl.store(DUp_ptr + offsets, d_up.to(DUp_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# 3. Autograd Wrappers for Triton Kernels
# ---------------------------------------------------------------------------


class TritonRMSNormFunction(torch.autograd.Function):
    """Custom autograd Function dispatching forward/backward RMSNorm to @triton.jit kernels."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x_2d = x.contiguous().view(-1, n_cols)
        w_1d = weight.contiguous()
        n_rows = x_2d.shape[0]

        y_2d = torch.empty_like(x_2d)
        rrms = torch.empty((n_rows,), dtype=torch.float32, device=x.device)
        block_size = next_power_of_2(n_cols)

        _rmsnorm_fwd_kernel[(n_rows,)](
            x_2d,
            w_1d,
            y_2d,
            rrms,
            x_2d.stride(0),
            y_2d.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(x_2d, w_1d, rrms)
        ctx.block_size = block_size
        ctx.orig_shape = orig_shape
        return y_2d.view(orig_shape)

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x_2d, w_1d, rrms = ctx.saved_tensors
        n_rows, n_cols = x_2d.shape
        dy_2d = dy.contiguous().view(n_rows, n_cols)

        dx_2d = torch.empty_like(x_2d)
        dw_partial = torch.empty((n_rows, n_cols), dtype=torch.float32, device=x_2d.device)

        _rmsnorm_bwd_dx_kernel[(n_rows,)](
            dy_2d,
            x_2d,
            w_1d,
            rrms,
            dx_2d,
            dw_partial,
            x_2d.stride(0),
            n_cols,
            BLOCK_SIZE=ctx.block_size,
        )
        dw = dw_partial.sum(dim=0).to(dtype=w_1d.dtype)
        return dx_2d.view(ctx.orig_shape), dw, None


class TritonSwiGLUFunction(torch.autograd.Function):
    """Custom autograd Function dispatching forward/backward SwiGLU to @triton.jit kernels."""

    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        out = torch.empty_like(gate_c)
        n_elements = gate_c.numel()
        block_size = 1024
        grid = ((n_elements + block_size - 1) // block_size,)

        _swiglu_fwd_kernel[grid](
            gate_c,
            up_c,
            out,
            n_elements,
            BLOCK_SIZE=block_size,
        )
        ctx.save_for_backward(gate_c, up_c)
        ctx.block_size = block_size
        return out

    @staticmethod
    def backward(ctx: Any, dout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_c, up_c = ctx.saved_tensors
        dout_c = dout.contiguous()
        d_gate = torch.empty_like(gate_c)
        d_up = torch.empty_like(up_c)
        n_elements = gate_c.numel()
        block_size = ctx.block_size
        grid = ((n_elements + block_size - 1) // block_size,)

        _swiglu_bwd_kernel[grid](
            dout_c,
            gate_c,
            up_c,
            d_gate,
            d_up,
            n_elements,
            BLOCK_SIZE=block_size,
        )
        return d_gate, d_up


# ---------------------------------------------------------------------------
# 4. Deterministic Tiled SRAM Block Simulators (for CPU/MPS Unit Testing)
# ---------------------------------------------------------------------------


def simulate_tiled_rmsnorm_fwd_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the exact Triton 1D row-program + power-of-2 BLOCK_SIZE mask equations on CPU/MPS.

    Verifies that our @triton.jit block indexing (`cols < N` padding with `0.0`) and analytical
    backward pass match PyTorch autograd to `< 1e-5` even when `d_model` is not a power of 2.
    """
    orig_shape = x.shape
    n_cols = orig_shape[-1]
    block_size = next_power_of_2(n_cols)
    pad = block_size - n_cols

    x_2d = x.detach().float().view(-1, n_cols)
    dy_2d = dy.detach().float().view(-1, n_cols)
    w_1d = weight.detach().float()

    # Simulate `tl.load(..., mask=cols < N, other=0.0)`
    x_blk = F.pad(x_2d, (0, pad), value=0.0)
    dy_blk = F.pad(dy_2d, (0, pad), value=0.0)
    w_blk = F.pad(w_1d, (0, pad), value=0.0)

    mean_sq = (x_blk * x_blk).sum(dim=-1, keepdim=True) / float(n_cols)
    rrms = torch.rsqrt(mean_sq + eps)
    y_blk = (x_blk * rrms) * w_blk

    # Analytical backward in block registers
    x_hat = x_blk * rrms
    wdy = w_blk * dy_blk
    c1 = (x_hat * wdy).sum(dim=-1, keepdim=True) / float(n_cols)
    dx_blk = rrms * (wdy - x_hat * c1)
    dw_blk = (dy_blk * x_hat).sum(dim=0)

    # Simulate `tl.store(..., mask=cols < N)`
    y = y_blk[:, :n_cols].to(dtype=x.dtype).view(orig_shape)
    dx = dx_blk[:, :n_cols].to(dtype=x.dtype).view(orig_shape)
    dw = dw_blk[:n_cols].to(dtype=weight.dtype)
    return y, dx, dw


def simulate_tiled_swiglu_fwd_bwd(
    gate: torch.Tensor,
    up: torch.Tensor,
    dout: torch.Tensor,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the exact Triton 1D block-tiled SwiGLU forward and backward math on CPU/MPS."""
    orig_shape = gate.shape
    n_elements = gate.numel()
    n_blocks = (n_elements + block_size - 1) // block_size
    padded_len = n_blocks * block_size
    pad = padded_len - n_elements

    g_flat = F.pad(gate.detach().float().reshape(-1), (0, pad), value=0.0)
    u_flat = F.pad(up.detach().float().reshape(-1), (0, pad), value=0.0)
    dout_flat = F.pad(dout.detach().float().reshape(-1), (0, pad), value=0.0)

    sig_g = torch.sigmoid(g_flat)
    silu_g = g_flat * sig_g
    out_flat = silu_g * u_flat

    d_up_flat = dout_flat * silu_g
    d_silu = sig_g * (1.0 + g_flat * (1.0 - sig_g))
    d_gate_flat = dout_flat * u_flat * d_silu

    out = out_flat[:n_elements].to(dtype=gate.dtype).view(orig_shape)
    d_gate = d_gate_flat[:n_elements].to(dtype=gate.dtype).view(orig_shape)
    d_up = d_up_flat[:n_elements].to(dtype=up.dtype).view(orig_shape)
    return out, d_gate, d_up


# ---------------------------------------------------------------------------
# 5. Public Dispatchers & HBM Bandwidth Calculators
# ---------------------------------------------------------------------------


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    backend: KernelBackend = "auto",
) -> torch.Tensor:
    """Apply RMSNorm using @triton.jit on CUDA or PyTorch reference on CPU/MPS."""
    if backend == "triton":
        if not is_triton_available(x.device):
            raise RuntimeError(
                "backend='triton' requires Triton and a CUDA tensor, "
                f"got device={x.device.type}, HAS_TRITON={HAS_TRITON}"
            )
        return TritonRMSNormFunction.apply(x, weight, eps)  # type: ignore[no-any-return]
    if backend == "auto" and is_triton_available(x.device):
        return TritonRMSNormFunction.apply(x, weight, eps)  # type: ignore[no-any-return]
    return ref_rmsnorm(x, weight, eps)


def fused_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    backend: KernelBackend = "auto",
) -> torch.Tensor:
    """Apply fused SwiGLU (`silu(gate) * up`) using @triton.jit on CUDA or PyTorch fallback."""
    if backend == "triton":
        if not is_triton_available(gate.device):
            raise RuntimeError(
                "backend='triton' requires Triton and a CUDA tensor, "
                f"got device={gate.device.type}, HAS_TRITON={HAS_TRITON}"
            )
        return TritonSwiGLUFunction.apply(gate, up)  # type: ignore[no-any-return]
    if backend == "auto" and is_triton_available(gate.device):
        return TritonSwiGLUFunction.apply(gate, up)  # type: ignore[no-any-return]
    return ref_swiglu(gate, up)


def compute_rmsnorm_hbm_bytes(
    num_tokens: int,
    d_model: int,
    bytes_per_elem: int = 2,
    include_backward: bool = False,
) -> int:
    """Compute theoretical minimum HBM traffic (bytes) for fused RMSNorm.

    - Forward reads `x` (`num_tokens * d_model`) + `weight` (`d_model`), writes `y`
      (`num_tokens * d_model`) plus `rrms` (`num_tokens * 4` bytes fp32).
    - Backward reads `dy`, `x`, `weight`, `rrms` and writes `dx`, `dw`.
    """
    fwd_bytes = (2 * num_tokens * d_model + d_model) * bytes_per_elem + (num_tokens * 4)
    if not include_backward:
        return fwd_bytes
    bwd_bytes = (3 * num_tokens * d_model + 2 * d_model) * bytes_per_elem + (num_tokens * 4)
    return fwd_bytes + bwd_bytes


def compute_swiglu_hbm_bytes(
    num_tokens: int,
    d_ff: int,
    bytes_per_elem: int = 2,
    include_backward: bool = False,
) -> int:
    """Compute theoretical minimum HBM traffic (bytes) for fused SwiGLU.

    - Forward reads `gate` and `up`, writes `out`: `3 * num_tokens * d_ff * bytes_per_elem`.
    - Backward reads `dout`, `gate`, `up`, and writes `d_gate`, `d_up`:
      `5 * num_tokens * d_ff * bytes_per_elem`.
    """
    fwd_bytes = 3 * num_tokens * d_ff * bytes_per_elem
    if not include_backward:
        return fwd_bytes
    bwd_bytes = 5 * num_tokens * d_ff * bytes_per_elem
    return fwd_bytes + bwd_bytes
