# Run Summary: `h2_mfu_waterfall`

- **Experiment:** `h2_mfu_waterfall` (`H2`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-100m` (`100,682,496` total / `75,516,672` non-emb)
- **Dataset:** `fineweb-edu-4b`
- **Hardware:** `nvidia-h100-sxm-80gb` (`$2.69/hr`)
- **Git SHA:** `uncommitted` (dirty: `False`)

## Hypothesis & Prediction
- **Hypothesis:** Cumulative systems optimizations (fp32 eager -> bf16 autocast -> SDPA/Flash -> fused AdamW -> torch.compile -> custom @triton.jit FusedRMSNorm & FusedSwiGLU kernels) yield monotonic throughput improvements and achieve >= 30% MFU on 1x H100 80GB SXM for sml-100m.
- **Prediction:** Throughput increases by >= 2.5x from fp32 eager baseline to compiled bf16 + SDPA + fused AdamW + custom @triton.jit kernels, reaching >= 30% MFU (~415,000 tok/s on H100 SXM).
- **Acceptance Criterion:** `measured_mfu >= 0.30 and speedup_vs_eager_fp32 >= 2.5`

## Final Metrics
- **`stages_profiled`:** `5`
- **`baseline_tokens_per_sec`:** `34850.26`
- **`final_tokens_per_sec`:** `472080.81`
- **`best_tokens_per_sec`:** `472080.81`
- **`final_mfu`:** `0.342236`
- **`best_mfu`:** `0.342236`
- **`max_speedup_vs_baseline`:** `13.546`
- **`tokens_processed`:** `6553600`
- **`tokens_per_sec`:** `472080.81`
- **`mfu`:** `0.342236`

## Compute & Cost
- **Active GPU-Hours:** `0.029385`
- **Billed GPU-Hours:** `0.029385`
- **Billed Cost (USD):** `$0.0790` (pre-run estimate: `$0.0947`)

## Conclusion
H2 MFU waterfall profiled across 5 stages: baseline=34850.3 tok/s -> best=472080.8 tok/s (13.55x speedup, MFU=0.3422).
