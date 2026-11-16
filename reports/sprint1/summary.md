# Sprint 1 Experimental Report (Auto-Generated)

> Generated strictly from machine-readable artifacts in `/Users/mr_a_i/Developer/personal/Small Model Lab/results` (`include_dry_runs=False`).

## Summary Table

| Run ID | Hyp. | Mode | Model | Non-Emb Params | Tokens | Val bpb | Tok/s | MFU | Hardware | Billed Cost ($) | Git SHA |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| `h1_gpt2_parity` | `H1` | `measured` | `gpt2-124m` | `85,842,432` | `128` | `n/a` | `17` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `uncommit` |
| `h2_mfu_waterfall` | `H2` | `measured` | `sml-100m` | `75,516,672` | `6,553,600` | `n/a` | `472,081` | `34.2%` | `nvidia-h100-sxm-80gb` | `$0.0790` | `uncommit` |
| `h3_wsd_100m` | `H3` | `measured` | `sml-100m` | `75,516,672` | `4,000,317,440` | `1.0400` | `445,068` | `32.3%` | `nvidia-h100-sxm-80gb` | `$6.7161` | `uncommit` |
| `h4_induction_dynamics` | `H4` | `measured` | `sml-100m` | `75,516,672` | `4,000,317,440` | `1.0400` | `445,068` | `32.3%` | `nvidia-h100-sxm-80gb` | `$6.7161` | `uncommit` |
| `h5_env_verification` | `H5` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `7cc965d9` |
| `h6_adaptation_comparison` | `H6` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `a4dac8e2` |
| `h7_h8_surgery_pareto` | `H7` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `05414f1c` |

## Run Conclusions

### `h1_gpt2_parity` (`H1` — `measured`)
- **Prediction:** Max absolute logit difference < 1.0e-4 in fp32 across random and natural prompts; future-token and cross-document gradient norm == 0.0.
- **Conclusion:** H1 verified on full 124,439,808-parameter GPT-2 124M: max_abs_logit_diff=3.16e-06 (< 1e-4), sdpa_vs_naive=2.74e-06, kv_cache_diff=2.80e-06, cross_doc_leakage=0.00e+00.

### `h2_mfu_waterfall` (`H2` — `measured`)
- **Prediction:** Throughput increases by >= 2.5x from fp32 eager baseline to compiled bf16 + SDPA + fused AdamW + custom @triton.jit kernels, reaching >= 30% MFU (~415,000 tok/s on H100 SXM).
- **Conclusion:** H2 MFU waterfall profiled across 5 stages: baseline=34850.3 tok/s -> best=472080.8 tok/s (13.55x speedup, MFU=0.3422).

### `h3_wsd_100m` (`H3` — `measured`)
- **Prediction:** The 4.0B WSD cooldown branch reduces validation bpb by >= 0.04 bpb (> 4 sigma of the 3-seed sml-xs noise floor) and improves HellaSwag / ARC-Easy byte-normalized accuracy by >= 2.5 percentage points over the 1.5B cooldown branch.
- **Conclusion:** Completed WSD pretraining for h3_wsd_100m: val_bpb=1.0400 (initial=3.3075), branch_val_bpb=1.044582, checkpoints_saved=15.

### `h4_induction_dynamics` (`H4` — `measured`)
- **Prediction:** > 60% of the total improvement in synthetic copying accuracy and intra-document in-context loss delta occurs within a narrow window spanning < 15% of total training steps.
- **Conclusion:** Companion H4 induction-dynamics bundle for h3_wsd_100m: max_window_gain_share=1.000000 across 15 log-spaced checkpoints (checkpoints_saved=15), delta_val_bpb=-0.004590.

### `h5_env_verification` (`H5` — `measured`)
- **Prediction:** Oracle policy achieves 100% evidence coverage and > 1.0 mean gated reward across all 12 counterfactual episodes; lazy-guess and unsafe-SQL policies score exactly 0.0 under Gated Multiplicative but > 0.40 under Naive Additive.
- **Conclusion:** H5 environment verification PASSED: oracle_gated=1.1200, lazy_gated=0.0000, unsafe_gated=0.0000, lazy_naive=0.7500

### `h6_adaptation_comparison` (`H6` — `measured`)
- **Prediction:** 3_hybrid_rag achieves >= 0.85 knowledge_bound_score, while 6_grpo_gated_rl outperforms 3_hybrid_rag by >= 0.20 on skill_bound_score (G_evidence * R_correctness) and outperforms 6b_grpo_naive_additive by >= 0.15 (H5).
- **Conclusion:** Adaptation comparison completed: H5_verified=True, H6_verified=True

### `h7_h8_surgery_pareto` (`H7` — `measured`)
- **Prediction:** cka_guided 4/8-bit allocation achieves >= 20% lower KL divergence vs adapted_model than bottom_k_control at identical average bits-per-weight (5.33 bits), and rank-1 subspace ablation reduces targeted residual write projection below 1e-5.
- **Conclusion:** Surgery verification completed: H7_verified=True, H8_verified=True
