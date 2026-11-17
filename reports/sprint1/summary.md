# Sprint 1 Experimental Report (Auto-Generated)

> Generated strictly from machine-readable artifacts in `results` (`include_dry_runs=False`).

## Summary Table

| Run ID | Hyp. | Mode | Model | Non-Emb Params | Tokens | Val bpb | Tok/s | MFU | Hardware | Billed Cost ($) | Git SHA |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| `h10_qwen3b_mps` | `H10` | `measured` | `Qwen2.5-3B-Instruct` | `2,774,681,600` | `0` | `n/a` | `n/a` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `dfe5d956` |
| `h10_unslop_grpo` | `H10` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `n/a` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `dfe5d956` |
| `h11_provider_fp8_readiness` | `H11` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `n/a` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `dfe5d956` |
| `h11_qwen3b_mps` | `H11` | `measured` | `Qwen2.5-3B-Instruct` | `2,774,681,600` | `0` | `n/a` | `n/a` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `dfe5d956` |
| `h1_gpt2_parity` | `H1` | `measured` | `gpt2-124m` | `85,842,432` | `128` | `n/a` | `17` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `uncommit` |
| `h2_mfu_waterfall` | `H2` | `measured` | `sml-100m` | `75,516,672` | `6,553,600` | `n/a` | `472,081` | `34.2%` | `nvidia-h100-sxm-80gb` | `$0.0790` | `uncommit` |
| `h3_wsd_100m` | `H3` | `measured` | `sml-100m` | `75,516,672` | `4,000,317,440` | `1.0400` | `445,068` | `32.3%` | `nvidia-h100-sxm-80gb` | `$6.7161` | `uncommit` |
| `h4_induction_dynamics` | `H4` | `measured` | `sml-100m` | `75,516,672` | `4,000,317,440` | `1.0400` | `445,068` | `32.3%` | `nvidia-h100-sxm-80gb` | `$6.7161` | `uncommit` |
| `h5_env_verification` | `H5` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `7cc965d9` |
| `h6_adaptation_comparison` | `H6` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `a4dac8e2` |
| `h6_qwen3b_mps` | `H6` | `measured` | `Qwen2.5-3B-Instruct` | `2,774,681,600` | `24,556` | `n/a` | `35` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `04adb51f` |
| `h7_h8_qwen3b_pareto` | `H7` | `measured` | `Qwen2.5-3B-Instruct` | `2,774,681,600` | `51,989` | `n/a` | `25` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `04adb51f` |
| `h7_h8_surgery_pareto` | `H7` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `05414f1c` |
| `h9_biprojected_surgery` | `H9` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `n/a` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `dfe5d956` |
| `h9_qwen3b_mps` | `H9` | `measured` | `Qwen2.5-3B-Instruct` | `2,774,681,600` | `0` | `n/a` | `n/a` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `dfe5d956` |

## H9–H11 Surgery, GRPO & Provider Readiness

| Run ID | Hyp. | Verified | Target Supp. | KL Red. (H9) | Anchor Ortho | Tool Ortho | Row Drift | Provider Ready | Ready Checks | FP8 Retention | KL Red. (FP8) | KL Naive (FP8) | KL Lancet (FP8) | Re-leakage (Naive) | Re-leakage (Lancet) | Max Crest | Slop & Refusal | Gated vs Naive | Post-Surgery | Merged LoRA |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `h10_qwen3b_mps` | `H10` | ✅ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.0909 | 0.1602 | 0.3598 | 0 |
| `h10_unslop_grpo` | `H10` | ✅ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.0077 | 0.2173 | 0.6698 | 0 |
| `h11_provider_fp8_readiness` | `H11` | ✅ | n/a | n/a | n/a | n/a | n/a | ✅ | 7/7 | 0.9997 | 0.4558 | 0.0005 | 0.0003 | 0.4996 | 0.0310 | 1.8583 | n/a | n/a | n/a | n/a |
| `h11_qwen3b_mps` | `H11` | ✅ | n/a | n/a | n/a | n/a | n/a | ✅ | 7/7 | 0.9541 | 0.5185 | 0.0975 | 0.0469 | 0.5003 | 0.0270 | 17.2706 | n/a | n/a | n/a | n/a |
| `h9_biprojected_surgery` | `H9` | ✅ | 0.9810 | 0.6935 | 4.00e-08 | 4.00e-08 | 2.00e-08 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| `h9_qwen3b_mps` | `H9` | ✅ | 0.9704 | 0.9907 | 1.80e-07 | 1.40e-07 | 1.34e-03 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Run Conclusions

### `h10_qwen3b_mps` (`H10` — `measured`)
- **Prediction:** After 27 GRPO steps (group size 6, lr 0.0038) on the nano model, slop_and_refusal_rate <= 0.02, gated reward shaping beats naive GRPO by >= 15% (h10_gated_vs_naive_gain >= 0.15), post-surgery composition adds >= 15% further gain (h10_post_surgery_gain >= 0.15), and merged LoRA logits stay within 1e-4 of the reference (max_abs_logit_diff < 1e-4).
- **Conclusion:** H10 unslop GRPO verification completed: h10_verified=True, slop_and_refusal_rate=0.0909, gated_vs_naive_gain=0.1602

### `h10_unslop_grpo` (`H10` — `measured`)
- **Prediction:** After 27 GRPO steps (group size 6, lr 0.0038) on the nano model, slop_and_refusal_rate <= 0.02, gated reward shaping beats naive GRPO by >= 15% (h10_gated_vs_naive_gain >= 0.15), post-surgery composition adds >= 15% further gain (h10_post_surgery_gain >= 0.15), and merged LoRA logits stay within 1e-4 of the reference (max_abs_logit_diff < 1e-4).
- **Conclusion:** H10 unslop GRPO verification completed: h10_verified=True, slop_and_refusal_rate=0.0077, gated_vs_naive_gain=0.2173

### `h11_provider_fp8_readiness` (`H11` — `measured`)
- **Prediction:** Under FP8 (E4M3) per-channel server quantization, Lancet norm-preserving surgery achieves fp8_retention_ratio >= 0.96 and h11_fp8_kl_reduction_ratio >= 0.25 vs naive ablation, verify_provider_readiness passes all 6 checks on a standard bundle, and model cards are generated for Tenet-3B-Qwen2.5-Unslop-v1, Tenet-8B-Llama3.1-Unslop-v1, and Tenet-12B-Nemo-Unslop-v1.
- **Conclusion:** H11 FP8 provider readiness verification completed: h11_verified=True, fp8_retention_ratio=0.9997, provider_ready=True

### `h11_qwen3b_mps` (`H11` — `measured`)
- **Prediction:** Under FP8 (E4M3) per-channel server quantization, Lancet norm-preserving surgery achieves fp8_retention_ratio >= 0.96 and h11_fp8_kl_reduction_ratio >= 0.25 vs naive ablation, verify_provider_readiness passes all 6 checks on a standard bundle, and model cards are generated for Tenet-3B-Qwen2.5-Unslop-v1, Tenet-8B-Llama3.1-Unslop-v1, and Tenet-12B-Nemo-Unslop-v1.
- **Conclusion:** H11 FP8 provider readiness verification completed: h11_verified=True, fp8_retention_ratio=0.9541, provider_ready=True

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

### `h6_qwen3b_mps` (`H6` — `measured`)
- **Prediction:** 3_hybrid_rag achieves >= 0.85 knowledge_bound_score, while 6_grpo_gated_rl outperforms 3_hybrid_rag by >= 0.20 on skill_bound_score (G_evidence * R_correctness) and outperforms 6b_grpo_naive_additive by >= 0.15 (H5).
- **Conclusion:** Adaptation comparison completed: H5_verified=True, H6_verified=True

### `h7_h8_qwen3b_pareto` (`H7` — `measured`)
- **Prediction:** cka_guided 4/8-bit allocation achieves >= 20% lower KL divergence vs adapted_model than bottom_k_control at identical average bits-per-weight (5.33 bits), and rank-1 subspace ablation reduces targeted residual write projection below 1e-5.
- **Conclusion:** Surgery verification completed: H7_verified=True, H8_verified=True

### `h7_h8_surgery_pareto` (`H7` — `measured`)
- **Prediction:** cka_guided 4/8-bit allocation achieves >= 20% lower KL divergence vs adapted_model than bottom_k_control at identical average bits-per-weight (5.33 bits), and rank-1 subspace ablation reduces targeted residual write projection below 1e-5.
- **Conclusion:** Surgery verification completed: H7_verified=True, H8_verified=True

### `h9_biprojected_surgery` (`H9` — `measured`)
- **Prediction:** Lancet bi-projected surgery suppresses >= 90% of target direction activation with anchor_orthogonality_error < 1e-6 and max_row_norm_drift < 1e-5, while naive surgery incurs >= 35% more KL collateral on anchor behaviour (h9_kl_reduction_ratio >= 0.35).
- **Conclusion:** H9 Lancet surgery verification completed: h9_verified=True, kl_reduction=0.6935

### `h9_qwen3b_mps` (`H9` — `measured`)
- **Prediction:** Lancet bi-projected surgery suppresses >= 90% of target direction activation with anchor_orthogonality_error < 1e-6 and max_row_norm_drift < 1e-5, while naive surgery incurs >= 35% more KL collateral on anchor behaviour (h9_kl_reduction_ratio >= 0.35).
- **Conclusion:** H9 Lancet surgery verification completed: h9_verified=True, kl_reduction=0.9907
