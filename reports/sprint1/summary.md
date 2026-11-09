# Sprint 1 Experimental Report (Auto-Generated)

> Generated strictly from machine-readable artifacts in `results` (`include_dry_runs=False`).

## Summary Table

| Run ID | Hyp. | Mode | Model | Non-Emb Params | Tokens | Val bpb | Tok/s | MFU | Hardware | Billed Cost ($) | Git SHA |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |
| `h1_gpt2_parity` | `H1` | `measured` | `gpt2-124m` | `85,842,432` | `128` | `n/a` | `30` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `f122a0c9` |
| `h5_env_verification` | `H5` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `7cc965d9` |
| `h6_adaptation_comparison` | `H6` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `a4dac8e2` |
| `h7_h8_surgery_pareto` | `H7` | `measured` | `sml-nano` | `3,148,032` | `0` | `n/a` | `0` | `n/a` | `apple-m4-pro-48gb` | `$0.0000` | `05414f1c` |

## Run Conclusions

### `h1_gpt2_parity` (`H1` — `measured`)
- **Prediction:** Max absolute logit difference < 1.0e-4 in fp32 across random and natural prompts; future-token and cross-document gradient norm == 0.0.
- **Conclusion:** H1 verified on full 124,439,808-parameter GPT-2 124M: max_abs_logit_diff=0.00e+00 (< 1e-4), sdpa_vs_naive=5.96e-06, kv_cache_diff=5.13e-06, cross_doc_leakage=0.00e+00.

### `h5_env_verification` (`H5` — `measured`)
- **Prediction:** Oracle policy achieves 100% evidence coverage and > 1.0 mean gated reward across all 12 counterfactual episodes; lazy-guess and unsafe-SQL policies score exactly 0.0 under Gated Multiplicative but > 0.40 under Naive Additive.
- **Conclusion:** H5 environment verification PASSED: oracle_gated=1.1200, lazy_gated=0.0000, unsafe_gated=0.0000, lazy_naive=0.7500

### `h6_adaptation_comparison` (`H6` — `measured`)
- **Prediction:** 3_hybrid_rag achieves >= 0.85 knowledge_bound_score, while 6_grpo_gated_rl outperforms 3_hybrid_rag by >= 0.20 on skill_bound_score (G_evidence * R_correctness) and outperforms 6b_grpo_naive_additive by >= 0.15 (H5).
- **Conclusion:** Adaptation comparison completed: H5_verified=True, H6_verified=True

### `h7_h8_surgery_pareto` (`H7` — `measured`)
- **Prediction:** cka_guided 4/8-bit allocation achieves >= 20% lower KL divergence vs adapted_model than bottom_k_control at identical average bits-per-weight (5.33 bits), and rank-1 subspace ablation reduces targeted residual write projection below 1e-5.
- **Conclusion:** Surgery verification completed: H7_verified=True, H8_verified=True
