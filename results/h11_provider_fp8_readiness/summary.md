# Run Summary: `h11_provider_fp8_readiness`

- **Experiment:** `h11_provider_fp8_readiness` (`H11`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-nano` (`11,536,640` total / `3,148,032` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `dfe5d9569adc67d62b9ee37039d03d21f2066b8f` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Lancet bi-projected norm-preserving surgery + gated unslop GRPO produces Tenet checkpoints that survive vLLM server-side FP8 (E4M3) quantization with >= 96% quality retention (fp8_retention_ratio >= 0.96) and >= 25% lower FP8 KL degradation than naive ablation without row-norm preservation (h11_fp8_kl_reduction_ratio >= 0.25), while passing strict OpenRouter provider readiness checks (standard LlamaForCausalLM/Qwen2ForCausalLM/MistralForCausalLM architecture, zero trust_remote_code/auto_map, valid chat_template, no NaN/Inf or unmerged lora_ keys, max_abs_weight < 50.0) and generating HF Hub model cards for the Tenet lineup.
- **Prediction:** Under FP8 (E4M3) per-channel server quantization, Lancet norm-preserving surgery achieves fp8_retention_ratio >= 0.96 and h11_fp8_kl_reduction_ratio >= 0.25 vs naive ablation, verify_provider_readiness passes all 6 checks on a standard bundle, and model cards are generated for Tenet-3B-Qwen2.5-Unslop-v1, Tenet-8B-Llama3.1-Unslop-v1, and Tenet-12B-Nemo-Unslop-v1.
- **Acceptance Criterion:** `fp8_retention_ratio >= 0.96 AND h11_fp8_kl_reduction_ratio >= 0.25 AND provider_ready == True AND h11_verified == True`

## Final Metrics
- **`h11_verified`:** `True`
- **`fp8_retention_ratio`:** `0.9997473955154419`
- **`h11_fp8_kl_reduction_ratio`:** `0.4558221147859871`
- **`kl_naive_fp8`:** `0.0004642039421014488`
- **`kl_lancet_fp8`:** `0.0002526095195207745`
- **`fp8_subspace_releakage_naive`:** `0.4995908019129473`
- **`fp8_subspace_releakage_lancet`:** `0.03103304917040631`
- **`fp8_subspace_releakage_naive_per_tensor`:** `0.4999280722916585`
- **`fp8_subspace_releakage_lancet_per_tensor`:** `0.03027544454590196`
- **`kl_naive_fp8_per_tensor`:** `0.0004566615098156035`
- **`kl_lancet_fp8_per_tensor`:** `0.000248587952228263`
- **`provider_ready`:** `True`
- **`readiness_checks`:** `{'standard_architecture': True, 'no_trust_remote_code': True, 'valid_chat_template': True, 'no_nan_inf': True, 'no_unmerged_lora': True, 'weight_outliers_ok': True, 'acceptable_row_crest_factor': True}`
- **`max_abs_weight`:** `0.47617220878601074`
- **`max_row_crest_factor`:** `1.8583072423934937`
- **`backend`:** `nano`
- **`model_name`:** `Qwen/Qwen2.5-3B-Instruct`
- **`model_cards`:** `['Tenet-3B-Qwen2.5-Unslop-v1', 'Tenet-8B-Llama3.1-Unslop-v1', 'Tenet-12B-Nemo-Unslop-v1']`

## Compute & Cost
- **Active GPU-Hours:** `4.7e-05`
- **Billed GPU-Hours:** `4.7e-05`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H11 FP8 provider readiness verification completed: h11_verified=True, fp8_retention_ratio=0.9997, provider_ready=True
