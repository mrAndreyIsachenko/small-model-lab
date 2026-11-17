# Run Summary: `h11_nemo12b_h100`

- **Experiment:** `h11_provider_fp8_readiness` (`H11`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Mistral-Nemo-Instruct-2407` (`12,772,070,400` total / `11,429,893,120` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `uncommitted` (dirty: `False`)

## Hypothesis & Prediction
- **Hypothesis:** Lancet bi-projected norm-preserving surgery + gated unslop GRPO produces Tenet checkpoints that survive vLLM server-side FP8 (E4M3) quantization with >= 96% quality retention (fp8_retention_ratio >= 0.96) and >= 25% lower FP8 KL degradation than naive ablation without row-norm preservation (h11_fp8_kl_reduction_ratio >= 0.25), while passing strict OpenRouter provider readiness checks (standard LlamaForCausalLM/Qwen2ForCausalLM/MistralForCausalLM architecture, zero trust_remote_code/auto_map, valid chat_template, no NaN/Inf or unmerged lora_ keys, max_abs_weight < 50.0) and generating HF Hub model cards for the Tenet lineup.
- **Prediction:** Under FP8 (E4M3) per-channel server quantization, Lancet norm-preserving surgery achieves fp8_retention_ratio >= 0.96 and h11_fp8_kl_reduction_ratio >= 0.25 vs naive ablation, verify_provider_readiness passes all 6 checks on a standard bundle, and model cards are generated for Tenet-3B-Qwen2.5-Unslop-v1, Tenet-8B-Llama3.1-Unslop-v1, and Tenet-12B-Nemo-Unslop-v1.
- **Acceptance Criterion:** `fp8_retention_ratio >= 0.96 AND h11_fp8_kl_reduction_ratio >= 0.25 AND provider_ready == True AND h11_verified == True`

## Final Metrics
- **`h11_verified`:** `True`
- **`fp8_retention_ratio`:** `0.9678030014038086`
- **`h11_fp8_kl_reduction_ratio`:** `0.4892105870710418`
- **`kl_naive_fp8`:** `3.172443389892578`
- **`kl_lancet_fp8`:** `1.620450496673584`
- **`fp8_subspace_releakage_naive`:** `0.499638818799388`
- **`fp8_subspace_releakage_lancet`:** `0.183885410347825`
- **`fp8_subspace_releakage_naive_per_tensor`:** `0.4997547968103573`
- **`fp8_subspace_releakage_lancet_per_tensor`:** `0.18395628866969374`
- **`kl_naive_fp8_per_tensor`:** `3.1780343055725098`
- **`kl_lancet_fp8_per_tensor`:** `1.6264135837554932`
- **`provider_ready`:** `True`
- **`readiness_checks`:** `{'standard_architecture': True, 'no_trust_remote_code': True, 'valid_chat_template': True, 'no_nan_inf': True, 'no_unmerged_lora': True, 'weight_outliers_ok': True, 'acceptable_row_crest_factor': True}`
- **`max_abs_weight`:** `0.91796875`
- **`max_row_crest_factor`:** `10.65080738067627`
- **`backend`:** `llm`
- **`model_name`:** `mistralai/Mistral-Nemo-Instruct-2407`
- **`model_cards`:** `['Tenet-3B-Qwen2.5-Unslop-v1', 'Tenet-8B-Llama3.1-Unslop-v1', 'Tenet-12B-Nemo-Unslop-v1']`

## Compute & Cost
- **Active GPU-Hours:** `0.030211`
- **Billed GPU-Hours:** `0.030211`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H11 FP8 provider readiness verification completed: h11_verified=True, fp8_retention_ratio=0.9678, provider_ready=True
