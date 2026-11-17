# Run Summary: `h11_llama8b_h100`

- **Experiment:** `h11_provider_fp8_readiness` (`H11`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Llama-3.1-8B-Instruct` (`8,030,261,248` total / `6,979,588,096` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `uncommitted` (dirty: `False`)

## Hypothesis & Prediction
- **Hypothesis:** Lancet bi-projected norm-preserving surgery + gated unslop GRPO produces Tenet checkpoints that survive vLLM server-side FP8 (E4M3) quantization with >= 96% quality retention (fp8_retention_ratio >= 0.96) and >= 25% lower FP8 KL degradation than naive ablation without row-norm preservation (h11_fp8_kl_reduction_ratio >= 0.25), while passing strict OpenRouter provider readiness checks (standard LlamaForCausalLM/Qwen2ForCausalLM/MistralForCausalLM architecture, zero trust_remote_code/auto_map, valid chat_template, no NaN/Inf or unmerged lora_ keys, max_abs_weight < 50.0) and generating HF Hub model cards for the Tenet lineup.
- **Prediction:** Under FP8 (E4M3) per-channel server quantization, Lancet norm-preserving surgery achieves fp8_retention_ratio >= 0.96 and h11_fp8_kl_reduction_ratio >= 0.25 vs naive ablation, verify_provider_readiness passes all 6 checks on a standard bundle, and model cards are generated for Tenet-3B-Qwen2.5-Unslop-v1, Tenet-8B-Llama3.1-Unslop-v1, and Tenet-12B-Nemo-Unslop-v1.
- **Acceptance Criterion:** `fp8_retention_ratio >= 0.96 AND h11_fp8_kl_reduction_ratio >= 0.25 AND provider_ready == True AND h11_verified == True`

## Final Metrics
- **`h11_verified`:** `True`
- **`fp8_retention_ratio`:** `0.9935780763626099`
- **`h11_fp8_kl_reduction_ratio`:** `0.4265360257965603`
- **`kl_naive_fp8`:** `0.032248564064502716`
- **`kl_lancet_fp8`:** `0.01849338971078396`
- **`fp8_subspace_releakage_naive`:** `0.5000227482473887`
- **`fp8_subspace_releakage_lancet`:** `0.02778303384129374`
- **`fp8_subspace_releakage_naive_per_tensor`:** `0.5000298591859549`
- **`fp8_subspace_releakage_lancet_per_tensor`:** `0.027781071648231535`
- **`kl_naive_fp8_per_tensor`:** `0.03702026605606079`
- **`kl_lancet_fp8_per_tensor`:** `0.01817736029624939`
- **`provider_ready`:** `True`
- **`readiness_checks`:** `{'standard_architecture': True, 'no_trust_remote_code': True, 'valid_chat_template': True, 'no_nan_inf': True, 'no_unmerged_lora': True, 'weight_outliers_ok': True, 'acceptable_row_crest_factor': True}`
- **`max_abs_weight`:** `0.9140625`
- **`max_row_crest_factor`:** `6.324390411376953`
- **`backend`:** `llm`
- **`model_name`:** `meta-llama/Llama-3.1-8B-Instruct`
- **`model_cards`:** `['Tenet-3B-Qwen2.5-Unslop-v1', 'Tenet-8B-Llama3.1-Unslop-v1', 'Tenet-12B-Nemo-Unslop-v1']`

## Compute & Cost
- **Active GPU-Hours:** `0.019987`
- **Billed GPU-Hours:** `0.019987`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H11 FP8 provider readiness verification completed: h11_verified=True, fp8_retention_ratio=0.9936, provider_ready=True
