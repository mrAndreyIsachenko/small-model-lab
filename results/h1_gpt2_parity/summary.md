# Run Summary: `h1_gpt2_parity`

- **Experiment:** `h1_gpt2_parity` (`H1`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `gpt2-124m` (`124,439,808` total / `85,842,432` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `f122a0c97aceefe3e2a74746e01c074615e4d69e` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Our custom Transformer implementation in gpt2-compat mode is mathematically equivalent to Hugging Face GPT2LMHeadModel (124M), naive O(T^2) GQA matches SDPA, KV-cache decoding matches full-sequence forward logits, and intra-document masking prevents cross-document gradient leakage across <|eos|> boundaries.
- **Prediction:** Max absolute logit difference < 1.0e-4 in fp32 across random and natural prompts; future-token and cross-document gradient norm == 0.0.
- **Acceptance Criterion:** `max_abs_logit_diff < 1.0e-4 and cross_doc_grad_norm == 0.0`

## Final Metrics
- **`h1_passed`:** `True`
- **`max_abs_logit_diff_sdpa`:** `0.0`
- **`max_abs_logit_diff_naive`:** `5.96e-06`
- **`max_diff_sdpa_vs_naive`:** `5.96e-06`
- **`max_diff_kv_cache`:** `5.13e-06`
- **`cross_doc_leakage`:** `0.0`
- **`verified_total_params`:** `124439808`
- **`tokens_processed`:** `128`
- **`tokens_per_sec`:** `30.3`

## Compute & Cost
- **Active GPU-Hours:** `0.001174`
- **Billed GPU-Hours:** `0.001174`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H1 verified on full 124,439,808-parameter GPT-2 124M: max_abs_logit_diff=0.00e+00 (< 1e-4), sdpa_vs_naive=5.96e-06, kv_cache_diff=5.13e-06, cross_doc_leakage=0.00e+00.
