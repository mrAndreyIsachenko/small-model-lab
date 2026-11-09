# Run Summary: `h5_env_verification`

- **Experiment:** `h5_env_verification` (`H5`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-nano` (`11,536,640` total / `3,148,032` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `7cc965d9e9f90fe8464974bc9ec972d001feaf80` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** A Gated Multiplicative Rubric (R = G_safety * G_complete * R_correctness * G_evidence * (1 + alpha * R_efficiency)) on multi-seed counterfactual DuckDB scenarios eliminates reward hacking by lazy/guessing and unsafe SQL policies (scoring 0.0) while rewarding verified analytical trajectories (> 1.0), whereas a Naive Additive Rubric rewards unverified guessing (> 0.45).
- **Prediction:** Oracle policy achieves 100% evidence coverage and > 1.0 mean gated reward across all 12 counterfactual episodes; lazy-guess and unsafe-SQL policies score exactly 0.0 under Gated Multiplicative but > 0.40 under Naive Additive.
- **Acceptance Criterion:** `oracle_gated_reward > 1.0 AND lazy_guess_gated_reward == 0.0 AND unsafe_sql_gated_reward == 0.0 AND lazy_guess_naive_reward > 0.40`

## Final Metrics
- **`h5_passed`:** `True`
- **`oracle_gated_mean`:** `1.12`
- **`oracle_naive_mean`:** `0.98`
- **`lazy_gated_mean`:** `0.0`
- **`lazy_naive_mean`:** `0.75`
- **`unsafe_gated_mean`:** `0.0`
- **`unsafe_naive_mean`:** `0.55`
- **`episodes_run`:** `12`
- **`tokens_processed`:** `0`
- **`tokens_per_sec`:** `0`

## Compute & Cost
- **Active GPU-Hours:** `0.000121`
- **Billed GPU-Hours:** `0.000121`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H5 environment verification PASSED: oracle_gated=1.1200, lazy_gated=0.0000, unsafe_gated=0.0000, lazy_naive=0.7500
