# Run Summary: `h7_h8_qwen3b_pareto`

- **Experiment:** `h7_h8_surgery_pareto` (`H7`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Qwen2.5-3B-Instruct` (`3,085,846,528` total / `2,774,681,600` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `04adb51fcbe177d0bd0e6b7ddfd7e1146dab6866` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Allocating 8-bit precision to the top-k highest CKA-drift layers and 4-bit to low-drift layers yields >20% lower KL divergence than an equal-bit bottom-k control (H7), while rank-1 subspace projection W' = W - r(r^T W) zeroes out targeted residual writes (<1e-5) with minimal collateral KL (H8).
- **Prediction:** cka_guided 4/8-bit allocation achieves >= 20% lower KL divergence vs adapted_model than bottom_k_control at identical average bits-per-weight (5.33 bits), and rank-1 subspace ablation reduces targeted residual write projection below 1e-5.
- **Acceptance Criterion:** `h7_verified == True AND h8_verified == True AND pareto_dominance_verified == True`

## Final Metrics
- **`h7_verified`:** `True`
- **`h7_kl_reduction_ratio`:** `0.301`
- **`h8_verified`:** `True`
- **`pareto_dominance_verified`:** `True`
- **`cka_guided_kl`:** `0.165203`
- **`bottom_k_control_kl`:** `0.236338`
- **`backend`:** `llm`
- **`model_name`:** `Qwen/Qwen2.5-3B-Instruct`
- **`tokens_processed`:** `51989`
- **`tokens_per_sec`:** `25.2`

## Compute & Cost
- **Active GPU-Hours:** `0.571972`
- **Billed GPU-Hours:** `0.571972`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
Surgery verification completed: H7_verified=True, H8_verified=True
