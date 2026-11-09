# Run Summary: `h7_h8_surgery_pareto`

- **Experiment:** `h7_h8_surgery_pareto` (`H7`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-nano` (`11,536,640` total / `3,148,032` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `05414f1c53740116c42af299e2051f5828fc5245` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Allocating 8-bit precision to the top-k highest CKA-drift layers and 4-bit to low-drift layers yields >20% lower KL divergence than an equal-bit bottom-k control (H7), while rank-1 subspace projection W' = W - r(r^T W) zeroes out targeted residual writes (<1e-5) with minimal collateral KL (H8).
- **Prediction:** cka_guided 4/8-bit allocation achieves >= 20% lower KL divergence vs adapted_model than bottom_k_control at identical average bits-per-weight (5.33 bits), and rank-1 subspace ablation reduces targeted residual write projection below 1e-5.
- **Acceptance Criterion:** `h7_verified == True AND h8_verified == True AND pareto_dominance_verified == True`

## Final Metrics
- **`h7_verified`:** `True`
- **`h7_kl_reduction_ratio`:** `0.7333`
- **`h8_verified`:** `True`
- **`pareto_dominance_verified`:** `True`
- **`cka_guided_kl`:** `0.005168`
- **`bottom_k_control_kl`:** `0.019381`
- **`tokens_processed`:** `0`
- **`tokens_per_sec`:** `0`

## Compute & Cost
- **Active GPU-Hours:** `6.8e-05`
- **Billed GPU-Hours:** `6.8e-05`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
Surgery verification completed: H7_verified=True, H8_verified=True
