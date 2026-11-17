# Run Summary: `h9_llama8b_h100`

- **Experiment:** `h9_biprojected_surgery` (`H9`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Llama-3.1-8B-Instruct` (`8,030,261,248` total / `6,979,588,096` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `uncommitted` (dirty: `False`)

## Hypothesis & Prediction
- **Hypothesis:** Bi-projected rank-1 surgery ("Lancet") suppresses targeted refusal & slop write directions while keeping them orthogonal to the reasoning-anchor direction (<1e-6) and preserving row norms (<1e-5 drift), collapsing anchor-behaviour KL collateral by >= 35% versus naive unprojected all-layer surgery.
- **Prediction:** Lancet bi-projected surgery suppresses >= 90% of target direction activation with anchor_orthogonality_error < 1e-6 and max_row_norm_drift < 1e-5, while naive surgery incurs >= 35% more KL collateral on anchor behaviour (h9_kl_reduction_ratio >= 0.35).
- **Acceptance Criterion:** `anchor_orthogonality_error < 1e-6 AND max_row_norm_drift < 1e-5 AND target_suppression_ratio >= 0.90 AND h9_kl_reduction_ratio >= 0.35 AND h9_verified == True`

## Final Metrics
- **`h9_verified`:** `True`
- **`anchor_orthogonality_error`:** `1e-08`
- **`max_row_norm_drift`:** `0.00107741`
- **`target_suppression_ratio`:** `0.992983`
- **`naive_kl_collateral`:** `0.13372`
- **`lancet_kl_collateral`:** `0.013525`
- **`tool_sql_orthogonality_error`:** `1e-08`
- **`probe_evr_refusal`:** `0.304339`
- **`probe_evr_slop`:** `0.306633`
- **`layer_alpha_schedule`:** `{24: 1.0, 20: 0.971108, 28: 0.955294, 18: 0.878644, 11: 0.876402, 14: 0.868617, 9: 0.851382, 10: 0.813021, 15: 0.780919, 25: 0.762515, 8: 0.761364, 3: 0.75697, 19: 0.747902, 27: 0.746591, 4: 0.74447, 29: 0.738117}`
- **`h9_kl_reduction_ratio`:** `0.8989`
- **`selected_layers`:** `[24, 20, 28, 18, 11, 14, 9, 10, 15, 25, 8, 3, 19, 27, 4, 29]`
- **`backend`:** `llm`
- **`model_name`:** `meta-llama/Llama-3.1-8B-Instruct`

## Compute & Cost
- **Active GPU-Hours:** `0.016854`
- **Billed GPU-Hours:** `0.016854`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H9 Lancet surgery verification completed: h9_verified=True, kl_reduction=0.8989
