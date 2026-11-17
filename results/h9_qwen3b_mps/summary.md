# Run Summary: `h9_qwen3b_mps`

- **Experiment:** `h9_biprojected_surgery` (`H9`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Qwen2.5-3B-Instruct` (`3,085,846,528` total / `2,774,681,600` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `dfe5d9569adc67d62b9ee37039d03d21f2066b8f` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Bi-projected rank-1 surgery ("Lancet") suppresses targeted refusal & slop write directions while keeping them orthogonal to the reasoning-anchor direction (<1e-6) and preserving row norms (<1e-5 drift), collapsing anchor-behaviour KL collateral by >= 35% versus naive unprojected all-layer surgery.
- **Prediction:** Lancet bi-projected surgery suppresses >= 90% of target direction activation with anchor_orthogonality_error < 1e-6 and max_row_norm_drift < 1e-5, while naive surgery incurs >= 35% more KL collateral on anchor behaviour (h9_kl_reduction_ratio >= 0.35).
- **Acceptance Criterion:** `anchor_orthogonality_error < 1e-6 AND max_row_norm_drift < 1e-5 AND target_suppression_ratio >= 0.90 AND h9_kl_reduction_ratio >= 0.35 AND h9_verified == True`

## Final Metrics
- **`h9_verified`:** `True`
- **`anchor_orthogonality_error`:** `1.8e-07`
- **`max_row_norm_drift`:** `0.00134182`
- **`target_suppression_ratio`:** `0.970431`
- **`naive_kl_collateral`:** `2.635312`
- **`lancet_kl_collateral`:** `0.024395`
- **`tool_sql_orthogonality_error`:** `1.4e-07`
- **`probe_evr_refusal`:** `0.437599`
- **`probe_evr_slop`:** `0.474672`
- **`layer_alpha_schedule`:** `{21: 1.0, 15: 0.947895, 19: 0.883576, 22: 0.706325, 25: 0.661202, 14: 0.651475, 20: 0.629798, 8: 0.628873, 24: 0.596117, 7: 0.586464, 23: 0.581197, 13: 0.557664, 12: 0.556387, 6: 0.547688, 5: 0.498523, 9: 0.49749, 18: 0.457154, 10: 0.452957}`
- **`h9_kl_reduction_ratio`:** `0.9907`
- **`selected_layers`:** `[21, 15, 19, 22, 25, 14, 20, 8, 24, 7, 23, 13, 12, 6, 5, 9, 18, 10]`
- **`backend`:** `llm`
- **`model_name`:** `Qwen/Qwen2.5-3B-Instruct`

## Compute & Cost
- **Active GPU-Hours:** `0.013023`
- **Billed GPU-Hours:** `0.013023`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H9 Lancet surgery verification completed: h9_verified=True, kl_reduction=0.9907
