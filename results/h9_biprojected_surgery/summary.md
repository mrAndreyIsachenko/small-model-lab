# Run Summary: `h9_biprojected_surgery`

- **Experiment:** `h9_biprojected_surgery` (`H9`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-nano` (`11,536,640` total / `3,148,032` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `eab01f01b3dcdcccce05d31156c37c1ce3bd7000` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Bi-projected rank-1 surgery ("Lancet") suppresses targeted refusal & slop write directions while keeping them orthogonal to the reasoning-anchor direction (<1e-6) and preserving row norms (<1e-5 drift), collapsing anchor-behaviour KL collateral by >= 35% versus naive unprojected all-layer surgery.
- **Prediction:** Lancet bi-projected surgery suppresses >= 90% of target direction activation with anchor_orthogonality_error < 1e-6 and max_row_norm_drift < 1e-5, while naive surgery incurs >= 35% more KL collateral on anchor behaviour (h9_kl_reduction_ratio >= 0.35).
- **Acceptance Criterion:** `anchor_orthogonality_error < 1e-6 AND max_row_norm_drift < 1e-5 AND target_suppression_ratio >= 0.90 AND h9_kl_reduction_ratio >= 0.35 AND h9_verified == True`

## Final Metrics
- **`h9_verified`:** `True`
- **`anchor_orthogonality_error`:** `6e-08`
- **`max_row_norm_drift`:** `2e-08`
- **`target_suppression_ratio`:** `0.98542`
- **`naive_kl_collateral`:** `0.001317`
- **`lancet_kl_collateral`:** `0.000515`
- **`tool_sql_orthogonality_error`:** `4e-08`
- **`probe_evr_refusal`:** `0.215511`
- **`probe_evr_slop`:** `0.279798`
- **`layer_alpha_schedule`:** `{5: 1.0, 4: 0.979185, 0: 0.829126}`
- **`h9_kl_reduction_ratio`:** `0.6092`
- **`selected_layers`:** `[5, 4, 0]`
- **`backend`:** `nano`
- **`model_name`:** `sml-surgery-6l`

## Compute & Cost
- **Active GPU-Hours:** `0.000145`
- **Billed GPU-Hours:** `0.000145`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H9 Lancet surgery verification completed: h9_verified=True, kl_reduction=0.6092
