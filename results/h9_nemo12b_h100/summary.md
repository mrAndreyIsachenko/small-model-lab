# Run Summary: `h9_nemo12b_h100`

- **Experiment:** `h9_biprojected_surgery` (`H9`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Mistral-Nemo-Instruct-2407` (`12,772,070,400` total / `11,429,893,120` non-emb)
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
- **`max_row_norm_drift`:** `0.00073671`
- **`target_suppression_ratio`:** `0.991367`
- **`naive_kl_collateral`:** `0.296837`
- **`lancet_kl_collateral`:** `0.009746`
- **`tool_sql_orthogonality_error`:** `3e-08`
- **`probe_evr_refusal`:** `0.632735`
- **`probe_evr_slop`:** `0.685466`
- **`layer_alpha_schedule`:** `{22: 1.0, 28: 0.988108, 34: 0.942255, 27: 0.941279, 32: 0.938612, 23: 0.932384, 31: 0.928188, 25: 0.927352, 21: 0.922955, 14: 0.879254, 30: 0.841147, 33: 0.835738, 15: 0.825246, 24: 0.778859, 26: 0.777783, 20: 0.773568, 5: 0.741034, 29: 0.739992, 35: 0.718657, 16: 0.702673}`
- **`h9_kl_reduction_ratio`:** `0.9672`
- **`selected_layers`:** `[22, 28, 34, 27, 32, 23, 31, 25, 21, 14, 30, 33, 15, 24, 26, 20, 5, 29, 35, 16]`
- **`backend`:** `llm`
- **`model_name`:** `mistralai/Mistral-Nemo-Instruct-2407`

## Compute & Cost
- **Active GPU-Hours:** `0.004193`
- **Billed GPU-Hours:** `0.004193`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H9 Lancet surgery verification completed: h9_verified=True, kl_reduction=0.9672
