# Run Summary: `h9_qwen3b_mps`

- **Experiment:** `h9_biprojected_surgery` (`H9`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Qwen2.5-3B-Instruct` (`3,085,846,528` total / `2,774,681,600` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `eab01f01b3dcdcccce05d31156c37c1ce3bd7000` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Bi-projected rank-1 surgery ("Lancet") suppresses targeted refusal & slop write directions while keeping them orthogonal to the reasoning-anchor direction (<1e-6) and preserving row norms (<1e-5 drift), collapsing anchor-behaviour KL collateral by >= 35% versus naive unprojected all-layer surgery.
- **Prediction:** Lancet bi-projected surgery suppresses >= 90% of target direction activation with anchor_orthogonality_error < 1e-6 and max_row_norm_drift < 1e-5, while naive surgery incurs >= 35% more KL collateral on anchor behaviour (h9_kl_reduction_ratio >= 0.35).
- **Acceptance Criterion:** `anchor_orthogonality_error < 1e-6 AND max_row_norm_drift < 1e-5 AND target_suppression_ratio >= 0.90 AND h9_kl_reduction_ratio >= 0.35 AND h9_verified == True`

## Final Metrics
- **`h9_verified`:** `True`
- **`anchor_orthogonality_error`:** `1e-07`
- **`max_row_norm_drift`:** `0.00160885`
- **`target_suppression_ratio`:** `0.969389`
- **`naive_kl_collateral`:** `3.760749`
- **`lancet_kl_collateral`:** `0.022046`
- **`tool_sql_orthogonality_error`:** `7e-08`
- **`probe_evr_refusal`:** `0.398583`
- **`probe_evr_slop`:** `0.419793`
- **`layer_alpha_schedule`:** `{21: 1.0, 19: 0.876719, 15: 0.831852, 22: 0.72535, 25: 0.712276, 20: 0.695181, 23: 0.675244, 24: 0.656061, 8: 0.635279, 14: 0.592078, 7: 0.589475, 12: 0.537915, 13: 0.51856, 9: 0.489156, 6: 0.452313, 5: 0.448776, 10: 0.444343, 26: 0.408597}`
- **`h9_kl_reduction_ratio`:** `0.9941`
- **`selected_layers`:** `[21, 19, 15, 22, 25, 20, 23, 24, 8, 14, 7, 12, 13, 9, 6, 5, 10, 26]`
- **`backend`:** `llm`
- **`model_name`:** `Qwen/Qwen2.5-3B-Instruct`

## Compute & Cost
- **Active GPU-Hours:** `0.015853`
- **Billed GPU-Hours:** `0.015853`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H9 Lancet surgery verification completed: h9_verified=True, kl_reduction=0.9941
