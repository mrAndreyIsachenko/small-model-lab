# Run Summary: `h12_nemo12b_h100`

- **Experiment:** `h12_subspace_safe_merge` (`H12`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-nano` (`11,536,640` total / `3,148,032` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `uncommitted` (dirty: `False`)

## Hypothesis & Prediction
- **Hypothesis:** Naive linear merge (Delta W_SQL + Delta W_unslop) of two specialized GRPO adapters causes destructive gradient interference — degrading G_evidence * R_correctness on SQL incidents or re-leaking slop/refusals. Orthogonalized TIES / Gram-Schmidt Subspace-Safe Merge (Delta W_unslop^perp projected orthogonally to both U_{refusal/slop} and top singular directions of Delta W_SQL with row-norm preservation) retains >= 95% of both skills and passes the H11 FP8 E4M3 gate.
- **Prediction:** TIES biprojected subspace-safe merge retains sql_incident_retention_ratio >= 0.95 and unslop_retention_ratio >= 0.95, keeps slop_and_refusal_rate <= 0.02 (nano) / <= 0.10 (3B), subspace_safe_merge_releakage < 1e-5 (nano), achieves ties_biprojected_vs_naive_merge_gain >= 0.15 on the joint benchmark (StatefulToolEnv + TenetUnslopRubric), and passes provider_ready == True (Row Crest-Factor < 25.0).
- **Acceptance Criterion:** `sql_incident_retention_ratio >= 0.95 and slop_and_refusal_rate <= 0.02 and subspace_safe_merge_releakage < 1e-5 and ties_biprojected_vs_naive_merge_gain >= 0.15 and provider_ready == True`

## Final Metrics
- **`h12_verified`:** `True`
- **`sql_incident_retention_ratio`:** `1.0`
- **`unslop_retention_ratio`:** `1.0`
- **`slop_and_refusal_rate`:** `1.1496781165953264e-07`
- **`subspace_safe_merge_releakage`:** `6.630873272683857e-08`
- **`ties_biprojected_vs_naive_merge_gain`:** `0.7805732948796269`
- **`provider_ready`:** `True`
- **`max_row_crest_factor`:** `10.650956388733396`
- **`backend`:** `llm`
- **`model_name`:** `mistralai/Mistral-Nemo-Instruct-2407`

## Compute & Cost
- **Active GPU-Hours:** `0.120646`
- **Billed GPU-Hours:** `0.120646`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H12 subspace-safe merge verification completed: h12_verified=True, sql_incident_retention_ratio=1.0000, unslop_retention_ratio=1.0000, ties_biprojected_vs_naive_merge_gain=0.7806
