# Run Summary: `h4_induction_dynamics`

- **Experiment:** `h4_induction_dynamics` (`H4`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-100m` (`100,682,496` total / `75,516,672` non-emb)
- **Dataset:** `fineweb-edu-4b`
- **Hardware:** `nvidia-h100-sxm-80gb` (`$2.69/hr`)
- **Git SHA:** `uncommitted` (dirty: `False`)

## Hypothesis & Prediction
- **Hypothesis:** In-context induction capability emerges via a sharp early phase transition across log-spaced checkpoints.
- **Prediction:** > 60% of the total improvement in synthetic copying accuracy and intra-document in-context loss delta occurs within a narrow window spanning < 15% of total training steps.
- **Acceptance Criterion:** `max_window_gain_share >= 0.60 and checkpoints_saved >= 15`

## Final Metrics
- **`initial_val_bpb`:** `3.307487`
- **`val_bpb`:** `1.039992`
- **`val_loss_nats`:** `3.324193`
- **`final_in_context_loss_delta`:** `-0.179115`
- **`final_synthetic_copy_accuracy`:** `0.25`
- **`final_synthetic_copy_prob`:** `0.134456`
- **`tokens_processed`:** `4000317440`
- **`tokens_per_sec`:** `445067.74`
- **`mfu`:** `0.322653`
- **`checkpoints_saved`:** `15`
- **`branch_val_bpb`:** `1.044582`
- **`branch_start_step`:** `2861`
- **`max_window_gain_share`:** `1.0`
- **`delta_val_bpb`:** `-0.00459`

## Compute & Cost
- **Active GPU-Hours:** `2.496697`
- **Billed GPU-Hours:** `2.496697`
- **Billed Cost (USD):** `$6.7161` (pre-run estimate: `$8.3067`)

## Conclusion
Companion H4 induction-dynamics bundle for h3_wsd_100m: max_window_gain_share=1.000000 across 15 log-spaced checkpoints (checkpoints_saved=15), delta_val_bpb=-0.004590.
