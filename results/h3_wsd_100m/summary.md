# Run Summary: `h3_wsd_100m`

- **Experiment:** `h3_wsd_100m` (`H3`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-100m` (`100,682,496` total / `75,516,672` non-emb)
- **Dataset:** `fineweb-edu-4b`
- **Hardware:** `nvidia-h100-sxm-80gb` (`$2.69/hr`)
- **Git SHA:** `uncommitted` (dirty: `False`)

## Hypothesis & Prediction
- **Hypothesis:** Overtraining sml-100m (75.52M non-embedding parameters) from the Chinchilla compute-optimal regime (1.5B tokens, ~20 tok/non-emb param) to 4.0B tokens (~53 tok/non-emb param) using Warmup-Stable-Decay (WSD) cooldown branching produces a statistically significant reduction in validation bits-per-byte (bpb) and higher zero-shot accuracy at identical inference cost.
- **Prediction:** The 4.0B WSD cooldown branch reduces validation bpb by >= 0.04 bpb (> 4 sigma of the 3-seed sml-xs noise floor) and improves HellaSwag / ARC-Easy byte-normalized accuracy by >= 2.5 percentage points over the 1.5B cooldown branch.
- **Acceptance Criterion:** `delta_val_bpb <= -0.04 and abs(delta_val_bpb) > 2 * seed_noise_sigma`

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
Completed WSD pretraining for h3_wsd_100m: val_bpb=1.0400 (initial=3.3075), branch_val_bpb=1.044582, checkpoints_saved=15.
