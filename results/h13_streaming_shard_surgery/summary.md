# Run Summary: `h13_streaming_shard_surgery`

- **Experiment:** `h13_streaming_shard_surgery` (`H13`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-nano` (`11,536,640` total / `3,148,032` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `ff9b373b868437c4f8df21ae1c0db5bc145116b5` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Layer-streaming safetensors shard surgery (apply_lancet_recipe_streaming) and shard-streaming TIES subspace-safe merge (stream_subspace_safe_ties_merge) with float32 Modified Gram-Schmidt and per-row L2-norm restoration match in-memory surgery and merge within < 1e-6 on sml-nano while bounding peak memory < 16 GB, local scratch disk delta < 15 GB, and FP8 E4M3 Row Crest-Factor < 25.0 for 32B-70B checkpoints on a single H100 80GB SXM.
- **Prediction:** Streaming surgery and merge achieve streaming_surgery_max_abs_diff < 1e-6 and streaming_merge_max_abs_diff < 1e-6 on sml-nano, keep peak_memory_gb < 16.0 and peak_disk_shard_gb < 15.0, and satisfy max_row_crest_factor < 25.0 and provider_ready == True across 32b-h100 (Qwen/Qwen2.5-32B-Instruct) and 70b-h100 (meta-llama/Llama-3.3-70B-Instruct).
- **Acceptance Criterion:** `streaming_surgery_max_abs_diff < 1e-6 and streaming_merge_max_abs_diff < 1e-6 and peak_memory_gb < 16.0 and peak_disk_shard_gb < 15.0 and max_row_crest_factor < 25.0 and provider_ready == True`

## Final Metrics
- **`h13_verified`:** `True`
- **`streaming_surgery_max_abs_diff`:** `7.450580596923828e-09`
- **`streaming_merge_max_abs_diff`:** `0.0`
- **`peak_memory_gb`:** `3.149360418319702e-05`
- **`peak_disk_shard_gb`:** `0.03564857691526413`
- **`max_row_crest_factor`:** `1.1926119327545166`
- **`shards_processed`:** `3.0`
- **`tensors_merged`:** `8.0`
- **`seed`:** `2026.0`
- **`num_shards`:** `3.0`
- **`sql_svd_rank`:** `2.0`
- **`ties_density`:** `0.85`
- **`backend`:** `nano`
- **`model_name`:** `Qwen/Qwen2.5-32B-Instruct`

## Compute & Cost
- **Active GPU-Hours:** `0.000285`
- **Billed GPU-Hours:** `0.000285`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H13 streaming shard surgery verification completed: h13_verified=True, backend=nano
