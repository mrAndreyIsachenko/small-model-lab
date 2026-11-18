# ADR-008: Layer-Streaming Lancet Surgery & Subspace-Safe Merge for 32B–70B Models

## Status

Accepted

## Context

Sprints 5–8 (`H9`–`H12`) established the Lancet biprojected surgery, subspace-safe LoRA merge, FP8 E4M3 readiness gate (`Row Crest-Factor < 25.0`), and TIES / Gram-Schmidt cross-domain adapter merge on `3B–12B` checkpoints (`Qwen2.5-3B-Instruct`, `Llama-3.1-8B-Instruct`, `Mistral-Nemo-Instruct-2407`). Scaling this pipeline to `32B` (`Qwen/Qwen2.5-32B-Instruct`, `~65 GB BF16`) and `70B` (`meta-llama/Llama-3.3-70B-Instruct`, `~140 GB BF16`) within a single `H100 80GB SXM` node and a strict local scratch disk budget (`< 15 GB`) breaks the in-memory materialization path:

1. **Full-checkpoint RAM/VRAM overflow.** `AutoModelForCausalLM.from_pretrained(...)` followed by `copy.deepcopy(model)` in `apply_lancet_recipe` (`src/sml/surgery/subspace.py`) requires `~140 GB` just to hold one `70B` `BF16` copy and `~280 GB` when cloned, exceeding both `80 GB` H100 VRAM and standard host memory limits.
2. **Local disk budget (`< 15 GB`).** Storing an entire `70B` unsharded copy or keeping duplicate full checkpoints on local scratch disk exceeds the `< 15 GB` ephemeral disk budget.
3. **FP8 E4M3 outlier amplification at `32B`/`70B` scale.** Larger hidden dimensions (`d_model = 5120`–`8192`) and deeper MLP/attention projections amplify extreme activation and weight outliers (`Row Crest-Factor >= 25.0`) when deltas are merged or projected without per-row L2-norm restoration and `float32` orthonormalization.

## Decision

We introduce **Layer-Streaming Safetensors Shard Surgery & Subspace-Safe Merge (`H13`)** so `32B` and `70B` checkpoints are modified on-the-fly shard-by-shard without ever loading the full model into RAM/VRAM simultaneously:

1. **Safetensors Shard-Streaming Execution (`model-00001-of-000XX.safetensors`).**
   - `apply_lancet_recipe_streaming` (`src/sml/surgery/subspace.py`) and `stream_subspace_safe_ties_merge` (`src/sml/adaptation/merge.py`) resolve `model.safetensors.index.json` (with `model.safetensors` single-file fallback) and process **exactly one shard at a time** on CPU (`safetensors.torch.load_file` -> transform matching `o_proj.weight` / `down_proj.weight` or adapter-matched 2D weights -> `safetensors.torch.save_file` -> explicit `del` + `gc.collect()`).
   - Peak host/GPU memory stays **below `16 GB`** (bounded by one `~4–5 GB` shard plus its `float32` working tensor), and optional `cleanup_input_shards=True` keeps local scratch disk delta **below `15 GB`**.
2. **Float32 Modified Gram-Schmidt & Row-Norm Preservation.**
   - Every target tensor in the active shard is cast to `float32` for Modified Gram-Schmidt orthonormalization (`_orthonormalize_basis`), orthogonal projection (`project_out_subspace_norm_preserving`), top-k SVD (`extract_top_singular_subspace`), and TIES trim/sign-election/disjoint merge (`subspace_safe_ties_merge`), then rescaled to its pre-surgery/pre-merge per-row L2 norm (`preserve_row_norm=True`) and cast back to the shard's original dtype (`bf16`/`fp16`/`fp32`).
   - This guarantees mathematical equivalence with the in-memory pipeline (`max_abs_diff < 1e-6` on `sml-nano`).
3. **32B/70B FP8 E4M3 Crest-Factor Hygiene Gate.**
   - During shard streaming, `_row_crest_factor` (`max(row_l2) / mean(row_l2)`) is tracked incrementally across every 2D weight tensor in every shard. Any checkpoint exceeding `Row Crest-Factor >= 25.0` fails the `H13` provider-readiness gate before upload/serving, ensuring clean `FP8 E4M3` quantization at `32B`/`70B` scale.
4. **Remote Foundry Profiles & Verification (`sml verify-h13`).**
   - `scripts/remote/run_tenet_foundry.sh` adds `32b-h100` (`Qwen/Qwen2.5-32B-Instruct`) and `70b-h100` (`meta-llama/Llama-3.3-70B-Instruct`) profiles.
   - `sml verify-h13` (`experiments/h13_streaming_shard_surgery.yaml`) verifies shard-streaming vs. in-memory equivalence (`< 1e-6`), peak memory (`< 16 GB`), and `Row Crest-Factor < 25.0` on `sml-nano` and `32B`/`70B` configurations.

## Consequences

- **70B surgery & merge on a single H100 80GB SXM.** Peak memory drops from `~140–280 GB` to `< 16 GB`, eliminating OOMs and `copy.deepcopy` duplication.
- **Strict `< 15 GB` local disk footprint.** Processing one `~4–5 GB` shard at a time (with optional input-shard cleanup) fits comfortably within ephemeral container disk limits.
- **Zero numerical drift vs. in-memory.** Running Modified Gram-Schmidt and TIES in `float32` per tensor yields `< 1e-6` equivalence to in-memory `apply_lancet_recipe` and `subspace_safe_ties_merge`.
- **Sequential shard I/O overhead.** Processing shards one-by-one trades parallel shard I/O for bounded memory/disk; this is accepted because surgery/merge runs once during materialization.
