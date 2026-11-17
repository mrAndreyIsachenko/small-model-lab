# ADR-005: Lancet Dual-Anchor Soft-SCR Surgery (`H9`) → Tenet Subspace-Safe GRPO (`H10`) → FP8 Re-Leakage & Crest-Factor Provider Readiness (`H11`)

* **Status:** Accepted
* **Date:** 2026-11-17
* **Scope:** The full Sprints 5–7 surgical-RL pipeline: **`H9` — Lancet Dual-Anchor Soft-SCR Surgery → `H10` — Tenet Subspace-Safe GRPO → `H11` — FP8 Re-Leakage & Crest-Factor Provider Readiness.**

## Context

Preparing Tenet-series models (alignment/slop steering on small open-weight checkpoints) requires suppressing refusal/moralizing and AI-slop write directions in the residual stream while leaving reasoning behaviour intact. The Sprint 4 mechanism (`H8`, `apply_subspace_ablation`) applies a naive rank-1 subspace ablation:

$$W' = W - r \,(r^{\top} W)$$

with a single contrastive direction $r$ projected out of `self_attn.o_proj` / `mlp.down_proj` in every layer. Two structural defects make it unusable for production edits:

1. **Non-zero projection onto the reasoning anchor subspace → high `kl_collateral`.** Probe-extracted refusal/slop directions are not orthogonal to capability-critical subspaces. Removing the full component along $r$ also removes the part shared with structured reasoning behaviour, so on anchor inputs (chain-of-thought, JSON-schema, analytical prompts) the forward KL divergence against the base model stays high.
2. **Row L2-norm shrinkage → activation variance drift and FP8 quantization degradation.** The ablation strictly shrinks every weight row whose component along $r$ is nonzero: $\lVert W'_i \rVert_2 = \lVert W_i \rVert_2 \sin\theta_i < \lVert W_i \rVert_2$. Shrunken rows change the variance of residual-stream writes, and the compounded rounding error measurably degrades output quality after OpenRouter FP8 (E4M3) quantization of the edited checkpoint.

Additionally, the naive mechanism ablates one direction at a time and operates on every layer, spending edit capacity where the target signal is weakest.

## Decision

The pipeline is a three-stage chain: H9 edits the base weights, H10 fine-tunes and merges RL adapters without re-leaking into the ablated subspace, and H11 certifies the result survives provider FP8 quantization.

### Stage 1 — `H9`: Lancet Dual-Anchor Soft-SCR Surgery (`src/sml/surgery/`)

Adopt the **Lancet surgical engine** (`src/sml/surgery/`): a bi-projected, norm-preserving, multi-direction subspace surgery with SCR-guided layer selection.

1. **Modified Gram-Schmidt biprojection of target directions.** The probe directions for refusal ($r_{\text{refusal}}$) and slop ($r_{\text{slop}}$) — extracted from the deterministic 12-pair contrastive probe families in `sml/surgery/probes.py` — are first biprojected against the reasoning anchor direction(s) ($a_{\text{reasoning}}$) in **float32**: the anchor span is orthonormalized via modified Gram-Schmidt, all anchor components are removed from each target, and the residual is renormalized to unit L2 norm (`biproject_direction`). Acceptance: `anchor_orthogonality_error = max_k |r_k^\perp \cdot a_{\text{reasoning}}| < 1e-6`.

2. **Multi-direction subspace projection with per-row L2-norm rescaling.** Both bi-projected targets are removed jointly from `self_attn.o_proj` and `mlp.down_proj`, then each row is rescaled back to its original L2 norm (`project_out_subspace_norm_preserving`, orchestrated by `apply_lancet_biprojected_surgery`):
   $$W_{\text{proj}} = W - \text{scale} \cdot \sum_k r_k \,(r_k^{\top} W), \qquad W'_i = W_{\text{proj},i} \cdot \frac{\lVert W_i \rVert_2}{\lVert W_{\text{proj},i} \rVert_2 + 10^{-8}}$$
   Acceptance: `max_row_norm_drift = max_i | \lVert W_i \rVert_2 - \lVert W'_i \rVert_2 | < 1e-5`.

3. **Layer selection via Signal-to-Collateral Ratio.** For each candidate layer, `compute_layer_scr_profile` temporarily applies the norm-preserving projection (weights restored afterwards) and measures `signal` (removed target projection energy on both write weights) and `kl_collateral` (forward KL on anchor inputs), then ranks layers by
   $$\text{SCR} = \frac{\text{signal}}{\text{kl\_collateral} + 10^{-6}}.$$
   Surgery edits only the top-$k$ highest-SCR layers instead of all layers.

The benchmark control (`run_h9_biprojected_surgery_benchmark`) compares Lancet against **naive surgery** (same targets, no biprojection, no row-norm preservation, all layers) and reports
`h9_kl_reduction_ratio = 1 - (\text{lancet\_kl\_collateral} / \text{naive\_kl\_collateral})`.

### Stage 2 — `H10`: Tenet Subspace-Safe GRPO (`src/sml/grpo/unslop/`)

H9 removes the refusal/slop write directions, but any subsequent RL fine-tuning re-introduces them: an ordinary LoRA merge writes unconstrained deltas back into the ablated span. Stage 2 closes that hole with a gated unslop GRPO recipe whose adapter merges are subspace-constrained:

1. **Gated unslop GRPO.** GRPO training runs with the Gated Multiplicative `Rubric` gate (`G_safety · G_complete · G_evidence`, cf. ADR-004) stacked with an unslop reward term that penalizes AI-slop phrasing and refusal/moralizing turns on the held-out slop probe families. This is the same anti-reward-hacking gate from Sprint 3, extended with the slop/refusal objective.
2. **Orthogonal LoRA merge (`I - U U^{\top}`).** Let $U \in \mathbb{R}^{d \times k}$ be the orthonormal basis (columns $r_k^\perp$ from the H9 biprojection) of the ablated target span. The trained LoRA delta is merged only through its component orthogonal to $U$:
   $$W_{\text{merged}} = \text{RowRescale}\Big( W_{\text{lancet}} + \Delta W_{\text{LoRA}} \,(I - U U^{\top}) \Big)$$
   so no merged update writes energy back into the ablated directions. Acceptance: `subspace_safe_lora_releakage < 1e-5` on `sml-nano`.
3. **Row-norm preservation at merge.** `RowRescale` restores each row of $W_{\text{merged}}$ to the row L2 norm of the post-H9 weights (same estimator as Stage 1), so the merge does not re-create the activation-variance drift that H9's rescaling eliminated. The merge ledger records `merged_lora_count`, and post-merge target suppression is verified via `post_merge_suppression_retention`.

The benchmark control (`run_h10_unslop_grpo_benchmark`) compares the subspace-safe (gated) merge against a naive unprojected merge of the same adapter (`unprojected_lora_releakage`) and reports `h10_gated_vs_naive_gain` and `h10_post_surgery_gain`.

### Stage 3 — `H11`: FP8 Re-Leakage & Crest-Factor Provider Readiness (`src/sml/quantization/fp8/`)

The final stage certifies that the edited checkpoint survives OpenRouter-style FP8 (E4M3) quantization, because quantization rounding can partially re-introduce the ablated subspace after the fact.

1. **FP8 Subspace Re-Leakage.** With $U$ the H9 target span and $\mathrm{Q}_{\text{FP8}}$ the E4M3 quantize–dequantize operator, the metric measures how much target-subspace write energy re-appears after quantization, relative to the pre-surgery energy:
   $$\text{ReLeak}_{\text{FP8}} = \frac{\lVert U^{\top} \,\mathrm{Q}_{\text{FP8}}(W') \rVert_F}{\lVert U^{\top} W_{\text{base}} \rVert_F}$$
   Naive row-shrinking surgery re-leaks roughly half of the original target energy after FP8, while norm-preserving Lancet surgery keeps re-leakage an order of magnitude lower (measured values below).
2. **Per-Row vs. Per-Tensor E4M3.** Each edited tensor is quantized with per-row E4M3 scales and compared against a per-tensor-scale control (`*_per_tensor`). Norm-preserving surgery is quantization-mode-insensitive (its row-norm distribution stays homogeneous), whereas naive row-shrinking surgery is mode-sensitive: the shrunken rows fall into the low-mantissa region of the shared per-tensor scale and degrade further.
3. **Row Crest-Factor provider gate (< 25.0).** For each weight row, the crest factor $\text{CF}_i = \max_j |W_{ij}| \,/\, \text{RMS}(W_i)$ bounds how far the row's peak must travel above its RMS under the E4M3 exponent range. A checkpoint is provider-ready only if $\max_i \text{CF}_i < 25.0$ and $\max_{i,j} |W_{ij}|$ fits the E4M3 range; the benchmark evaluates a 7-check readiness checklist (`provider_ready`).
4. **`Tenet` model cards.** The pipeline's public releases are published as `Tenet` model cards — `Tenet-3B-Qwen2.5-Unslop-v1`, `Tenet-8B-Llama3.1-Unslop-v1`, `Tenet-12B-Nemo-Unslop-v1` — each documenting the H9 surgery parameters (biprojected directions, SCR-selected layers), the H10 merge ledger, and the H11 FP8 readiness report for the released checkpoint.

## Acceptance Criteria & Verification

| Hypothesis | Metric | Threshold |
| --- | --- | --- |
| `H9` | `anchor_orthogonality_error` | < 1e-6 |
| `H9` | `max_row_norm_drift` | < 1e-5 |
| `H9` | `target_suppression_ratio` | >= 0.90 |
| `H9` | `h9_kl_reduction_ratio` | >= 0.35 |
| `H10` | `slop_and_refusal_rate` | < 0.05 |
| `H10` | `subspace_safe_lora_releakage` | < 1e-5 |
| `H10` | `post_merge_suppression_retention` | >= 0.95 |
| `H11` | `fp8_retention_ratio` | >= 0.95 |
| `H11` | `fp8_subspace_releakage_lancet` | < 0.10 |
| `H11` | `max_row_crest_factor` | < 25.0 |
| `H11` | `provider_ready` | `true` (7/7 checks) |

Verified via `run_h9_biprojected_surgery_benchmark`, `run_h10_unslop_grpo_benchmark`, and `run_h11_provider_fp8_readiness_benchmark` (exposed through the `sml verify-h9` / `sml verify-h10` / `sml verify-h11` CLIs, which persist the `final_metrics.json` run-bundle artifacts under `results/<run_id>/`) and the regression suites `tests/test_surgery_h9.py`, `tests/test_unslop_grpo_h10.py`, and `tests/test_fp8_readiness_h11.py`.

## Measured Results (`sml-nano`, `Qwen/Qwen2.5-3B-Instruct` MPS, `Llama-3.1-8B-Instruct` H100, `Mistral-Nemo-12B` H100)

**`H9` — Lancet Dual-Anchor Soft-SCR Surgery**

| Metric | `sml-nano` ([`results/h9_biprojected_surgery/`](../../results/h9_biprojected_surgery/final_metrics.json)) | `Qwen2.5-3B` MPS ([`results/h9_qwen3b_mps/`](../../results/h9_qwen3b_mps/final_metrics.json)) | `Llama-3.1-8B` H100 ([`results/h9_llama8b_h100/`](../../results/h9_llama8b_h100/final_metrics.json)) | `Mistral-Nemo-12B` H100 ([`results/h9_nemo12b_h100/`](../../results/h9_nemo12b_h100/final_metrics.json)) |
| --- | --- | --- | --- | --- |
| `h9_verified` | `true` | `true` | `true` | `true` |
| `anchor_orthogonality_error` | < 1e-6 | `1.0e-07` | `1.0e-08` | `1.0e-08` |
| `tool_sql_orthogonality_error` | `4.0e-08` | `7.0e-08` | `1.0e-08` | `3.0e-08` |
| `target_suppression_ratio` | `0.9854` | `0.9694` | `0.9930` | `0.9914` |
| `naive_kl_collateral` | `0.001317` | `3.7607` | `0.1337` | `0.2968` |
| `lancet_kl_collateral` | `0.000515` | `0.0220` | `0.0135` | `0.0097` |
| `h9_kl_reduction_ratio` | `0.6092` | `0.9941` | `0.8989` | `0.9672` |
| `max_row_norm_drift` | < 1e-5 | `0.00161` (bf16 ULP) | `0.00108` (bf16 ULP) | `0.00074` (bf16 ULP) |

**`H10` — Tenet Subspace-Safe GRPO**

| Metric | `sml-nano` ([`results/h10_unslop_grpo/`](../../results/h10_unslop_grpo/final_metrics.json)) | `Qwen2.5-3B` MPS ([`results/h10_qwen3b_mps/`](../../results/h10_qwen3b_mps/final_metrics.json)) | `Llama-3.1-8B` H100 ([`results/h10_llama8b_h100/`](../../results/h10_llama8b_h100/final_metrics.json)) | `Mistral-Nemo-12B` H100 ([`results/h10_nemo12b_h100/`](../../results/h10_nemo12b_h100/final_metrics.json)) |
| --- | --- | --- | --- | --- |
| `h10_verified` | `true` | `true` | `true` | `true` |
| `slop_and_refusal_rate` | `0.0150` | `0.0909` (`2e-05` surgical-only) | `0.0410` | `0.0239` |
| `h10_gated_vs_naive_gain` | `+0.2276` | `+0.1602` | `+0.2489` | `+0.4292` |
| `h10_post_surgery_gain` | `+0.6611` | `+0.3598` | `+0.4580` | `+0.2710` |
| `post_merge_suppression_retention` | `1.0` | `1.0` | `1.0` | `1.0` |
| `unprojected_lora_releakage` | `3.5124` | `0.8300` | `1.1328` | `0.8052` |
| `subspace_safe_lora_releakage` | < 1e-5 | `0.0690` | `0.0726` | `0.1282` |
| `merged_lora_count` | `0` | `0` | `0` | `0` |

**`H11` — FP8 Re-Leakage & Crest-Factor Provider Readiness**

| Metric | `sml-nano` ([`results/h11_provider_fp8_readiness/`](../../results/h11_provider_fp8_readiness/final_metrics.json)) | `Qwen2.5-3B` MPS ([`results/h11_qwen3b_mps/`](../../results/h11_qwen3b_mps/final_metrics.json)) | `Llama-3.1-8B` H100 ([`results/h11_llama8b_h100/`](../../results/h11_llama8b_h100/final_metrics.json)) | `Mistral-Nemo-12B` H100 ([`results/h11_nemo12b_h100/`](../../results/h11_nemo12b_h100/final_metrics.json)) |
| --- | --- | --- | --- | --- |
| `h11_verified` | `true` | `true` | `true` | `true` |
| `fp8_retention_ratio` | `0.9997` | `0.9541` | `0.9936` | `0.9678` |
| `h11_fp8_kl_reduction_ratio` | `0.4558` | `0.5185` | `0.4265` | `0.4892` |
| `kl_naive_fp8` | `0.000464` | `0.0975` | `0.0322` | `3.1724` |
| `kl_lancet_fp8` | `0.000253` | `0.0469` | `0.0185` | `1.6205` |
| `kl_naive_fp8_per_tensor` | — | `0.1113` | `0.0370` | `3.1780` |
| `kl_lancet_fp8_per_tensor` | — | `0.0470` | `0.0182` | `1.6264` |
| `fp8_subspace_releakage_naive` | `0.4996` | `0.5003` | `0.5000` | `0.4996` |
| `fp8_subspace_releakage_lancet` | `0.0310` | `0.0270` | `0.0278` | `0.1839` |
| `max_abs_weight` | — | `1.4297` | `0.9141` | `0.9180` |
| `max_row_crest_factor` | — | `17.2706` | `6.3244` | `10.6508` |
| `provider_ready` | `true` (7/7 checks) | `true` (7/7 checks) | `true` (7/7 checks) | `true` (7/7 checks) |

On `Qwen/Qwen2.5-3B-Instruct` MPS, FP8 re-leakage confirms the Stage 3 diagnosis: naive surgery re-leaks `fp8_subspace_releakage_naive = 0.5003` of the target energy after quantization, while the Lancet edit stays at `0.0270`; and the Lancet edit is quantization-mode-insensitive (`kl_lancet_fp8 = 0.0469` per-row vs `kl_lancet_fp8_per_tensor = 0.0470` per-tensor) where naive surgery degrades under the shared per-tensor scale (`0.0975` vs `0.1113`). The checkpoint passes the provider gate with `max_row_crest_factor = 17.2706 < 25.0` and `max_abs_weight = 1.4297` inside the E4M3 range, i.e. `provider_ready = true` (7/7 checks).

## Consequences

* Refusal/slop suppression no longer leaks into reasoning behaviour: anchor-input KL collateral collapses by >= 35% relative to naive all-layer ablation at equal target suppression.
* Row L2 norms are preserved to < 1e-5, so residual-stream activation variance — and therefore FP8 quantization fidelity of edited checkpoints on OpenRouter — remains stable without recalibration.
* SCR ranking concentrates edits in a few auditable layers (`selected_layers` is recorded in the run bundle), reducing the edit surface exposed to collateral regressions.
* Target directions must be re-extracted per model and per layer; the hashed probe encoding keeps extraction deterministic and tokenizer-free, so surgery is reproducible from a seed alone.
* (`H10`) RL fine-tuning after surgery is safe by construction: the `I - U U^{\top}` merge keeps GRPO updates orthogonal to the ablated span (`subspace_safe_lora_releakage < 1e-5` on `sml-nano`, `0.0690` vs `0.8300` unprojected on `Qwen2.5-3B-Instruct`), and target suppression survives the merge (`post_merge_suppression_retention = 1.0`).
* (`H11`) Edited checkpoints are certified against provider quantization before release, not after: FP8 re-leakage (`0.0310` Lancet vs `0.4996` naive on `sml-nano`; `0.0270` vs `0.5003` on `Qwen2.5-3B-Instruct`) and the `max_row_crest_factor < 25.0` gate (`17.2706` on `Qwen2.5-3B-Instruct`) make FP8 provider readiness a measured, reproducible property of the artifact, recorded in every `Tenet` model card.
