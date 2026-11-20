# ADR-010: Tenet v2 — Phase-Segmented Tri-Anchor Lancet (H15) & Dual-Window Length-Calibrated GRPO (H16)

- **Status:** Accepted (`Tenet-4B-Qwen3-Unslop-v2` and `Tenet-8B-Qwen3-Unslop-v2` 10/10 gates passed on Apple M4 Pro 48GB & published to Hugging Face Hub)
- **Date:** 2026-11-20
- **Hypotheses:** `H15` ([experiments/h15_cot_tri_anchor_lancet.yaml](../../experiments/h15_cot_tri_anchor_lancet.yaml)), `H16` ([experiments/h16_length_calibrated_grpo.yaml](../../experiments/h16_length_calibrated_grpo.yaml))
- **Related:** [ADR-005: Lancet Bi-Projected Surgery](005-lancet-biprojected-surgery.md), [ADR-009: GGUF Quantization Survival](009-gguf-imatrix-vs-cka-survival.md), [Tenet Technical Report](../../reports/tenet_technical_report.md)

---

## Context

Following the `Tenet v1` release (`H1–H14`) — where `Tenet-3B-Qwen2.5-Unslop-v1` and `Tenet-8B-Llama3.1-Unslop-v1` placed **#1** in both divisions (Minors and Majors) at Model Fight Card 16 — post-mortem analysis of `results/h10_llama8b_h100/final_metrics.json` and tournament logs surfaced three structural gaps for `Tenet v2`:

1. **Deliberation-Channel Vulnerability (`<think>...</think>` / `<reasoning>` / harmony channels):** Modern reasoning models (`Qwen3`, `DeepSeek-R1-Distill`, `GPT-OSS`) separate internal deliberation from surface answers. Whole-sequence abliteration or unsegmented contrast extraction damages the Chain-of-Thought (CoT) subspace, causing skipped closing tags, broken self-correction, or token loops under 4-bit GGUF quantization.
2. **`length_collapse` and `overthinking`:** In `H10`, a linear/hard slop penalty compressed 8B outputs to `mean_response_words = 31.5` with `length_collapse_rate = 8.33%`. Conversely, `<think>` models suffer from *overthinking* — burning excessive deliberation tokens on trivial factual queries.
3. **100% Artifact Survival:** Tournament and production evaluations require mechanically verifiable technical artifacts (parseable code blocks, read-only SQL, valid JSON, or exact numeric keys) after `</think>`, with zero tag leakage into structured outputs.

---

## Decision

### 1. H15 — Model-Agnostic Phase-Segmented Tri-Anchor Lancet (`src/sml/surgery/`)

1. **`PhaseSegmenter` (`src/sml/surgery/probes.py`):**
   - Configurable delimiter abstraction (`deliberation_start`, `deliberation_end`, optional explicit token IDs, and format presets: `think`, `reasoning`, `harmony`).
   - Produces boolean `(deliberation_mask, answer_mask)` tensors over `(B, T)` and validates tag balance/nesting via `segment_text`.
2. **Phase-Segmented Extraction & `Tri-Anchor` Basis (`src/sml/surgery/subspace.py`):**
   - `extract_contrastive_direction` supports `pool_mode: Literal["all", "deliberation_only", "answer_only"]`.
   - Inside $\mathcal{I}_{\text{cot}}$ (`deliberation_only`), extract the protected **`CoT Self-Correction Anchor`** ($a_{\text{cot}}$) from `COT_REASONING_ANCHOR_PAIRS` (multi-step verification with backtracking vs. truncated/looping stubs).
   - Combine $a_{\text{cot}}$, $a_{\text{math\_logic}}$ (`REASONING_ANCHOR_PAIRS`), and $a_{\text{tool\_sql}}$ (`TOOL_SQL_ANCHOR_PAIRS`) into an orthonormal `Tri-Anchor` basis $U_{\text{anchor}}^{(\ell)}$ via two-pass Modified Gram-Schmidt in `float32`.
   - Extract `refusal` and `slop` directions strictly from the post-`</think>` answer phase ($\mathcal{I}_{\text{ans}}$, `answer_only`), bi-project them into $(I - U_{\text{anchor}}^T U_{\text{anchor}})$, and apply row-norm-preserving projection on `o_proj` and `down_proj`.
3. **CoT-Weighted KL Collateral (`compute_layer_scr_profile`):**
   - Replace single-corpus KL with $\lambda_{\text{cot}} \text{KL}_{\text{cot}} + (1 - \lambda_{\text{cot}}) \text{KL}_{\text{anchor}}$, where $\text{KL}_{\text{cot}}$ explicitly weights `<think>` / `</think>` tag-logit divergence so layers critical to deliberation delimiters are protected.

### 2. H16 — Dual-Window Length-Calibrated & Artifact-Completeness Gated GRPO (`src/sml/envs/unslop_rubric.py`, `src/sml/adaptation/grpo.py`)

Extend `UnslopPromptSpec` and `TenetUnslopRubric` (keeping `v1` `gated`/`additive` modes 100% backwards-compatible) with `mode="v2"`:

1. **`G_cot_integrity` ($\{0, 1\}$):** Enforces balanced `<think>...</think>` tags, zero tag leakage into final JSON/SQL/code artifacts, and absence of repetition loops inside `<think>`.
2. **`G_length_cal` (Dual-Window Calibration):**
   - Asymmetric log-normal window around task-type target $w_{\text{ans}}^*$ (`concise_factual`, `analytical_sql`, `deep_technical_explanation`) providing a smooth positive gradient for $0 < w_{\text{ans}} < w_{\text{ans}}^*$ (eliminating `length_collapse`) and gentle decay for $w_{\text{ans}} > w_{\text{ans}}^*$.
   - Soft deliberation budget $w_{\text{cot}}^*$ penalizing `overthinking` beyond budget.
3. **`G_antislop_density`:** Hard penalty on introductory/concluding clichés in the first and last paragraphs of the surface answer, plus per-100-word normalization for internal hits.
4. **`R_info_density`:** Combines mechanical artifact validity (`json`, `sql_readonly`, `code`, `exact_numeric`) with lexical diversity without filler.

---

## Acceptance Criteria

- `tri_anchor_orthogonality_error < 1e-6`
- `cot_tag_integrity_rate >= 0.995`
- `length_collapse_rate < 0.01` (`< 1%`)
- `slop_and_refusal_rate <= 0.02`
- `max_row_crest_factor < 25.0`
- Full backwards compatibility with `Tenet v1` (`H1–H14`) test suite.

---

## Local Foundry Validation (`Tenet-4B-Qwen3-Unslop-v2` & `Tenet-8B-Qwen3-Unslop-v2`, Apple M4 Pro 48GB)

Executed via [scripts/run_tenet_v2_foundry.py](../../scripts/run_tenet_v2_foundry.py) on `Qwen/Qwen3-4B` (`bfloat16` MPS + `llama.cpp` Metal) and audited in [results/tenet_releases/Tenet-4B-Qwen3-Unslop-v2/foundry_summary.json](../../results/tenet_releases/Tenet-4B-Qwen3-Unslop-v2/foundry_summary.json) — **`10/10` release gates passed**, published to [`mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-4B-Qwen3-Unslop-v2):

- **H15 + H16 Weight Hygiene:** `max_row_crest_factor = 3.1766` (`< 25.0`), `fp8_subspace_releakage = 2.4946%` (`< 3.0%`).
- **H17 Arena House & Peer Eval (`i1-CKA-Q4_K_M`):** `arena_house_hp = 100.00` (`>= 95.0`), `house_artifact_classification_rate = 1.0000` (`24/24` on `f16`, `cka_q4_k_m`, and `i1-CKA-Q4_K_M`), `length_collapse_rate = 0.0` (`< 0.01`), `slop_and_refusal_rate = 0.0` (`<= 0.02`), `unslop_pass_rate = 1.0` (`24/24`), `sql_readonly_rate = 1.0` (`6/6`).
- **CoT Tag Integrity:** `cot_tag_integrity_rate = 1.0000` on `f16` and `i1-CKA-Q4_K_M` (60 prompts; 100% of `sql_readonly` and `unslop_audit` prompts preserved clean thinking separation).
- **External Evals ([results/external_evals/Tenet-4B-Qwen3-Unslop-v2.json](../../results/external_evals/Tenet-4B-Qwen3-Unslop-v2.json)):** `gsm8k_cot_flexible` delta = `0.0 pp` (`0.15` vs `0.15`, `>= -0.5 pp`).
- **`Tenet-8B-Qwen3-Unslop-v2` (`Qwen/Qwen3-8B`) — `10/10 gates passed`** ([results/tenet_releases/Tenet-8B-Qwen3-Unslop-v2/foundry_summary.json](../../results/tenet_releases/Tenet-8B-Qwen3-Unslop-v2/foundry_summary.json); published to [`mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2`](https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Qwen3-Unslop-v2)):
  - **H15 Surgery & Weight Hygiene:** `selected_layers = [20, 17, 9, 5, 18, 21, 7, 19]` (CoT-weighted SCR, `lambda_cot = 0.5`), `max_row_crest_factor = 4.3688` (`< 25.0`), `fp8_subspace_releakage = 2.7243%` (`< 3.0%`).
  - **H17 Arena House & Behavioral Evals:** `cot_tag_integrity_rate = 1.0000` (on `f16` and `i1_cka_q4_k_m`), `arena_house_hp = 100.00`, `house_artifact_classification_rate = 1.0000` (`24/24`), `length_collapse_rate = 0.0`, `slop_and_refusal_rate = 0.0`, `unslop_pass_rate = 1.0000`, `sql_readonly_rate = 1.0000` (`6/6`), `Mean KLD vs F16 = 0.01314` (Domain `i1_cka_q4_k_m`).
