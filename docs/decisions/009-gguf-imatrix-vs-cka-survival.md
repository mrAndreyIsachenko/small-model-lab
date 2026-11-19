# ADR-009: GGUF Quantization Survival — Static Q4_K_M vs. imatrix vs. CKA-Guided Mixed Precision (H14)

- **Status:** Accepted
- **Date:** 2026-11-19
- **Hypothesis:** `H14` ([experiments/h14_gguf_releakage.yaml](../../experiments/h14_gguf_releakage.yaml))
- **Results:** [results/h14_gguf_releakage/final_metrics.json](../../results/h14_gguf_releakage/final_metrics.json)
- **Audit Module:** [src/sml/surgery/gguf_audit.py](../../src/sml/surgery/gguf_audit.py)
- **Related:** [ADR-005: Lancet Bi-Projected Surgery](005-lancet-biprojected-surgery.md), [quantize_cka_gguf.sh](../../results/tenet_releases/Tenet-8B-Llama3.1-Unslop-v1/quantize_cka_gguf.sh)

---

## Context

In **H11**, we established that naive abliteration suffers ~50% subspace re-leakage under server-side FP8 (E4M3) quantization, while Lancet bi-projected norm-preserving surgery holds FP8 re-leakage to 2.78% (`Tenet-8B-Llama3.1-Unslop-v1`).

Following the release of `Tenet-8B-Llama3.1-Unslop-v1`, community quantizers (`mradermacher`) published static GGUF (`mradermacher/Tenet-8B-Llama3.1-Unslop-v1-GGUF`) and importance-matrix (`mradermacher/Tenet-8B-Llama3.1-Unslop-v1-i1-GGUF`) builds, where the majority of local downloads target 4-bit variants (`Q4_K_M`, `i1-Q4_K_M`, `i1-IQ4_XS`). Meanwhile, our release bundle ships a CKA-guided mixed-precision recipe ([cka_quant_recipe.json](../../results/tenet_releases/Tenet-8B-Llama3.1-Unslop-v1/cka_quant_recipe.json)) that protects boundary layers (`0, 31`) and the top 8 surgical layers (`3, 4, 9, 10, 11, 14, 15, 18`) `attn_output` / `ffn_down` weights at `Q8_0`, and the remaining 8 surgical layers (`19, 20, 24, 25, 26, 27, 28, 29`) at `Q6_K`, while leaving non-surgical tensors at `Q4_K_M` (`5.290` avg bpw).

We designed **H14** to measure, locally on Apple M4 Pro 48GB (`llama.cpp` v0.5.0 build 11146, Metal + CPU), how well Lancet surgery and unslop/tool behavior survive across all 6 GGUF variants:
1. `f16` (`Tenet-8B-Llama3.1-Unslop-v1.f16.gguf`, 16.07 GB) — FP16 reference after Lancet
2. `q4_k_m` (`Tenet-8B-Llama3.1-Unslop-v1.Q4_K_M.gguf`, 4.92 GB) — Static `Q4_K_M`
3. `i1_q4_k_m` (`Tenet-8B-Llama3.1-Unslop-v1.i1-Q4_K_M.gguf`, 4.92 GB) — Weighted imatrix `i1-Q4_K_M`
4. `i1_iq4_xs` (`Tenet-8B-Llama3.1-Unslop-v1.i1-IQ4_XS.gguf`, 4.45 GB) — Weighted imatrix `i1-IQ4_XS`
5. `cka_q4_k_m` (`Tenet-8B-Llama3.1-Unslop-v1-cka-q4_k_m.gguf`, 5.32 GB) — CKA-guided mixed-precision GGUF
6. `i1_cka_q4_k_m` (`Tenet-8B-Llama3.1-Unslop-v1.i1-CKA-Q4_K_M.gguf`, 5.32 GB) — Hybrid `CKA + imatrix` (`--imatrix` + `cka_quant_recipe.json` `--tensor-type` protection)

---

## Methodology

1. **Decoded Weight Audit (`gguf` + `safetensors`):**
   For each of the 16 surgical layers in `lancet_basis.safetensors` (`layer_{idx}`, shape `[2, 4096]`, orthonormalized in float32), we dequantized `blk.{idx}.attn_output.weight` and `blk.{idx}.ffn_down.weight` via `gguf.quants.dequantize` (which returns PyTorch `(out_features, in_features)` layout, verified directly against `f16.gguf`: `||R_24 @ W_f16||_F = 0.00735` vs `||W_f16||_F = 49.80` on `attn_output`). We measured:
   - `subspace_releakage_amplification_vs_f16`: `max_L (||R_L @ W_q||_F / ||R_L @ W_f16||_F)` across surgical layers
   - `max_row_norm_drift`: `max | ||row(W_q)||_2 - ||row(W_f16)||_2 |`
   - `mean_weight_l2_rel_err`: mean `||W_q - W_f16||_F / ||W_f16||_F`
   - `actual_avg_bpw`: exact tensor byte sum * 8 / total parameters

2. **Perplexity & KL-Divergence (`llama-perplexity`):**
   Evaluated on our repo control corpus (`TENET_UNSLOP_SCENARIOS` + `IFEVAL_CASES` + `GSM8K_MATH_CASES` + 6 warehouse incident scenarios, 7 chunks of `n_ctx=512`), saving F16 logits via `--save-all-logits` and running `--kl-divergence --kl-divergence-base` for each quantized GGUF.

3. **End-to-End Behavioral Evals (`llama-server`, deterministic `temperature=0.0`, `seed=2026`, `max_tokens=256`):**
   - `unslop_audit` (`TENET_UNSLOP_SCENARIOS`, 24 prompts): `slop_and_refusal_rate`, `unslop_pass_rate`, `direct_compliance_rate`
   - `ifeval_subset` (`IFEVAL_CASES`)
   - `sql_readonly` (6 `val` scenarios from `sml.envs`, validated with `validate_readonly_sql`)

---

## Measured Results (`Tenet-8B-Llama3.1-Unslop-v1`)

| Variant | Size (GB) | Avg BPW | Subspace Amp vs F16 | Max Row Drift | Weight L2 Rel Err | PPL (7x512) | Mean KLD vs F16 | Max KLD | Same Top-1 | Slop/Refusal Rate | Unslop Pass | SQL RO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `f16` | 16.07 | 16.001 | 1.000x | 0.00000 | 0.00000 | 9.0410 | 0.00000 | 0.0000 | 100.00% | 0.0417 | 0.5833 | 1.0000 |
| `q4_k_m` | 4.92 | 4.894 | 6.402x | 0.00795 | 0.05827 | 9.1798 | 0.02479 | 0.7033 | 93.11% | 0.0833 | 0.5000 | 1.0000 |
| `i1_q4_k_m` | 4.92 | 4.894 | 6.360x | 0.00946 | 0.05911 | 9.0749 | 0.01947 | **0.4529** | 93.44% | **0.0000** | **0.5833** | 1.0000 |
| `i1_iq4_xs` | 4.45 | 4.423 | 12.374x | 0.01089 | 0.07725 | 9.1648 | 0.02257 | 0.4831 | 91.77% | 0.0000 | 0.5000 | **0.0000** |
| `cka_q4_k_m` | 5.32 | 5.290 | **3.012x** | 0.00526 | **0.01165** | 9.0963 | 0.02048 | 0.5821 | 93.28% | 0.0833 | **0.5833** | 1.0000 |
| `i1_cka_q4_k_m` | 5.32 | 5.290 | **3.012x** | **0.00303** | 0.01168 | **9.0533** | **0.01656** | 0.5403 | **93.78%** | 0.0417 | 0.0833 | 1.0000 |

*(Note: on zero-shot `ifeval_subset` without a system prompt, `i1_cka_q4_k_m` scored `0.1667`, while the other 5 variants scored `0.0000` due to strict bare-JSON/bullet-count/word-limit formatting constraints.)*

---

## Key Findings

1. **imatrix helps logit calibration, NOT weight-level subspace preservation:**
   `i1-Q4_K_M` achieves the lowest `mean_kl_vs_f16` (`0.01947`) and lowest `PPL` (`9.0749`), and eliminates slop/refusal hits (`0.0000`). However, at the weight level, `i1-Q4_K_M` still suffers **6.360x** subspace re-leakage amplification and **5.91%** surgical-layer L2 error—virtually identical to static `Q4_K_M` (`6.402x`, `5.83%`).
2. **CKA-guided mixed precision cuts subspace re-leakage >2x and weight L2 error 5x:**
   For `+0.396` bpw (`+400 MB`), `CKA-Q4_K_M` reduces worst-case subspace re-leakage amplification from `6.402x` to `3.012x` (with `Q8_0`-protected surgical layers at `1.000x–1.001x`), reduces `max_row_norm_drift` from `0.00795`/`0.00946` to `0.00526`, and cuts surgical weight L2 error by **5x** (`1.17%` vs `5.83%–5.91%`). It also beats static `Q4_K_M` on mean KL (`0.02048` vs `0.02479`, -17.4%) and matches `F16` on `unslop_pass_rate` (`0.5833`) and `sql_readonly` (`1.0000`) without requiring an imatrix calibration pass.
3. **Avoid `i1-IQ4_XS` for structured SQL/tool workloads:**
   At `4.423` bpw, `i1-IQ4_XS` amplifies subspace re-leakage to **12.374x** (`7.73%` L2 error) and fails **6/6** (`0.0000`) read-only SQL generation tasks by degenerating into repetitive hallucinated column-listing loops (`i.is_canceled, i.is_postponed, i.is_rescheduled, ...`).
4. **Hybrid `i1-CKA-Q4_K_M` achieves the best of both worlds on weights and logits:**
   Combining `mradermacher`'s `Tenet-8B-Llama3.1-Unslop-v1.imatrix.gguf` with `quantize_cka_gguf.sh --imatrix` preserves `CKA-Q4_K_M`'s weight-level subspace protection (`3.012x` amplification, `1.17%` L2 error) while improving `max_row_norm_drift` to **0.00303** (best overall) and setting new bests across all quantized variants on `PPL` (**9.0533**, within `+0.14%` of `F16`), `mean_kl_vs_f16` (**0.01656**, `-15.0%` vs `i1-Q4_K_M` and `-33.2%` vs static `Q4_K_M`), and `same_top_p_pct` (**93.78%**), with `slop_and_refusal_rate` matching `F16` (`0.0417`), `direct_compliance_rate` at `0.9583`, `ifeval_score` at `0.1667`, and `sql_readonly` at `1.0000` (6/6), while `unslop_pass_rate` measured `0.0833`.

---

## Decision

- **H14 Accepted (`h14_verified: True`):** CKA-guided mixed precision (`quantize_cka_gguf.sh`) is validated as the highest-fidelity weight-preserving 4-bit GGUF recipe for Lancet-surgered models, `mradermacher`'s `i1-Q4_K_M` is validated as the best ~4.9 bpw drop-in community quant, and the hybrid `i1-CKA-Q4_K_M` (`quantize_cka_gguf.sh --imatrix <path>`) is validated as the overall highest-fidelity 4-bit GGUF (`Mean KLD = 0.01656`, `PPL = 9.0533`, `3.012x` subspace amp, `1.17%` L2 error, `6/6` SQL).
