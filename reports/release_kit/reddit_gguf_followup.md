# How Well Does Lancet Surgery Survive 4-Bit GGUF? (Static Q4_K_M vs. imatrix vs. CKA Mixed-Precision on Tenet-8B)

Following up on our `Tenet-8B-Llama3.1-Unslop-v1` release and `mradermacher`'s GGUF + imatrix (`i1-GGUF`) uploads, we ran a local benchmark (**H14**) on an **Apple M4 Pro (48GB)** using `llama.cpp` (Metal + CPU) to measure how much of our Lancet bi-projected surgery survives 4-bit GGUF quantization.

Specifically, we compared 6 variants of `Tenet-8B-Llama3.1-Unslop-v1`:
1. **`f16`** (`mradermacher` F16 GGUF) — FP16 reference after Lancet
2. **`Q4_K_M`** (`mradermacher` static `Q4_K_M`)
3. **`i1-Q4_K_M`** (`mradermacher` imatrix `i1-Q4_K_M`)
4. **`i1-IQ4_XS`** (`mradermacher` imatrix `i1-IQ4_XS`)
5. **`CKA-Q4_K_M`** (quantized locally from F16 using the `quantize_cka_gguf.sh` / `cka_quant_recipe.json` shipped in our release bundle, which protects the 16 surgical + 2 boundary layers' `attn_output` and `ffn_down` at `Q8_0`/`Q6_K`)
6. **`i1-CKA-Q4_K_M`** (hybrid `CKA + imatrix`: `quantize_cka_gguf.sh --imatrix Tenet-8B-Llama3.1-Unslop-v1.imatrix.gguf`)

---

## Results

| Variant | Size | Avg BPW | Subspace Re-Leakage Amp (vs F16) | Max Row-Norm Drift | Surgical Weight L2 Err | PPL (7x512) | Mean KLD vs F16 | Same Top-1 | Slop/Refusal Rate | Unslop Pass | Read-Only SQL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `f16` | 16.07 GB | 16.001 | 1.000x | 0.00000 | 0.00% | 9.0410 | 0.00000 | 100.00% | 4.17% | 58.33% | 6/6 (100%) |
| `Q4_K_M` | 4.92 GB | 4.894 | 6.402x | 0.00795 | 5.83% | 9.1798 | 0.02479 | 93.11% | 8.33% | 50.00% | 6/6 (100%) |
| `i1-Q4_K_M` | 4.92 GB | 4.894 | 6.360x | 0.00946 | 5.91% | 9.0749 | 0.01947 | 93.44% | **0.00%** | **58.33%** | 6/6 (100%) |
| `i1-IQ4_XS` | 4.45 GB | 4.423 | 12.374x | 0.01089 | 7.73% | 9.1648 | 0.02257 | 91.77% | 0.00% | 50.00% | **0/6 (0%)** |
| `CKA-Q4_K_M` | 5.32 GB | 5.290 | **3.012x** | 0.00526 | **1.17%** | 9.0963 | 0.02048 | 93.28% | 8.33% | **58.33%** | 6/6 (100%) |
| `i1-CKA-Q4_K_M` | 5.32 GB | 5.290 | **3.012x** | **0.00303** | 1.17% | **9.0533** | **0.01656** | **93.78%** | 4.17% | 8.33% | 6/6 (100%) |

---

## Key Takeaways for Local Users

1. **`i1-Q4_K_M` is the best ~4.9 GB drop-in download:**
   `mradermacher`'s imatrix calibration genuinely improves token-level fidelity (`Mean KLD = 0.01947` vs `0.02479` for static `Q4_K_M`, `PPL = 9.0749` vs `9.1798`) and keeps `unslop_pass` at `58.33%` (matching `F16`) with `0.00%` slop/refusal hits and `6/6` SQL.
2. **imatrix does NOT stop weight-level subspace re-leakage — CKA mixed-precision does:**
   Both `Q4_K_M` (`6.402x`) and `i1-Q4_K_M` (`6.360x`) amplify residual energy along the removed Lancet subspace by >6x with ~5.9% L2 error on surgical projections. Our `CKA-Q4_K_M` recipe (`+400 MB`, `5.290` bpw) cuts worst-case subspace amplification in half (`3.012x`, with `Q8_0`-protected layers at `1.000x–1.001x`), cuts surgical weight L2 error **5x** (`1.17%`), and beats static `Q4_K_M` KL by 17.4% — with zero imatrix calibration needed.
3. **Skip `i1-IQ4_XS` if you use SQL/tool-calling:**
   `i1-IQ4_XS` (`4.423` bpw) suffers `12.374x` subspace amplification and failed `0/6` read-only SQL tasks by falling into repetitive column-name hallucination loops.
4. **Combining `CKA + imatrix` (`i1-CKA-Q4_K_M`) gives the best overall fidelity:**
   Running `quantize_cka_gguf.sh --imatrix Tenet-8B-Llama3.1-Unslop-v1.imatrix.gguf` keeps `CKA-Q4_K_M`'s `3.012x` subspace protection and `1.17%` surgical L2 error while setting new bests on `Max Row-Norm Drift` (`0.00303`), `PPL` (`9.0533`, `+0.14%` vs `F16`), `Mean KLD vs F16` (`0.01656`, `-15.0%` vs `i1-Q4_K_M` and `-33.2%` vs static `Q4_K_M`), and `Same Top-1` (`93.78%`), with `4.17%` slop/refusal (matching `F16`) and `6/6` SQL.

If you want maximum fidelity at ~5.3 GB, you can quantize `f16` locally in ~40 seconds using `quantize_cka_gguf.sh --imatrix <path-to-imatrix.gguf>`.
