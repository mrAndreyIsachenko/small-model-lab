# Why naive abliterated models silently break under vLLM FP8 quantization (50% subspace re-leakage & Row Crest-Factor > 25) — and how we fixed it with bi-projected norm-preserving Lancet surgery across 3B, 8B, and 12B

**TL;DR:** Naive orthogonal abliteration ($W' = W - \hat{r}\hat{r}^\top W$) looks fine in FP16 synthetic refusal tests, then silently breaks in production under **three failure modes**:
1. **Collateral KL Explosion** on reasoning/tool-use subspaces (up to `3.7607` KL on `Qwen2.5-3B-Instruct`).
2. **~50% Subspace Re-Leakage under FP8 (`E4M3`)** (`0.4996–0.5003` across `3B`, `8B`, `12B`) because 3-bit mantissa rounding without row-$L_2$-norm preservation re-induces the ablated direction, while weight outliers can blow past `Row Crest-Factor > 25.0`.
3. **RL & Adapter Merge Re-Leakage** (`0.805–1.133` re-leakage when running GRPO or merging LoRAs naively).

We open-sourced the full **Lancet + Tenet** pipeline ([GitHub: `mrAndreyIsachenko/tenet-lab`](https://github.com/mrAndreyIsachenko/tenet-lab)), an interactive zero-upload **FP8 & Abliteration Inspector** ([HF Space: `mr-andrey-isachenko/fp8-abliteration-inspector`](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector)), and the **Tenet v1** release bundles (`3B`, `8B`, `12B`, plus our `16K` Nano verification repo) on Hugging Face: [`mr-andrey-isachenko`](https://huggingface.co/mr-andrey-isachenko).

---

## 1. The Post-Mortem: Why Naive Abliteration Fails in Production

Most open-source abliterated checkpoints only check two things in `bf16`/`fp16`: *"Did the refusal rate drop?"* and *"Does MMLU look okay?"* When you deploy those weights behind **vLLM / OpenRouter FP8 (`E4M3`)** or compose them with domain LoRAs, three things go wrong:

### Failure 1: Reasoning Anchor Collision (`H9`)
In instruction-tuned models (`Qwen2.5-3B`, `Llama-3.1-8B`, `Mistral-Nemo-12B`), refusal and verbose "slop" directions are **not** orthogonal to structured reasoning or SQL/tool-calling directions. Subtracting $\hat{r}\hat{r}^\top W$ directly strips useful variance:
* `Qwen2.5-3B-Instruct`: `naive_kl_collateral = 3.7607`
* `Mistral-Nemo-Instruct-2407`: `naive_kl_collateral = 0.2968`
* `Llama-3.1-8B-Instruct`: `naive_kl_collateral = 0.1337`

### Failure 2: `FP8 (E4M3)` Subspace Re-Leakage (~50%) & Row Crest-Factor (`H11`)
`FP8 E4M3` only has **3 mantissa bits**. Without per-row $L_2$-norm restoration and outlier control ($\text{Row Crest-Factor} = \max_i \|W_{i,:}\|_\infty / \text{RMS}(W_{i,:}) < 25.0$), quantizing an abliterated matrix onto the E4M3 grid re-introduces a large projection along the supposedly zeroed direction. Across **every** tested scale (`nano`, `3B`, `8B`, `12B`), naive abliteration measured **`fp8_subspace_releakage_naive = 0.4996–0.5003` (~50%)**.

### Failure 3: Post-Surgery RL & LoRA Merge Re-Leakage (`H10`, `H12`)
Running GRPO or merging a domain adapter ($\Delta W_{\text{SQL}} + \Delta W_{\text{unslop}}$) directly onto abliterated weights writes straight back into the ablated subspace (`unprojected_lora_releakage = 0.8300` on `3B`, `1.1328` on `8B`, `0.8052` on `12B`), and singular-subspace interference collapses naive cross-domain merges.

---

## 2. The Fix: Lancet (`H9`) → Subspace-Safe GRPO (`H10`) → FP8 Gate (`H11`) → TIES-GS Merge (`H12`) → Streaming Surgery (`H13`)

1. **`H9` — Lancet Dual-Anchor Soft-SCR Surgery:**
   - Extracts contrastive refusal/slop directions alongside a **reasoning anchor** and **tool-SQL anchor** in `float32`.
   - Orthogonalizes target directions against the anchors via **Modified Gram-Schmidt** (`anchor_orthogonality_error <= 1e-7`).
   - Selects top-$k$ layers by **Signal-to-Collateral-Ratio (SCR)** and applies joint projection to `self_attn.o_proj` & `mlp.down_proj` with **exact per-row $L_2$-norm restoration**.
2. **`H10` — Tenet Subspace-Safe Unslop GRPO:**
   - Trains GRPO with a **Gated Multiplicative Rubric** ($R_{\text{total}} = R_{\text{outcome}} \cdot R_{\text{evidence}} \cdot R_{\text{process}}$, eliminating lazy-guess reward hacking) + slop/refusal penalty.
   - Merges LoRA updates only through the orthogonal complement $(I - U_{\text{ablate}} U_{\text{ablate}}^\top)\Delta W_{\text{LoRA}}$ with row-norm preservation (`post_merge_suppression_retention = 1.0` across all scales).
3. **`H11` — FP8 (`E4M3`) Survival & Crest-Factor Hygiene:**
   - Tests both Per-Row and Per-Tensor E4M3 scales and enforces `max_row_crest_factor < 25.0` (7/7 provider readiness checks).
4. **`H12` — Cross-Domain TIES + Gram-Schmidt Merge:**
   - Projects $\Delta W_{\text{unslop}}$ against both $U_{\text{refusal/slop}}$ and the top-$k$ left singular vectors $U_{\text{SQL}}$ of $\Delta W_{\text{SQL}}$ before TIES trim/sign-election/disjoint merge.
5. **`H13` — Streaming 32B/70B Shard Surgery:**
   - Shard-by-shard `safetensors` streaming architecture (`Qwen/Qwen2.5-32B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct` targets, `< 16 GB` RAM / `< 15 GB` disk), verified to `7.45e-09` max abs surgery diff and `0.0` merge diff against in-memory execution.

---

## 3. Measured Results (`3B` / `8B` / `12B`)

All numbers below come directly from `results/h{9..13}*/final_metrics.json` and `reports/tenet_technical_report.md`:

| Model | H9 Target Supp. | H9 KL Red. (Naive → Lancet) | H10 Slop/Refusal (Gated IFEval) | H11 FP8 Re-Leak (Naive → Lancet) | H11 FP8 Retention | Max Row Crest (`< 25`) | H12 TIES vs Naive Gain |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Tenet-3B-Qwen2.5-Unslop-v1`** (`Qwen2.5-3B-Instruct`, M4 Pro MPS) | `96.94%` | `99.41%` (`3.7607 → 0.0220`) | `0.0909` (`0.7230`) | `0.5003 → 0.0270` | `95.41%` | `17.2706` | `+78.48 pp` (`SQL 1.0000`) |
| **`Tenet-8B-Llama3.1-Unslop-v1`** (`Llama-3.1-8B-Instruct`, H100) | `99.30%` | `89.89%` (`0.1337 → 0.0135`) | `0.0410` (`0.8114`) | `0.5000 → 0.0278` | `99.36%` | `6.3244` | `+79.65 pp` (`SQL 0.9799`) |
| **`Tenet-12B-Nemo-Unslop-v1`** (`Mistral-Nemo-12B`, H100) | `99.14%` | `96.72%` (`0.2968 → 0.0097`) | `0.0239` (`0.6272`) | `0.4996 → 0.1839` | `96.78%` | `10.6508` | `+78.06 pp` (`SQL 1.0000`) |

---

## 4. Try It / Inspect Your Own Checkpoints

* **Interactive Space (Zero-Upload Browser Audit):** [FP8 & Abliteration Brain-Damage Inspector](https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector) — drop in any HF repo ID or local `safetensors` + `lancet_recipe.json` to check `Row Crest-Factor < 25.0`, FP8 survival, and generate CKA-safe `llama-quantize --tensor-type` flags (`~5.28 bpw` default).
* **Hugging Face Family:**
  * [`mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1)
  * [`mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1)
  * [`mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1)
  * [`mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1`](https://huggingface.co/mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1) (16K-param Nano verification & `H1–H13` artifacts)
* **Code & Technical Report:** [`github.com/mrAndreyIsachenko/tenet-lab`](https://github.com/mrAndreyIsachenko/tenet-lab) (Apache-2.0)

Happy to answer any questions on the Modified Gram-Schmidt biprojection math, the `FP8 E4M3` crest-factor gate, or the `StatefulToolEnv` GRPO setup!
