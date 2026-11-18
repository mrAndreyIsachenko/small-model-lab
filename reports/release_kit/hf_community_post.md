# Hugging Face Post / X Thread: Tenet v1 (`3B` / `8B` / `12B`) & FP8 Abliteration Inspector

## Short HF Community Post

Introducing **Tenet v1** — bi-projected, norm-preserving **Lancet** unslop/de-refusal surgery + **Subspace-Safe GRPO** across **3B, 8B, and 12B** (`Qwen2.5-3B`, `Llama-3.1-8B`, `Mistral-Nemo-12B`), certified for **vLLM / OpenRouter FP8 (`E4M3`)**!

Naive abliteration loses **~50% of its effect under FP8 (`E4M3`) quantization** (`0.4996–0.5003` re-leakage) and spikes reasoning KL (`3.7607` on `Qwen2.5-3B`). With Dual-Anchor Soft-SCR + row-$L_2$-norm preservation + `Row Crest-Factor < 25.0` hygiene:
- **H9 Target Suppression / KL Reduction:** `96.94%–99.30%` suppression with `89.89%–99.41%` lower collateral KL
- **H10 Subspace-Safe GRPO:** `slop_and_refusal_rate` down to `0.0239–0.0909`, `100%` suppression retained post-merge
- **H11 FP8 (`E4M3`) Survival:** `95.41%–99.36%` retention, `2.70%–18.39%` re-leakage (vs `~50%` naive), `Crest-Factor 6.32–17.27 < 25.0`
- **H12 TIES + Gram-Schmidt Merge:** `+78.06` to `+79.65 pp` over naive LoRA merge (`97.99%–100%` SQL retention)
- **H13 Streaming Surgery:** `7.45e-09` max abs diff vs in-memory for 32B/70B shard streaming

- **Interactive Space:** https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector
- **HF Family & Collection:** https://huggingface.co/mr-andrey-isachenko
- **Models:**
  - `mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1`
  - `mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1`
  - `mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1`
  - `mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1`
- **GitHub:** https://github.com/mrAndreyIsachenko/tenet-lab

---

## X / Twitter Thread (6 Posts)

**1/6**
Naive abliterated LLMs silently break under vLLM FP8 (`E4M3`) quantization:
- Up to `3.76` KL collateral damage on reasoning/tool-use
- **~50% subspace re-leakage** (`0.500`) after E4M3 3-bit mantissa rounding
- `0.80–1.13` re-leakage if you run RL or merge a LoRA afterward

We fixed all 3 with **Lancet + Tenet v1** (`3B`, `8B`, `12B`). 🧵👇

**2/6**
Why does naive abliteration ($W' = W - \hat{r}\hat{r}^\top W$) break?
1) Refusal/slop directions collide with reasoning & SQL/tool subspaces.
2) Without per-row $L_2$-norm restoration, FP8 E4M3 rounding re-induces the ablated direction (~50% re-leakage across 3B, 8B, and 12B!).

**3/6**
Our pipeline (`H9`–`H13`):
- **H9 Lancet:** Dual-Anchor Modified Gram-Schmidt (`ortho err <= 1e-7`) + SCR layer selection + row-norm restoration (`89.89%–99.41%` KL reduction, `96.9%–99.3%` suppression)
- **H10 Tenet GRPO:** Gated Multiplicative Rubric + orthogonal LoRA merge $(I - U U^\top)\Delta W$ (`1.0` suppression retained)

**4/6**
Production **H11 FP8 (`E4M3`)** & **H12 Merge** numbers:
- `Tenet-3B-Qwen2.5`: `95.41%` FP8 retention, re-leakage `0.5003 → 0.0270`, Crest `17.27 < 25`
- `Tenet-8B-Llama3.1`: `99.36%` FP8 retention, re-leakage `0.5000 → 0.0278`, Crest `6.32 < 25`
- `Tenet-12B-Nemo`: `96.78%` FP8 retention, re-leakage `0.4996 → 0.1839`, Crest `10.65 < 25`
- **H12 TIES + GS Merge:** `+78.1` to `+79.7 pp` over naive merge

**5/6**
Want to audit your own abliterated checkpoint or LoRA merge without downloading 20GB+ weights?
Try our zero-upload static HF Space **FP8 & Abliteration Brain-Damage Inspector**:
- 7/7 FP8 provider readiness checks + `Row Crest-Factor < 25.0`
- CKA-guided `llama-quantize --tensor-type` generator (`~5.28 bpw`)
🔗 https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector

**6/6**
Everything is Apache-2.0 open-source with machine-verified JSON manifests (`H1–H13`):
- Models & Space: https://huggingface.co/mr-andrey-isachenko
- Code & Technical Report: https://github.com/mrAndreyIsachenko/tenet-lab
