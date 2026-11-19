# Hugging Face Post / X Thread: Tenet v1 (`3B` / `8B` / `12B`) & FP8 Abliteration Inspector

## Short HF Community Post

Introducing **Tenet v1** — bi-projected, norm-preserving **Lancet** unslop/de-refusal surgery + **Subspace-Safe GRPO** across **3B, 8B, and 12B** (`Qwen2.5-3B`, `Llama-3.1-8B`, `Mistral-Nemo-12B`), certified for **vLLM / OpenRouter FP8 (`E4M3`)**!

Naive abliteration loses **~50% of its effect under FP8 (`E4M3`) quantization** (`0.4996–0.5003` re-leakage) and spikes reasoning KL (`3.7607` on `Qwen2.5-3B`). With Dual-Anchor Soft-SCR + row-$L_2$-norm preservation + `Row Crest-Factor < 25.0` hygiene:
- **H9 Target Suppression / KL Reduction:** `96.94%–99.30%` suppression with `89.89%–99.41%` lower collateral KL
- **H10 Subspace-Safe GRPO:** `slop_and_refusal_rate` down to `0.0239–0.0909`, `100%` suppression retained post-merge
- **H11 FP8 (`E4M3`) Survival:** `95.41%–99.36%` retention, `2.70%–18.39%` re-leakage (vs `~50%` naive), `Crest-Factor 6.32–17.27 < 25.0`
- **H12 TIES + Gram-Schmidt Merge:** `+78.06` to `+79.65 pp` over naive LoRA merge (`97.99%–100%` SQL retention)
- **H13 Streaming Surgery:** `7.45e-09` max abs diff vs in-memory for 32B/70B shard streaming
- **External Benchmarks (`lm-eval`, Base vs Naive vs Tenet v1):**
  - `Tenet-3B-Qwen2.5-Unslop-v1` (`n=40`): `IFEval (strict)` `0.525 -> 0.500 (naive) -> 0.600` (`+10.0 pp` vs naive, `+7.5 pp` vs base); `GSM8K CoT (flex)` `0.525 -> 0.350 (naive) -> 0.450` (`+10.0 pp` vs naive); `Unslop Pass` `1.00 -> 0.96 -> 1.00`
  - `Tenet-8B-Llama3.1-Unslop-v1` (`n=30`): `GSM8K CoT (strict)` `0.667 -> 0.700` (`+3.3 pp` vs base), `GSM8K CoT (flex)` `0.667 -> 0.733` (`+6.7 pp` vs base); `IFEval (strict)` `0.600` (`0.733` inst); `Unslop Pass` `1.00`
- **Community GGUF (`mradermacher`, recommended `i1-Q4_K_M` / `i1-Q6_K`):**
  - [`Tenet-3B-Qwen2.5-Unslop-v1-i1-GGUF`](https://huggingface.co/mradermacher/Tenet-3B-Qwen2.5-Unslop-v1-i1-GGUF) | [`Tenet-8B-Llama3.1-Unslop-v1-i1-GGUF`](https://huggingface.co/mradermacher/Tenet-8B-Llama3.1-Unslop-v1-i1-GGUF) | [`Tenet-12B-Nemo-Unslop-v1-i1-GGUF`](https://huggingface.co/mradermacher/Tenet-12B-Nemo-Unslop-v1-i1-GGUF)

- **Interactive Space:** https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector
- **HF Collection:** https://huggingface.co/collections/mr-andrey-isachenko/tenet-v1-bi-projected-unslop-and-fp8-safe-llm-family-6afed2048ba13a74fb7e0fac
- **Models:**
  - `mr-andrey-isachenko/Tenet-32B-Qwen2.5-Unslop-v1` (H13 streaming shard surgery)
  - `mr-andrey-isachenko/Tenet-12B-Nemo-Unslop-v1`
  - `mr-andrey-isachenko/Tenet-8B-Llama3.1-Unslop-v1`
  - `mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1`
  - `mr-andrey-isachenko/Tenet-16K-Nano-Lab-v1`
- **GitHub:** https://github.com/mrAndreyIsachenko/tenet-lab

---

## X / Twitter Thread (5 Posts, Indie-Lab Style, <= 250 chars each)

**1/5** (attach `reports/release_kit/tenet_v1_x_card.png`)
Most abliterated LLMs lose ~50% of their effect the moment you quantize them to vLLM FP8.

We're open-sourcing Tenet v1 (3B, 8B, 12B, 32B) + Lancet — norm-preserving weight surgery built to survive FP8 & LoRA merges:
• 50% → 2.7% FP8 re-leakage
• Up to 99.4% lower reasoning KL

🧵👇

**2/5**
Why does standard abliteration (W' = W - r r^T W) break in production?

1) Refusal/slop vectors collide with reasoning & SQL/tool-use subspaces (spiking KL up to 3.76).
2) Without per-row L2-norm restoration, 3-bit FP8 (E4M3) mantissa rounding re-induces ~50% of the ablated direction.

**3/5**
How Lancet + Tenet fixes it:
• Dual-Anchor Gram-Schmidt protects reasoning & tool-use (89.9%–99.4% lower KL, 96.9%–99.3% suppression)
• Per-row L2-norm restoration + Crest < 25 gives 95.4%–99.4% FP8 retention
• Orthogonal LoRA/GRPO merge keeps 100% suppression

**4/5**
Want to check if your own checkpoint or LoRA merge survives FP8 or GGUF?

Try our zero-upload browser tool — FP8 & Abliteration Inspector:
• 8/8 vLLM/OpenRouter FP8 readiness + Row Crest-Factor < 25
• CKA-guided llama-quantize flag generator (~5.28 bpw)
https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector

**5/5**
All code, streaming 32B/70B shard surgery (7.45e-9 diff), and checkpoints (Qwen2.5 3B & 32B, Llama-3.1 8B, Mistral-Nemo 12B) are out under Apache-2.0:

• HF Collection: https://huggingface.co/collections/mr-andrey-isachenko/tenet-v1-bi-projected-unslop-and-fp8-safe-llm-family-6afed2048ba13a74fb7e0fac
• GitHub: https://github.com/mrAndreyIsachenko/tenet-lab
