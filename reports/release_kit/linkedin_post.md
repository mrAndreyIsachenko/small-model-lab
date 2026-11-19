# LinkedIn Release Post: Tenet v1 (`3B` / `8B` / `12B` / `32B`) & Lancet

**Recommended attachments (attach all 3 images in this order):**
1. `reports/release_kit/tenet_v1_x_card.png`
2. `reports/release_kit/tenet_v1_x_models_card.png`
3. `reports/release_kit/tenet_v1_x_space_card.png`

---

## Post Text (Plain Text for LinkedIn — No Markdown Stars/Backticks)

Most weight-edited ("abliterated") open-weight LLMs look fine in local FP16 tests — and then silently lose ~50% of their effect the moment you deploy them behind vLLM in FP8 (E4M3).

I'm open-sourcing Tenet v1 (3B, 8B, 12B, 32B) + Lancet (Apache-2.0): a norm-preserving weight surgery and subspace-safe post-training pipeline built to survive production FP8 inference and multi-adapter LoRA merges.

Why standard rank-1 orthogonal projection (W' = W - r r^T W) breaks in production:

1) Collateral Reasoning & Tool-Use Damage: Refusal and boilerplate ("slop") directions aren't isolated — projecting them out naively collides with reasoning and SQL/tool-calling subspaces, spiking KL divergence up to 3.76 on Qwen2.5-3B.

2) ~50% FP8 (E4M3) Subspace Re-Leakage: Without per-row L2-norm restoration, 3-bit mantissa rounding in FP8 E4M3 re-induces ~50% of the ablated direction across 3B, 8B, and 12B checkpoints.

3) Post-RL & LoRA Merge Interference: Running standard GRPO or merging a domain LoRA afterward overwrites the surgical null-space (0.80–1.13 re-leakage).

How we fixed all three in Lancet + Tenet v1:
• Dual-Anchor Modified Gram-Schmidt (Lancet): Orthogonalizes target directions against reasoning & SQL/tool anchor subspaces before surgery (ortho error <= 1e-7), cutting collateral KL by 89.9%–99.4% while keeping 96.9%–99.3% target suppression.
• FP8 E4M3 & Row Crest-Factor (< 25.0) Hygiene: Per-row L2-norm restoration cuts FP8 re-leakage from ~50.0% down to 2.7%, retaining 95.4%–99.4% of the surgical effect across Qwen2.5-3B, Llama-3.1-8B, and Mistral-Nemo-12B.
• Subspace-Safe GRPO & TIES Merge: Projecting LoRA/RL weight deltas into the orthogonal complement (I - U U^T) ΔW locks post-merge suppression retention at 100% (+78.1 to +79.7 pp over naive LoRA merging, with 98.0%–100% SQL accuracy retained).
• Streaming 32B/70B Shard Surgery: Processes multi-shard .safetensors checkpoints (Tenet-32B-Qwen2.5-Unslop-v1) in < 16 GB RAM with 7.45e-9 max absolute diff vs. in-memory execution.

We also built a zero-upload browser tool on Hugging Face Spaces — FP8 & Abliteration Inspector — where you can audit any checkpoint for vLLM/OpenRouter FP8 readiness (8/8 checks + Row Crest-Factor < 25.0) and generate CKA-guided llama-quantize --tensor-type flags (~5.28 bpw mixed precision) without uploading weights to a server.

Links:
• HF Collection (3B / 8B / 12B / 32B + Space): https://huggingface.co/collections/mr-andrey-isachenko/tenet-v1-bi-projected-unslop-and-fp8-safe-llm-family-6afed2048ba13a74fb7e0fac
• Interactive Inspector (HF Space): https://huggingface.co/spaces/mr-andrey-isachenko/fp8-abliteration-inspector
• GitHub & Technical Report (Apache-2.0): https://github.com/mrAndreyIsachenko/tenet-lab

#LLM #vLLM #FP8 #Quantization #OpenSourceAI #MachineLearning #AIInfrastructure
