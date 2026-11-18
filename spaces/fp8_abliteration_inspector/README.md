---
title: FP8 & Abliteration Brain-Damage Inspector
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: static
pinned: true
license: apache-2.0
short_description: Audit abliterated & Tenet LLMs for FP8 re-leakage & crest
---

# 🔬 FP8 & Abliteration Brain-Damage Inspector

Interactive Hugging Face Space & CLI toolkit by **Tenet Lab (Small Model Lab)** that audits **abliterated checkpoints**, **LoRA merges**, and **Lancet recipe bundles** (`ADR-005` / `ADR-006` / `ADR-007`) for **vLLM / OpenRouter FP8 (E4M3) deployment survival** — without loading multi-GB checkpoints into server RAM.

## Features

1. **Live Tenet v1 Benchmark Explorer (`3B` / `8B` / `12B`):** Compare measured **Naive Abliteration** vs **Tenet Lancet** (`H9` Bi-Projected Surgery, `H10` Subspace-Safe Unslop GRPO, `H11` FP8 E4M3 Survival, `H12` Cross-Domain Subspace-Safe Merge) across:
   - `Tenet-3B-Qwen2.5-Unslop-v1` (`Qwen/Qwen2.5-3B-Instruct`, Apple M4 Pro MPS)
   - `Tenet-8B-Llama3.1-Unslop-v1` (`meta-llama/Llama-3.1-8B-Instruct`, NVIDIA H100)
   - `Tenet-12B-Nemo-Unslop-v1` (`mistralai/Mistral-Nemo-Instruct-2407`, NVIDIA H100)
2. **Zero-Upload Browser & Hub Inspector:** Inspect any Hugging Face repository ID (`org/model`) via live Hub API or drop local `config.json`, `tokenizer_config.json`, `provider_readiness.json`, `lancet_recipe.json`, and `.safetensors` files directly in your browser (7/7 provider readiness gates + `Row Crest-Factor < 25.0` audit).
3. **CKA-Guided & Lancet-Safe Quantization Generator:** Generate `llama-quantize --tensor-type` flags (protecting high-SCR surgical layers in `Q8_0`/`Q6_K`), `vllm serve --quantization fp8`, and `mlx_lm` commands.

## Local CLI Usage

```bash
python app.py --inspect path/to/checkpoint_or_bundle
```
