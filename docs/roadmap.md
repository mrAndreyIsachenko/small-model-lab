# Small Model Lab — Master Roadmap & Sprint 1 Specification

## 1. Strategic Objective

**Small Model Lab** is an end-to-end language model systems and applied research project designed to investigate:
1. How small and efficient a language model can become while retaining specialized capability for production data-pipeline incident reasoning.
2. When updating model weights (LoRA, SFT, Rejection-Sampled Distillation, RLVR/GRPO) outperforms enriching runtime context (Prompting, Hybrid RAG) on `knowledge-bound` vs. `skill-bound` failures.
3. How pretraining mechanics, post-training weight changes, and quantization interact under production GPU serving constraints (`vLLM`).

## 2. Four-Sprint Program Structure (~10–12 Weeks @ 10h/week)

| Sprint | Target Duration | Core Engineering & Research Scope | Primary Public Artifacts |
| :--- | :--- | :--- | :--- |
| **Sprint 1: Core Transformer & GPU Pretraining (`Phase I Lean`)** | Weeks 1–4 (~35–40h) | Implement Llama/Qwen-compatible decoder (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, `KV-cache`, intra-doc causal mask), verify `gpt2` (`124M`) logit parity (`< 1e-4`), train 32k BPE, stream-shard `FineWeb-Edu`, profile MFU waterfall, train `sml-100m` (`4.0B` WSD tokens + `1.5B` Chinchilla branch). | `sml-100m` weights + 15 log-spaced checkpoints + 32k BPE + shard manifest on HF Hub + Report #1. |
| **Sprint 2: Executable Incident Benchmark** | Weeks 5–6 (~20–25h) | Build deterministic fault-injection sandbox on `dbt Core + DuckDB + sqlglot`, separate `knowledge-bound` vs. `skill-bound` failure modes, implement programmatic verification + bootstrap 95% CI harness. | Standalone benchmark dataset + executable evaluator + Report #2. |
| **Sprint 3: Adaptation & Distillation (`Context vs. Weights`)** | Weeks 7–9 (~25–30h) | Compare Prompting, Hybrid RAG, LoRA/SFT, Teacher (`30B+`) Rejection-Sampled Distillation, and lightweight verifiable-reward RL (`GRPO`) into `1.5B–3B` open weights. | Distilled specialist model (`1.5B–3B`) + verified reasoning dataset + Report #3. |
| **Sprint 4: Specialist Surgery & Production Inference (`vLLM`)** | Weeks 10–11 (~20–25h) | Inspect layer-wise CKA drift (Base vs. Student), evaluate `FP8` / `INT4 AWQ` and layer-pruning degradation on structured outputs, benchmark `vLLM` continuous batching & prefix caching (`TTFT`, `TPOT`, `$/1k` incidents Pareto frontier). | Quantized specialist checkpoints + `vLLM` benchmark suite + Final Pareto Report #4. |

## 3. Sprint 1 Milestone Plan

* **`M0` (Week 1, Local Mac):** Repository scaffold, `uv` Python 3.12 lockfile, typed config/math engine, `results/<run_id>/` manifest & cost tracker, `sml report` compiler, GitHub Actions CI.
* **`M1` (Week 1, Local Mac):** Transformer implementation (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, `KV-cache`, `gpt2_adapter`, `hooks/`), `H1` logit parity test (`< 1e-4`), and local `sml-nano` MPS smoke run on `TinyStories`.
* **`M2` (Week 2, Local Mac):** 32k byte-level BPE training, streaming `FineWeb-Edu` `uint16` shard pipeline (`< 15 GB` peak local disk), 13-gram decontamination audit, upload shards to HF Hub.
* **`M3` (Week 2, Local Mac):** Trainer (`AdamW`, WSD schedule with cooldown branching, `bf16`, `torch.compile`, deterministic resume, async HF checkpoint uploader) and evaluation harness (`bpb`, MC zero-shot tasks, induction metric).
* **`M4` (Week 3, Cloud 24 GB GPU):** Ephemeral GPU bootstrap (`RTX 4090` / `L4`), idle watchdog test, `H2` throughput/MFU waterfall via `torch.profiler`.
* **`M5` (Week 3, Cloud `1x H100 80GB SXM`):** 3-seed `sml-xs` noise baseline (`0.4B` tokens each) + main `sml-100m` WSD run (`4.0B` tokens + `1.5B` Chinchilla cooldown branch, 15 log-spaced checkpoints).
* **`M6` (Week 4, Local + Cloud Eval):** Zero-shot suite (`HellaSwag`, `ARC-Easy`, `PIQA`, `BLiMP`), `lm-evaluation-harness` cross-check, one-time `test` split evaluation, public HF release, and Article #1.
