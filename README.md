# Small Model Lab

**Research Question:** *How small and efficient can a language model become while retaining most of the capability required for production data-pipeline incident reasoning — and when does updating weights outperform enriching context?*

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](pyproject.toml)

---

## Overview

**Small Model Lab** is an end-to-end language model systems and applied research project structured into four tightly coupled sprints:

1. **Sprint 1 — Core Transformer, Custom `@triton.jit` Kernels & GPU Pretraining (`Phase I Lean`):** Modern Llama/Qwen-compatible decoder (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, intra-document causal masking, `KV-cache`) + custom `@triton.jit` fused `RMSNorm`, `SwiGLU`, and chunked `CrossEntropy` forward/backward kernels (`src/sml/kernels/`), validated against OpenAI `gpt2` (`124M`) logits (`< 1e-4`), trained with a 32k BPE tokenizer on `FineWeb-Edu` (`4.0B` tokens with Warmup–Stable–Decay and a `1.5B` Chinchilla-optimal cooldown branch).
2. **Sprint 2 — Trainable Analytical & Incident RL Environment (`StatefulToolEnv`):** `verifiers`-compatible multi-turn environment (`StatefulToolEnv` + `load_environment`) built on `dbt Core + DuckDB + sqlglot` with isolated MCP tools, multi-seed counterfactual fault injection (`product_bug`, `data_pipeline_artifact`, `mix_shift` + confounders), and a Gated Multiplicative `Rubric` (`R_total = R_outcome · R_evidence · R_process`).
3. **Sprint 3 — Adaptation, Distillation & Multi-Turn RL (`GRPO`):** Controlled comparison on `1.5B–3B` open weights: Prompting vs. Hybrid RAG vs. LoRA/SFT vs. Teacher (`30B+`) Rejection-Sampled Distillation (via the hardened OpenRouter gateway in `src/sml/gateway/` with strict recursive secret/raw-row egress guards) and multi-turn `GRPO`, including a pre-registered reward-hacking ablation (`H5`: Gated Multiplicative vs. Naive Additive reward).
4. **Sprint 4 — Specialist Surgery & Production Inference (`vLLM`):** Layer-wise representation drift (`CKA`), CKA-guided mixed-precision allocation (RCO principle), rank-1 subspace weight surgery (W' = W - r(r^T W)), quantization sensitivity (`FP8` / `INT4 AWQ`), layer pruning, component-level Gated Rubric degradation audit (H7, H8), and load-tested serving economics (`TTFT`, `TPOT`, `req/s`, `$/1k` incidents Pareto frontier).

> **Zero Placeholder Numbers Policy:** No synthetic or estimated numbers are ever written into result tables. All tables in `reports/` are compiled directly from machine-readable experiment manifests in `results/<run_id>/` via `sml report`.

---

## Pre-Registered Hypotheses & Status (Sprints 1–4)

| ID | Hypothesis | Pre-Registered Acceptance Criterion | Status |
| :--- | :--- | :--- | :--- |
| **H1** | **Mathematical & Causal Equivalence:** Custom `Transformer` in `gpt2-compat` mode matches `GPT2LMHeadModel` (`124M`) logits; naive `O(T^2)` GQA matches `SDPA`; `KV-cache` matches full forward; intra-document mask blocks cross-`<\|eos\|>` leakage. | `max_abs_logit_diff < 1e-4` (`fp32`); cross-doc leakage `== 0.0` | **Verified** (`sdpa=0.00e+00`, `naive=5.96e-6`, `kv_cache=5.13e-6`, `leakage=0.00e+00` in [`results/h1_gpt2_parity/`](results/h1_gpt2_parity/final_metrics.json)) |
| **H2** | **GPU Throughput & `@triton.jit` Waterfall:** Cumulative system optimizations (`fp32 naive` → `fp32 sdpa` → `bf16 sdpa` → `bf16 fused kernels` → `torch.compile`) yield measured throughput gains and high hardware utilization. | Measured 5-stage waterfall in `reports/sprint1/`; `MFU >= 30%` on `1x H100 SXM` (`1.40x` local MPS speedup verified) | **Code Ready (`sml profile-h2`)** |
| **H3** | **Chinchilla Optimal (`1.5B`) vs. Overtrained (`4.0B`) WSD Cooldown:** Overtraining `sml-100m` (`75.52M` non-embedding params) past `20x` tokens/param via WSD cooldown branching improves `val bpb` and zero-shot accuracy at identical inference cost. | Improvement exceeds `2σ` of 3-seed `sml-xs` noise floor | **Code Ready (`sml train`)** |
| **H4** | **In-Context Induction Phase Transition:** Log-spaced checkpoints reveal an abrupt phase transition in intra-document `loss(late) - loss(early)` and synthetic sequence copying. | Phase transition localized across 15 log-spaced checkpoints | **Code Ready (`sml train`)** |
| **H5** | **Gated Multiplicative vs. Naive Additive Rubric (Anti-Reward-Hacking):** `G_evidence · R_correctness` prevents premature guessing and unsafe SQL during multi-turn `GRPO`. | `oracle_gated > 1.0`, `lazy_gated == 0.0` vs. `lazy_naive == 0.75`, and `H5_skill_difference >= 0.15` | **Verified** (`Δskill = +0.8625` in [`results/h5_env_verification/`](results/h5_env_verification/final_metrics.json) & [`results/h6_adaptation_comparison/`](results/h6_adaptation_comparison/adaptation_comparison.json)) |
| **H6** | **Context Enrichment (`Hybrid RAG`) vs. Weight Updates (`Distillation + GRPO`):** Hybrid BM25 + char 3-gram RRF resolves knowledge-bound table lookups (`>= 85%`), while weight updates dominate skill-bound multi-turn hypothesis elimination (`>= 20 pp` gain). | `rag_knowledge_score >= 0.85` AND `H6_skill_difference >= 0.20` | **Verified** (`rag_kb = 1.00`, `Δskill = +0.5050` in [`results/h6_adaptation_comparison/`](results/h6_adaptation_comparison/adaptation_comparison.json)) |
| **H7** | **Representation-Constrained Compression (CKA-Guided Mixed-Precision RCO):** Allocating 8-bit precision to the top-k highest per-block CKA-drift layers reduces KL divergence by `>= 20%` vs. an equal-bit bottom-k control (`5.33 bits/weight`). | `h7_kl_reduction_ratio >= 0.20` at identical average bits/weight | **Verified** (`73.33%` KL reduction: `0.005168` vs. `0.019381` at `5.33 bits` in [`results/h7_h8_surgery_pareto/`](results/h7_h8_surgery_pareto/surgery_pareto.json)) |
| **H8** | **Rank-1 Subspace Projection (`W' = W - r(r^T W)`):** Closed-form orthogonal projection zeroes out targeted residual stream writes along contrastive direction `r` (`< 1e-5`). | `max_ortho_residual < 1e-5` AND `layer_write_proj_after < 1e-5` | **Verified** (`ortho_residual = 0.0`, `write_proj: 7.91e-3 → 2.57e-6` in [`results/h7_h8_surgery_pareto/`](results/h7_h8_surgery_pareto/surgery_pareto.json)) |

---

## Verified Model Configurations (Sprint 1)

| Model | `d_model` | Layers | Q / KV Heads | `d_ff` | Vocab | Non-Emb Params | Total Params (Tied) | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sml-nano` | 256 | 4 | 4 / 2 | 768 | 32,768 | `3,148,032` | `11,536,640` | Local CPU/MPS unit tests & TinyStories smoke check |
| `sml-xs` | 512 | 8 | 8 / 4 | 1,408 | 32,768 | `23,601,664` | `40,378,880` | 24 GB GPU pilot & 3-seed noise floor (`0.4B` tokens) |
| **`sml-100m`** | **768** | **12** | **12 / 4** | **2,048** | **32,768** | **`75,516,672`** | **`100,682,496`** | **Primary Sprint 1 model (`1.5B` branch + `4.0B` full)** |
| `gpt2-124m` | 768 | 12 | 12 / 12 | 3,072 | 50,257 | `85,842,432` | `124,439,808` | Reference parity target for `H1` (`gpt2-compat`) |

---

## Quickstart (Local macOS / Apple Silicon & Ephemeral Cloud GPU)

```bash
# 1. Install dependencies into an isolated Python 3.12 virtualenv
make sync

# 2. Run all 32 unit & integration tests and ruff linter
make test
make lint

# 3. Verify H1 (GPT-2 124M parity), H5 (StatefulToolEnv), H6 (Context vs. Weights), and H7/H8 (Weight Surgery & Pareto)
make verify-h1
make verify-env
make compare-adaptation
make run-surgery
make report

# 4. Run end-to-end M2+M3 local pretraining smoke test on Apple Silicon MPS / CPU
make train-smoke

# 5. Profile the 5-stage H2 throughput waterfall locally
make profile-h2-local
```

### Running `M4–M5` on an Ephemeral Cloud GPU (`RTX 4090` / `H100 SXM`)

```bash
# 1. Bootstrap Ubuntu + NVIDIA GPU instance (installs uv, Python 3.12, locked deps, runs pytest)
./scripts/remote/bootstrap.sh

# 2. Run the full M4-M5 GPU experiment pipeline (with 15-min idle GPU auto-shutdown watchdog):
#    - H1 parity verification
#    - H2 5-stage MFU & Triton fused-kernel waterfall
#    - FineWeb-Edu streaming BPE + uint16 shard generation (< 15 GB disk discipline)
#    - H3/H4 WSD pretraining with cooldown branching & 15 log-spaced checkpoints
#    - Markdown report compilation to reports/sprint1/summary.md
GPU_PROFILE=h100 ./scripts/remote/run_m4_m5_gpu.sh
# Or on a 24 GB RTX 4090 pilot instance:
GPU_PROFILE=rtx4090 MAX_DOCS=50000 MAX_STEPS=2000 ./scripts/remote/run_m4_m5_gpu.sh
```

---

## Architecture & Decisions

* [Master Roadmap & Sprint Specifications](docs/roadmap.md)
* [Sprint 1 Measured Results Summary](reports/sprint1/summary.md)
* [ADR-001: Four-Sprint Architecture vs. Eight Loose Phases](docs/decisions/001-four-sprint-architecture.md)
* [ADR-002: Grouped-Query Attention (GQA) & WSD Cooldown Branching](docs/decisions/002-gqa-and-wsd-schedule.md)
* [ADR-003: Streaming Data Shards & `< 15 GB` Local Disk Discipline](docs/decisions/003-streaming-disk-discipline.md)
* [ADR-004: Multi-Turn Verifiable RL Environment (`StatefulToolEnv` + Gated Anti-Hacking `Rubric`)](docs/decisions/004-verifiable-rl-analytics-environment.md)
