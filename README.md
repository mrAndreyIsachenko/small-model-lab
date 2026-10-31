# Small Model Lab

**Research Question:** *How small and efficient can a language model become while retaining most of the capability required for production data-pipeline incident reasoning — and when does updating weights outperform enriching context?*

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](pyproject.toml)

---

## Overview

**Small Model Lab** is an end-to-end language model systems and applied research project structured into four tightly coupled sprints:

1. **Sprint 1 — Core Transformer & GPU Pretraining (`Phase I Lean`):** Modern Llama/Qwen-compatible decoder (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, intra-document causal masking, `KV-cache`) implemented from scratch in PyTorch, validated against OpenAI `gpt2` (`124M`) logits (`< 1e-4`), trained with a 32k BPE tokenizer on `FineWeb-Edu` (`4.0B` tokens with Warmup–Stable–Decay and a `1.5B` Chinchilla-optimal cooldown branch).
2. **Sprint 2 — Executable Data Incident Benchmark:** Deterministic fault-injection sandbox built on `dbt Core + DuckDB + sqlglot`, separating `knowledge-bound` and `skill-bound` failures with programmatic grading and bootstrap 95% confidence intervals.
3. **Sprint 3 — Adaptation & Knowledge Distillation:** Controlled comparison on `1.5B–3B` open weights: Prompting vs. Hybrid RAG vs. LoRA/SFT vs. Teacher (`30B+`) Rejection-Sampled Distillation and verifiable-reward RL (`GRPO`).
4. **Sprint 4 — Specialist Surgery & Production Inference (`vLLM`):** Layer-wise representation drift (`CKA`), quantization sensitivity (`FP8` / `INT4 AWQ`), layer pruning, and load-tested serving economics (`TTFT`, `TPOT`, `req/s`, `$/1k` incidents Pareto frontier).

> **Zero Placeholder Numbers Policy:** No synthetic or estimated numbers are ever written into result tables. All tables in `reports/` are compiled directly from machine-readable experiment manifests in `results/<run_id>/` via `sml report`.

---

## Sprint 1 Pre-Registered Hypotheses

| ID | Hypothesis | Pre-Registered Acceptance Criterion | Status |
| :--- | :--- | :--- | :--- |
| **H1** | **Mathematical & Causal Equivalence:** Custom `Transformer` in `gpt2-compat` mode matches `GPT2LMHeadModel` (`124M`) logits; naive `O(T^2)` GQA matches `SDPA`; `KV-cache` matches full forward; intra-document mask blocks cross-`<\|eos\|>` gradients. | `max_abs_logit_diff < 1e-4` (`fp32`); future & cross-doc grad `== 0.0` | `M0 Ready / M1 Next` |
| **H2** | **GPU Throughput Waterfall:** Cumulative system optimizations (`fp32 eager` → `bf16` → `SDPA` → `fused AdamW` → `torch.compile`) yield measured throughput gains and high hardware utilization. | Measured waterfall in `reports/sprint1/`; `MFU >= 30%` on `1x H100 SXM` | `Planned (M4)` |
| **H3** | **Chinchilla Optimal (`1.5B`) vs. Overtrained (`4.0B`) WSD Cooldown:** Overtraining `sml-100m` (`75.52M` non-embedding params) past `20x` tokens/param via WSD cooldown branching improves `val bpb` and zero-shot accuracy at identical inference cost. | Improvement exceeds `2σ` of 3-seed `sml-xs` noise floor | `Planned (M5)` |
| **H4** | **In-Context Induction Phase Transition:** Log-spaced checkpoints reveal an abrupt phase transition in intra-document `loss(pos=500) - loss(pos=50)` and synthetic sequence copying. | Phase transition localized across 15 checkpoints | `Planned (M5-M6)` |

---

## Verified Model Configurations (Sprint 1)

| Model | `d_model` | Layers | Q / KV Heads | `d_ff` | Vocab | Non-Emb Params | Total Params (Tied) | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sml-nano` | 256 | 4 | 4 / 2 | 768 | 32,768 | `3,148,032` | `11,536,640` | Local CPU/MPS unit tests & TinyStories smoke check |
| `sml-xs` | 512 | 8 | 8 / 4 | 1,408 | 32,768 | `23,601,664` | `40,378,880` | 24 GB GPU pilot & 3-seed noise floor (`0.4B` tokens) |
| **`sml-100m`** | **768** | **12** | **12 / 4** | **2,048** | **32,768** | **`75,516,672`** | **`100,682,496`** | **Primary Sprint 1 model (`1.5B` branch + `4.0B` full)** |
| `gpt2-124m` | 768 | 12 | 12 / 12 | 3,072 | 50,257 | `85,842,432` | `124,439,808` | Reference parity target for `H1` (`gpt2-compat`) |

---

## Quickstart (`M0` Scaffold)

```bash
# 1. Install dependencies into an isolated Python 3.12 virtualenv
make sync

# 2. Run unit tests and linter
make test
make lint

# 3. Inspect exact closed-form parameter, KV-cache, and FLOP math for sml-100m
uv run sml inspect-config --model configs/model/sml_100m.yaml

# 4. Run M0 dry-run pipeline and compile a preview report from results/
make smoke
```

## Architecture & Decisions

* [Master Roadmap & Sprint 1 Specification](docs/roadmap.md)
* [ADR-001: Four-Sprint Architecture vs. Eight Loose Phases](docs/decisions/001-four-sprint-architecture.md)
* [ADR-002: Grouped-Query Attention (GQA) & WSD Cooldown Branching](docs/decisions/002-gqa-and-wsd-schedule.md)
* [ADR-003: Streaming Data Shards & `< 15 GB` Local Disk Discipline](docs/decisions/003-streaming-disk-discipline.md)
