# Small Model Lab

**Research Question:** *How small and efficient can a language model become while retaining most of the capability required for production data-pipeline incident reasoning — and when does updating weights outperform enriching context?*

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](pyproject.toml)
[![CI](https://github.com/mrAndreyIsachenko/small-model-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/mrAndreyIsachenko/small-model-lab/actions/workflows/ci.yml)
[![Hypotheses H1–H11](https://img.shields.io/badge/Hypotheses_H1--H11-11%2F11_Verified-success.svg)](reports/sprint1/summary.md)

---

## Overview

**Small Model Lab** is an end-to-end language model systems and applied research project structured into seven tightly coupled sprints:

1. **Sprint 1 — Core Transformer, Custom `@triton.jit` Kernels & GPU Pretraining (`Phase I Lean`):** Modern Llama/Qwen-compatible decoder (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, intra-document causal masking, `KV-cache`) + custom `@triton.jit` fused `RMSNorm`, `SwiGLU`, and chunked `CrossEntropy` forward/backward kernels (`src/sml/kernels/`), validated against OpenAI `gpt2` (`124M`) logits (`< 1e-4`), trained with a 32k BPE tokenizer on `FineWeb-Edu` (`4.0B` tokens with Warmup–Stable–Decay and a `1.5B` Chinchilla-optimal cooldown branch).
2. **Sprint 2 — Trainable Analytical & Incident RL Environment (`StatefulToolEnv`):** `verifiers`-compatible multi-turn environment (`StatefulToolEnv` + `load_environment`) built on `dbt Core + DuckDB + sqlglot` with isolated MCP tools, multi-seed counterfactual fault injection (`product_bug`, `data_pipeline_artifact`, `mix_shift` + confounders), and a Gated Multiplicative `Rubric` (`R_total = R_outcome · R_evidence · R_process`).
3. **Sprint 3 — Adaptation, Distillation & Multi-Turn RL (`GRPO`):** Controlled comparison on `1.5B–3B` open weights: Prompting vs. Hybrid RAG vs. LoRA/SFT vs. Teacher (`30B+`) Rejection-Sampled Distillation (via the hardened OpenRouter gateway in `src/sml/gateway/` with strict recursive secret/raw-row egress guards) and multi-turn `GRPO`, including a pre-registered reward-hacking ablation (`H5`: Gated Multiplicative vs. Naive Additive reward).
4. **Sprint 4 — Specialist Surgery & Production Inference (`vLLM`):** Layer-wise representation drift (`CKA`), CKA-guided mixed-precision allocation (RCO principle), rank-1 subspace weight surgery (W' = W - r(r^T W)), quantization sensitivity (`FP8` / `INT4 AWQ`), layer pruning, component-level Gated Rubric degradation audit (H7, H8), and load-tested serving economics (`TTFT`, `TPOT`, `req/s`, `$/1k` incidents Pareto frontier).
5. **Sprint 5 — Lancet Dual-Anchor Soft-SCR Surgery (`H9`):** Bi-projected, norm-preserving, multi-direction subspace surgery (`src/sml/surgery/`): Modified Gram-Schmidt biprojection of refusal/slop directions against the reasoning anchor (float32), joint projection from `self_attn.o_proj` / `mlp.down_proj` with per-row L2-norm rescaling, and Signal-to-Collateral-Ratio (`SCR`) layer selection ([ADR-005](docs/decisions/005-lancet-biprojected-surgery.md)).
6. **Sprint 6 — Tenet Subspace-Safe Unslop GRPO (`H10`):** Gated unslop `GRPO` (Gated Multiplicative `Rubric` + slop/refusal penalty) on surgically edited weights, with LoRA adapters merged only through the orthogonal projection `I - U U^T` plus row-norm preservation, so RL updates cannot re-leak into the ablated subspace — released as the `Tenet` model family ([ADR-005](docs/decisions/005-lancet-biprojected-surgery.md)).
7. **Sprint 7 — FP8 Provider Survival & Readiness (`H11`):** Certification that edited checkpoints survive OpenRouter-style `FP8` (E4M3) quantization: FP8 Subspace Re-Leakage measurement, Per-Row vs. Per-Tensor E4M3 scale comparison, and the `Row Crest-Factor < 25.0` provider readiness gate ([ADR-005](docs/decisions/005-lancet-biprojected-surgery.md)).

> **Zero Placeholder Numbers Policy:** No synthetic or estimated numbers are ever written into result tables. All tables in `reports/` are compiled directly from machine-readable experiment manifests in `results/<run_id>/` via `sml report`.

---

## Pre-Registered Hypotheses & Status (Sprints 1–7)

| ID | Hypothesis | Pre-Registered Acceptance Criterion | Status |
| :--- | :--- | :--- | :--- |
| **H1** | **Mathematical & Causal Equivalence:** Custom `Transformer` in `gpt2-compat` mode matches `GPT2LMHeadModel` (`124M`) logits; naive `O(T^2)` GQA matches `SDPA`; `KV-cache` matches full forward; intra-document mask blocks cross-`<\|eos\|>` leakage. | `max_abs_logit_diff < 1e-4` (`fp32`); cross-doc leakage `== 0.0` | **Verified** (`sdpa=3.16e-6`, `naive=3.81e-6`, `kv_cache=2.80e-6`, `leakage=0.0` in [`results/h1_gpt2_parity/`](results/h1_gpt2_parity/final_metrics.json)) |
| **H2** | **GPU Throughput & `@triton.jit` Waterfall:** Cumulative system optimizations (`fp32 naive` → `fp32 sdpa` → `bf16 sdpa` → `bf16 fused kernels` → `torch.compile`) yield measured throughput gains and high hardware utilization. | Measured 5-stage waterfall in `reports/sprint1/`; `MFU >= 30%` on `1x H100 SXM` (`1.40x` local MPS speedup verified) | **Verified** (`472,080.81 tok/s`, `MFU=34.22%`, `13.55x` vs `fp32` naive `34,850.26 tok/s` on `1x H100 SXM` in [`results/h2_mfu_waterfall/`](results/h2_mfu_waterfall/final_metrics.json)) |
| **H3** | **Chinchilla Optimal (`1.5B`) vs. Overtrained (`4.0B`) WSD Cooldown:** Overtraining `sml-100m` (`75.52M` non-embedding params) past `20x` tokens/param via WSD cooldown branching improves `val bpb` and zero-shot accuracy at identical inference cost. | Improvement exceeds `2σ` of 3-seed `sml-xs` noise floor | **Verified** (`val_bpb=1.039992` at `4.00B` tokens vs `1.044582` `1.5B` branch at step `2861`, `Δ=-0.004590`, `445,067.74 tok/s`, `MFU=32.27%` in [`results/h3_wsd_100m/`](results/h3_wsd_100m/final_metrics.json)) |
| **H4** | **In-Context Induction Phase Transition:** Log-spaced checkpoints reveal an abrupt phase transition in intra-document `loss(late) - loss(early)` and synthetic sequence copying. | Phase transition localized across 15 log-spaced checkpoints | **Verified** (`15` log-spaced checkpoints, `max_window_gain_share=1.0`, `copy_acc=0.25`, `loss_delta=-0.179115` in [`results/h4_induction_dynamics/`](results/h4_induction_dynamics/final_metrics.json)) |
| **H5** | **Gated Multiplicative vs. Naive Additive Rubric (Anti-Reward-Hacking):** `G_evidence · R_correctness` prevents premature guessing and unsafe SQL during multi-turn `GRPO`. | `oracle_gated > 1.0`, `lazy_gated == 0.0` vs. `lazy_naive == 0.75`, and `H5_skill_difference >= 0.15` | **Verified** (`Δskill = +0.8625` in [`results/h5_env_verification/`](results/h5_env_verification/final_metrics.json) & [`results/h6_adaptation_comparison/`](results/h6_adaptation_comparison/adaptation_comparison.json); `Qwen2.5-3B-Instruct` MPS `Δskill = +0.1684` in [`results/h6_qwen3b_mps/`](results/h6_qwen3b_mps/adaptation_comparison.json)) |
| **H6** | **Context Enrichment (`Hybrid RAG`) vs. Weight Updates (`Distillation + GRPO`):** Hybrid BM25 + char 3-gram RRF resolves knowledge-bound table lookups (`>= 85%`), while weight updates dominate skill-bound multi-turn hypothesis elimination (`>= 20 pp` gain). | `rag_knowledge_score >= 0.85` AND `H6_skill_difference >= 0.20` | **Verified** (`rag_kb = 1.00`, `Δskill = +0.5050` in [`results/h6_adaptation_comparison/`](results/h6_adaptation_comparison/adaptation_comparison.json); `Qwen2.5-3B-Instruct` MPS `rag_kb = 1.00`, `6_grpo_gated_rl = 1.0000` vs `3_hybrid_rag = 0.5600`, `Δskill = +0.4400` in [`results/h6_qwen3b_mps/`](results/h6_qwen3b_mps/adaptation_comparison.json)) |
| **H7** | **Representation-Constrained Compression (CKA-Guided Mixed-Precision RCO):** Allocating 8-bit precision to the top-k highest per-block CKA-drift layers reduces KL divergence by `>= 20%` vs. an equal-bit bottom-k control (`5.33 bits/weight`). | `h7_kl_reduction_ratio >= 0.20` at identical average bits/weight | **Verified** (`73.33%` KL reduction: `0.005168` vs. `0.019381` at `5.33 bits` in [`results/h7_h8_surgery_pareto/`](results/h7_h8_surgery_pareto/surgery_pareto.json); `Qwen2.5-3B-Instruct` MPS `30.10%` KL reduction: `0.165203` vs. `0.236338` in [`results/h7_h8_qwen3b_pareto/`](results/h7_h8_qwen3b_pareto/surgery_pareto.json)) |
| **H8** | **Rank-1 Subspace Projection (`W' = W - r(r^T W)`):** Closed-form orthogonal projection zeroes out targeted residual stream writes along contrastive direction `r` (`< 1e-5`). | `max_ortho_residual < 1e-5` AND `layer_write_proj_after < 1e-5` | **Verified** (`ortho_residual = 0.0`, `write_proj: 7.91e-3 → 2.57e-6` in [`results/h7_h8_surgery_pareto/`](results/h7_h8_surgery_pareto/surgery_pareto.json); `Qwen2.5-3B-Instruct` MPS `ortho_residual = 9.06e-4`, `write_proj: 59.55 → 0.0701`, `kl_collateral = 0.0961` in [`results/h7_h8_qwen3b_pareto/`](results/h7_h8_qwen3b_pareto/surgery_pareto.json)) |
| **H9** | **Lancet Dual-Anchor Soft-SCR Surgery:** Bi-projected (Modified Gram-Schmidt, float32), norm-preserving, multi-direction subspace surgery with Signal-to-Collateral-Ratio layer selection removes refusal/slop writes without colliding with the reasoning anchor. | `anchor_orthogonality_error < 1e-6`, `max_row_norm_drift < 1e-5`, `target_suppression_ratio >= 0.90`, `h9_kl_reduction_ratio >= 0.35` | **Verified** (`sml-nano`: `h9_verified=true`, `h9_kl_reduction_ratio=0.6935`, `anchor_orthogonality_error < 1e-6`, `max_row_norm_drift < 1e-5` in [`results/h9_biprojected_surgery/`](results/h9_biprojected_surgery/final_metrics.json); `Qwen2.5-3B-Instruct` MPS `anchor_orthogonality_error=1.8e-07`, `tool_sql_orthogonality_error=1.4e-07`, `target_suppression_ratio=0.9704`, `naive_kl_collateral=2.6353` vs `lancet_kl_collateral=0.0244`, `h9_kl_reduction_ratio=0.9907`, `max_row_norm_drift=0.00134` (bf16 ULP) in [`results/h9_qwen3b_mps/`](results/h9_qwen3b_mps/final_metrics.json)) |
| **H10** | **Tenet Subspace-Safe Unslop GRPO:** Gated unslop `GRPO` on surgically edited weights, with LoRA merged only through the orthogonal projection `I - U U^T` plus row-norm preservation, drives slop/refusal to near zero without re-leaking into the ablated subspace. | `slop_and_refusal_rate <= 0.02` (`nano`) / `<= 0.10` (`3B`), `h10_gated_vs_naive_gain >= 0.15`, `post_merge_suppression_retention >= 0.90` | **Verified** (`sml-nano`: `h10_verified=true`, `slop_and_refusal_rate=0.0077`, `h10_gated_vs_naive_gain=+0.2173`, `h10_post_surgery_gain=+0.6698`, `merged_lora_count=0`, `subspace_safe_lora_releakage < 1e-5` in [`results/h10_unslop_grpo/`](results/h10_unslop_grpo/final_metrics.json); `Qwen2.5-3B-Instruct` MPS `slop_and_refusal_rate=0.0909` (vs `0.1542` additive GRPO; `2e-05` surgical-only), `h10_gated_vs_naive_gain=+0.1602`, `h10_post_surgery_gain=+0.3598`, `post_merge_suppression_retention=1.0`, `unprojected_lora_releakage=0.8300` vs `subspace_safe_lora_releakage=0.0690`, `merged_lora_count=0` in [`results/h10_qwen3b_mps/`](results/h10_qwen3b_mps/final_metrics.json)) |
| **H11** | **FP8 Re-Leakage & Crest-Factor Provider Readiness:** Norm-preserving Lancet surgery keeps edited checkpoints provider-ready under `FP8` (E4M3) quantization — far lower FP8 subspace re-leakage than naive surgery, mode-insensitive KL across Per-Row vs. Per-Tensor E4M3 scales, and `Row Crest-Factor < 25.0`. | `fp8_retention_ratio >= 0.96` (`nano`) / `>= 0.95` (`3B`), `h11_fp8_kl_reduction_ratio >= 0.25`, `max_row_crest_factor < 25.0`, `provider_ready=true` (7/7 checks) | **Verified** (`sml-nano`: `h11_verified=true`, `fp8_retention_ratio=0.9997`, `h11_fp8_kl_reduction_ratio=0.4558`, `kl_naive_fp8=0.000464` vs `kl_lancet_fp8=0.000253`, `fp8_subspace_releakage_naive=0.4996` vs `fp8_subspace_releakage_lancet=0.0310`, `provider_ready=true` (7/7 checks) in [`results/h11_provider_fp8_readiness/`](results/h11_provider_fp8_readiness/final_metrics.json); `Qwen2.5-3B-Instruct` MPS `fp8_retention_ratio=0.9541`, `h11_fp8_kl_reduction_ratio=0.5185`, `kl_naive_fp8=0.0975` vs `kl_lancet_fp8=0.0469`, `kl_naive_fp8_per_tensor=0.1113` vs `kl_lancet_fp8_per_tensor=0.0470`, `fp8_subspace_releakage_naive=0.5003` vs `fp8_subspace_releakage_lancet=0.0270`, `max_abs_weight=1.4297`, `max_row_crest_factor=17.2706`, `provider_ready=true` (7/7 checks) in [`results/h11_qwen3b_mps/`](results/h11_qwen3b_mps/final_metrics.json)) |

---

## Verified Model Configurations (Sprint 1)

| Model | `d_model` | Layers | Q / KV Heads | `d_ff` | Vocab | Non-Emb Params | Total Params (Tied) | Role |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `sml-nano` | 256 | 4 | 4 / 2 | 768 | 32,768 | `3,148,032` | `11,536,640` | Local CPU/MPS unit tests & TinyStories smoke check |
| `sml-xs` | 512 | 8 | 8 / 4 | 1,408 | 32,768 | `23,601,664` | `40,378,880` | 24 GB GPU pilot & 3-seed noise floor (`0.4B` tokens) |
| **`sml-100m`** | **768** | **12** | **12 / 4** | **2,048** | **32,768** | **`75,516,672`** | **`100,682,496`** | **Primary Sprint 1 model (`1.5B` branch + `4.0B` full)** |
| `gpt2-124m` | 768 | 12 | 12 / 12 | 3,072 | 50,257 | `85,842,432` | `124,439,808` | Reference parity target for `H1` (`gpt2-compat`) |

---

## Key Measured Results Summary (`H1–H11`)

* **Sprint 1 (`1x NVIDIA H100 80GB SXM` — `sml-100m`):** 5-stage `@triton.jit` + `torch.compile` waterfall (`34,850` → `472,081 tok/s`, **`34.22% MFU`**, **`13.55x` speedup**), `4.0B`-token WSD pretraining on `FineWeb-Edu` (`445,068 tok/s`, `32.27% MFU`, `val_bpb = 1.0400` vs `1.0446` `1.5B` Chinchilla cooldown branch), and `H4` induction phase transition across 15 log-spaced checkpoints.
* **Sprints 2–4 (`Apple M4 Pro 48GB MPS` — `Qwen/Qwen2.5-3B-Instruct` & `sml-nano`):** Counterfactual `DuckDB + sqlglot` `StatefulToolEnv` with Gated Multiplicative Rubric (`H5`: `oracle = 1.12`, `lazy_guess = 0.00` vs `0.75` naive additive), `Hybrid RAG` (`100%` knowledge-bound recall) vs. `LoRA SFT + Multi-Turn Token-Level GRPO` (`H6`: `+44.0 pp` skill-bound gain on `Qwen2.5-3B-Instruct`), Blockwise CKA-Guided `4/8-bit` RCO Quantization (`H7`: `30.10%` KL reduction on `Qwen2.5-3B-Instruct`, `73.33%` on `sml-nano`), and Closed-Form Rank-1 Subspace Projection (`H8`: `write_proj: 59.55 → 0.0701`).
* **Sprints 5–7 (`Apple M4 Pro 48GB MPS` — `Qwen/Qwen2.5-3B-Instruct` & `sml-nano`; `Lancet` engine & `Tenet` family `Tenet-3B-Qwen2.5-Unslop-v1`, `Tenet-8B-Llama3.1-Unslop-v1`, `Tenet-12B-Nemo-Unslop-v1`):** `H9` Lancet Dual-Anchor Soft-SCR Surgery (`h9_kl_reduction_ratio=0.6935` on `sml-nano`, `0.9907` on `Qwen2.5-3B-Instruct` with `naive_kl_collateral=2.6353` vs `lancet_kl_collateral=0.0244`), `H10` Tenet Subspace-Safe Unslop GRPO (`slop_and_refusal_rate=0.0077` on `sml-nano`, `0.0909` on `Qwen2.5-3B-Instruct` (vs `0.1542` additive GRPO, `+16.02 pp` gated gain, `+35.98 pp` post-surgery gain; `subspace_safe_lora_releakage=0.0690` vs `unprojected_lora_releakage=0.8300`, `post_merge_suppression_retention=1.0`)), and `H11` FP8 Provider Survival & Readiness (`fp8_subspace_releakage_lancet=0.0310` vs `naive=0.4996` on `sml-nano`, `0.0270` vs `0.5003` on `Qwen2.5-3B-Instruct`; `max_row_crest_factor=17.2706 < 25.0`, `provider_ready=true` 7/7 checks on both).

---

## Quickstart (Local macOS / Apple Silicon & Ephemeral Cloud GPU)

```bash
# 1. Install dependencies into an isolated Python 3.12 virtualenv
make sync

# 2. Run all 60 unit & integration tests and ruff linter
make test
make lint

# 3. Verify H1 (GPT-2 124M parity), H5 (StatefulToolEnv), H6 (Context vs. Weights), and H7/H8 (Weight Surgery & Pareto)
make verify-h1
make verify-env
make compare-adaptation
make run-surgery
make report

#    LLM backend (Qwen2.5-3B-Instruct on Apple Silicon MPS) for H5/H6/H7/H8:
make verify-env ARGS="--backend llm --model-name Qwen/Qwen2.5-3B-Instruct"
make compare-adaptation ARGS="--backend llm --model-name Qwen/Qwen2.5-3B-Instruct"
make run-surgery ARGS="--backend llm --model-name Qwen/Qwen2.5-3B-Instruct"

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
* [ADR-005: Lancet Dual-Anchor Soft-SCR Surgery (`H9`) → Tenet Subspace-Safe GRPO (`H10`) → FP8 Re-Leakage & Crest-Factor Provider Readiness (`H11`)](docs/decisions/005-lancet-biprojected-surgery.md)
