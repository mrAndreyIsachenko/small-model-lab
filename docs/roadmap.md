# Small Model Lab — Master Roadmap & Sprint Specifications

## 1. Strategic Objective

**Small Model Lab** is an end-to-end language model systems and applied research project designed to investigate:
1. How small and efficient a language model can become while retaining specialized capability for production data-pipeline incident and analytical root-cause reasoning.
2. How to turn static "AI Gold" analytical cases (e.g., *"Why did appointment completion drop last week?"*) into a **trainable, verifiable multi-turn RL environment (`StatefulToolEnv` + Gated `Rubric`)** with isolated MCP tools, deterministic fault injection, and anti-reward-hacking guarantees ([ADR-004](decisions/004-verifiable-rl-analytics-environment.md)).
3. When updating model weights (LoRA, SFT, Rejection-Sampled Distillation, multi-turn `GRPO`) outperforms enriching runtime context (Prompting, Hybrid RAG) on `knowledge-bound` vs. `skill-bound` failures.
4. How pretraining mechanics, post-training weight changes, and quantization interact under production GPU serving constraints (`vLLM`).

## 2. Four-Sprint Program Structure (~10–12 Weeks @ 10h/week)

| Sprint | Target Duration | Core Engineering & Research Scope | Primary Public Artifacts |
| :--- | :--- | :--- | :--- |
| **Sprint 1: Core Transformer, `@triton.jit` Kernels & GPU Pretraining (`Phase I Lean`)** | Weeks 1–4 (~35–40h) | Implement Llama/Qwen-compatible decoder (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, `KV-cache`, intra-doc causal mask), custom `@triton.jit` fused `RMSNorm` & `SwiGLU` forward/backward kernels (`src/sml/kernels/`), verify `gpt2` (`124M`) logit parity (`< 1e-4`), train 32k BPE, stream-shard `FineWeb-Edu`, profile MFU waterfall, train `sml-100m` (`4.0B` WSD tokens + `1.5B` Chinchilla branch). | `sml-100m` weights + 15 log-spaced checkpoints + `@triton.jit` kernel benchmarks + 32k BPE + shard manifest on HF Hub + Report #1. |
| **Sprint 2: Trainable Analytical & Incident RL Environment (`StatefulToolEnv`)** | Weeks 5–6 (~20–25h) | Build `verifiers`-compatible multi-turn environment (`StatefulToolEnv` + `load_environment`) on `dbt Core + DuckDB + sqlglot`, isolated MCP tools (`query_warehouse`, `get_metric_lineage`, `inspect_changes`, `submit_diagnosis`), multi-seed counterfactual fault injection (`product_bug`, `data_pipeline_artifact`, `mix_shift` + confounders), and Gated Multiplicative `Rubric` (`G_safety · G_complete · R_correctness · G_evidence · (1 + α R_efficiency)`). | Installable `verifiers` environment package (`src/sml/envs/`) + counterfactual scenario benchmark + anti-hacking unit suite + Report #2. |
| **Sprint 3: Adaptation, Distillation & Multi-Turn RL (`Context vs. Weights`)** | Weeks 7–9 (~25–30h) | Baseline eval $\rightarrow$ trajectory collection $\rightarrow$ Teacher (`30B+`) Rejection-Sampled SFT $\rightarrow$ Multi-Turn `GRPO` into `1.5B–3B` open weights; run controlled ablation comparing Gated Multiplicative `Rubric` vs. Naive Additive reward on held-out fault seeds (`H5`). | Distilled & RL-tuned specialist models (`1.5B–3B`) + verified multi-turn trajectory dataset + Report #3. |
| **Sprint 4: Specialist Surgery & Production Inference (`vLLM`)** | Weeks 10–11 (~20–25h) | Inspect layer-wise CKA drift (Base vs. SFT vs. GRPO Student), evaluate `FP8` / `INT4 AWQ` and layer-pruning degradation on multi-turn tool calls, benchmark `vLLM` continuous batching & prefix caching (`TTFT`, `TPOT`, `$/1k` incidents Pareto frontier). | Quantized specialist checkpoints + `vLLM` benchmark suite + Final Pareto Report #4. |

## 3. Sprint 1 Milestone Plan

* **`M0` (Week 1, Local Mac):** Repository scaffold, `uv` Python 3.12 lockfile, typed config/math engine, `results/<run_id>/` manifest & cost tracker, `sml report` compiler, GitHub Actions CI.
* **`M1` (Week 1, Local Mac):** Transformer implementation (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, `KV-cache`, `gpt2_adapter`, `hooks/`), `H1` logit parity test (`< 1e-4`), and local `sml-nano` MPS smoke run on `TinyStories`.
* **`M2` (Week 2, Local Mac):** 32k byte-level BPE training, streaming `FineWeb-Edu` `uint16` shard pipeline (`< 15 GB` peak local disk), 13-gram decontamination audit, upload shards to HF Hub.
* **`M3` (Week 2, Local Mac):** Trainer (`AdamW`, WSD schedule with cooldown branching, `bf16`, `torch.compile`, deterministic resume, async HF checkpoint uploader) and evaluation harness (`bpb`, MC zero-shot tasks, induction metric).
* **`M4` (Week 3, Cloud 24 GB GPU):** Ephemeral GPU bootstrap (`RTX 4090` / `L4`), idle watchdog test, `H2` throughput/MFU waterfall via `torch.profiler`.
* **`M5` (Week 3, Cloud `1x H100 80GB SXM`):** 3-seed `sml-xs` noise baseline (`0.4B` tokens each) + main `sml-100m` WSD run (`4.0B` tokens + `1.5B` Chinchilla cooldown branch, 15 log-spaced checkpoints).
* **`M6` (Week 4, Local + Cloud Eval):** Zero-shot suite (`HellaSwag`, `ARC-Easy`, `PIQA`, `BLiMP`), `lm-evaluation-harness` cross-check, one-time `test` split evaluation, public HF release, and Article #1.

## 4. Sprint 2 & Sprint 3 Specification: Multi-Turn MCP RL Environment ([ADR-004](decisions/004-verifiable-rl-analytics-environment.md))

### 4.1 Environment Contract (`PrimeIntellect-ai/verifiers` Compatible)

The environment module (`src/sml/envs/`) implements the standard `verifiers` contract so it can be executed both standalone and inside async RL training loops (`prime-rl`, `TRL` / `verl`):
* **Entrypoint:** `load_environment(split: str = "train", max_steps: int = 8, rubric_mode: Literal["gated", "additive"] = "gated") -> StatefulToolEnv`
* **Per-Rollout Isolation (`setup_state(state)`):** Instantiates an isolated `DuckDB` database seeded with deterministic tables (`mart_appointments`, `raw_notification_events`, `dbt_pipeline_audit`, `infra_deploys_and_flags`), injects 1 hidden root cause + 2 plausible confounders, and attaches the hidden `GroundTruthSpec` to `state`.
* **Guardrailed MCP Tools (`update_tool_args`):**
  1. `query_warehouse(sql: str)` — Validated via `sqlglot` AST (blocks `DDL`/`DML`, enforces `LIMIT 50`, injects deterministic scan-budget errors on unpartitioned full-table scans).
  2. `get_metric_lineage(metric_name: str)` — Returns `dbt` SQL definition and upstream dependencies.
  3. `inspect_changes(start_ds: str, end_ds: str, service: str | None)` — Queries deployment, feature-flag, and pipeline audit logs.
  4. `submit_diagnosis(diagnosis: DiagnosisPayload)` — Terminal structured action reporting `root_cause_category`, `primary_segment`, `estimated_impact_pp`, `supporting_change_id`, `ruled_out_hypotheses`, and verbatim `evidence_numbers`.

### 4.2 Gated Anti-Reward-Hacking `Rubric` (`src/sml/envs/rubric.py`)

To prevent reward hacking during multi-turn RL (`GRPO`), the environment scores trajectories $\tau$ with a hierarchical multiplicative gate:

$$R(\tau) = G_{\text{safety}}(\tau) \cdot G_{\text{complete}}(\tau) \cdot \Big[ R_{\text{correctness}}(\tau) \cdot G_{\text{evidence}}(\tau) \cdot \big(1 + \alpha \cdot R_{\text{efficiency}}(\tau)\big) \Big]$$

* **`G_safety` & `G_complete` ($\{0, 1\}$):** Hard gates on read-only tool compliance, step/token budgets, and valid terminal schema submission.
* **`G_evidence` ($[0, 1]$):** Parses executed SQL queries via `sqlglot` to verify that ground-truth tables and slice columns were actually queried, and checks that cited `evidence_numbers` exist in tool outputs. Collapses to `0.0` if required evidence tables were skipped (blocking lucky guesses).
* **`R_correctness` ($[0, 1]$):** Weighted verification of root-cause category (`0.30`), segment localization (`0.25`), quantitative impact within tolerance (`0.20`), supporting change ID (`0.15`), and ruled-out confounders (`0.10`).
* **`R_efficiency` ($[0, 1]$):** Smooth step-budget bonus active *only* when $R_{\text{correctness}} \cdot G_{\text{evidence}} > 0$.

### 4.3 Sprint 2 & Sprint 3 Pre-Registered Hypotheses

| ID | Hypothesis | Pre-Registered Acceptance Criterion | Sprint |
| :--- | :--- | :--- | :--- |
| **H5** | **Gated vs. Additive Reward Hacking in Multi-Turn `GRPO`:** Training a `1.5B–3B` policy with a naive additive reward induces premature termination / ungrounded guessing, whereas the Gated Multiplicative `Rubric` increases verified root-cause accuracy and maintains `< 5%` ungrounded citation rate on held-out seeds. | Gated `GRPO` outperforms Naive Additive `GRPO` by `>= 15 pp` on held-out `G_evidence * R_correctness` with `p < 0.05` (bootstrap 95% CI). | `Sprint 3` |
| **H6** | **Context Enrichment (`Hybrid RAG`) vs. Weight Updates (`Distillation + GRPO`):** Enriching context resolves `knowledge-bound` schema/lineage lookups, but multi-turn `Distillation + GRPO` significantly outperforms prompting + RAG on `skill-bound` multi-step hypothesis elimination and confounder rejection. | `Distillation + GRPO` improves `skill-bound` episode pass rate by `>= 20 pp` over `Hybrid RAG` on the same `1.5B–3B` base model. | `Sprint 3` |
