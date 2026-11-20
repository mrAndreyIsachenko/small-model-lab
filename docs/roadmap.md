# Small Model Lab — Master Roadmap & Sprint Specifications

## 1. Strategic Objective

**Small Model Lab** is an end-to-end language model systems and applied research project designed to investigate:
1. How small and efficient a language model can become while retaining specialized capability for production data-pipeline incident and analytical root-cause reasoning.
2. How to turn static "AI Gold" analytical cases (e.g., *"Why did appointment completion drop last week?"*) into a **trainable, verifiable multi-turn RL environment (`StatefulToolEnv` + Gated `Rubric`)** with isolated MCP tools, deterministic fault injection, and anti-reward-hacking guarantees ([ADR-004](decisions/004-verifiable-rl-analytics-environment.md)).
3. When updating model weights (LoRA, SFT, Rejection-Sampled Distillation, multi-turn `GRPO`) outperforms enriching runtime context (Prompting, Hybrid RAG) on `knowledge-bound` vs. `skill-bound` failures.
4. How pretraining mechanics, post-training weight changes, and quantization interact under production GPU serving constraints (`vLLM`).

## 2. Sprint Program Structure (~10–12 Weeks @ 10h/week for Sprints 1–4, plus Sprints 5–7)

| Sprint | Target Duration | Core Engineering & Research Scope | Primary Public Artifacts |
| :--- | :--- | :--- | :--- |
| **Sprint 1: Core Transformer, `@triton.jit` Kernels & GPU Pretraining (`Phase I Lean`)** | Weeks 1–4 (~35–40h) | Implement Llama/Qwen-compatible decoder (`RMSNorm`, `RoPE`, `GQA`, `SwiGLU`, `KV-cache`, intra-doc causal mask), custom `@triton.jit` fused `RMSNorm` & `SwiGLU` forward/backward kernels (`src/sml/kernels/`), verify `gpt2` (`124M`) logit parity (`< 1e-4`), train 32k BPE, stream-shard `FineWeb-Edu`, profile MFU waterfall, train `sml-100m` (`4.0B` WSD tokens + `1.5B` Chinchilla branch). | `sml-100m` weights + 15 log-spaced checkpoints + `@triton.jit` kernel benchmarks + 32k BPE + shard manifest on HF Hub + Report #1. |
| **Sprint 2: Trainable Analytical & Incident RL Environment (`StatefulToolEnv`)** | Weeks 5–6 (~20–25h) | Build `verifiers`-compatible multi-turn environment (`StatefulToolEnv` + `load_environment`) on `dbt Core + DuckDB + sqlglot`, isolated MCP tools (`query_warehouse`, `get_metric_lineage`, `inspect_changes`, `submit_diagnosis`), multi-seed counterfactual fault injection (`product_bug`, `data_pipeline_artifact`, `mix_shift` + confounders), and Gated Multiplicative `Rubric` (`G_safety · G_complete · R_correctness · G_evidence · (1 + α R_efficiency)`). | Installable `verifiers` environment package (`src/sml/envs/`) + counterfactual scenario benchmark + anti-hacking unit suite + Report #2. |
| **Sprint 3: Adaptation, Distillation & Multi-Turn RL (`Context vs. Weights`)** | Weeks 7–9 (~25–30h) | Baseline eval $\rightarrow$ trajectory collection $\rightarrow$ Teacher (`30B+`) Rejection-Sampled SFT $\rightarrow$ Multi-Turn `GRPO` into `1.5B–3B` open weights; run controlled ablation comparing Gated Multiplicative `Rubric` vs. Naive Additive reward on held-out fault seeds (`H5`). | Distilled & RL-tuned specialist models (`1.5B–3B`) + verified multi-turn trajectory dataset + Report #3. |
| **Sprint 4: Specialist Surgery & Production Inference (`vLLM`)** | Weeks 10–11 (~20–25h) | Inspect layer-wise CKA drift (Base vs. SFT vs. GRPO Student), evaluate `FP8` / `INT4 AWQ` and layer-pruning degradation on multi-turn tool calls, benchmark `vLLM` continuous batching & prefix caching (`TTFT`, `TPOT`, `$/1k` incidents Pareto frontier), test CKA-guided non-uniform mixed-precision bit allocation (RCO principle), surgical rank-1 weight projection (W' = W - r(r^T W)) sensitivity across Gated Rubric components. | Quantized specialist checkpoints + `vLLM` benchmark suite + Final Pareto Report #4. |
| **Sprint 5: Lancet Dual-Anchor Soft-SCR Surgery (`H9`)** | Extension (`H9`) | Modified Gram-Schmidt biprojection of refusal & slop probes against reasoning + tool/SQL anchors, per-row L2-norm-preserving projection (`o_proj`, `down_proj`), and soft-SCR layer selection ([ADR-005](decisions/005-lancet-biprojected-surgery.md)). | Lancet surgical engine (`src/sml/surgery/`) + `H9` verification bundles (`nano` & `Qwen2.5-3B-Instruct`). |
| **Sprint 6: Tenet Subspace-Safe Unslop GRPO (`H10`)** | Extension (`H10`) | Gated unslop `GRPO` (`TenetUnslopRubric`), orthogonal LoRA merge (`I - U U^T`) with row-norm preservation, and merge equivalence verification (`max_abs_logit_diff < 1e-4`, `merged_lora_count == 0`). | `Tenet` family (`Tenet-3B-Qwen2.5-Unslop-v1`, `Tenet-8B-Llama3.1-Unslop-v1`, `Tenet-12B-Nemo-Unslop-v1`) + `H10` bundles. |
| **Sprint 7: FP8 Provider Survival & Readiness (`H11`)** | Extension (`H11`) | Server-side FP8 (E4M3) survival (`per_channel` vs. `per_tensor`), `FP8 Subspace Re-Leakage`, `Row Crest-Factor < 25.0` weight hygiene gate, and provider readiness validation (`vLLM` / OpenRouter). | Provider readiness validator (`src/sml/hub/`) + `Tenet` model cards + `H11` bundles. |
| **Sprint 8: Cross-Domain Subspace-Safe Adapter Merge (`H12`)** | Extension (`H12`) | TIES / Gram-Schmidt subspace-safe composition of independently trained GRPO adapters (`Delta W_SQL` + `Delta W_unslop`): orthogonal projection of `Delta W_unslop` against `U_{refusal/slop}` and top-k left singular vectors `U_SQL` of `Delta W_SQL`, TIES trim/sign-election/disjoint merge, row-norm restoration, and the `H11` FP8 (E4M3) readiness gate (`Row Crest-Factor < 25.0`) ([ADR-007](decisions/007-cross-domain-subspace-safe-adapter-merge.md)). | Subspace-safe merge engine (`src/sml/surgery/`) + `H12` verification bundles (`nano` & `Qwen2.5-3B-Instruct`). |
| **Sprint 9: Layer-Streaming Lancet Surgery & Subspace-Safe Merge for 32B–70B (`H13`)** | Extension (`H13`) | Shard-by-shard `safetensors` streaming (`model-00001-of-000XX.safetensors`) for Lancet surgery (`float32` Modified Gram-Schmidt) and TIES subspace-safe merge with row-norm preservation and `FP8 E4M3` outlier control (`Row Crest-Factor < 25.0`), fitting `32B`/`70B` within `1x H100 80GB SXM`, `< 16 GB` peak RAM, and `< 15 GB` local disk ([ADR-008](decisions/008-streaming-shard-surgery-32b-70b.md)). | Streaming surgery/merge engine (`subspace.py`, `merge.py`) + `32b-h100`/`70b-h100` foundry profiles + `sml verify-h13` + `sml-nano` equivalence tests. |

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

## 5. Sprint 4 Specification: Weight Surgery, Mixed-Precision Allocation & Serving Pareto

### 5.1 CKA-Guided Mixed-Precision Allocation (RCO Principle) & Post-Quantization Weight Surgery

Using layer-wise Linear CKA drift (Base vs. SFT vs. GRPO Student) to guide non-uniform bit allocation across tensors (preserving higher precision in high-drift tool-routing/reasoning layers while compressing low-sensitivity layers, inspired by RCO / GSQ-RCO) and testing surgical rank-1 subspace updates W' = W - r(r^T W) on target projections (o_proj, down_proj) without full retraining.

### 5.2 Component-Level Degradation Audit on StatefulToolEnv

Measuring degradation across individual Gated Rubric factors (G_safety, G_complete, G_evidence, R_correctness, and confounder rejection) rather than only static perplexity or single-turn benchmarks.

### 5.3 Sprint 4 Pre-Registered Hypotheses

| ID | Hypothesis | Pre-Registered Acceptance Criterion | Sprint |
| :--- | :--- | :--- | :--- |
| **H7** | **CKA-Guided Mixed-Precision vs. Uniform Low-Bit Quantization on Multi-Turn Tool Use:** protecting top-25% CKA-drift layers recovers >= 85% of BF16 G_evidence * R_correctness and beats uniform low-bit quantization by >= 10 pp on held-out episodes. | CKA-guided mixed-precision preserves >= 85% of BF16 performance on G_evidence * R_correctness while beating uniform low-bit by >= 10 pp on held-out episodes. | `Sprint 4` |
| **H8** | **Rank-1 Subspace Weight Projection Sensitivity W' = W - r(r^T W):** selective mid-to-late layer rank-1 projection shifts verification behavior while keeping G_safety * G_complete >= 0.95, whereas all-layer projection degrades G_evidence and confounder rejection first. | Selective mid-to-late layer rank-1 projection maintains G_safety * G_complete >= 0.95 while all-layer projection degrades G_evidence and confounder rejection first. | `Sprint 4` |

## 6. Sprints 5–7 Specification: Lancet Engine & Tenet Provider-Ready Release ([ADR-005](decisions/005-lancet-biprojected-surgery.md))

| ID | Hypothesis | Pre-Registered Acceptance Criterion | Sprint |
| :--- | :--- | :--- | :--- |
| **H9** | **Lancet Dual-Anchor Soft-SCR Surgery:** Bi-projecting refusal & slop probes against reasoning/tool anchors with per-row L2-norm preservation and SCR layer selection suppresses target writes while cutting anchor KL collateral vs. naive ablation. | `anchor_orthogonality_error < 1e-6`, `max_row_norm_drift < 1e-5` (`fp32`) / `< 2e-3` (`bf16`), `target_suppression_ratio >= 0.90`, `h9_kl_reduction_ratio >= 0.35`. | `Sprint 5` |
| **H10** | **Tenet Subspace-Safe Unslop GRPO:** Gated unslop GRPO eliminates residual slop/refusal mass, and projecting LoRA deltas onto `I - U U^T` before in-place merge prevents subspace re-leakage. | `slop_and_refusal_rate <= 0.02`, `h10_gated_vs_naive_gain >= 0.15`, `h10_post_surgery_gain >= 0.15`, `post_merge_suppression_retention >= 0.90`, `merged_lora_count == 0`, `subspace_safe_lora_releakage < 1e-5` (`nano`) / `< 0.10` (`3B`). | `Sprint 6` |
| **H11** | **FP8 Provider Survival & Readiness:** Lancet norm-preserving surgery survives E4M3 quantization with lower KL drift and lower `FP8 Subspace Re-Leakage` than naive ablation, and passes provider hygiene (`Row Crest-Factor < 25.0`). | `fp8_retention_ratio >= 0.96` (`nano`) / `>= 0.95` (`3B`), `h11_fp8_kl_reduction_ratio >= 0.25`, `max_row_crest_factor < 25.0`, `provider_ready == True`. | `Sprint 7` |
| **H12** | **Cross-Domain Subspace-Safe Adapter Merge:** Naive linear merge (`Delta W_SQL + Delta W_unslop`) of specialized GRPO adapters causes destructive interference along dominant singular directions of `Delta W_SQL` and re-leaks mass into `U_{refusal/slop}`; the TIES / Gram-Schmidt Subspace-Safe Merge (projection of `Delta W_unslop^perp` against both subspaces, TIES trim/sign-election/disjoint merge, row-norm restoration) preserves SQL skill and refusal/slop suppression with FP8-ready weight hygiene. | `h12_naive_interference_ratio` measured (naive linear merge degrades SQL skill and/or raises slop/refusal vs. each adapter alone), `subspace_safe_merge_orthogonality_error < 1e-6` (float32), `subspace_safe_merge_releakage < 1e-5` (`nano`) / `< 0.10` (`3B`), `max_row_norm_drift < 1e-5` (`fp32`) / `< 2e-3` (`bf16`), `h12_skill_retention_ratio >= 0.95`, `post_merge_suppression_retention >= 0.90`, `max_row_crest_factor < 25.0`, `provider_ready == True`. | `Sprint 8` |
| **H13** | **Layer-Streaming Lancet Surgery & Subspace-Safe Merge (32B–70B):** Streaming `safetensors` shards one-by-one with `float32` Modified Gram-Schmidt projection, TIES subspace-safe merge, and per-row L2-norm restoration achieves numerical equivalence with in-memory surgery/merge while keeping peak memory `< 16 GB`, local disk overhead `< 15 GB`, and `FP8 E4M3` `Row Crest-Factor < 25.0` on a single `H100 80GB SXM`. | `streaming_surgery_max_abs_diff < 1e-6` (`nano`), `streaming_merge_max_abs_diff < 1e-6` (`nano`), `peak_memory_gb < 16.0`, `max_row_crest_factor < 25.0`, `provider_ready == True`. | `Sprint 9` |
| **H15** | **Phase-Segmented Tri-Anchor Lancet (`Tenet v2`):** Configurable `PhaseSegmenter` (`<think>...</think>`, `<reasoning>`, harmony channels), `CoT Self-Correction Anchor` ($a_{\text{cot}}$) extracted from deliberation tokens and orthonormalized with $a_{\text{math\_logic}}$ and $a_{\text{tool\_sql}}$ via 2-pass float32 MGS, post-`</think>` refusal/slop extraction, and CoT-weighted KL collateral (`compute_layer_scr_profile`) preserve reasoning-channel integrity and FP8 readiness. | `tri_anchor_orthogonality_error < 1e-6`, `biprojected_orthogonality_error < 1e-6`, `max_row_crest_factor < 25.0`, `h15_verified == True`. | `Sprint 10` |
| **H16** | **Dual-Window Length-Calibrated & Artifact-Completeness Gated GRPO (`Tenet v2`):** `G_cot_integrity`, asymmetric log-normal `G_length_cal` (eliminating `length_collapse` and `overthinking`), paragraph/density `G_antislop_density`, and verifiable `R_info_density` (code, read-only SQL, JSON, exact numeric) drive `length_collapse_rate < 0.01`, `cot_tag_integrity_rate >= 0.995`, and `slop_and_refusal_rate <= 0.02`. | `cot_tag_integrity_rate >= 0.995`, `length_collapse_rate < 0.01`, `slop_and_refusal_rate <= 0.02`, `artifact_completeness_rate >= 0.99`, `h16_verified == True`. | `Sprint 11` |

## 6.4 Sprint 8 — Cross-Domain Subspace-Safe Adapter Merge (H12) ([ADR-007](decisions/007-cross-domain-subspace-safe-adapter-merge.md))

After Sprint 6, the Tenet family carries two independently trained LoRA deltas on the same Lancet-surgered base: `Delta W_SQL` (multi-turn incident/SQL GRPO, `H5`/`H6`/`H10`) and `Delta W_unslop` (gated unslop GRPO, `H10`). Sprint 8 defines how they are composed into one checkpoint:

1. **Naive-merge failure mode (measured, not assumed).** The baseline for `H12` is the naive linear merge `Delta W = Delta W_SQL + Delta W_unslop`. `Delta W_unslop` is not orthogonal to the top-k left singular vectors `U_SQL` of `Delta W_SQL`, so linear summation destructively interferes along the dominant, highest-energy SQL directions; it also re-injects mass into the ablated `U_{refusal/slop}` subspace — the same re-leakage family measured in `H10`, now caused by the merge itself.
2. **Subspace-safe projection.** `Delta W_unslop^perp = (I - P_{refusal/slop} - P_{U_SQL}) Delta W_unslop`, built with Modified Gram-Schmidt in float32 (`H9` convention): the unslop delta is projected orthogonally against both the refusal/slop basis (stored in the Lancet recipe) and the SVD-derived `U_SQL` basis of the SQL adapter delta.
3. **TIES merge.** The protected deltas are combined with TIES: magnitude trim per delta, coordinate-wise sign election by total signed magnitude, and disjoint merge of surviving coordinates.
4. **Row-norm restoration.** Every row of the merged matrix is rescaled to its pre-merge L2 norm, preserving the `H9`/`H10` norm-preservation invariant.
5. **H11 FP8 readiness gate.** The merged checkpoint must pass `max_row_crest_factor < 25.0` and the full 7/7 `verify_provider_readiness` set before publish; a merge that violates the gate fails during ADR-006 materialization rather than at serving time.

Artifacts: subspace-safe cross-domain merge engine in `src/sml/surgery/`, naive-vs-safe ablation bundles for `sml-nano` and `Qwen2.5-3B-Instruct`, and a new recipe step between "Unslop GRPO LoRA Merge" and "Post-Merge Lancet Guard" in the ADR-006 materialization chain.

## 6.5 Sprint 9 — Layer-Streaming Lancet Surgery & Subspace-Safe Merge for 32B–70B (H13) ([ADR-008](decisions/008-streaming-shard-surgery-32b-70b.md))

Scaling Lancet surgery (`H9`) and TIES subspace-safe merge (`H12`) to `32B` (`Qwen/Qwen2.5-32B-Instruct`) and `70B` (`meta-llama/Llama-3.3-70B-Instruct`, `~140 GB BF16`) on a single `H100 80GB SXM` with `< 15 GB` local scratch disk requires eliminating full-checkpoint materialization:

1. **Shard-by-shard `safetensors` streaming.** `apply_lancet_recipe_streaming` (`src/sml/surgery/subspace.py`) and `stream_subspace_safe_ties_merge` (`src/sml/adaptation/merge.py`) resolve `model.safetensors.index.json` and process one `model-00001-of-000XX.safetensors` shard at a time, freeing each shard immediately (`gc.collect()`), bounding peak RAM/VRAM to `< 16 GB` and local disk overhead to `< 15 GB`.
2. **Float32 Modified Gram-Schmidt & TIES equivalence.** Each target tensor (`o_proj.weight`, `down_proj.weight`, or adapter-matched 2D weights) is projected/merged in `float32` with per-row L2-norm restoration and cast back to original dtype, matching in-memory results within `< 1e-6`.
3. **32B/70B FP8 outlier & crest-factor control.** Incremental `Row Crest-Factor < 25.0` tracking across all streamed shards prevents extreme activation/weight outliers from breaking `FP8 E4M3` provider readiness at `32B`/`70B` scale.

Artifacts: streaming engines in `src/sml/surgery/subspace.py` and `src/sml/adaptation/merge.py`, `32b-h100` and `70b-h100` profiles in `scripts/remote/run_tenet_foundry.sh`, `experiments/h13_streaming_shard_surgery.yaml`, `sml verify-h13`, and `sml-nano` shard-equivalence tests.

## 6.6 Sprint 10 — Tenet v2: Phase-Segmented Tri-Anchor Lancet (H15) ([ADR-010](decisions/010-tenet-v2-thinking-surgery-and-length-calibrated-grpo.md))

To extend Lancet surgery to reasoning models (`<think>...</think>`, `<reasoning>`, harmony channels) without damaging Chain-of-Thought self-correction or delimiter logits:

1. **Model-agnostic `PhaseSegmenter`.** Splits sequences into `deliberation_span` and `surface_answer_span` across token IDs, tokenizers, or raw text (`src/sml/surgery/probes.py`).
2. **Tri-Anchor basis.** Extracts $a_{\text{cot}}$ from `COT_REASONING_ANCHOR_PAIRS` (`pool_mode="deliberation_only"`), combines with $a_{\text{math\_logic}}$ and $a_{\text{tool\_sql}}$, and orthonormalizes via two-pass Modified Gram-Schmidt in `float32` (`src/sml/surgery/subspace.py`).
3. **Post-`</think>` target extraction & CoT-weighted SCR.** Extracts `refusal` and `slop` strictly from `pool_mode="answer_only"`, bi-projects against the Tri-Anchor basis with row-norm preservation, and scores layers with $\lambda_{\text{cot}} \text{KL}_{\text{cot}} + (1 - \lambda_{\text{cot}}) \text{KL}_{\text{anchor}}$ protecting `<think>`/`</think>` logits.

Artifacts: `PhaseSegmenter` and `COT_REASONING_ANCHOR_PAIRS` in `src/sml/surgery/probes.py`, phase-segmented extraction and Tri-Anchor SCR in `src/sml/surgery/subspace.py`, `run_h15_cot_tri_anchor_lancet_benchmark` in `src/sml/surgery/pareto.py`, and `experiments/h15_cot_tri_anchor_lancet.yaml`.

## 6.7 Sprint 11 — Tenet v2: Dual-Window Length-Calibrated & Artifact-Gated GRPO (H16) ([ADR-010](decisions/010-tenet-v2-thinking-surgery-and-length-calibrated-grpo.md))

To eliminate `H10` `length_collapse` (`8.33%` on 8B) and `<think>` `overthinking` while guaranteeing 100% mechanically verifiable technical artifacts after `</think>`:

1. **`G_cot_integrity` ($\{0, 1\}$).** Verifies balanced `<think>...</think>` tags, zero tag leakage into JSON/SQL/code outputs, and absence of repetition loops inside deliberation.
2. **Dual-window `G_length_cal`.** Asymmetric log-normal window around task-type target $w_{\text{ans}}^*$ (`concise_factual`, `analytical_sql`, `deep_technical_explanation`) with smooth positive gradient for $w < w_{\text{ans}}^*$ plus a soft deliberation budget $w_{\text{cot}}^*$.
3. **`G_antislop_density` & `R_info_density`.** Hard first/last-paragraph cliché gating + per-100-word internal normalization, paired with verifiable artifact checks (`code`, `sql_readonly`, `json`, `exact_numeric`) and lexical diversity.

Artifacts: `TenetUnslopRubric(mode="v2")` in `src/sml/envs/unslop_rubric.py`, `run_h16_length_calibrated_grpo_benchmark` in `src/sml/adaptation/grpo.py`, `experiments/h16_length_calibrated_grpo.yaml`, and `tests/test_tenet_v2_h15_h16.py`.

## 7. Success Criteria Summary

| ID | Success Criterion | Sprint |
| :--- | :--- | :--- |
| **H9** | Bi-projected, norm-preserving Lancet surgery suppresses target writes with `h9_kl_reduction_ratio >= 0.35` and `anchor_orthogonality_error < 1e-6`. | `Sprint 5` |
| **H10** | Gated unslop GRPO drives `slop_and_refusal_rate <= 0.02` (`nano`) / `<= 0.10` (`3B`) with subspace-safe LoRA merging (`merged_lora_count == 0`, re-leakage within bound). | `Sprint 6` |
| **H11** | Edited checkpoints stay provider-ready under FP8 (E4M3): `max_row_crest_factor < 25.0`, `provider_ready == True` (7/7 checks). | `Sprint 7` |
| **H12** | Cross-domain TIES / Gram-Schmidt Subspace-Safe Merge retains SQL skill (`h12_skill_retention_ratio >= 0.95`) and refusal/slop suppression (`post_merge_suppression_retention >= 0.90`) without re-leakage, with FP8-ready weight hygiene (`max_row_crest_factor < 25.0`, `provider_ready == True`). | `Sprint 8` |
| **H13** | Layer-streaming `safetensors` surgery & TIES merge matches in-memory (`< 1e-6` on `sml-nano`) with `< 16 GB` peak memory, `< 15 GB` local disk overhead, and `Row Crest-Factor < 25.0` (`provider_ready == True`) for `32B`–`70B` checkpoints. | `Sprint 9` |
| **H14** | CKA-guided mixed-precision `GGUF` (`5.290` avg bpw) achieves lower surgical subspace re-leakage amplification (`3.012x` vs `6.402x`) and lower mean KLD vs `F16` (`0.02048` vs `0.02479`) than uniform `Q4_K_M`, keeping `slop_and_refusal_rate <= 0.15` ([ADR-009](decisions/009-gguf-imatrix-vs-cka-survival.md)). | `Post-Release (H14)` |
| **H15** | Phase-segmented Tri-Anchor Lancet preserves deliberation integrity with `tri_anchor_orthogonality_error < 1e-6`, `biprojected_orthogonality_error < 1e-6`, `max_row_crest_factor < 25.0`, and CoT-weighted SCR protection ([ADR-010](decisions/010-tenet-v2-thinking-surgery-and-length-calibrated-grpo.md)). | `Sprint 10` |
| **H16** | Dual-window length-calibrated & artifact-gated GRPO achieves `cot_tag_integrity_rate >= 0.995`, `length_collapse_rate < 0.01`, `slop_and_refusal_rate <= 0.02`, and `artifact_completeness_rate >= 0.99` ([ADR-010](decisions/010-tenet-v2-thinking-surgery-and-length-calibrated-grpo.md)). | `Sprint 11` |