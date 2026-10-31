# ADR-001: Four-Sprint Architecture vs. Eight Loose Phases

* **Status:** Accepted
* **Date:** 2026-10-31

## Context

The initial program draft proposed eight sequential phases spanning pretraining (`Phase I`), mechanistic interpretability on a tiny model (`Phase II`), toy model surgery (`Phase III`), benchmark creation (`Phase IV`), RAG vs. LoRA/SFT (`Phase V`), knowledge distillation (`Phase VI`), model release (`Phase VII`), and inference optimization (`Phase VIII`). At 10 hours/week, executing eight isolated phases takes 9–12 months and spends the first 4–5 months entirely on a `~100M` general web-text model that cannot solve the target domain task (data-pipeline incident reasoning).

## Decision

Consolidate the program into **four high-cohesion sprints (~10–12 weeks @ 10h/week)**:
1. **Sprint 1:** Core Transformer & GPU Pretraining (`Phase I Lean`).
2. **Sprint 2:** Executable Data Incident Benchmark (`dbt + DuckDB + sqlglot`).
3. **Sprint 3:** Adaptation, Knowledge Distillation & Verifiable Rewards (`Context vs. Weights`).
4. **Sprint 4:** Specialist Surgery, Internals Drift (`CKA`) & Production Serving (`vLLM`).

Instead of running standalone interpretability and pruning phases on a `100M` toy model in a vacuum, reusable instrumentation (`src/sml/hooks/` and `src/sml/evals/`) is built in Sprint 1 with Hugging Face `Llama`/`Qwen`-compatible module naming and applied directly to the distilled `1.5B–3B` domain specialist in Sprint 4. Furthermore, Sprint 1 saves 15 log-spaced checkpoints of `sml-100m` to Hugging Face Hub so deeper pretraining dynamics remain available for analysis at any time.

## Consequences

* Reaches the novel applied research core (executable benchmark + distillation + production `vLLM` Pareto curve) within ~8–10 weeks instead of 6+ months.
* Every sprint produces a self-contained, publishable artifact on GitHub and Hugging Face.
* Eliminates throwaway tooling by aligning `sml` layer/projection names (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with standard open-weight architectures from day one.
