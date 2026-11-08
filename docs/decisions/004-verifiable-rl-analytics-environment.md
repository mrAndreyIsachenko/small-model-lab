# ADR-004: Multi-Turn Verifiable RL Environment (`StatefulToolEnv` + Gated Anti-Hacking `Rubric`)

* **Status:** Accepted
* **Date:** 2026-11-01

## Context

In production analytics and data-engineering workflows (e.g., *"Why did appointment completion drop last week?"* or *"Why did the `mart_appointments` SLA fail?"*), static golden datasets (`question -> reference_answer`) fail to evaluate or train the **causal investigation process**:
1. **Pretraining Prior Guessing:** Models can guess plausible-sounding root causes without querying the underlying tables or ruling out confounders.
2. **Reward Hacking under RL (`GRPO`):** When post-training a multi-turn tool-using agent with a naive additive reward ($R = w_1 R_{\text{correct}} + w_2 R_{\text{evidence}} + w_3 R_{\text{cost}} + w_4 R_{\text{safety}}$), the policy rapidly converges to degenerate local optima—such as issuing a single trivial query and terminating early to maximize `safety`, `completion`, and `cost` bonuses, or hallucinating metric numbers that were never returned by tool outputs.
3. **Ecosystem Interoperability:** To serve as a reusable, frontier-grade applied RL benchmark and training harness, the environment should adhere to the open **`verifiers`** (`PrimeIntellect-ai/verifiers`) interface standard (`MultiTurnEnv` / `StatefulToolEnv`, `Rubric`, and `load_environment(...)`) alongside isolated MCP tool schemas.

## Decision

1. **Upgrade Sprint 2 from a Static Evaluator to a `verifiers`-Compatible `StatefulToolEnv` (`src/sml/envs/`):**
   * Expose `load_environment(split, seed, max_steps, ...)` returning a `StatefulToolEnv` (`MultiTurnEnv` subclass) that initializes an isolated per-rollout state (`setup_state(state)`) backed by an ephemeral in-memory/snapshot `DuckDB` handle, seeded fault injection, and planted confounders.
   * Route all agent interactions through guardrailed MCP-compatible tools (`query_warehouse`, `get_metric_lineage`, `inspect_changes`, `submit_diagnosis`) via `update_tool_args(tool_name, args, messages, state)`.
   * Enforce deterministic execution limits (`max_steps = 8`, `max_sql_rows = 50`, `sqlglot` read-only AST validation, and deterministic scan-budget errors on unpartitioned queries).

2. **Enforce Anti-Guessing via Multi-Seed Counterfactual Scenarios:**
   * Each analytical prompt (e.g., *"Why did appointment completion drop last week?"*) is instantiated across multiple deterministic seeds where the surface symptom is identical but the hidden causal state changes across:
     * `product_bug` (e.g., release `v4.18.2` breaking Android SMS reminder webhooks),
     * `data_pipeline_artifact` (e.g., upstream EHR schema drift `'COMPLETED' -> 'FINISHED'` unmapped in `dbt`),
     * `mix_shift` (e.g., Simpson's paradox from clinic/booking-horizon volume shifts),
   * Alongside 2 planted confounders per episode that must be explicitly queried and ruled out.

3. **Use a Gated Multiplicative `Rubric` Instead of an Additive Sum:**
   * Implement the trajectory verifier (`src/sml/envs/rubric.py`) as a `vf.Rubric` with individual diagnostic reward functions and a **multiplicative gated total reward**:
     $$R(\tau) = G_{\text{safety}}(\tau) \cdot G_{\text{complete}}(\tau) \cdot \Big[ R_{\text{correctness}}(\tau) \cdot G_{\text{evidence}}(\tau) \cdot \big(1 + \alpha \cdot R_{\text{efficiency}}(\tau)\big) \Big]$$
   * **Hard Safety & Budget Gates ($G_{\text{safety}}, G_{\text{complete}} \in \{0, 1\}$):** Any forbidden tool call, non-`SELECT` SQL AST (`DROP`, `UPDATE`, `COPY`, `ATTACH`), missing terminal `submit_diagnosis`, or step-budget overrun yields `0.0`.
   * **AST & Output Evidence Gate ($G_{\text{evidence}} \in [0, 1]$):** Parses all executed SQL queries in the trajectory via `sqlglot` to verify that the required ground-truth tables and slice columns (`GROUP BY` / `WHERE`) were actually queried, and cross-checks that `evidence_numbers` in `submit_diagnosis` appear verbatim in the tool observation history. If required tables were not queried, $G_{\text{evidence}} = 0$.
   * **Gated Efficiency Bonus ($R_{\text{efficiency}}$):** Step/cost efficiency only scales the reward when $R_{\text{correctness}} \cdot G_{\text{evidence}} > 0$, eliminating the "cheap silent policy" reward hack.

4. **Pre-Register a Reward-Hacking Ablation in Sprint 3 (`H5`):**
   * In Sprint 3 (`GRPO` post-training on `1.5B–3B` open weights), run a controlled ablation comparing the **Gated Multiplicative Rubric** against a **Naive Additive Rubric** on held-out out-of-distribution fault seeds, measuring trajectory evidence coverage, hallucinated citation rate, and root-cause accuracy.

## Consequences

* Turns static "AI Gold" analytical cases into a reproducible, trainable multi-turn RLVR environment (`src/sml/envs/`) ready for both Rejection-Sampled SFT and multi-turn `GRPO`.
* Provides plug-and-play compatibility with Prime Intellect's `verifiers` (`StatefulToolEnv` + `Rubric` + `load_environment`) and standard MCP tool schemas.
* Produces a rigorous empirical study in Sprint 3 demonstrating how reward structure design prevents reward hacking in multi-turn analytical agents.
