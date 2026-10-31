# ADR-002: Grouped-Query Attention (GQA) & WSD Cooldown Branching

* **Status:** Accepted
* **Date:** 2026-10-31

## Context

Two design choices govern the pretraining experiments in Sprint 1:
1. **Attention mechanism:** Classic educational implementations (`nanoGPT`, `llm.c`) use standard Multi-Head Attention (`MHA`, `n_kv_heads == n_heads`). However, modern open-weight families (`Llama 3`, `Qwen 2.5/3`, `Gemma 2`, `Mistral`) use **Grouped-Query Attention (`GQA`)** to reduce `KV-cache` memory capacity and HBM bandwidth bottlenecks during autoregressive serving.
2. **Schedule & scaling experiments:** Fitting a power-law scaling ladder across 4 tiny model sizes (`23M` to `110M`) plus 12 architectural micro-ablations requires 16+ training runs and suffers from severe embedding-parameter dilution at `V = 32,768` (`>50%` of parameters in a `23M` model sit in the token embedding table).

## Decision

1. **Adopt `GQA` as the default attention mechanism** (`12` query heads / `4` KV heads in `sml-100m`, `3:1` ratio; `2:1` in `sml-xs` and `sml-nano`), while supporting `n_kv_heads == n_heads` in configuration for `gpt2-124m` reference parity (`H1`).
2. **Use the Warmup–Stable–Decay (`WSD`) learning-rate schedule** instead of fixed-horizon cosine decay.
3. **Test `H3` via WSD Cooldown Branching on `sml-100m`:** Run a single primary `sml-100m` (`75.52M` non-embedding parameters) trajectory up to `4.0B` tokens (`~53` tokens per non-embedding parameter), branching a short cooldown decay at `1.5B` tokens (`~20` tokens per non-embedding parameter, Chinchilla compute-optimal regime) and comparing it against the full `4.0B` overtrained cooldown against a 3-seed noise baseline on `sml-xs`.

## Consequences

* Directly connects Sprint 1 pretraining code to Sprint 4 `KV-cache` memory math (`3x` smaller KV-cache in `sml-100m`).
* Tests the core modern Small Language Model hypothesis—how much task quality overtraining buys at zero extra inference cost—on the **exact same architecture, seed, and optimizer trajectory** for `~$1` of extra compute.
* Always reports both `total_params` and `non_embedding_params` in configurations and reports.
