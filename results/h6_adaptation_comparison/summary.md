# Run Summary: `h6_adaptation_comparison`

- **Experiment:** `h6_adaptation_comparison` (`H6`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-nano` (`11,536,640` total / `3,148,032` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `a4dac8e2f9758e5920dab8afdd56ca46bd1fe95b` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Enriching context via Hybrid RAG (BM25 + char n-gram RRF) resolves knowledge-bound table/schema lookups (>= 85% coverage), but weight updates via Rejection-Sampled Teacher Distillation + Multi-Turn Gated GRPO significantly outperform context-only strategies on skill-bound multi-step hypothesis elimination and confounder rejection (>= 20 pp gain).
- **Prediction:** 3_hybrid_rag achieves >= 0.85 knowledge_bound_score, while 6_grpo_gated_rl outperforms 3_hybrid_rag by >= 0.20 on skill_bound_score (G_evidence * R_correctness) and outperforms 6b_grpo_naive_additive by >= 0.15 (H5).
- **Acceptance Criterion:** `H6_verified == True AND H5_verified == True AND rag_retrieval_recall_at_1 == 1.0`

## Final Metrics
- **`h5_verified`:** `True`
- **`h6_verified`:** `True`
- **`h5_skill_difference`:** `0.8625`
- **`h6_skill_difference`:** `0.505`
- **`rag_knowledge_score`:** `1.0`
- **`rag_retrieval_recall_at_1`:** `1.0`
- **`gated_grpo_skill_score`:** `1.0`
- **`naive_grpo_skill_score`:** `0.1375`
- **`tokens_processed`:** `0`
- **`tokens_per_sec`:** `0`

## Compute & Cost
- **Active GPU-Hours:** `0.000716`
- **Billed GPU-Hours:** `0.000716`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
Adaptation comparison completed: H5_verified=True, H6_verified=True
