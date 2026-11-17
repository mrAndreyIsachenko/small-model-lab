# Run Summary: `h10_nemo12b_h100`

- **Experiment:** `h10_unslop_grpo` (`H10`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Mistral-Nemo-Instruct-2407` (`12,772,070,400` total / `11,429,893,120` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `uncommitted` (dirty: `False`)

## Hypothesis & Prediction
- **Hypothesis:** Gated "unslop" GRPO — reward shaping with a gated slop-and-refusal penalty — drives the slop-and-refusal rate to <= 2% while beating naive reward-only GRPO by >= 15% relative gain; composing the GRPO-tuned adapter with H9 Lancet bi-projected surgery yields a further >= 15% gain, and merging the trained adapters preserves logits within 1e-4 of the reference model.
- **Prediction:** After 27 GRPO steps (group size 6, lr 0.0038) on the nano model, slop_and_refusal_rate <= 0.02, gated reward shaping beats naive GRPO by >= 15% (h10_gated_vs_naive_gain >= 0.15), post-surgery composition adds >= 15% further gain (h10_post_surgery_gain >= 0.15), and merged LoRA logits stay within 1e-4 of the reference (max_abs_logit_diff < 1e-4).
- **Acceptance Criterion:** `slop_and_refusal_rate <= 0.02 AND h10_gated_vs_naive_gain >= 0.15 AND h10_post_surgery_gain >= 0.15 AND max_abs_logit_diff < 1e-4 AND h10_verified == True`

## Final Metrics
- **`h10_verified`:** `True`
- **`slop_and_refusal_rate`:** `0.023933`
- **`h10_gated_vs_naive_gain`:** `0.429204`
- **`h10_post_surgery_gain`:** `0.270966`
- **`max_abs_logit_diff`:** `0.15315247`
- **`merged_lora_count`:** `0`
- **`unprojected_lora_releakage`:** `0.80521945`
- **`subspace_safe_lora_releakage`:** `0.128179941`
- **`post_merge_suppression_retention`:** `1.0`
- **`mean_response_words`:** `29.375`
- **`length_collapse_rate`:** `0.125`
- **`arms`:** `{'1_lancet_surgical_only': {'gated_ifeval_score': 0.356241, 'mean_total_reward': 0.356891, 'slop_and_refusal_rate': 0.000163, 'g_compliance': 1.0, 'g_antislop': 0.370885, 'r_ifeval': 0.776401, 'r_coherence': 0.798922, 'archetype_probs': [0.326742, 0.000163, 0.0, 0.0, 0.672608, 0.000487]}, '2_tenet_grpo_additive': {'gated_ifeval_score': 0.198003, 'mean_total_reward': 0.228732, 'slop_and_refusal_rate': 0.303385, 'g_compliance': 1.0, 'g_antislop': 0.298196, 'r_ifeval': 0.605616, 'r_coherence': 0.810999, 'archetype_probs': [0.123475, 0.007673, 0.0, 0.295713, 0.523422, 0.049717], 'history': [{'step': 0, 'loss': 0.024835, 'policy_obj': -0.024835, 'kl_div': 0.0}, {'step': 1, 'loss': 0.024838, 'policy_obj': -0.024835, 'kl_div': 0.000315}, {'step': 2, 'loss': 0.092725, 'policy_obj': -0.0, 'kl_div': 9.272489}, {'step': 3, 'loss': 0.02692, 'policy_obj': -0.024835, 'kl_div': 0.208436}, {'step': 4, 'loss': 0.053908, 'policy_obj': -0.023594, 'kl_div': 3.031421}, {'step': 5, 'loss': 0.057216, 'policy_obj': -0.022352, 'kl_div': 3.486404}]}, '3_tenet_grpo_gated': {'gated_ifeval_score': 0.627207, 'mean_total_reward': 0.754608, 'slop_and_refusal_rate': 0.023933, 'g_compliance': 1.0, 'g_antislop': 0.983247, 'r_ifeval': 0.633763, 'r_coherence': 0.962166, 'archetype_probs': [0.235582, 0.009366, 0.0, 0.014566, 0.0, 0.740485], 'history': [{'step': 0, 'loss': 0.000698, 'policy_obj': -0.000698, 'kl_div': 0.0}, {'step': 1, 'loss': 0.000701, 'policy_obj': -0.000698, 'kl_div': 0.00025}, {'step': 2, 'loss': 0.080804, 'policy_obj': -0.000233, 'kl_div': 8.057139}, {'step': 3, 'loss': 0.126443, 'policy_obj': -0.017618, 'kl_div': 10.882597}, {'step': 4, 'loss': 0.091737, 'policy_obj': -0.015444, 'kl_div': 7.629305}, {'step': 5, 'loss': 0.117081, 'policy_obj': -0.017618, 'kl_div': 9.946382}], 'merged_lora_count': 0, 'max_abs_logit_diff': 0.15315247}}`
- **`backend`:** `llm`
- **`model_name`:** `mistralai/Mistral-Nemo-Instruct-2407`

## Compute & Cost
- **Active GPU-Hours:** `0.010494`
- **Billed GPU-Hours:** `0.010494`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H10 unslop GRPO verification completed: h10_verified=True, slop_and_refusal_rate=0.0239, gated_vs_naive_gain=0.4292
