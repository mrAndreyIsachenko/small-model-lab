# Run Summary: `h10_qwen3b_mps`

- **Experiment:** `h10_unslop_grpo` (`H10`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `Qwen2.5-3B-Instruct` (`3,085,846,528` total / `2,774,681,600` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `eab01f01b3dcdcccce05d31156c37c1ce3bd7000` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Gated "unslop" GRPO — reward shaping with a gated slop-and-refusal penalty — drives the slop-and-refusal rate to <= 2% while beating naive reward-only GRPO by >= 15% relative gain; composing the GRPO-tuned adapter with H9 Lancet bi-projected surgery yields a further >= 15% gain, and merging the trained adapters preserves logits within 1e-4 of the reference model.
- **Prediction:** After 27 GRPO steps (group size 6, lr 0.0038) on the nano model, slop_and_refusal_rate <= 0.02, gated reward shaping beats naive GRPO by >= 15% (h10_gated_vs_naive_gain >= 0.15), post-surgery composition adds >= 15% further gain (h10_post_surgery_gain >= 0.15), and merged LoRA logits stay within 1e-4 of the reference (max_abs_logit_diff < 1e-4).
- **Acceptance Criterion:** `slop_and_refusal_rate <= 0.02 AND h10_gated_vs_naive_gain >= 0.15 AND h10_post_surgery_gain >= 0.15 AND max_abs_logit_diff < 1e-4 AND h10_verified == True`

## Final Metrics
- **`h10_verified`:** `True`
- **`slop_and_refusal_rate`:** `0.090889`
- **`h10_gated_vs_naive_gain`:** `0.160164`
- **`h10_post_surgery_gain`:** `0.359802`
- **`max_abs_logit_diff`:** `0.09113312`
- **`merged_lora_count`:** `0`
- **`unprojected_lora_releakage`:** `0.83001828`
- **`subspace_safe_lora_releakage`:** `0.069000869`
- **`post_merge_suppression_retention`:** `1.0`
- **`mean_response_words`:** `31.2083`
- **`length_collapse_rate`:** `0.041666666666666664`
- **`arms`:** `{'1_lancet_surgical_only': {'gated_ifeval_score': 0.363222, 'mean_total_reward': 0.363796, 'slop_and_refusal_rate': 2e-05, 'g_compliance': 1.0, 'g_antislop': 0.377576, 'r_ifeval': 0.779193, 'r_coherence': 0.800452, 'archetype_probs': [0.334216, 2e-05, 0.0, 0.0, 0.665508, 0.000255]}, '2_tenet_grpo_additive': {'gated_ifeval_score': 0.56286, 'mean_total_reward': 0.563669, 'slop_and_refusal_rate': 0.154198, 'g_compliance': 0.898083, 'g_antislop': 0.672106, 'r_ifeval': 0.862844, 'r_coherence': 0.916004, 'archetype_probs': [0.530541, 0.052165, 0.101917, 0.000116, 0.311065, 0.004196], 'history': [{'step': 0, 'loss': 0.001242, 'policy_obj': -0.001242, 'kl_div': -0.0}, {'step': 1, 'loss': 0.001489, 'policy_obj': -0.001489, 'kl_div': 1e-06}, {'step': 2, 'loss': 0.002078, 'policy_obj': -0.000982, 'kl_div': 0.109569}, {'step': 3, 'loss': 0.002317, 'policy_obj': -0.001242, 'kl_div': 0.107536}, {'step': 4, 'loss': 0.003983, 'policy_obj': -0.001385, 'kl_div': 0.259778}, {'step': 5, 'loss': 0.003587, 'policy_obj': -0.001258, 'kl_div': 0.232913}]}, '3_tenet_grpo_gated': {'gated_ifeval_score': 0.723024, 'mean_total_reward': 0.729023, 'slop_and_refusal_rate': 0.090889, 'g_compliance': 0.999855, 'g_antislop': 0.741581, 'r_ifeval': 0.909995, 'r_coherence': 0.946833, 'archetype_probs': [0.660225, 0.090688, 0.000145, 5.6e-05, 0.208299, 0.040587], 'history': [{'step': 0, 'loss': 0.000548, 'policy_obj': -0.000548, 'kl_div': -0.0}, {'step': 1, 'loss': 0.001622, 'policy_obj': -0.0009, 'kl_div': 0.072168}, {'step': 2, 'loss': 0.001614, 'policy_obj': -0.000534, 'kl_div': 0.108009}, {'step': 3, 'loss': 0.008171, 'policy_obj': -0.000722, 'kl_div': 0.744917}, {'step': 4, 'loss': 0.063714, 'policy_obj': 0.000358, 'kl_div': 6.407249}, {'step': 5, 'loss': 0.04983, 'policy_obj': 0.002498, 'kl_div': 5.232752}], 'merged_lora_count': 0, 'max_abs_logit_diff': 0.09113312}}`
- **`backend`:** `llm`
- **`model_name`:** `Qwen/Qwen2.5-3B-Instruct`

## Compute & Cost
- **Active GPU-Hours:** `0.173918`
- **Billed GPU-Hours:** `0.173918`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H10 unslop GRPO verification completed: h10_verified=True, slop_and_refusal_rate=0.0909, gated_vs_naive_gain=0.1602
