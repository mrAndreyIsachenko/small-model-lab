# Run Summary: `h10_unslop_grpo`

- **Experiment:** `h10_unslop_grpo` (`H10`)
- **Mode:** `measured`
- **Status:** `completed`
- **Model:** `sml-nano` (`11,536,640` total / `3,148,032` non-emb)
- **Dataset:** `tinystories-smoke`
- **Hardware:** `apple-m4-pro-48gb` (`$0.00/hr`)
- **Git SHA:** `eab01f01b3dcdcccce05d31156c37c1ce3bd7000` (dirty: `True`)

## Hypothesis & Prediction
- **Hypothesis:** Gated "unslop" GRPO — reward shaping with a gated slop-and-refusal penalty — drives the slop-and-refusal rate to <= 2% while beating naive reward-only GRPO by >= 15% relative gain; composing the GRPO-tuned adapter with H9 Lancet bi-projected surgery yields a further >= 15% gain, and merging the trained adapters preserves logits within 1e-4 of the reference model.
- **Prediction:** After 27 GRPO steps (group size 6, lr 0.0038) on the nano model, slop_and_refusal_rate <= 0.02, gated reward shaping beats naive GRPO by >= 15% (h10_gated_vs_naive_gain >= 0.15), post-surgery composition adds >= 15% further gain (h10_post_surgery_gain >= 0.15), and merged LoRA logits stay within 1e-4 of the reference (max_abs_logit_diff < 1e-4).
- **Acceptance Criterion:** `slop_and_refusal_rate <= 0.02 AND h10_gated_vs_naive_gain >= 0.15 AND h10_post_surgery_gain >= 0.15 AND max_abs_logit_diff < 1e-4 AND h10_verified == True`

## Final Metrics
- **`h10_verified`:** `True`
- **`slop_and_refusal_rate`:** `0.015034`
- **`h10_gated_vs_naive_gain`:** `0.227591`
- **`h10_post_surgery_gain`:** `0.661059`
- **`max_abs_logit_diff`:** `5.72e-06`
- **`merged_lora_count`:** `0`
- **`unprojected_lora_releakage`:** `3.51243305`
- **`subspace_safe_lora_releakage`:** `7.5e-07`
- **`post_merge_suppression_retention`:** `1.0`
- **`mean_response_words`:** `37.375`
- **`length_collapse_rate`:** `0.0`
- **`arms`:** `{'1_lancet_surgical_only': {'gated_ifeval_score': 0.318507, 'mean_total_reward': 0.349186, 'slop_and_refusal_rate': 0.731508, 'g_compliance': 0.806276, 'g_antislop': 0.599649, 'r_ifeval': 0.707052, 'r_coherence': 0.952251, 'archetype_probs': [0.085212, 0.409292, 0.193724, 0.128492, 0.052796, 0.130484]}, '2_tenet_grpo_additive': {'gated_ifeval_score': 0.751975, 'mean_total_reward': 0.752274, 'slop_and_refusal_rate': 0.365606, 'g_compliance': 0.999985, 'g_antislop': 0.752798, 'r_ifeval': 0.998261, 'r_coherence': 0.999288, 'archetype_probs': [0.630056, 0.365566, 1.5e-05, 2.5e-05, 0.002696, 0.001642], 'history': [{'step': 0, 'loss': 0.0, 'policy_obj': -0.0, 'kl_div': 0.0}, {'step': 1, 'loss': 1e-06, 'policy_obj': 0.0, 'kl_div': 1.4e-05}, {'step': 2, 'loss': 3e-06, 'policy_obj': -0.0, 'kl_div': 7.3e-05}, {'step': 3, 'loss': 7e-06, 'policy_obj': 0.0, 'kl_div': 0.000177}, {'step': 4, 'loss': 1.7e-05, 'policy_obj': 1e-06, 'kl_div': 0.000443}, {'step': 5, 'loss': 3.4e-05, 'policy_obj': 0.0, 'kl_div': 0.000848}, {'step': 6, 'loss': 6.4e-05, 'policy_obj': -0.0, 'kl_div': 0.001595}, {'step': 7, 'loss': 9.9e-05, 'policy_obj': 0.0, 'kl_div': 0.00249}, {'step': 8, 'loss': -0.002178, 'policy_obj': 0.002344, 'kl_div': 0.004141}, {'step': 9, 'loss': 0.000274, 'policy_obj': -0.0, 'kl_div': 0.006847}, {'step': 10, 'loss': 0.000456, 'policy_obj': -0.0, 'kl_div': 0.011404}, {'step': 11, 'loss': 0.000559, 'policy_obj': 0.0, 'kl_div': 0.013976}, {'step': 12, 'loss': 0.000869, 'policy_obj': -0.0, 'kl_div': 0.021715}, {'step': 13, 'loss': 0.00109, 'policy_obj': 0.0, 'kl_div': 0.027248}, {'step': 14, 'loss': 0.001683, 'policy_obj': 0.0, 'kl_div': 0.042073}, {'step': 15, 'loss': 0.002774, 'policy_obj': -0.0, 'kl_div': 0.069356}, {'step': 16, 'loss': 0.002982, 'policy_obj': 0.0, 'kl_div': 0.074563}, {'step': 17, 'loss': 0.004571, 'policy_obj': 0.0, 'kl_div': 0.114265}, {'step': 18, 'loss': 0.003984, 'policy_obj': 0.002344, 'kl_div': 0.158202}, {'step': 19, 'loss': 0.010877, 'policy_obj': 0.0, 'kl_div': 0.271937}, {'step': 20, 'loss': 0.013557, 'policy_obj': 0.002344, 'kl_div': 0.397529}, {'step': 21, 'loss': 0.016806, 'policy_obj': 0.002344, 'kl_div': 0.478751}, {'step': 22, 'loss': 0.010847, 'policy_obj': 0.002344, 'kl_div': 0.329777}, {'step': 23, 'loss': 0.00621, 'policy_obj': 0.004688, 'kl_div': 0.272439}, {'step': 24, 'loss': 0.0148, 'policy_obj': 0.0, 'kl_div': 0.370014}, {'step': 25, 'loss': 0.008304, 'policy_obj': 0.0, 'kl_div': 0.207612}, {'step': 26, 'loss': 0.014211, 'policy_obj': 0.0, 'kl_div': 0.355291}]}, '3_tenet_grpo_gated': {'gated_ifeval_score': 0.979566, 'mean_total_reward': 0.982706, 'slop_and_refusal_rate': 0.015034, 'g_compliance': 0.999975, 'g_antislop': 0.987759, 'r_ifeval': 0.990996, 'r_coherence': 0.999162, 'archetype_probs': [0.964978, 0.014869, 2.5e-05, 0.000139, 0.00213, 0.017858], 'history': [{'step': 0, 'loss': 0.0, 'policy_obj': -0.0, 'kl_div': 0.0}, {'step': 1, 'loss': 0.0, 'policy_obj': 0.0, 'kl_div': 9e-06}, {'step': 2, 'loss': 2e-06, 'policy_obj': 0.0, 'kl_div': 4.5e-05}, {'step': 3, 'loss': 4e-06, 'policy_obj': 0.0, 'kl_div': 0.000104}, {'step': 4, 'loss': 1e-05, 'policy_obj': -0.0, 'kl_div': 0.000241}, {'step': 5, 'loss': 2e-05, 'policy_obj': 0.0, 'kl_div': 0.000498}, {'step': 6, 'loss': 3.8e-05, 'policy_obj': 0.0, 'kl_div': 0.000951}, {'step': 7, 'loss': 6.3e-05, 'policy_obj': 0.0, 'kl_div': 0.001567}, {'step': 8, 'loss': 0.002434, 'policy_obj': -0.002344, 'kl_div': 0.002256}, {'step': 9, 'loss': 0.00015, 'policy_obj': 0.0, 'kl_div': 0.003742}, {'step': 10, 'loss': 0.000264, 'policy_obj': 0.0, 'kl_div': 0.006591}, {'step': 11, 'loss': 0.000315, 'policy_obj': 0.0, 'kl_div': 0.007876}, {'step': 12, 'loss': 0.00046, 'policy_obj': -0.0, 'kl_div': 0.011506}, {'step': 13, 'loss': 0.000623, 'policy_obj': 0.0, 'kl_div': 0.015567}, {'step': 14, 'loss': 0.001125, 'policy_obj': 0.0, 'kl_div': 0.028134}, {'step': 15, 'loss': 0.001326, 'policy_obj': 0.0, 'kl_div': 0.033155}, {'step': 16, 'loss': 0.002067, 'policy_obj': -0.0, 'kl_div': 0.051671}, {'step': 17, 'loss': 0.002044, 'policy_obj': 0.0, 'kl_div': 0.051106}, {'step': 18, 'loss': 0.006001, 'policy_obj': -0.002344, 'kl_div': 0.091436}, {'step': 19, 'loss': 0.003586, 'policy_obj': 0.0, 'kl_div': 0.089661}, {'step': 20, 'loss': 0.005647, 'policy_obj': 0.0, 'kl_div': 0.14118}, {'step': 21, 'loss': 0.004663, 'policy_obj': 0.001206, 'kl_div': 0.14672}, {'step': 22, 'loss': 0.016704, 'policy_obj': 0.0, 'kl_div': 0.417608}, {'step': 23, 'loss': 0.017661, 'policy_obj': 0.001206, 'kl_div': 0.471668}, {'step': 24, 'loss': 0.026547, 'policy_obj': -0.0, 'kl_div': 0.663683}, {'step': 25, 'loss': 0.049205, 'policy_obj': -0.0, 'kl_div': 1.230129}, {'step': 26, 'loss': 0.076034, 'policy_obj': -0.0, 'kl_div': 1.900845}], 'merged_lora_count': 0, 'max_abs_logit_diff': 5.72e-06}}`
- **`backend`:** `nano`
- **`model_name`:** `sml-tenet-rehab`

## Compute & Cost
- **Active GPU-Hours:** `0.000165`
- **Billed GPU-Hours:** `0.000165`
- **Billed Cost (USD):** `$0.0000` (pre-run estimate: `$0.0000`)

## Conclusion
H10 unslop GRPO verification completed: h10_verified=True, slop_and_refusal_rate=0.0150, gated_vs_naive_gain=0.2276
