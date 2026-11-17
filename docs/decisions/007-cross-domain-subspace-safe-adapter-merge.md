# ADR-007: Cross-Domain Subspace-Safe Adapter Merge (MergeKit / TIES + Lancet)

## Status

Accepted

## Context

After Sprint 6 (`H10`), the Tenet family carries two independently trained LoRA deltas on top of the same Lancet-surgered base:

- `Delta W_SQL` — the multi-turn GRPO adapter trained on the `StatefulToolEnv` incident/SQL rubric (`H5`/`H6`/`H10`).
- `Delta W_unslop` — the gated unslop GRPO adapter (`H10`) that pushes slop/refusal behavior to near zero.

A natural temptation is the naive linear merge `Delta W = Delta W_SQL + Delta W_unslop` (a plain MergeKit `linear` composition, or concatenating both adapters into one combined delta before the in-place merge). This is unsafe for two reasons:

1. **Destructive gradient interference along dominant singular directions of `Delta W_SQL`.** Each adapter was trained against its own reward landscape, so their updates are not independent: `Delta W_unslop` has non-zero components along the top-k left singular vectors `U_SQL` of `Delta W_SQL`. Adding the deltas linearly lets the unslop update partially cancel or perturb the highest-energy SQL directions — exactly the low-rank directions that carry the incident-reasoning skill — producing interference that no per-adapter evaluation catches (each adapter is fine alone; only the sum is broken).
2. **Re-leakage of mass into `U_{refusal/slop}`.** Neither adapter is orthogonal to the ablated refusal/slop subspace removed by Lancet surgery (`H9`) and guarded during the `H10` merge. A linear sum re-injects mass into `U_{refusal/slop}` — the same re-leakage failure mode measured in `H10` (`unprojected_lora_releakage = 0.83` vs. `subspace_safe_lora_releakage = 0.069`), but now introduced *by the merge itself* rather than by skipping projection.

Additionally, any merged checkpoint must remain provider-ready under FP8 (E4M3) quantization (`H11`): merging two deltas changes row norms, and a single high-crest row can break the `Row Crest-Factor < 25.0` readiness gate.

## Decision

Cross-domain adapter merges are performed with the **TIES / Gram-Schmidt Subspace-Safe Merge** instead of naive linear addition. For every merged weight matrix `W`:

1. **Subspace-safe projection of the second delta.** Compute `Delta W_unslop^perp` by projecting `Delta W_unslop` orthogonally against *both*:
   - the refusal/slop basis `U_{refusal/slop}` (stored in the Lancet recipe, `ADR-005`/`ADR-006`), and
   - the top-k left singular vectors `U_SQL` of `Delta W_SQL`, obtained via SVD of the SQL adapter delta.

   Concretely, `Delta W_unslop^perp = (I - P_{refusal/slop} - P_{U_SQL}) Delta W_unslop`, where the projection operator is built with Modified Gram-Schmidt (float32, `H9` convention) so the residual basis stays orthonormal and the projection removes any component of the unslop delta that interferes with the SQL skill subspace or re-leaks into the ablated subspace.
2. **TIES merge of the deltas.** The protected deltas (`Delta W_SQL` and `Delta W_unslop^perp`) are combined with the TIES procedure rather than plain summation: magnitude **trim** (drop the smallest-magnitude coordinates per delta), **sign election** across deltas (resolve coordinate-wise sign conflicts by total signed magnitude), and **disjoint merge** (average only the coordinates that survived trim and agree with the elected sign).
3. **Row-norm restoration.** After the merge, every row of `W + Delta W` is rescaled to its pre-merge L2 row norm (the same norm-preservation invariant as `H9`/`H10`), so the merge cannot silently grow specific rows and shift the model's activation scale.
4. **H11 FP8 readiness gate.** The merged checkpoint passes the FP8 (E4M3) quantization readiness gate before publish: `max_row_crest_factor < 25.0` and the full `verify_provider_readiness` check set (7/7) from `H11`, so a cross-domain merge can never ship a checkpoint that breaks E4M3 serving.

This merge is implemented as a recipe step in the ADR-006 materialization chain (between "Unslop GRPO LoRA Merge" and "Post-Merge Lancet Guard") so it stays deterministic, recipe-pinned, and rebuildable.

## Consequences

- **Skill retention is protected by construction.** The SQL adapter's dominant singular directions are excluded from the second delta before any summation, so the unslop update can no longer cancel the highest-energy incident-reasoning directions; cross-domain interference is measured (`H12`) rather than assumed away.
- **The refusal/slop ablation survives merging.** Orthogonal projection against `U_{refusal/slop}` keeps post-merge suppression retention at the `H10` invariant (`>= 0.90`, re-leakage within the registered bound) even when a second adapter is composed on top.
- **Merges remain FP8-provider-safe.** The `Row Crest-Factor < 25.0` gate plus row-norm restoration guarantee the composed checkpoint quantizes cleanly; merges that violate the gate fail fast during materialization instead of at serving time.
- **Slightly lower expressivity for the second delta.** Trimming and projecting `Delta W_unslop` discards its components along `U_SQL` and low-magnitude coordinates. This is accepted: those components are precisely the interference terms, and the unslop objective is preserved on its own subspace.
- **Reproducibility cost is bounded.** The extra SVD of `Delta W_SQL` and Gram-Schmidt projection run during recipe materialization only; both the SQL basis and the refusal/slop basis are stored in the `< 500 KB` Lancet recipe bundle, keeping ADR-006's rebuild contract intact.
- **Verified under `H12`.** Sprint 8 pre-registers the acceptance criteria for this merge; a merge that fails any criterion is rejected rather than patched post hoc. On `Qwen/Qwen2.5-3B-Instruct` (MPS, `results/h12_qwen3b_mps/final_metrics.json`): `h12_verified=true`, `sql_incident_retention_ratio=1.0`, `unslop_retention_ratio=1.0`, `slop_and_refusal_rate=5.55e-07`, `subspace_safe_merge_releakage=4.71e-07`, `ties_biprojected_vs_naive_merge_gain=+0.7848`, `max_row_crest_factor=17.2706`, `provider_ready=true`.
