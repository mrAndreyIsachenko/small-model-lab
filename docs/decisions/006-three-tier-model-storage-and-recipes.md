# ADR-006: Three-Tier Model Storage, Deterministic Lancet Recipes (< 500 KB), and On-Demand Checkpoint Materialization

## Status

Accepted

## Context

The Tenet model family is scaling from 3B to 8B, 12B, 32B, and 70B+ parameters. Two constraints make it impossible to treat full checkpoints as first-class, versioned assets:

- **Local disk discipline (ADR-003).** The lab operates under a `< 15 GB` local disk budget for model storage. A single 70B checkpoint in BF16 exceeds 140 GB; even 8B and 12B models overflow the budget after one or two sizes are kept locally.
- **Third-party hub risk.** Official base weights live on external hubs (Hugging Face Hub, and ModelScope for some families). Hub-side moderation, licensing disputes, or takedowns can erase the upstream inputs our builds depend on. Storing our own multi-gigabyte derived blobs on those hubs inherits that fragility and multiplies the blast radius.

What is genuinely ours — and what can be regenerated deterministically — is the *delta* on top of official base models: the Lancet surgery bases (H9), the Unslop GRPO LoRA adapters (H10), and the FP8 (E4M3) export configuration (H11). Those deltas are tiny relative to the checkpoints they produce.

## Decision

Full multi-gigabyte checkpoints published to the Hugging Face Hub are treated as **disposable, deterministic build artifacts**, never as source assets. Any release can be deleted and rebuilt from:

> **Materialized Tenet Checkpoint = Official Base Model + Optional LoRA Delta + LancetRecipe (<= 500 KB)**

Storage is organized in three tiers:

### Tier 1 - Single GitHub Monorepo

One repository holds everything needed to rebuild any release:

- Engine code (`src/sml/`).
- Declarative model family specs (sizes, base model identifiers, layer selections).
- Every LancetRecipe artifact, each under 500 KB: `lancet_recipe.json` plus `lancet_basis.safetensors`. Recipes are small enough to be Git-tracked, diffed, reviewed, and pinned by commit hash.

### Tier 2 - Delta Vault

Private object storage (Cloudflare R2 / S3) holds compact binary deltas that are too large for Git yet still tiny next to checkpoints:

- Compact LoRA adapters produced by Unslop GRPO (H10): 30-150 MB each.
- Recipe bundles (recipe JSON + basis + adapter packaged together) for archival and off-site redundancy.

### Tier 3 - Disposable Materialized Hub Repos

One Hugging Face Hub (or ModelScope) repository per released model/size, named `Tenet-<size>B-<base>-Unslop-v<N>`. These repositories are a cache, not canon:

- They are assembled on demand in the canonical order:
  1. Base (official upstream checkpoint).
  2. Lancet Surgery (H9).
  3. Unslop GRPO LoRA Merge (H10).
  4. Post-Merge Lancet Guard.
  5. FP8 (E4M3) Export (H11).
- Each materialized repo is validated by `verify_provider_readiness` before publish.
- Released repos are grouped into HF Collections so a deleted repository can be re-materialized into the same catalog slot.

## Consequences

- **Resilience to external hub takedowns.** Losing any single Hub repository — ours or upstream — becomes a rebuild rather than a disaster: Tier 1 plus Tier 2 contain everything required to re-materialize the artifact from the same base revision.
- **Git stays light.** Only engine code, specs, and `< 500 KB` recipes are tracked in version control; no multi-gigabyte blobs ever enter the monorepo.
- **Disk stays under budget.** The standing local footprint remains within the `< 15 GB` discipline of ADR-003: base weights and deltas are fetched transiently during materialization, and materialized checkpoints are exported and deleted.
- **Reproducibility is contractual.** Model cards document the reconstruction path (`## Reproducible Reconstruction (ADR-006)`), and `verify_recipe_provenance` binds a materialized checkpoint's `config.json` (hidden size, layer count) to its `lancet_recipe.json` / `lancet_basis.safetensors`, so any published artifact can be traced back to its recipe.
- **Accepted trade-off.** Materialization adds an automated build step (base download + surgery + merge + FP8 export) before a model is usable. This is accepted because the build is deterministic and far cheaper than storing every size.
