---
title: FP8 Abliteration Inspector
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
---

# 🔬 FP8 Abliteration Inspector

A self-contained Hugging Face Space that audits **abliterated checkpoints** and
**Lancet recipe bundles** for **FP8 quantization safety** — without ever loading
a whole model into RAM.

## Why

Naive abliteration writes rows that collapse onto the injected *refusal
direction*. Under FP8 (per-tensor scales) those outlier rows clip the direction
back in: the model **re-leaks refusals**. The detector is the **row crest
factor** of each 2D weight tensor:

```
row_rms   = sqrt(mean(w², dim=1) + 1e-8)
row_crest = max(abs(w).amax(dim=1) / row_rms)   # per tensor
```

- `max_row_crest_factor < 25.0` → **FP8 PASS**
- `max_abs_weight < 50.0` → weight-outlier gate

## Tabs

1. **Live Demo Comparison** — real Qwen2.5-3B measurements: *Naive Abliteration*
   (H9 KL 2.6353, H11 FP8 KL 0.0975, FP8 re-leakage 0.5003, crest ≥ 25.0 → FAIL)
   vs *Tenet-3B-Qwen2.5-Unslop-v1* (H9 KL 0.0244, H11 FP8 KL 0.0469, re-leakage
   0.0270, FP8 retention 0.9541, crest 17.27 → PASS; top-3 SCR layers 21/15/19).
2. **Inspect Local Checkpoint / Recipe Bundle** — streams `*.safetensors`
   tensor-by-tensor (`safetensors.safe_open`), enforces the ADR-003 `< 15 GB`
   bundle limit, runs the 7 `verify_provider_readiness` checks
   (`standard_architecture`, `no_trust_remote_code`, `valid_chat_template`,
   `no_nan_inf`, `no_unmerged_lora`, `weight_outliers_ok`,
   `acceptable_row_crest_factor`) plus **recipe provenance** when
   `lancet_recipe.json` exists, and falls back to metadata JSONs
   (`provider_readiness.json`, `final_metrics.json`, `release_manifest.json`,
   `openrouter_onboarding.json`) when no weight files are present. Renders the
   crest-factor distribution + per-layer SCR profile as inline SVG.
3. **CKA / Lancet-Safe GGUF Command Generator** — emits a ready-to-paste bash
   pipeline (recipe application → FP8 crest gate → llama.cpp GGUF → quantize →
   optional `vllm serve --quantization fp8` / HF upload).

## Recipe provenance

`lancet_recipe.json` (schema `1.0`) + `lancet_basis.safetensors` are verified:
basis tensors must be finite with **unit-norm rows** (tolerance `1e-4`), and
`config.json` `hidden_size` / `num_hidden_layers` must agree with the recipe and
the basis shapes.

## Usage

```bash
pip install -r requirements.txt
python app.py                          # UI on :7860
python app.py --inspect /path/to/ckpt  # CLI: pure functions, no gradio needed
```

`app.py` imports cleanly without gradio/torch/safetensors — the pure functions
(`inspect_directory_or_bundle`, `verify_recipe_provenance`,
`generate_gguf_commands`, `render_demo_svg`) stay usable in CI. Torch is
optional (numpy fallback), `src.sml` is **not** imported; constants mirror
`src/sml/hub/validator.py`.
