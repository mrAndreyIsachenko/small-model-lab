"""Model card generation for Tenet checkpoints.

Tenet models are produced by **Lancet**, a bi-projected norm-preserving
subspace surgery engine followed by gated unslop GRPO, and are published
as provider-ready FP8 checkpoints for vLLM and OpenRouter providers.
"""

from __future__ import annotations

import re
from typing import Any

CANONICAL_TENET_NAMES = (
    "Tenet-3B-Qwen2.5-Unslop-v1",
    "Tenet-8B-Llama3.1-Unslop-v1",
    "Tenet-12B-Nemo-Unslop-v1",
    "Tenet-32B-Qwen2.5-Unslop-v1",
    "Tenet-70B-Llama3.3-Unslop-v1",
)

_TENET_NAME_PATTERN = re.compile(
    r"^Tenet-\d+(?:\.\d+)?B-[A-Za-z0-9](?:[A-Za-z0-9.\-]*[A-Za-z0-9])?-Unslop-v\d+$"
)

_TAGS = ("tenet", "lancet", "unslop", "fp8", "vllm", "openrouter", "grpo")


def generate_tenet_model_card(
    model_id: str,
    base_model: str,
    metrics: dict[str, Any],
    *,
    recipe_metadata: dict[str, Any] | None = None,
) -> str:
    """Render a Hugging Face model card for a Tenet checkpoint.

    ``model_id`` must follow the canonical Tenet naming scheme
    ``Tenet-<size>B-<base>-Unslop-v<revision>``, e.g.
    ``Tenet-3B-Qwen2.5-Unslop-v1``, ``Tenet-8B-Llama3.1-Unslop-v1`` or
    ``Tenet-12B-Nemo-Unslop-v1``.

    ``metrics`` may provide benchmark numbers keyed as ``h9_bf16``,
    ``h9_fp8``, ``h9_naive``, ``h10_*`` and ``h11_*``; missing entries
    fall back to safe ``n/a`` placeholders.

    ``recipe_metadata`` may provide the Lancet recipe fields published
    alongside the checkpoint (``probes_sha256``, ``selected_layers``);
    when supplied they are rendered into the ADR-006 reconstruction
    section.
    """
    _validate_model_id(model_id)

    h9_bf16 = _metric(metrics, "h9_bf16")
    h9_fp8 = _metric(metrics, "h9_fp8")
    h9_naive = _metric(metrics, "h9_naive")
    h10_bf16 = _metric(metrics, "h10_bf16")
    h10_fp8 = _metric(metrics, "h10_fp8")
    h10_naive = _metric(metrics, "h10_naive")
    h11_bf16 = _metric(metrics, "h11_bf16")
    h11_fp8 = _metric(metrics, "h11_fp8")
    h11_naive = _metric(metrics, "h11_naive")
    max_row_crest_factor = _metric(metrics, "max_row_crest_factor")

    tags = "\n".join(f"- {tag}" for tag in _TAGS)
    recipe_details = _recipe_reconstruction_details(recipe_metadata)

    return f"""---
license: apache-2.0
base_model: {base_model}
pipeline_tag: text-generation
tags:
{tags}
---

# {model_id}

## Model overview

**{model_id}** is a Tenet fine-tune of [`{base_model}`]({_hf_url(base_model)}),
surgical de-refusal and de-slop work by the **Lancet** engine, finalized
with gated unslop GRPO and exported as a provider-ready FP8 (E4M3)
checkpoint. No custom code, no `trust_remote_code`, no `auto_map`:
the repo drops straight into vLLM, OpenRouter providers, GGUF and MLX.

## Lancet surgery engine

Lancet is not naive abliteration. It performs **bi-projected
norm-preserving subspace surgery**: the refusal direction is identified
in both the input (residual stream) and output (unembedding) projections,
the component is removed bi-directionally, and the residual norm is
restored exactly, so activation magnitudes and downstream token
distributions stay on-manifold.

The surgery is followed by **gated unslop GRPO**: a preference group
optimization pass whose reward gates penalize slop patterns (purple
prose, "tapestry" vocabulary, moralizing preambles) only where the
surgery weakened refusal circuitry, so the base model's reasoning and
coding ability is reinforced rather than damaged.

## Intended Use & Safety Alignment Scope

**{model_id}** is intended for aligned assistant use. Lancet subspace
steering mitigates false-positive over-refusals (blanket refusals of
benign prompts) and stylistic AI-slop drift (purple prose, moralizing
preambles, "tapestry" vocabulary) while preserving the analytical,
coding and structured-reasoning anchor subspace of the base model.

The model is not intended or tuned for offensive cyber operations,
malware generation, or unlawful content of any kind; such uses are
explicitly out of scope and unsupported by this release.

## Benchmarks

Benchmarks are reported for the BF16 surgery output, the shipped FP8
(E4M3) export, and a naive-abliteration baseline of the same base model.

### H9 - refusal & slop retention

Lower is better: residual refusal-direction alignment and slop density
after surgery.

| Variant | H9 |
| --- | --- |
| BF16 | {h9_bf16} |
| FP8 (E4M3) | {h9_fp8} |
| Naive Abliteration | {h9_naive} |

### H10 - downstream capability retention

Higher is better: held-out reasoning, coding and instruction-following
accuracy relative to the untouched base model.

| Variant | H10 |
| --- | --- |
| BF16 | {h10_bf16} |
| FP8 (E4M3) | {h10_fp8} |
| Naive Abliteration | {h10_naive} |

### H11 - FP8 quantization robustness

Higher is better: task accuracy of each variant after FP8 (E4M3)
quantization, measuring how cleanly the surgery survives deployment
precision.

| Variant | H11 |
| --- | --- |
| BF16 | {h11_bf16} |
| FP8 (E4M3) | {h11_fp8} |
| Naive Abliteration | {h11_naive} |

Max row crest-factor: {max_row_crest_factor}

## Deployment & inference

### vLLM (FP8)

```bash
vllm serve {model_id} --quantization fp8
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="{model_id}",
    messages=[{{"role": "user", "content": "Hello!"}}],
)
print(response.choices[0].message.content)
```

### Ollama / GGUF

```text
# Modelfile
FROM ./{model_id}-Q4_K_M.gguf
PARAMETER temperature 0.7
```

```bash
ollama create {model_id.lower()} -f Modelfile
ollama run {model_id.lower()}
```

### MLX

```bash
mlx_lm.generate \\
    --model {model_id} \\
    --prompt "Explain norm-preserving subspace surgery." \\
    --max-tokens 512
```

### OpenRouter

{model_id} is served on OpenRouter by FP8-native providers with zero
`trust_remote_code` required - the config ships no `auto_map` and no
custom modeling code, so hosts need only standard support:

- **DeepInfra** - FP8 (E4M3) vLLM deployment
- **Novita AI** - FP8 vLLM deployment
- **Chutes** - serverless FP8 deployment
- **Featherless** - managed vLLM deployment

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="$OPENROUTER_API_KEY",
)
response = client.chat.completions.create(
    model="{model_id}",
    messages=[{{"role": "user", "content": "Hello!"}}],
)
```

## Reproducible Reconstruction (ADR-006)

Per **ADR-006**, this checkpoint is fully reproducible from its
upstream base model plus a compact steering artifact. The repository
ships:

- `lancet_recipe.json` - the recipe: `schema_version`, `d_model`,
  `num_layers`, `selected_layers` and per-layer `layer_weights`.
- `lancet_basis.safetensors` - the compact basis artifact with one
  orthonormal `layer_<idx>` tensor (shape ``(k, d_model)``) per
  steered layer.

Deterministic reconstruction:

1. Start from the untouched upstream base checkpoint
   [`{base_model}`]({_hf_url(base_model)}).
2. Apply the bi-projected norm-preserving subspace surgery using the
   `layer_<idx>` basis rows from `lancet_basis.safetensors` for each
   layer listed in `selected_layers`, restoring the residual norm
   exactly so activation magnitudes stay on-manifold.
3. Verify against `probes_sha256`: the probe manifest checksum must
   match the published recipe, making the reconstruction
   deterministic relative to the released BF16 surgery output.
{recipe_details}
"""


def _recipe_reconstruction_details(recipe_metadata: dict[str, Any] | None) -> str:
    """Render the ADR-006 recipe metadata bullet lines (may be empty)."""
    if not recipe_metadata:
        return ""
    lines: list[str] = []
    probes_sha256 = recipe_metadata.get("probes_sha256")
    if isinstance(probes_sha256, str) and probes_sha256.strip():
        lines.append(f"- Probe manifest checksum (`probes_sha256`): `{probes_sha256.strip()}`")
    selected_layers = recipe_metadata.get("selected_layers")
    if isinstance(selected_layers, (list, tuple)) and selected_layers:
        rendered = []
        for layer in selected_layers:
            try:
                rendered.append(str(int(layer)))
            except (TypeError, ValueError):
                rendered.append(str(layer))
        lines.append("- Steered layers (`selected_layers`): " + ", ".join(rendered))
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def _validate_model_id(model_id: str) -> None:
    """Raise ``ValueError`` for non-canonical Tenet names."""
    if not isinstance(model_id, str) or not _TENET_NAME_PATTERN.match(model_id):
        examples = ", ".join(CANONICAL_TENET_NAMES)
        raise ValueError(
            "model_id must follow the canonical Tenet naming scheme "
            "'Tenet-<size>B-<base>-Unslop-v<revision>', e.g. " + examples
        )


def _metric(metrics: dict[str, Any], key: str) -> str:
    """Format a benchmark metric, falling back to ``n/a``."""
    value = metrics.get(key)
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _hf_url(repo_id: str) -> str:
    """Return the Hugging Face hub URL for ``repo_id``."""
    return f"https://huggingface.co/{repo_id}"
