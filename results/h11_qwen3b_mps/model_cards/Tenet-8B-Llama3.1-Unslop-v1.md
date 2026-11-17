---
license: apache-2.0
base_model: meta-llama/Llama-3.1-8B-Instruct
pipeline_tag: text-generation
tags:
- tenet
- lancet
- unslop
- fp8
- vllm
- openrouter
- grpo
---

# Tenet-8B-Llama3.1-Unslop-v1

## Model overview

**Tenet-8B-Llama3.1-Unslop-v1** is a Tenet fine-tune of [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct),
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

**Tenet-8B-Llama3.1-Unslop-v1** is intended for aligned assistant use. Lancet subspace
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
| BF16 | n/a |
| FP8 (E4M3) | n/a |
| Naive Abliteration | n/a |

### H10 - downstream capability retention

Higher is better: held-out reasoning, coding and instruction-following
accuracy relative to the untouched base model.

| Variant | H10 |
| --- | --- |
| BF16 | n/a |
| FP8 (E4M3) | n/a |
| Naive Abliteration | n/a |

### H11 - FP8 quantization robustness

Higher is better: task accuracy of each variant after FP8 (E4M3)
quantization, measuring how cleanly the surgery survives deployment
precision.

| Variant | H11 |
| --- | --- |
| BF16 | 1.00 |
| FP8 (E4M3) | 0.95 |
| Naive Abliteration | 0.90 |

Max row crest-factor: 17.27

## Deployment & inference

### vLLM (FP8)

```bash
vllm serve Tenet-8B-Llama3.1-Unslop-v1 --quantization fp8
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="Tenet-8B-Llama3.1-Unslop-v1",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Ollama / GGUF

```text
# Modelfile
FROM ./Tenet-8B-Llama3.1-Unslop-v1-Q4_K_M.gguf
PARAMETER temperature 0.7
```

```bash
ollama create tenet-8b-llama3.1-unslop-v1 -f Modelfile
ollama run tenet-8b-llama3.1-unslop-v1
```

### MLX

```bash
mlx_lm.generate \
    --model Tenet-8B-Llama3.1-Unslop-v1 \
    --prompt "Explain norm-preserving subspace surgery." \
    --max-tokens 512
```

### OpenRouter

Tenet-8B-Llama3.1-Unslop-v1 is served on OpenRouter by FP8-native providers with zero
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
    model="Tenet-8B-Llama3.1-Unslop-v1",
    messages=[{"role": "user", "content": "Hello!"}],
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
   [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
2. Apply the bi-projected norm-preserving subspace surgery using the
   `layer_<idx>` basis rows from `lancet_basis.safetensors` for each
   layer listed in `selected_layers`, restoring the residual norm
   exactly so activation magnitudes stay on-manifold.
3. Verify against `probes_sha256`: the probe manifest checksum must
   match the published recipe, making the reconstruction
   deterministic relative to the released BF16 surgery output.

