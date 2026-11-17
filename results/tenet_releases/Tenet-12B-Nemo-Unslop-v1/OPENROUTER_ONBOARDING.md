# OpenRouter Onboarding — tenet-lab/Tenet-12B-Nemo-Unslop-v1

## Repository

- **HF repo id:** `tenet-lab/Tenet-12B-Nemo-Unslop-v1`
- **Base model:** `mistralai/Mistral-Nemo-Instruct-2407`
- **Target providers:** Featherless, Chutes, Novita, DeepInfra

## vLLM launch command

```bash
vllm serve --model tenet-lab/Tenet-12B-Nemo-Unslop-v1 --quantization fp8 --dtype bfloat16 --enable-auto-tool-choice
```

Settings: `quantization=fp8`, `dtype=bfloat16`, `trust_remote_code=False`, `enable-auto-tool-choice=True`.

## FP8 Re-Leakage & Row Crest-Factor proof

| Proof | Value |
| --- | --- |
| H11 FP8 retention ratio | 0.967803 |
| H11 FP8 KL reduction ratio | 0.489211 |
| H11 FP8 subspace Re-Leakage (Lancet) | 0.183885 |
| H11 FP8 subspace Re-Leakage (naive) | 0.499639 |
| H9 Lancet KL reduction ratio | 0.9672 |
| H10 subspace-safe LoRA Re-Leakage | 0.12818 |
| Max row crest factor | 10.6508 |
| Max abs weight | 0.917969 |

## Architectural compatibility checklist

- [x] Standard, first-party architecture (no custom `auto_map`)
- [x] No `trust_remote_code` escape hatches
- [x] Chat template ready
- Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM
