# OpenRouter Onboarding — tenet-lab/Tenet-8B-Llama3.1-Unslop-v1

## Repository

- **HF repo id:** `tenet-lab/Tenet-8B-Llama3.1-Unslop-v1`
- **Base model:** `meta-llama/Llama-3.1-8B-Instruct`
- **Target providers:** Featherless, Chutes, Novita, DeepInfra

## vLLM launch command

```bash
vllm serve --model tenet-lab/Tenet-8B-Llama3.1-Unslop-v1 --quantization fp8 --dtype bfloat16 --enable-auto-tool-choice
```

Settings: `quantization=fp8`, `dtype=bfloat16`, `trust_remote_code=False`, `enable-auto-tool-choice=True`.

## FP8 Re-Leakage & Row Crest-Factor proof

| Proof | Value |
| --- | --- |
| H11 FP8 retention ratio | 0.993578 |
| H11 FP8 KL reduction ratio | 0.426536 |
| H11 FP8 subspace Re-Leakage (Lancet) | 0.027783 |
| H11 FP8 subspace Re-Leakage (naive) | 0.500023 |
| H9 Lancet KL reduction ratio | 0.8989 |
| H10 subspace-safe LoRA Re-Leakage | 0.0726127 |
| Max row crest factor | 6.32439 |
| Max abs weight | 0.914062 |

## Architectural compatibility checklist

- [x] Standard, first-party architecture (no custom `auto_map`)
- [x] No `trust_remote_code` escape hatches
- [x] Chat template ready
- Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM
