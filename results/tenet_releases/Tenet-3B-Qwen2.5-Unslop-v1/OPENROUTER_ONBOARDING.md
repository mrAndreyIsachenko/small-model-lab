# OpenRouter Onboarding — tenet-lab/Tenet-3B-Qwen2.5-Unslop-v1

## Repository

- **HF repo id:** `tenet-lab/Tenet-3B-Qwen2.5-Unslop-v1`
- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Target providers:** Featherless, Chutes, Novita, DeepInfra

## vLLM launch command

```bash
vllm serve --model tenet-lab/Tenet-3B-Qwen2.5-Unslop-v1 --quantization fp8 --dtype bfloat16 --enable-auto-tool-choice
```

Settings: `quantization=fp8`, `dtype=bfloat16`, `trust_remote_code=False`, `enable-auto-tool-choice=True`.

## FP8 Re-Leakage & Row Crest-Factor proof

| Proof | Value |
| --- | --- |
| H11 FP8 retention ratio | 0.999747 |
| H11 FP8 KL reduction ratio | 0.455822 |
| H11 FP8 subspace Re-Leakage (Lancet) | 0.031033 |
| H11 FP8 subspace Re-Leakage (naive) | 0.499591 |
| H9 Lancet KL reduction ratio | 0.6935 |
| H10 subspace-safe LoRA Re-Leakage | 7.5e-07 |
| Max row crest factor | 3.68282 |
| Max abs weight | 0.0848047 |

## Architectural compatibility checklist

- [x] Standard, first-party architecture (no custom `auto_map`)
- [x] No `trust_remote_code` escape hatches
- [x] Chat template ready
- Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM
