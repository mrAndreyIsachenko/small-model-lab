# OpenRouter Onboarding — mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1

## Repository

- **HF repo id:** `mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1`
- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Target providers:** Featherless, Chutes, Novita, DeepInfra

## vLLM launch command

```bash
vllm serve --model mr-andrey-isachenko/Tenet-3B-Qwen2.5-Unslop-v1 --quantization fp8 --dtype bfloat16 --enable-auto-tool-choice
```

Settings: `quantization=fp8`, `dtype=bfloat16`, `trust_remote_code=False`, `enable-auto-tool-choice=True`.

## FP8 Re-Leakage & Row Crest-Factor proof

| Proof | Value |
| --- | --- |
| H11 FP8 retention ratio | 0.954145 |
| H11 FP8 KL reduction ratio | 0.518488 |
| H11 FP8 subspace Re-Leakage (Lancet) | 0.0270057 |
| H11 FP8 subspace Re-Leakage (naive) | 0.500251 |
| H9 Lancet KL reduction ratio | 0.9941 |
| H10 subspace-safe LoRA Re-Leakage | 0.0690009 |
| Max row crest factor | 17.2706 |
| Max abs weight | 1.42969 |

## Architectural compatibility checklist

- [x] Standard, first-party architecture (no custom `auto_map`)
- [x] No `trust_remote_code` escape hatches
- [x] Chat template ready
- Supported architectures: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM

## External Benchmarks

| Benchmark | Score |
| --- | --- |
| IFEval strict | 0.6 |
| GSM8K / MATH-500 | 0 |
| SQL-Incident Eval | 0 |
| Direct-Compliance / Unslop Audit | 1 |
| Gsm8K Cot Flexible | 0.45 |
| Gsm8K Cot Strict | 0 |
| Ifeval Inst Strict | 0.666667 |
| Slop And Refusal Rate | 0 |
| Sql Incident Reward | 0 |
| Unslop Pass Rate | 1 |
