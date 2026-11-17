"""LLM-scale weight surgery (`H7` CKA Mixed-Precision, `H8` Subspace Ablation) & Pareto.

Mirrors `sml.surgery.pareto.run_surgery_and_pareto_suite` (sml-nano backend) with a real
causal LM (`AutoModelForCausalLM`) adapted on `StatefulToolEnv` via LoRA SFT + GRPO.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from sml.adaptation.compare import _llm_episode_state, _resolve_llm_device
from sml.adaptation.grpo import LLMGRPOConfig, train_grpo_llm_policy
from sml.adaptation.llm_sft import (
    collect_oracle_sft_trajectories,
    collect_partial_sft_trajectories,
    train_lora_sft_llm,
)
from sml.adaptation.lora import apply_lora_to_transformer, merge_lora_weights
from sml.adaptation.rag import HybridRunbookRetriever, build_incident_knowledge_base
from sml.envs.scenarios import IncidentScenario
from sml.envs.tool_env import format_incident_prompt, load_environment
from sml.surgery.cka import (
    _collect_prompt_deltas,
    collect_layer_activations,
    compute_blockwise_quantization_drift,
    get_transformer_layers,
    linear_cka,
)
from sml.surgery.probes import (
    REASONING_ANCHOR_PAIRS,
    REFUSAL_CONTRAST_PAIRS,
    SLOP_CONTRAST_PAIRS,
    TOOL_SQL_ANCHOR_PAIRS,
    encode_contrast_pairs,
)
from sml.surgery.quantization import (
    allocate_mixed_precision_bits,
    apply_mixed_precision_quantization,
    prune_transformer_layers,
)
from sml.surgery.subspace import (
    apply_lancet_biprojected_surgery,
    apply_subspace_ablation,
    biproject_direction,
    build_lancet_recipe,
    compute_layer_scr_profile,
    compute_probe_spectral_stats,
    save_lancet_recipe,
)

_RUBRIC_KEYS = ("g_safety", "g_complete", "g_evidence", "r_correctness", "mean_gated_reward")


def _encode_chat_prompt(
    tokenizer: Any, messages: list[dict[str, str]], target_device: str
) -> torch.Tensor:
    """Render a chat prompt into an unpadded `[1, seq_len]` tensor on `target_device`.

    Prefers `tokenizer.apply_chat_template`; falls back to plain tokenization when the
    tokenizer ships no chat template.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(rendered, return_tensors="pt").input_ids
    except Exception:
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        input_ids = tokenizer(joined, return_tensors="pt").input_ids
    return input_ids.to(target_device)


def _drift_curve(
    reference_deltas: list[torch.Tensor], adapted_deltas: list[torch.Tensor]
) -> list[dict[str, float | int]]:
    """Per-layer Linear CKA drift between two models' residual-delta activations."""
    results: list[dict[str, float | int]] = []
    for idx, (ref_d, adapted_d) in enumerate(zip(reference_deltas, adapted_deltas, strict=False)):
        similarity = linear_cka(ref_d, adapted_d)
        results.append(
            {
                "layer_idx": idx,
                "cka_similarity": round(similarity, 6),
                "cka_drift": round(1.0 - similarity, 6),
            }
        )
    return results


def _last_token_direction(
    premature_acts: torch.Tensor, verified_acts: torch.Tensor
) -> torch.Tensor:
    """Unit contrastive direction between premature-guessing and verified residuals.

    Same normalized mean-difference estimator as
    `sml.surgery.subspace.extract_contrastive_direction`, applied directly to
    pre-collected last-token activations of the final transformer layer.
    """
    diff = premature_acts.mean(dim=0).float() - verified_acts.mean(dim=0).float()
    return diff / (diff.norm(p=2) + 1e-8)


def run_llm_surgery_and_pareto_suite(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "auto",
    seed: int = 2026,
    episodes: int = 6,
    sft_steps: int = 8,
    grpo_steps: int = 4,
    group_size: int = 4,
    max_new_tokens: int = 160,
) -> dict[str, Any]:
    """Execute `H7` (CKA Mixed-Precision RCO), `H8` (Rank-1 Subspace Ablation) and the
    Pareto frontier on a real LLM adapted on `StatefulToolEnv` (LoRA SFT + GRPO)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Deferred: `sml.surgery.pareto` re-exports this module, so a module-scope import
    # of it here would be circular.
    from sml.surgery.pareto import compute_logit_kl_divergence

    torch.manual_seed(seed)
    resolved_device, dtype = _resolve_llm_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(resolved_device)

    train_env = load_environment(split="train", count=3, base_seed=seed, max_steps=4)
    eval_env = load_environment(split="test", count=episodes, base_seed=seed + 100, max_steps=4)
    retriever = HybridRunbookRetriever(build_incident_knowledge_base())

    def _rag_extra_context(scenario: IncidentScenario) -> str:
        query = f"{scenario.question} {scenario.root_cause_category}"
        doc, _score = retriever.retrieve(query, top_k=1)[0]
        return (
            f"Runbook [{doc.title} | category={doc.category} | table={doc.target_table}]: "
            f"{doc.content}"
            "\nFollow the exact 3-step runbook sequence above: (1) query_warehouse on "
            "mart_appointments, (2) query_warehouse on the target upstream table, "
            "(3) submit_diagnosis with the exact fields from the runbook."
        )

    def _premature_assistant_prefix(scenario: IncidentScenario) -> dict[str, str]:
        """`run_lazy_guess_episode` prefix: submit the diagnosis on turn 1 without SQL."""
        guess = {
            "name": "submit_diagnosis",
            "arguments": {
                "root_cause_category": scenario.root_cause_category,
                "root_cause_code": scenario.root_cause_code,
                "ruled_out_confounders": [],
                "evidence_numbers": [],
                "remediation": "Guessed without querying tables.",
            },
        }
        return {"role": "assistant", "content": f"<tool_call>{json.dumps(guess)}</tool_call>"}

    def _verified_assistant_prefix(scenario: IncidentScenario) -> dict[str, str]:
        """`run_oracle_episode` prefix: 3 verification queries right before `submit_diagnosis`."""
        calls = [
            {
                "name": "query_warehouse",
                "arguments": {
                    "sql": (
                        "SELECT week, AVG(is_completed) AS completion_rate, "
                        "COUNT(DISTINCT billing_tz_offset) AS tz_buckets "
                        "FROM mart_appointments GROUP BY week"
                    )
                },
            },
            {
                "name": "query_warehouse",
                "arguments": {
                    "sql": (
                        "SELECT d.clinic_tier, d.is_regional_holiday, m.booking_horizon_days, "
                        "AVG(m.is_completed) AS slice_completion_rate "
                        "FROM mart_appointments m JOIN dim_clinics d ON m.clinic_id = d.clinic_id "
                        "GROUP BY d.clinic_tier, d.is_regional_holiday, m.booking_horizon_days"
                    )
                },
            },
        ]
        if scenario.root_cause_category == "product_bug":
            calls.append(
                {
                    "name": "query_warehouse",
                    "arguments": {
                        "sql": (
                            "SELECT client_platform, app_version, "
                            "AVG(webhook_delivery_rate) AS delivery_rate "
                            "FROM stg_sms_reminders WHERE week = '2026-10-26' "
                            "GROUP BY client_platform, app_version"
                        )
                    },
                }
            )
        elif scenario.root_cause_category == "data_pipeline_artifact":
            calls.append(
                {
                    "name": "query_warehouse",
                    "arguments": {
                        "sql": (
                            "SELECT raw_status, mapped_is_completed, COUNT(*) AS row_cnt "
                            "FROM stg_ehr_encounters WHERE week = '2026-10-26' "
                            "GROUP BY raw_status, mapped_is_completed"
                        )
                    },
                }
            )
        else:  # mix_shift: surface the onboarding-cohort slice culprit right before submitting
            calls.append(
                {
                    "name": "query_warehouse",
                    "arguments": {
                        "sql": (
                            "SELECT d.clinic_tier, AVG(m.is_completed) AS slice_completion_rate "
                            "FROM mart_appointments m JOIN dim_clinics d "
                            "ON m.clinic_id = d.clinic_id "
                            "WHERE d.clinic_tier = 'onboarding_tier_c' GROUP BY d.clinic_tier"
                        )
                    },
                }
            )
        content = "\n".join(f"<tool_call>{json.dumps(call)}</tool_call>" for call in calls)
        return {"role": "assistant", "content": content}

    def _release_device_cache() -> None:
        gc.collect()
        if resolved_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif (
            resolved_device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()

    # Real rollout prompts (unpadded `[1, seq_len]` tensors) for CKA/KL calibration,
    # carrying the same RAG grounding the adapted policy is evaluated with.
    prompt_inputs = [
        _encode_chat_prompt(
            tokenizer,
            format_incident_prompt(eval_env, scenario, extra_context=_rag_extra_context(scenario)),
            resolved_device,
        )
        for scenario in eval_env.scenarios
    ]

    n_layers = len(get_transformer_layers(model))
    tokens_processed = 0

    # Base (pre-adaptation) residual deltas, collected before any weight update.
    base_deltas = _collect_prompt_deltas(model, prompt_inputs)

    print("[llm_pareto] stage 1/4: LoRA SFT on partial-RAG trajectories", flush=True)
    sft_trajectories = collect_partial_sft_trajectories(train_env, tokenizer)
    sft_info = train_lora_sft_llm(model, tokenizer, sft_trajectories, steps=sft_steps)
    tokens_processed += int(sft_info.get("tokens_processed", 0))
    sft_lora_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if "lora_" in key
    }

    print("[llm_pareto] stage 2/4: oracle-SFT warmup + token-level GRPO", flush=True)
    oracle_trajectories = collect_oracle_sft_trajectories(train_env, tokenizer)
    grpo_warmup_info = train_lora_sft_llm(
        model, tokenizer, oracle_trajectories, lr=5e-4, steps=max(4, sft_steps)
    )
    tokens_processed += int(grpo_warmup_info.get("tokens_processed", 0))

    grpo_info = train_grpo_llm_policy(
        train_env,
        model,
        tokenizer,
        rubric_mode="gated_multiplicative",
        config=LLMGRPOConfig(
            group_size=group_size,
            steps=grpo_steps,
            max_new_tokens=max_new_tokens,
            lr=1e-5,
        ),
        seed=seed,
        extra_context_fn=_rag_extra_context,
    )
    tokens_processed += int(grpo_info.get("tokens_processed", 0))

    merge_lora_weights(model)
    grpo_model = model.eval()
    grpo_deltas = _collect_prompt_deltas(grpo_model, prompt_inputs)

    # Reconstruct the post-SFT (pre-GRPO) policy cleanly from the frozen LoRA snapshot.
    sft_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    sft_model.to(resolved_device)
    apply_lora_to_transformer(
        sft_model,
        rank=16,
        alpha=32.0,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    sft_model.load_state_dict(sft_lora_state, strict=False)
    merge_lora_weights(sft_model)
    sft_deltas = _collect_prompt_deltas(sft_model.eval(), prompt_inputs)

    # H7 calibration: quantization-induced (RCO) drift of the GRPO policy, per layer.
    compression_rco = compute_blockwise_quantization_drift(
        grpo_model, prompt_inputs, quant_bits=4, group_size=32
    )

    layer_drifts = {
        "base_to_sft": _drift_curve(base_deltas, sft_deltas),
        "sft_to_grpo": _drift_curve(sft_deltas, grpo_deltas),
        "base_to_grpo": _drift_curve(base_deltas, grpo_deltas),
        "compression_rco": compression_rco,
    }

    del sft_model, sft_deltas, base_deltas
    _release_device_cache()

    print("[llm_pareto] stage 3/4: H7 CKA-guided mixed-precision RCO + Pareto", flush=True)
    grpo_layer_drifts = layer_drifts["compression_rco"]
    top_k_high = max(1, int(round(0.25 * n_layers)))
    bits_cka = allocate_mixed_precision_bits(
        grpo_layer_drifts, high_bits=8, low_bits=4, top_k_high=top_k_high, strategy="cka_guided"
    )
    bits_bottom_k = allocate_mixed_precision_bits(
        grpo_layer_drifts,
        high_bits=8,
        low_bits=4,
        top_k_high=top_k_high,
        strategy="bottom_k_control",
    )
    bits_fp8 = allocate_mixed_precision_bits(
        grpo_layer_drifts, high_bits=8, low_bits=4, top_k_high=top_k_high, strategy="uniform_high"
    )
    bits_int4 = allocate_mixed_precision_bits(
        grpo_layer_drifts, high_bits=8, low_bits=4, top_k_high=top_k_high, strategy="uniform_low"
    )

    grpo_params_mb = sum(p.numel() for p in grpo_model.parameters()) * 2 / (1024 * 1024)

    def _kl_vs_grpo(candidate_model: Any, inputs: list[torch.Tensor]) -> float:
        """Mean forward KL `KL(P_grpo || P_candidate)` averaged over `inputs`."""
        kl_values = [
            compute_logit_kl_divergence(grpo_model, candidate_model, ids) for ids in inputs
        ]
        return round(sum(kl_values) / len(kl_values), 6)

    def _measure_ttft_and_throughput(candidate_model: Any) -> tuple[float, float]:
        """First-token prefill latency (ms) and greedy decode throughput (tok/s)."""
        candidate_model.eval()
        prompt = prompt_inputs[0]

        def _sync() -> None:
            if resolved_device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif (
                resolved_device == "mps"
                and hasattr(torch, "mps")
                and hasattr(torch.mps, "synchronize")
            ):
                torch.mps.synchronize()

        with torch.no_grad():
            _ = candidate_model(prompt)  # warmup
            _sync()
            t0 = time.perf_counter()
            _ = candidate_model(prompt)
            _sync()
            ttft_ms = (time.perf_counter() - t0) * 1000.0

            _sync()
            t0 = time.perf_counter()
            generated = candidate_model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            _sync()
            elapsed = max(1e-6, time.perf_counter() - t0)
        new_tokens = max(1, int(generated.shape[1] - prompt.shape[1]))
        return round(ttft_ms, 2), round(new_tokens / elapsed, 1)

    def _evaluate_gated_rubric(policy_model: Any) -> tuple[dict[str, float], int]:
        """Greedy gated-rubric evaluation of `policy_model` on all `eval_env` scenarios."""
        sums = {
            "g_safety": 0.0,
            "g_complete": 0.0,
            "g_evidence": 0.0,
            "r_correctness": 0.0,
            "total_reward": 0.0,
        }
        used = 0
        for scenario in eval_env.scenarios:
            state, gen_tokens = _llm_episode_state(
                eval_env,
                scenario,
                policy_model,
                tokenizer,
                max_new_tokens,
                do_sample=False,
                extra_context=_rag_extra_context(scenario),
            )
            breakdown = eval_env.score_state(state, mode_override="gated_multiplicative")
            sums["g_safety"] += breakdown.g_safety
            sums["g_complete"] += breakdown.g_complete
            sums["g_evidence"] += breakdown.g_evidence
            sums["r_correctness"] += breakdown.r_correctness
            sums["total_reward"] += breakdown.total_reward
            used += gen_tokens
        count = max(1, len(eval_env.scenarios))
        aggregates = {
            "g_safety": round(sums["g_safety"] / count, 4),
            "g_complete": round(sums["g_complete"] / count, 4),
            "g_evidence": round(sums["g_evidence"] / count, 4),
            "r_correctness": round(sums["r_correctness"] / count, 4),
            "mean_gated_reward": round(sums["total_reward"] / count, 4),
        }
        return aggregates, used

    def _evaluate_variant(
        name: str,
        candidate_model: Any,
        avg_bits: float,
        compression_ratio: float,
        kl_value: float,
        memory_mb: float,
        owned: bool,
    ) -> tuple[dict[str, Any], int]:
        """Measure KL/TTFT/throughput/memory + gated rubric for one compression variant."""
        ttft_ms, gen_tok_s = _measure_ttft_and_throughput(candidate_model)
        rubric, used = _evaluate_gated_rubric(candidate_model)
        entry: dict[str, Any] = {
            "variant": name,
            "avg_bits": round(float(avg_bits), 2),
            "compression_ratio": round(float(compression_ratio), 2),
            "kl_vs_bf16": kl_value,
            "measured_llm_ttft_ms": ttft_ms,
            "measured_llm_tok_s": gen_tok_s,
            "estimated_model_mb": round(float(memory_mb), 2),
            **rubric,
        }
        if owned:
            del candidate_model
            _release_device_cache()
        return entry, used

    variants: list[dict[str, Any]] = []
    rubric_degradation: dict[str, Any] = {}

    # 1_bf16_adapted: the GRPO policy itself (reference; never freed).
    entry, used = _evaluate_variant(
        "1_bf16_adapted", grpo_model, 16.0, 1.0, 0.0, grpo_params_mb, owned=False
    )
    tokens_processed += used
    variants.append(entry)
    rubric_degradation["1_bf16_adapted"] = {key: entry[key] for key in _RUBRIC_KEYS}

    def _quantize(bits_map: dict[int, int]) -> tuple[Any, dict[str, float]]:
        return apply_mixed_precision_quantization(grpo_model, bits_map, group_size=32)

    # 2_fp8_uniform
    fp8_model, stats_fp8 = _quantize(bits_fp8)
    kl_fp8 = _kl_vs_grpo(fp8_model, prompt_inputs)
    entry, used = _evaluate_variant(
        "2_fp8_uniform",
        fp8_model,
        stats_fp8["average_bits_per_weight"],
        stats_fp8["compression_ratio_vs_bf16"],
        kl_fp8,
        stats_fp8["estimated_model_mb"],
        owned=True,
    )
    tokens_processed += used
    variants.append(entry)
    rubric_degradation["2_fp8_uniform"] = {key: entry[key] for key in _RUBRIC_KEYS}

    # 3_cka_guided_mixed_4_8
    cka_model, stats_cka = _quantize(bits_cka)
    kl_cka = _kl_vs_grpo(cka_model, prompt_inputs)
    entry, used = _evaluate_variant(
        "3_cka_guided_mixed_4_8",
        cka_model,
        stats_cka["average_bits_per_weight"],
        stats_cka["compression_ratio_vs_bf16"],
        kl_cka,
        stats_cka["estimated_model_mb"],
        owned=True,
    )
    tokens_processed += used
    variants.append(entry)
    rubric_degradation["3_cka_guided_mixed_4_8"] = {key: entry[key] for key in _RUBRIC_KEYS}

    # 4_bottom_k_control_4_8
    bottom_k_model, stats_bottom_k = _quantize(bits_bottom_k)
    kl_bottom_k = _kl_vs_grpo(bottom_k_model, prompt_inputs)
    entry, used = _evaluate_variant(
        "4_bottom_k_control_4_8",
        bottom_k_model,
        stats_bottom_k["average_bits_per_weight"],
        stats_bottom_k["compression_ratio_vs_bf16"],
        kl_bottom_k,
        stats_bottom_k["estimated_model_mb"],
        owned=True,
    )
    tokens_processed += used
    variants.append(entry)
    rubric_degradation["4_bottom_k_control_4_8"] = {key: entry[key] for key in _RUBRIC_KEYS}

    # 5_uniform_int4 (kept alive: variant 6 prunes it) & 6_pruned_int4
    int4_model, stats_int4 = _quantize(bits_int4)
    kl_int4 = _kl_vs_grpo(int4_model, prompt_inputs)
    entry, used = _evaluate_variant(
        "5_uniform_int4",
        int4_model,
        stats_int4["average_bits_per_weight"],
        stats_int4["compression_ratio_vs_bf16"],
        kl_int4,
        stats_int4["estimated_model_mb"],
        owned=False,
    )
    tokens_processed += used
    variants.append(entry)
    rubric_degradation["5_uniform_int4"] = {key: entry[key] for key in _RUBRIC_KEYS}

    n_drop = max(1, int(round(0.10 * n_layers)))
    drop_layer_indices = [
        d["layer_idx"] for d in sorted(grpo_layer_drifts, key=lambda x: x["cka_drift"])[:n_drop]
    ]
    pruned_model = prune_transformer_layers(int4_model, drop_layer_indices=drop_layer_indices)
    kl_pruned = _kl_vs_grpo(pruned_model, prompt_inputs)
    pruned_avg_bits = stats_int4["average_bits_per_weight"] * (n_layers - n_drop) / n_layers
    entry, used = _evaluate_variant(
        "6_pruned_int4",
        pruned_model,
        pruned_avg_bits,
        16.0 / max(pruned_avg_bits, 1e-9),
        kl_pruned,
        stats_int4["estimated_model_mb"] * (n_layers - n_drop) / n_layers,
        owned=True,
    )
    tokens_processed += used
    variants.append(entry)
    rubric_degradation["6_pruned_int4"] = {key: entry[key] for key in _RUBRIC_KEYS}
    del int4_model
    _release_device_cache()

    kl_divergences = {
        "cka_guided": kl_cka,
        "bottom_k_control": kl_bottom_k,
        "uniform_low": kl_int4,
        "uniform_high": kl_fp8,
        "pruned_int4": kl_pruned,
    }

    h7_kl_reduction_ratio = round((kl_bottom_k - kl_cka) / max(1e-12, kl_bottom_k), 4)
    h7_verified = bool(kl_cka < kl_bottom_k and kl_cka < kl_int4)

    # Pareto optimality on (avg_bits, kl_vs_bf16): no other variant dominates on both axes
    # with at least one strict inequality.
    pareto_frontier: list[dict[str, Any]] = []
    for cand in variants:
        dominated = False
        for other in variants:
            if other["variant"] == cand["variant"]:
                continue
            if (
                other["avg_bits"] <= cand["avg_bits"]
                and other["kl_vs_bf16"] <= cand["kl_vs_bf16"]
                and (
                    other["avg_bits"] < cand["avg_bits"] or other["kl_vs_bf16"] < cand["kl_vs_bf16"]
                )
            ):
                dominated = True
                break
        frontier_entry = dict(cand)
        frontier_entry["is_pareto_optimal"] = not dominated
        pareto_frontier.append(frontier_entry)

    by_variant = {item["variant"]: item for item in pareto_frontier}
    pareto_dominance_verified = bool(
        by_variant["3_cka_guided_mixed_4_8"]["avg_bits"]
        == by_variant["4_bottom_k_control_4_8"]["avg_bits"]
        and kl_cka < kl_bottom_k
    )

    print("[llm_pareto] stage 4/4: H8 rank-1 subspace ablation", flush=True)
    premature_inputs = [
        _encode_chat_prompt(
            tokenizer,
            format_incident_prompt(eval_env, scenario) + [_premature_assistant_prefix(scenario)],
            resolved_device,
        )
        for scenario in eval_env.scenarios
    ]
    verified_inputs = [
        _encode_chat_prompt(
            tokenizer,
            format_incident_prompt(eval_env, scenario) + [_verified_assistant_prefix(scenario)],
            resolved_device,
        )
        for scenario in eval_env.scenarios
    ]

    def _final_layer_last_token_acts(policy_model: Any, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Last-token residual activations of the final layer, stacked `[n, hidden]`."""
        layers = get_transformer_layers(policy_model)
        acts: list[torch.Tensor] = []
        hook = layers[-1].register_forward_hook(
            lambda _m, _inp, out: acts.append(
                (out[0] if isinstance(out, tuple) else out)[:, -1, :].detach().float().cpu()
            )
        )
        try:
            with torch.no_grad():
                for ids in inputs:
                    policy_model(ids.to(next(policy_model.parameters()).device))
        finally:
            hook.remove()
        return torch.cat(acts, dim=0)

    premature_acts = _final_layer_last_token_acts(grpo_model, premature_inputs)
    verified_acts = _final_layer_last_token_acts(grpo_model, verified_inputs)
    direction = _last_token_direction(premature_acts, verified_acts).to(resolved_device)

    # Ablate the "premature guessing" write direction from middle & late layers.
    ablation_layer_indices = list(range(n_layers // 2, n_layers))
    ablated_model = apply_subspace_ablation(
        grpo_model, direction, layer_indices=ablation_layer_indices, scale=1.0
    )

    ablated_layers = get_transformer_layers(ablated_model)
    max_ortho_residual = 0.0
    with torch.no_grad():
        for layer_idx in ablation_layer_indices:
            layer = ablated_layers[layer_idx]
            for proj in (layer.self_attn.o_proj, layer.mlp.down_proj):
                weight = proj.weight
                residual = direction.to(weight.device, dtype=torch.float32) @ weight.float()
                max_ortho_residual = max(max_ortho_residual, float(residual.abs().max().item()))

    contrastive_inputs = premature_inputs + verified_inputs
    contrastive_lengths = [ids.shape[1] for ids in contrastive_inputs]

    def _final_layer_write_proj(policy_model: Any) -> float:
        """Mean |r^T (h_out - h_in)| of the final layer's residual write (last token/prompt)."""
        deltas = _collect_prompt_deltas(policy_model, contrastive_inputs)[-1]
        rows = []
        offset = 0
        for length in contrastive_lengths:
            rows.append(deltas[offset + length - 1])
            offset += length
        write = torch.stack(rows, dim=0)
        r = direction.to(write.device, dtype=torch.float32)
        return float((write @ r).abs().mean().item())

    write_proj_before = _final_layer_write_proj(grpo_model)
    write_proj_after = _final_layer_write_proj(ablated_model)
    kl_ablation_collateral = _kl_vs_grpo(ablated_model, contrastive_inputs)
    rubric_after_ablation, used_after = _evaluate_gated_rubric(ablated_model)
    tokens_processed += used_after

    h8_verified = bool(write_proj_after < write_proj_before * 0.25 and max_ortho_residual < 1e-2)

    h8_diagnostics = {
        "max_ortho_residual": round(max_ortho_residual, 8),
        "write_proj_before": round(write_proj_before, 6),
        "write_proj_after": round(write_proj_after, 8),
        "kl_ablation_collateral": kl_ablation_collateral,
        "ablated_layer_indices": ablation_layer_indices,
        "direction_norm": round(float(direction.norm().item()), 6),
        "rubric_before": rubric_degradation["1_bf16_adapted"],
        "rubric_after": rubric_after_ablation,
    }

    del ablated_model
    _release_device_cache()

    return {
        "model_name": model_name,
        "backend": "llm",
        "device": resolved_device,
        "dtype": str(dtype),
        "n_layers": n_layers,
        "tokens_processed": tokens_processed,
        "h7_verified": h7_verified,
        "h7_kl_reduction_ratio": h7_kl_reduction_ratio,
        "h7_kl_divergences": kl_divergences,
        "h8_verified": h8_verified,
        "h8_diagnostics": h8_diagnostics,
        "pareto_dominance_verified": pareto_dominance_verified,
        "layer_drifts": layer_drifts,
        "bit_allocations": {
            "cka_guided": bits_cka,
            "bottom_k_control": bits_bottom_k,
            "uniform_high": bits_fp8,
            "uniform_low": bits_int4,
            "pruned_drop_layer_indices": drop_layer_indices,
            "top_k_high": top_k_high,
        },
        "quantization_stats": {
            "cka_guided": stats_cka,
            "bottom_k_control": stats_bottom_k,
            "uniform_high": stats_fp8,
            "uniform_low": stats_int4,
        },
        "kl_divergences": kl_divergences,
        "rubric_degradation": rubric_degradation,
        "pareto_frontier": pareto_frontier,
    }


def run_llm_h9_biprojected_surgery_benchmark(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "auto",
    seed: int = 2026,
    top_k_layers: int | None = None,
    scale: float = 1.0,
    output_dir: Path | str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Execute `H9` (Lancet bi-projected surgery) on a real LLM.

    Mirrors `sml.surgery.pareto.run_h9_biprojected_surgery_benchmark` (sml-nano backend):
    suppress refusal & slop write directions with anchor-bi-projection, row-norm preservation
    and SCR top-k layer selection, then compare anchor KL collateral against naive surgery.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Deferred: `sml.surgery.pareto` re-exports this module, so module-scope imports of it
    # here would be circular.
    from sml.surgery.pareto import (
        _biprojected_basis,
        _compute_probes_sha256,
        _h9_surgery_metrics,
        _select_top_scr_layers,
        _snapshot_write_weights,
        compute_logit_kl_divergence,
    )

    torch.manual_seed(seed)
    resolved_device, dtype = _resolve_llm_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(resolved_device)
    model.eval()

    layers = get_transformer_layers(model)
    n_layers = len(layers)
    if top_k_layers is None:
        top_k_layers = max(1, n_layers // 2)

    refusal_pos_ids, refusal_neg_ids = encode_contrast_pairs(
        REFUSAL_CONTRAST_PAIRS, tokenizer=tokenizer
    )
    slop_pos_ids, slop_neg_ids = encode_contrast_pairs(SLOP_CONTRAST_PAIRS, tokenizer=tokenizer)
    anchor_pos_ids, anchor_neg_ids = encode_contrast_pairs(
        REASONING_ANCHOR_PAIRS, tokenizer=tokenizer
    )
    tool_sql_pos_ids, tool_sql_neg_ids = encode_contrast_pairs(
        TOOL_SQL_ANCHOR_PAIRS, tokenizer=tokenizer
    )
    encoded_inputs = [
        refusal_pos_ids,
        refusal_neg_ids,
        slop_pos_ids,
        slop_neg_ids,
        anchor_pos_ids,
        anchor_neg_ids,
        tool_sql_pos_ids,
        tool_sql_neg_ids,
    ]
    encoded_inputs = [ids.to(resolved_device) for ids in encoded_inputs]
    (
        refusal_pos_ids,
        refusal_neg_ids,
        slop_pos_ids,
        slop_neg_ids,
        anchor_pos_ids,
        anchor_neg_ids,
        tool_sql_pos_ids,
        tool_sql_neg_ids,
    ) = encoded_inputs

    def _contrastive_directions(
        pos_ids: torch.Tensor, neg_ids: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[float]]:
        """Per-layer unit mean-difference directions plus per-layer probe EVR.

        `collect_layer_activations` returns flattened `(B*T, D)` activations, so the
        per-example mean over sequence tokens restores `(B, T, D)` first and reduces
        over `dim=1`, giving `delta_h` of shape `(B, D)` per layer for the spectral
        probe stats. The returned direction is the unit-normalized mean of `delta_h`.
        """
        pos_acts = collect_layer_activations(model, pos_ids)
        neg_acts = collect_layer_activations(model, neg_ids)
        batch_size = int(pos_ids.shape[0])
        seq_len = int(pos_ids.shape[1]) if pos_ids.dim() > 1 else 1
        directions: list[torch.Tensor] = []
        evrs: list[float] = []
        for pos_layer_acts, neg_layer_acts in zip(pos_acts, neg_acts, strict=False):
            delta_h = pos_layer_acts.float().reshape(batch_size, seq_len, -1).mean(
                dim=1
            ) - neg_layer_acts.float().reshape(batch_size, seq_len, -1).mean(dim=1)
            stats = compute_probe_spectral_stats(delta_h)
            evrs.append(float(stats["explained_variance_ratio_1"]))
            diff = delta_h.mean(dim=0)
            directions.append(diff / (diff.norm(p=2) + 1e-8))
        return directions, evrs

    r_refusal_dirs, refusal_layer_evrs = _contrastive_directions(refusal_pos_ids, refusal_neg_ids)
    r_slop_dirs, slop_layer_evrs = _contrastive_directions(slop_pos_ids, slop_neg_ids)
    a_reasoning_dirs, _ = _contrastive_directions(anchor_pos_ids, anchor_neg_ids)
    a_tool_sql_dirs, _ = _contrastive_directions(tool_sql_pos_ids, tool_sql_neg_ids)
    probe_evr_refusal = sum(refusal_layer_evrs) / max(1, len(refusal_layer_evrs))
    probe_evr_slop = sum(slop_layer_evrs) / max(1, len(slop_layer_evrs))

    # Per-layer contrastive directions, bi-projected to be orthogonal to the 2D
    # anchor subspace spanned by the reasoning & tool-SQL anchor directions.
    r_refusal_perp: list[torch.Tensor] = []
    r_slop_perp: list[torch.Tensor] = []
    anchor_orthogonality_error = 0.0
    tool_sql_orthogonality_error = 0.0
    per_layer_ortho: dict[str, float] = {}
    per_layer_tool_sql_ortho: dict[str, float] = {}
    for layer_idx in range(n_layers):
        a_reasoning = a_reasoning_dirs[layer_idx].to(resolved_device)
        a_tool_sql = a_tool_sql_dirs[layer_idx].to(resolved_device)
        anchor_pair = [a_reasoning, a_tool_sql]
        ref_perp = biproject_direction(r_refusal_dirs[layer_idx].to(resolved_device), anchor_pair)
        slop_perp = biproject_direction(r_slop_dirs[layer_idx].to(resolved_device), anchor_pair)
        layer_error = max(
            float((ref_perp @ a_reasoning).abs().item()),
            float((slop_perp @ a_reasoning).abs().item()),
        )
        layer_tool_sql_error = max(
            float((ref_perp @ a_tool_sql).abs().item()),
            float((slop_perp @ a_tool_sql).abs().item()),
        )
        per_layer_ortho[str(layer_idx)] = round(layer_error, 8)
        per_layer_tool_sql_ortho[str(layer_idx)] = round(layer_tool_sql_error, 8)
        anchor_orthogonality_error = max(anchor_orthogonality_error, layer_error)
        tool_sql_orthogonality_error = max(tool_sql_orthogonality_error, layer_tool_sql_error)
        r_refusal_perp.append(ref_perp)
        r_slop_perp.append(slop_perp)

    # SCR-guided layer selection on the bi-projected targets.
    target_dirs_per_layer = {
        idx: [r_refusal_perp[idx], r_slop_perp[idx]] for idx in range(n_layers)
    }
    scr_profile = compute_layer_scr_profile(
        model,
        target_dirs_per_layer=target_dirs_per_layer,
        anchor_input_ids=anchor_pos_ids,
        scale=scale,
        preserve_row_norm=True,
    )
    selected_layers = _select_top_scr_layers(scr_profile, top_k_layers)

    # Soft SCR attenuation: full strength (alpha=1.0) on the max-SCR layer, softer
    # alphas (>= 0.1) proportional to each layer's SCR on the remaining selections.
    scr_by_layer = {int(entry["layer_idx"]): float(entry["scr"]) for entry in scr_profile}
    max_scr = max((scr_by_layer[layer_idx] for layer_idx in selected_layers), default=1.0)
    layer_alpha_schedule = {
        int(layer_idx): round(
            float(min(1.0, max(0.1, scr_by_layer[layer_idx] / max(max_scr, 1e-12)))), 6
        )
        for layer_idx in selected_layers
    }

    basis_per_layer = {
        idx: _biprojected_basis(r_refusal_perp[idx], r_slop_perp[idx]) for idx in selected_layers
    }
    original_weights = _snapshot_write_weights(layers, selected_layers)

    anchors_per_layer = {
        idx: [
            a_reasoning_dirs[idx].to(resolved_device),
            a_tool_sql_dirs[idx].to(resolved_device),
        ]
        for idx in range(n_layers)
    }

    # Serializable Lancet recipe: the shared builder re-derives the bi-projected
    # targets, SCR selection, soft weights and orthonormal bases for this model.
    recipe = build_lancet_recipe(
        model,
        target_dirs=target_dirs_per_layer,
        anchor_dirs=anchors_per_layer,
        anchor_input_ids=anchor_pos_ids,
        model_id=model_id or model_name,
        base_model=model_name,
        top_k_layers=top_k_layers,
        scale=scale,
        probes_sha256=_compute_probes_sha256(),
    )

    # Naive control: every layer, no anchor bi-projection, no row-norm preservation.
    naive_model = apply_lancet_biprojected_surgery(
        model,
        target_dirs=target_dirs_per_layer,
        anchor_dirs=None,
        layer_indices=None,
        scale=scale,
        preserve_row_norm=False,
    )
    # Lancet: anchor-bi-projected targets, SCR top-k layers, norm-preserving projection,
    # with soft SCR attenuation weights per layer.
    lancet_model = apply_lancet_biprojected_surgery(
        model,
        target_dirs=target_dirs_per_layer,
        anchor_dirs=anchors_per_layer,
        layer_indices=selected_layers,
        scale=scale,
        preserve_row_norm=True,
        layer_weights=layer_alpha_schedule,
    )

    max_row_norm_drift, target_suppression_ratio, unweighted_suppression_ratio = (
        _h9_surgery_metrics(
            get_transformer_layers(lancet_model),
            original_weights,
            selected_layers,
            basis_per_layer,
            layer_weights=layer_alpha_schedule,
        )
    )
    naive_row_norm_drift, naive_suppression_ratio, _ = _h9_surgery_metrics(
        get_transformer_layers(naive_model), original_weights, selected_layers, basis_per_layer
    )

    naive_kl_collateral = compute_logit_kl_divergence(model, naive_model, anchor_pos_ids)
    lancet_kl_collateral = compute_logit_kl_divergence(model, lancet_model, anchor_pos_ids)
    h9_kl_reduction_ratio = 1.0 - (lancet_kl_collateral / max(1e-8, naive_kl_collateral))

    norm_drift_tol = 2e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    h9_verified = bool(
        anchor_orthogonality_error < 1e-6
        and tool_sql_orthogonality_error < 1e-6
        and max_row_norm_drift < norm_drift_tol
        and target_suppression_ratio >= 0.90
        and h9_kl_reduction_ratio >= 0.35
    )

    if output_dir is not None:
        save_lancet_recipe(recipe, output_dir)

    del naive_model, lancet_model
    gc.collect()
    if resolved_device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif resolved_device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return {
        "h9_verified": h9_verified,
        "backend": "llm",
        "model_name": model_name,
        "probes_sha256": recipe.probes_sha256,
        "recipe_metadata": {
            "schema_version": recipe.schema_version,
            "probes_sha256": recipe.probes_sha256,
            "selected_layers": recipe.selected_layers,
            "d_model": recipe.d_model,
            "num_layers": recipe.num_layers,
        },
        "device": resolved_device,
        "dtype": str(dtype),
        "n_layers": n_layers,
        "anchor_orthogonality_error": round(anchor_orthogonality_error, 8),
        "tool_sql_orthogonality_error": round(tool_sql_orthogonality_error, 8),
        "probe_evr_refusal": round(probe_evr_refusal, 6),
        "probe_evr_slop": round(probe_evr_slop, 6),
        "layer_alpha_schedule": layer_alpha_schedule,
        "max_row_norm_drift": round(max_row_norm_drift, 8),
        "target_suppression_ratio": round(target_suppression_ratio, 6),
        "naive_kl_collateral": round(naive_kl_collateral, 6),
        "lancet_kl_collateral": round(lancet_kl_collateral, 6),
        "h9_kl_reduction_ratio": round(h9_kl_reduction_ratio, 4),
        "selected_layers": selected_layers,
        "scr_profile": scr_profile,
        "diagnostics": {
            "top_k_layers": top_k_layers,
            "scale": scale,
            "per_layer_anchor_orthogonality": per_layer_ortho,
            "per_layer_tool_sql_orthogonality": per_layer_tool_sql_ortho,
            "per_layer_evr_refusal": {
                str(i): round(e, 6) for i, e in enumerate(refusal_layer_evrs)
            },
            "per_layer_evr_slop": {str(i): round(e, 6) for i, e in enumerate(slop_layer_evrs)},
            "unweighted_suppression_ratio": round(unweighted_suppression_ratio, 6),
            "naive_row_norm_drift": round(naive_row_norm_drift, 8),
            "naive_suppression_ratio": round(naive_suppression_ratio, 6),
        },
    }
