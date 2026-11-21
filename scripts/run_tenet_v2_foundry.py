#!/usr/bin/env python
"""Tenet v2 local foundry pipeline for Apple M4 Pro (MPS/Metal).

Four logged stages:

1. H15 Tri-Anchor Lancet surgery (bfloat16 on mps, cpu fallback).
2. H16 Subspace-Safe GRPO + norm-preserving merge over TENET_V2_SCENARIOS.
3. H14+ domain GGUF build (imatrix + CKA mixed quant) and GGUF audit.
4. Release gate + release_manifest.json / foundry_summary.json.

Example:
    .venv/bin/python scripts/run_tenet_v2_foundry.py \
        --model-name Qwen/Qwen3-4B --release-name Tenet-4B-Qwen3-Unslop-v2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from sml.adaptation.grpo import (
    TENET_V2_SCENARIOS,
    LoRALinear,
    TenetRehabPolicy,
    TenetUnslopRubric,
    _v2_candidate_responses,
    compute_group_normalized_advantages,
    compute_grpo_loss,
    compute_token_logprobs,
    merge_lora_weights_in_place,
)
from sml.hub.release import _write_gguf_recipe_bundle_files, build_cka_quant_recipe
from sml.surgery.cka import compute_blockwise_quantization_drift, get_transformer_layers
from sml.surgery.probes import (
    COT_REASONING_ANCHOR_PAIRS,
    REASONING_ANCHOR_PAIRS,
    PhaseSegmenter,
    encode_contrast_pairs,
)
from sml.surgery.quantization import quantize_fp8_e4m3
from sml.surgery.subspace import (
    apply_lancet_recipe,
    biproject_direction,
    build_lancet_recipe,
    compute_layer_scr_profile,
    extract_contrastive_direction_with_stats,
    extract_tri_anchor_directions,
    save_lancet_recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
GROUP_SIZE = TenetRehabPolicy.ARCHETYPE_COUNT  # 6 deterministic v2 archetypes
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_LAYERS = 4
GRPO_LR = 5e-4
GRPO_BETA_KL = 0.04
GRPO_CLIP_EPS = 0.2
GRPO_MAX_LENGTH = 512
PROBE_SEQ_LEN = 48
ANSWER_SEQ_LEN = 256
LAMBDA_COT = 0.5
CREST_LIMIT = 25.0
FP8_RELEAK_LIMIT_PCT = 3.0
V2_SLOP_REFUSAL_LIMIT = 0.10
V2_INTEGRITY_FLOOR = 0.90

STAGE_BAR = "=" * 78


def log_stage(stage: int, title: str) -> None:
    print(f"\n{STAGE_BAR}\nSTAGE {stage}: {title}\n{STAGE_BAR}", flush=True)


def log_step(msg: str) -> None:
    print(f"  [foundry] {msg}", flush=True)


def resolve_device() -> tuple[torch.device, torch.dtype]:
    """bfloat16 on cuda (H100) / mps (Apple M4 Pro / Metal), cpu fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    else:
        return torch.device("cpu"), torch.float32


def mps_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def enforce_mac_zero_local_weights_guard(model_name: str, allow_local_mac_weights: bool) -> None:
    """Zero-Local-Weights Policy (ADR-003/ADR-006) gate for macOS.

    Refuses to download/build full HF checkpoints on a Mac unless the caller
    explicitly overrides (``--allow-local-mac-weights`` or the
    ``TENET_ALLOW_LOCAL_MAC_WEIGHTS`` env var), the model is a nano stub, or
    ``model_name`` already resolves to a local directory.
    """
    if (
        sys.platform == "darwin"
        and not allow_local_mac_weights
        and not os.environ.get("TENET_ALLOW_LOCAL_MAC_WEIGHTS")
        and "nano" not in model_name.lower()
        and not Path(model_name).is_dir()
    ):
        raise RuntimeError(
            f"Zero-Local-Weights Policy (ADR-003/ADR-006): refusing to download/build "
            f"full checkpoint {model_name!r} on macOS. Run on remote H100 via "
            "scripts/remote/run_tenet_foundry.sh (or pass --allow-local-mac-weights "
            "to override)."
        )


def _read_hf_token() -> str | None:
    """Resolve the HF token from the environment, falling back to ``.env``."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "HF_TOKEN":
            return value.strip().strip("\"'")
    return None


def upload_and_drop_local_weights(
    release_dir: Path,
    release_name: str,
    hf_namespace: str,
    drop_local_weights: bool,
) -> dict[str, Any]:
    """Upload the release bundle to the HF Hub, then optionally drop local weights.

    Keeps ``lancet_basis.safetensors`` and JSON/recipe/README metadata on disk;
    deletes the heavy ``model*.safetensors`` / ``*.gguf`` / tokenizer files only
    when ``drop_local_weights`` is requested (after the upload has succeeded).
    """
    from huggingface_hub import HfApi  # local import: only needed for upload

    api = HfApi(token=_read_hf_token())
    repo_id = f"{hf_namespace}/{release_name}"
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(release_dir),
        ignore_patterns=["*-f16.gguf", "*.bin", "__pycache__/*"],
        commit_message=f"Release {release_name} (Tenet v2 Foundry)",
    )

    if drop_local_weights:
        dropped = 0
        for pattern in (
            "model*.safetensors",
            "*.gguf",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
        ):
            for path in sorted(release_dir.glob(pattern)):
                path.unlink()
                dropped += 1
        log_step(f"dropped {dropped} local weight file(s) from {release_dir}")

    return {
        "repo_id": repo_id,
        "commit_url": str(commit_info.commit_url),
        "dropped_local_weights": bool(drop_local_weights),
    }


def row_crest_factor(weight: torch.Tensor) -> float:
    """max row L2 norm / mean row L2 norm (1.0 for non-2D or empty tensors)."""
    if weight.ndim != 2 or weight.numel() == 0:
        return 1.0
    norms = weight.detach().float().norm(p=2, dim=-1)
    mean_norm = norms.mean().clamp_min(1e-8)
    return float((norms.max() / mean_norm).item())


def surgical_layer_metrics(
    model: torch.nn.Module,
    recipe: Any,
    pre_surgery: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    """Row crest factor + FP8 (E4M3) subspace re-leakage on surgical layers.

    Re-leakage: quantize each surgical ``o_proj`` / ``down_proj`` weight with
    ``quantize_fp8_e4m3(per_channel=True)``, project the quantization delta
    onto the stored Lancet basis (row space), and report the basis-projected
    delta mass as a percent of the mass the surgery removed (basis-projected
    pre-surgery weight mass). The gate limit is 3.0%.
    """
    layers = get_transformer_layers(model)
    max_crest = 0.0
    crest_by_layer: dict[str, float] = {}
    removed_total = 0.0
    releak_total = 0.0
    with torch.no_grad():
        for idx in recipe.selected_layers:
            layer = layers[idx]
            basis = recipe.basis_per_layer[idx].to(
                device=next(model.parameters()).device, dtype=torch.float32
            )
            for name, module in (
                ("self_attn.o_proj", layer.self_attn.o_proj),
                ("mlp.down_proj", layer.mlp.down_proj),
            ):
                weight = module.weight
                crest = row_crest_factor(weight)
                crest_by_layer[f"{idx}.{name}"] = round(crest, 4)
                max_crest = max(max_crest, crest)

                w = weight.detach().float()
                quantized = quantize_fp8_e4m3(w, per_channel=True)
                if not isinstance(quantized, torch.Tensor):
                    continue
                delta = quantized.to(torch.float32) - w
                # Column-space (output hidden dim) projection onto orthonormal Lancet basis.
                releak = float((basis @ delta).norm())
                pre_w = pre_surgery[idx][name].to(device=w.device, dtype=torch.float32)
                removed = float((basis @ pre_w).norm())
                releak_total += releak
                removed_total += removed
    releak_pct = 100.0 * releak_total / max(removed_total, 1e-12)
    return {
        "max_row_crest_factor": round(max_crest, 4),
        "crest_by_layer": crest_by_layer,
        "fp8_subspace_releakage_pct": round(releak_pct, 4),
        "fp8_releakage_mass": round(releak_total, 8),
        "fp8_removed_mass": round(removed_total, 8),
    }


def _import_from_candidates(attr: str, modules: tuple[str, ...]) -> Any:
    """Import ``attr`` from the first candidate module that defines it."""
    errors: list[str] = []
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=[attr])
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        if hasattr(module, attr):
            return getattr(module, attr)
        errors.append(f"{module_name}: no attribute {attr!r}")
    raise ImportError(f"Could not resolve {attr!r}; tried:\n  " + "\n  ".join(errors))


def resolve_run_h14_gguf_audit() -> Any:
    return _import_from_candidates(
        "run_h14_gguf_audit",
        (
            "sml.evaluation.gguf_audit",
            "sml.hub.gguf_audit",
            "sml.evaluation.h14_gguf_audit",
            "sml.evaluation",
            "sml.hub.audit",
        ),
    )


def resolve_build_tenet_domain_imatrix_corpus() -> Any:
    return _import_from_candidates(
        "build_tenet_domain_imatrix_corpus",
        (
            "sml.hub.imatrix",
            "sml.evaluation.imatrix",
            "sml.evaluation.gguf_audit",
            "sml.hub.gguf_audit",
            "sml.evaluation",
        ),
    )


def run_command(
    cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    """Run an external command, echoing it and failing loudly."""
    printable = " ".join(str(part) for part in cmd)
    log_step(f"$ {printable}")
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    subprocess.run(
        [str(part) for part in cmd],
        check=True,
        env=merged_env,
        cwd=str(cwd) if cwd else None,
    )


def resolve_tool(env_key: str, name: str) -> str:
    """Resolve a llama.cpp binary: env override, PATH, then .tools/llama.cpp."""
    override = os.environ.get(env_key)
    if override and Path(override).is_file():
        return override
    found = shutil.which(name)
    if found:
        return found
    bundled = REPO_ROOT / ".tools" / "llama.cpp" / name
    if bundled.is_file():
        return str(bundled)
    raise FileNotFoundError(
        f"{name} not found (set {env_key}, put it on PATH, or bundle it in .tools/llama.cpp)"
    )


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture_pre_surgery_weights(
    model: torch.nn.Module, selected_layers: list[int]
) -> dict[int, dict[str, torch.Tensor]]:
    """Snapshot surgical-layer weights (CPU float32) before the Lancet cut."""
    layers = get_transformer_layers(model)
    snapshot: dict[int, dict[str, torch.Tensor]] = {}
    for idx in selected_layers:
        layer = layers[idx]
        snapshot[idx] = {
            "self_attn.o_proj": layer.self_attn.o_proj.weight.detach().float().cpu().clone(),
            "mlp.down_proj": layer.mlp.down_proj.weight.detach().float().cpu().clone(),
        }
    return snapshot


def stage1_h15_tri_anchor_lancet(
    model: torch.nn.Module,
    tokenizer: Any,
    release_name: str,
    model_name: str,
    top_k_layers: int,
    release_dir: Path,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Stage 1: H15 Tri-Anchor Lancet surgery + FP8/crest audit."""
    log_stage(1, "H15 Tri-Anchor Lancet surgery")
    device = next(model.parameters()).device
    log_step(f"model device={device} dtype={next(model.parameters()).dtype}")

    seg = PhaseSegmenter.for_format("think")
    _start_ids = tokenizer.encode(seg.deliberation_start, add_special_tokens=False)
    _end_ids = tokenizer.encode(seg.deliberation_end, add_special_tokens=False)
    if len(_start_ids) == 1:
        seg.deliberation_start_id = int(_start_ids[0])
    if len(_end_ids) == 1:
        seg.deliberation_end_id = int(_end_ids[0])
    delib_tag_ids = [int(t) for t in (*_start_ids, *_end_ids)]
    log_step(f"PhaseSegmenter('think') deliberation token ids: {delib_tag_ids}")

    # --- Tri-Anchor directions (protected basis) ---------------------------
    log_step("extracting Tri-Anchor directions (CoT + math/logic + tool/SQL)")
    tri_basis = extract_tri_anchor_directions(model, phase_segmenter=seg, tokenizer=tokenizer)
    log_step(f"tri-anchor basis extracted for {len(tri_basis)} layers")

    # --- Post-</think> refusal & slop directions (v2 scenario archetypes) --
    # Positive = clean calibrated answer (archetype 0); negatives = refusal
    # (archetype 5) and overthinking+slop (archetype 2). All pooled with
    # pool_mode='answer_only' and bi-projected against the Tri-Anchor basis.
    clean_answers: list[str] = []
    refusal_answers: list[str] = []
    slop_answers: list[str] = []
    for spec in TENET_V2_SCENARIOS:
        candidates = _v2_candidate_responses(spec)
        clean_answers.append(candidates[0].split("</think>", 1)[-1].strip())
        refusal_answers.append(candidates[5].split("</think>", 1)[-1].strip())
        slop_answers.append(candidates[2].split("</think>", 1)[-1].strip())
    refusal_pairs = list(zip(clean_answers, refusal_answers, strict=False))
    slop_pairs = list(zip(clean_answers, slop_answers, strict=False))

    vocab_size = max(int(len(tokenizer)), 32768)
    ref_pos_ids, ref_neg_ids = encode_contrast_pairs(
        refusal_pairs,
        vocab_size=vocab_size,
        seq_len=ANSWER_SEQ_LEN,
        device=device,
        tokenizer=tokenizer,
    )
    slop_pos_ids, slop_neg_ids = encode_contrast_pairs(
        slop_pairs,
        vocab_size=vocab_size,
        seq_len=ANSWER_SEQ_LEN,
        device=device,
        tokenizer=tokenizer,
    )

    target_dirs: dict[int, list[torch.Tensor]] = {}
    forbidden_dir_map: dict[int, list[torch.Tensor]] = {}
    extraction_stats: dict[str, Any] = {}
    log_step("extracting refusal/slop directions (pool_mode='answer_only', bi-projected)")
    for idx, anchor_dirs in tri_basis.items():
        refusal_dir, ref_stats = extract_contrastive_direction_with_stats(
            model,
            ref_pos_ids,
            ref_neg_ids,
            layer_idx=idx,
            pool_mode="answer_only",
            phase_segmenter=seg,
            tokenizer=tokenizer,
            texts=(clean_answers, refusal_answers),
        )
        slop_dir, slop_stats = extract_contrastive_direction_with_stats(
            model,
            slop_pos_ids,
            slop_neg_ids,
            layer_idx=idx,
            pool_mode="answer_only",
            phase_segmenter=seg,
            tokenizer=tokenizer,
            texts=(clean_answers, slop_answers),
        )
        target_dirs[idx] = [
            biproject_direction(refusal_dir, anchor_dirs),
            biproject_direction(slop_dir, anchor_dirs),
        ]
        # Forbidden subspace for the LoRA fold: the layer's Tri-Anchor basis
        # plus its bi-projected refusal/slop directions (orthonormal union).
        forbidden_dir_map[idx] = [
            d.detach().float().cpu() for d in list(anchor_dirs) + target_dirs[idx]
        ]
        extraction_stats[str(idx)] = {
            "refusal": {k: round(v, 6) for k, v in ref_stats.items()},
            "slop": {k: round(v, 6) for k, v in slop_stats.items()},
        }

    # --- CoT-blended SCR layer profile ------------------------------------
    anchor_pos_ids, _anchor_neg_ids = encode_contrast_pairs(
        list(REASONING_ANCHOR_PAIRS),
        vocab_size=vocab_size,
        seq_len=PROBE_SEQ_LEN,
        device=device,
        tokenizer=tokenizer,
    )
    cot_pos_ids, _cot_neg_ids = encode_contrast_pairs(
        list(COT_REASONING_ANCHOR_PAIRS),
        vocab_size=vocab_size,
        seq_len=PROBE_SEQ_LEN,
        device=device,
        tokenizer=tokenizer,
    )
    log_step(
        "computing CoT-blended SCR layer profile "
        f"(lambda_cot={LAMBDA_COT}, tag ids={delib_tag_ids})"
    )
    scr_profile = compute_layer_scr_profile(
        model,
        target_dirs,
        anchor_pos_ids,
        scale=1.0,
        preserve_row_norm=True,
        cot_input_ids=cot_pos_ids,
        cot_tag_token_ids=delib_tag_ids,
        lambda_cot=LAMBDA_COT,
    )
    top_rows = scr_profile[: max(1, top_k_layers)]
    log_step(
        "top-SCR layers: " + ", ".join(f"L{row['layer_idx']}={row['scr']:.2f}" for row in top_rows)
    )

    # --- Build / save / apply the Lancet recipe ---------------------------
    log_step(f"building LancetRecipe (top_k_layers={top_k_layers})")
    recipe = build_lancet_recipe(
        model,
        target_dirs,
        tri_basis,
        anchor_pos_ids,
        model_id=release_name,
        base_model=model_name,
        extraction_mode="activation_contrastive",
        top_k_layers=top_k_layers,
        scale=1.0,
        preserve_row_norm=True,
    )
    save_lancet_recipe(recipe, release_dir)
    log_step(
        f"saved lancet_recipe.json + lancet_basis.safetensors "
        f"(selected layers: {recipe.selected_layers})"
    )

    pre_surgery = capture_pre_surgery_weights(model, recipe.selected_layers)
    apply_lancet_recipe(model, recipe, in_place=True)  # row-norm preserving cut
    log_step("applied Lancet recipe in place (preserve_row_norm=True)")

    # --- Weight-hygiene audit: crest factor + FP8 re-leakage --------------
    metrics = surgical_layer_metrics(model, recipe, pre_surgery)
    log_step("prompt-input KL quant drift (quant_bits=8, group_size=32) on all layers")
    prompt_rows = [anchor_pos_ids[i : i + 1] for i in range(anchor_pos_ids.shape[0])]
    fp8_drift = compute_blockwise_quantization_drift(
        model, prompt_rows, quant_bits=8, group_size=32
    )
    metrics["fp8_blockwise_kl_drift_mean"] = round(
        sum(float(row["leave_one_out_kl"]) for row in fp8_drift) / max(len(fp8_drift), 1), 6
    )
    metrics["selected_layers"] = list(recipe.selected_layers)
    metrics["layer_weights"] = {str(k): float(v) for k, v in recipe.layer_weights.items()}
    metrics["extraction_stats"] = extraction_stats
    log_step(
        f"max_row_crest_factor={metrics['max_row_crest_factor']} "
        f"(limit {CREST_LIMIT}), "
        f"fp8_subspace_releakage={metrics['fp8_subspace_releakage_pct']}% "
        f"(limit {FP8_RELEAK_LIMIT_PCT}%)"
    )
    return recipe, metrics, scr_profile, forbidden_dir_map


def stage2_h16_grpo_merge(
    model: torch.nn.Module,
    tokenizer: Any,
    recipe: Any,
    scr_profile: list[dict[str, Any]],
    pre_surgery: dict[int, dict[str, torch.Tensor]],
    forbidden_dir_map: dict[int, list[torch.Tensor]],
    grpo_steps: int,
    release_name: str,
    release_dir: Path,
) -> dict[str, Any]:
    """Stage 2: H16 subspace-safe GRPO + norm-preserving merge + bundle."""
    log_stage(2, "H16 Subspace-Safe GRPO + Norm-Preserving Merge")
    device = next(model.parameters()).device
    layers = get_transformer_layers(model)
    target_layer_indices = list(range(max(0, len(layers) - LORA_LAYERS), len(layers)))
    log_step(f"LoRA targets: layers {target_layer_indices} o_proj+down_proj (r={LORA_RANK})")

    # --- Freeze base, wrap the last 4 blocks' o_proj + down_proj ----------
    for param in model.parameters():
        param.requires_grad_(False)
    for idx in target_layer_indices:
        block = layers[idx]
        block.self_attn.o_proj = LoRALinear(
            block.self_attn.o_proj, rank=LORA_RANK, alpha=LORA_ALPHA
        )
        block.mlp.down_proj = LoRALinear(block.mlp.down_proj, rank=LORA_RANK, alpha=LORA_ALPHA)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.requires_grad_(True)
            module.lora_b.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log_step(f"trainable LoRA params: {sum(p.numel() for p in trainable)}")

    # --- Deterministic v2 rewards over the 6 archetypes -------------------
    v2_rubric = TenetUnslopRubric(mode="v2")
    num_scenarios = len(TENET_V2_SCENARIOS)
    v2_rewards = torch.zeros(num_scenarios, GROUP_SIZE)
    for row_idx, spec in enumerate(TENET_V2_SCENARIOS):
        for col_idx, text in enumerate(_v2_candidate_responses(spec)):
            v2_rewards[row_idx, col_idx] = v2_rubric.score(text, spec).total_reward
    log_step(f"v2 rubric rewards precomputed for {num_scenarios} scenarios")

    # --- Tokenize (prompt, candidate) pairs with completion masks ---------
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    tokenized: list[list[tuple[list[int], list[int]]]] = []
    for spec in TENET_V2_SCENARIOS:
        rows: list[tuple[list[int], list[int]]] = []
        for cand in _v2_candidate_responses(spec):
            prompt_ids = tokenizer(spec.prompt, add_special_tokens=True)["input_ids"]
            cand_ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
            prompt_len = min(len(prompt_ids), GRPO_MAX_LENGTH - 1)
            cand_trunc = cand_ids[: GRPO_MAX_LENGTH - prompt_len]
            rows.append(
                (
                    prompt_ids[:prompt_len] + cand_trunc,
                    [0] * prompt_len + [1] * len(cand_trunc),
                )
            )
        tokenized.append(rows)

    def _score_scenario(
        policy: torch.nn.Module, scenario_rows: list[tuple[list[int], list[int]]]
    ) -> torch.Tensor:
        batch_len = max(len(seq) for seq, _ in scenario_rows)
        ids = torch.full((len(scenario_rows), batch_len), pad_token_id, dtype=torch.long)
        comp_mask = torch.zeros((len(scenario_rows), batch_len), dtype=torch.float32)
        for row, (seq, comp) in enumerate(scenario_rows):
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            comp_mask[row, : len(comp)] = torch.tensor(comp, dtype=torch.float32)
        token_lp, shift_mask = compute_token_logprobs(policy, ids.to(device), comp_mask.to(device))
        seq_lp = (token_lp * shift_mask).sum(dim=1)
        n_completion = shift_mask.sum(dim=1).clamp(min=1.0)
        return torch.log_softmax(seq_lp / n_completion.sqrt(), dim=-1)

    def _score_candidates(policy: torch.nn.Module) -> torch.Tensor:
        """(num_scenarios, group_size) log-softmax over candidate log-probs."""
        return torch.stack([_score_scenario(policy, sr) for sr in tokenized])

    # --- GRPO training loop (h10 pattern) ---------------------------------
    torch.manual_seed(SEED)
    optimizer = torch.optim.AdamW(trainable, lr=GRPO_LR)
    with torch.no_grad():
        ref_all_logprobs = _score_candidates(model)
    history: list[dict[str, Any]] = []
    log_step(f"GRPO training for {grpo_steps} steps (lr={GRPO_LR}, beta_kl={GRPO_BETA_KL})")
    for step in range(grpo_steps):
        with torch.no_grad():
            old_all_logprobs = _score_candidates(model)
            old_probs = torch.exp(old_all_logprobs)
            actions = torch.multinomial(old_probs, num_samples=GROUP_SIZE, replacement=True)
            old_lp = old_all_logprobs.gather(1, actions)
            ref_lp = ref_all_logprobs.gather(1, actions)
            sampled_rewards = v2_rewards.to(device).gather(1, actions)
            advantages = torch.stack(
                [
                    compute_group_normalized_advantages(sampled_rewards[idx])
                    for idx in range(num_scenarios)
                ]
            )
        optimizer.zero_grad()
        total_loss = 0.0
        total_obj = 0.0
        total_kl = 0.0
        for s_idx in range(num_scenarios):
            s_logprobs = _score_scenario(model, tokenized[s_idx])
            s_lp = s_logprobs.gather(0, actions[s_idx]).unsqueeze(0)
            s_loss_dict = compute_grpo_loss(
                policy_logprobs=s_lp,
                old_logprobs=old_lp[s_idx : s_idx + 1],
                ref_logprobs=ref_lp[s_idx : s_idx + 1],
                advantages=advantages[s_idx : s_idx + 1],
                clip_eps=GRPO_CLIP_EPS,
                beta_kl=GRPO_BETA_KL,
            )
            (s_loss_dict["loss"] / num_scenarios).backward()
            total_loss += float(s_loss_dict["loss"].detach()) / num_scenarios
            total_obj += float(s_loss_dict["policy_obj"].detach()) / num_scenarios
            total_kl += float(s_loss_dict["kl_div"].detach()) / num_scenarios
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.5)
        optimizer.step()
        mps_cleanup()
        entry = {
            "step": step,
            "loss": round(total_loss, 6),
            "policy_obj": round(total_obj, 6),
            "kl_div": round(total_kl, 6),
        }
        history.append(entry)
        log_step(
            f"step {step}: loss={entry['loss']} policy_obj={entry['policy_obj']} "
            f"kl={entry['kl_div']}"
        )

    # --- Subspace-safe fold: forbidden dirs = Tri-Anchor + refusal/slop ---
    forbidden_dirs: dict[int, list[torch.Tensor]] = {}
    for idx in target_layer_indices:
        dirs = [d.detach().float().to(device) for d in forbidden_dir_map.get(idx, [])]
        if not dirs:
            raise RuntimeError(f"no forbidden directions resolved for LoRA layer {idx}")
        forbidden_dirs[idx] = dirs
    log_step(
        "merging LoRA in place (merge_lora_weights_in_place, "
        "forbidden_dirs=Tri-Anchor basis, preserve_row_norm=True)"
    )
    merge_lora_weights_in_place(model, forbidden_dirs=forbidden_dirs, preserve_row_norm=True)
    merged_lora_count = sum(isinstance(m, LoRALinear) for m in model.modules())
    if merged_lora_count != 0:
        raise RuntimeError("merge_lora_weights_in_place left LoRALinear wrappers")
    log_step("LoRA folded; 0 LoRALinear wrappers remain")

    # --- Post-merge Lancet guard + re-verify weight hygiene ---------------
    apply_lancet_recipe(model, recipe, in_place=True)
    log_step("post-merge guard: Lancet recipe re-applied in place")
    metrics = surgical_layer_metrics(model, recipe, pre_surgery)
    log_step(
        f"re-verified: max_row_crest_factor={metrics['max_row_crest_factor']} "
        f"(< {CREST_LIMIT}), "
        f"fp8_subspace_releakage={metrics['fp8_subspace_releakage_pct']}% "
        f"(< {FP8_RELEAK_LIMIT_PCT}%)"
    )

    # --- V2 rubric evaluation of the deployed model -----------------------
    with torch.no_grad():
        probs = torch.exp(_score_candidates(model))
    argmax_idx = [int(torch.argmax(probs[row])) for row in range(num_scenarios)]
    cot_integrity = 0
    collapsed = 0
    slop_refusal = 0
    artifact_complete = 0
    total_ans_words = 0
    total_cot_words = 0
    for row_idx, spec in enumerate(TENET_V2_SCENARIOS):
        chosen = _v2_candidate_responses(spec)[argmax_idx[row_idx]]
        breakdown = v2_rubric.score(chosen, spec)
        cot_integrity += float(breakdown.g_cot_integrity >= 1.0)
        if breakdown.ans_word_count < max(6, getattr(spec, "min_words", 12) or 12):
            collapsed += 1
        slop_refusal += float(breakdown.slop_hits > 0 or breakdown.refusal_hits > 0)
        artifact_complete += float(bool(breakdown.artifact_valid))
        total_ans_words += breakdown.ans_word_count
        total_cot_words += breakdown.cot_word_count
    v2_eval = {
        "cot_tag_integrity_rate": round(cot_integrity / num_scenarios, 6),
        "length_collapse_rate": round(collapsed / num_scenarios, 6),
        "slop_and_refusal_rate": round(slop_refusal / num_scenarios, 6),
        "artifact_completeness_rate": round(artifact_complete / num_scenarios, 6),
        "mean_ans_words": round(total_ans_words / num_scenarios, 4),
        "mean_cot_words": round(total_cot_words / num_scenarios, 4),
    }
    log_step(f"v2 argmax response audit: {v2_eval}")

    # --- Save HF checkpoint + tokenizer ------------------------------------
    model.to("cpu")
    if getattr(model.config, "auto_map", None):
        model.config.auto_map = None
    if getattr(model.config, "trust_remote_code", False):
        model.config.trust_remote_code = False
    model.save_pretrained(release_dir, safe_serialization=True)
    tokenizer.save_pretrained(release_dir)
    log_step(f"saved HF checkpoint + tokenizer to {release_dir}")

    # --- CKA quant recipe + GGUF bundle files ------------------------------
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", len(layers)))
    scr_rows = [{"layer": int(row["layer_idx"]), "scr": float(row["scr"])} for row in scr_profile]
    cka_recipe = build_cka_quant_recipe(
        list(recipe.selected_layers),
        num_hidden_layers,
        layer_scr_profile=scr_rows,
    )
    cka_recipe_path = release_dir / "cka_quant_recipe.json"
    _json_write(cka_recipe_path, cka_recipe)
    bundle = _write_gguf_recipe_bundle_files(release_dir, release_name, cka_recipe)
    log_step(
        f"wrote cka_quant_recipe.json (est. avg bpw={cka_recipe.get('estimated_avg_bpw')}), "
        f"Modelfile + quantize_cka_gguf.sh"
    )

    mps_cleanup()
    return {
        "grpo_history": history,
        "grpo_steps": grpo_steps,
        "merged_lora_count": merged_lora_count,
        "target_layers": target_layer_indices,
        "post_merge_guard": True,
        "max_row_crest_factor": metrics["max_row_crest_factor"],
        "fp8_subspace_releakage_pct": metrics["fp8_subspace_releakage_pct"],
        "v2_eval": v2_eval,
        "cka_recipe_summary": {
            "estimated_avg_bpw": cka_recipe.get("estimated_avg_bpw"),
            "q8_0_layers": cka_recipe.get("q8_0_layers"),
            "q6_k_layers": cka_recipe.get("q6_k_layers"),
            "boundary_layers": cka_recipe.get("boundary_layers"),
        },
        "num_hidden_layers": num_hidden_layers,
        "bundle_files": bundle,
    }


def stage3_domain_gguf_and_audit(
    release_dir: Path,
    release_name: str,
    ppl_chunks: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Stage 3: domain imatrix + f16 GGUF + CKA mixed quant + full GGUF audit."""
    from sml.surgery.gguf_audit import (
        build_tenet_domain_imatrix_corpus,
        run_h14_gguf_audit,
    )

    log_stage(3, "Domain GGUF build (imatrix + CKA mixed quant) + audit")

    f16_gguf = release_dir / f"{release_name}-f16.gguf"
    cka_gguf = release_dir / f"{release_name}-cka-q4_k_m.gguf"
    i1_cka_gguf = release_dir / f"{release_name}-i1-cka-q4_k_m.gguf"
    imatrix_path = release_dir / "tenet_domain.imatrix"
    imatrix_txt = release_dir / "tenet_domain_imatrix.txt"

    # --- Domain imatrix corpus --------------------------------------------
    build_tenet_domain_imatrix_corpus(imatrix_txt, repeats=4)
    log_step(f"built tenet domain imatrix corpus: {imatrix_txt}")

    # --- HF checkpoint -> f16 GGUF ----------------------------------------
    if f16_gguf.is_file():
        log_step(f"f16 GGUF already present, skipping conversion: {f16_gguf}")
    else:
        convert_script = REPO_ROOT / ".tools" / "llama.cpp" / "convert_hf_to_gguf.py"
        if not convert_script.is_file():
            raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")
        run_command(
            [
                sys.executable,
                str(convert_script),
                str(release_dir),
                "--outfile",
                str(f16_gguf),
                "--outtype",
                "f16",
            ]
        )
        log_step(f"converted HF checkpoint -> {f16_gguf}")

    # --- Domain imatrix over the f16 GGUF ---------------------------------
    if imatrix_path.is_file():
        log_step(f"imatrix already present, skipping: {imatrix_path}")
    else:
        run_command(
            [
                resolve_tool("LLAMA_IMATRIX", "llama-imatrix"),
                "-m",
                str(f16_gguf),
                "-f",
                str(imatrix_txt),
                "-o",
                str(imatrix_path),
                "--chunks",
                "32",
                "-ngl",
                "99",
            ]
        )
        log_step(f"computed domain imatrix -> {imatrix_path}")

    # --- CKA mixed-quant builds (plain + imatrix) --------------------------
    quantize_script = (release_dir / "quantize_cka_gguf.sh").resolve()
    imatrix_abs = imatrix_path.resolve()
    if not cka_gguf.is_file():
        run_command(
            ["bash", str(quantize_script)],
            env={"OUTPUT_GGUF": cka_gguf.name},
            cwd=release_dir,
        )
    log_step(f"CKA quant (no imatrix) -> {cka_gguf}")
    if not i1_cka_gguf.is_file():
        run_command(
            ["bash", str(quantize_script), "--imatrix", str(imatrix_abs)],
            env={"OUTPUT_GGUF": i1_cka_gguf.name},
            cwd=release_dir,
        )
    log_step(f"CKA quant (domain imatrix) -> {i1_cka_gguf}")

    # --- Full GGUF audit (inventory + weights + ppl + behavioral) ----------
    audit = run_h14_gguf_audit(
        variants={
            "f16": f16_gguf,
            "cka_q4_k_m": cka_gguf,
            "i1_cka_q4_k_m": i1_cka_gguf,
        },
        lancet_basis_path=release_dir / "lancet_basis.safetensors",
        f16_name="f16",
        output_dir=release_dir / "gguf_audit",
        run_perplexity=True,
        run_behavioral=True,
        ppl_chunks=ppl_chunks,
        use_chat_completions=True,
        run_arena_house=True,
        require_think=True,
        max_tokens=max_tokens,
    )
    log_step(f"GGUF audit written: {audit.get('final_metrics_path')}")
    return {
        "imatrix_corpus": str(imatrix_txt),
        "imatrix_path": str(imatrix_path),
        "f16_gguf": str(f16_gguf),
        "cka_q4_k_m_gguf": str(cka_gguf),
        "i1_cka_q4_k_m_gguf": str(i1_cka_gguf),
        "audit": audit,
    }


V2_GGUF_COT_INTEGRITY_FLOOR = 0.995
V2_HOUSE_CLASSIFICATION_TARGET = 1.0
V2_ARENA_HOUSE_HP_FLOOR = 95.0
V2_LENGTH_COLLAPSE_LIMIT = 0.01
V2_GGUF_SLOP_REFUSAL_LIMIT = 0.02
V2_SQL_READONLY_TARGET = 1.0


def _find_metric(payload: Any, key: str) -> Any | None:
    """Depth-first search for ``key`` in nested dicts/lists; first match wins."""
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_metric(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_metric(item, key)
            if found is not None:
                return found
    return None


def _variant_behavioral(
    stage3_metrics: dict[str, Any] | None, variant: str
) -> dict[str, Any] | None:
    """Return the ``behavioral`` block recorded for one audited GGUF variant."""
    if not stage3_metrics:
        return None
    audit = stage3_metrics.get("audit") or {}
    entry = (audit.get("variants") or {}).get(variant) or {}
    behavioral = entry.get("behavioral")
    if isinstance(behavioral, dict) and isinstance(behavioral.get("behavioral"), dict):
        behavioral = behavioral["behavioral"]
    return behavioral if isinstance(behavioral, dict) else None


def stage4_gates_and_manifest(
    release_dir: Path,
    args: argparse.Namespace,
    stage1_metrics: dict[str, Any],
    stage2_metrics: dict[str, Any],
    stage3_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Stage 4: evaluate every Tenet v2 release gate + write the manifests."""
    log_stage(4, "Release gates + manifest")
    gates: list[dict[str, Any]] = []

    def _add_gate(
        name: str,
        value: Any,
        limit_expr: str,
        passed: bool,
        detail: str = "",
    ) -> None:
        entry: dict[str, Any] = {
            "gate": name,
            "value": value,
            "limit": limit_expr,
            "passed": bool(passed),
        }
        if detail:
            entry["detail"] = detail
        gates.append(entry)

    # --- Weight-hygiene gates (final merged model = stage 2 numbers) -------
    post_merge = stage2_metrics.get("post_merge_metrics") or {}
    crest = stage2_metrics.get("max_row_crest_factor")
    if crest is None:
        crest = post_merge.get("max_row_crest_factor")
    _add_gate(
        "max_row_crest_factor",
        crest,
        f"< {CREST_LIMIT}",
        crest is not None and crest < CREST_LIMIT,
    )
    releak = stage2_metrics.get("fp8_subspace_releakage_pct")
    if releak is None:
        releak = post_merge.get("fp8_subspace_releakage_pct")
    _add_gate(
        "fp8_releakage_pct",
        releak,
        f"< {FP8_RELEAK_LIMIT_PCT}",
        releak is not None and releak < FP8_RELEAK_LIMIT_PCT,
    )

    audit = (stage3_metrics or {}).get("audit") or {}
    variants = audit.get("variants") or {}
    if stage3_metrics and variants:
        # --- CoT tag integrity on f16 and the deployed i1 CKA quant --------
        for variant in ("f16", "i1_cka_q4_k_m"):
            behavioral = _variant_behavioral(stage3_metrics, variant)
            rate = behavioral.get("cot_tag_integrity_rate") if behavioral else None
            _add_gate(
                f"cot_tag_integrity_rate[{variant}]",
                rate,
                f">= {V2_GGUF_COT_INTEGRITY_FLOOR}",
                rate is not None and rate >= V2_GGUF_COT_INTEGRITY_FLOOR,
            )

        i1_behavioral = _variant_behavioral(stage3_metrics, "i1_cka_q4_k_m")
        if i1_behavioral is None:
            _add_gate(
                "i1_cka_q4_k_m.behavioral",
                None,
                "available",
                False,
                detail="missing behavioral block for i1_cka_q4_k_m",
            )
        else:
            house_rate = i1_behavioral.get("house_artifact_classification_rate")
            _add_gate(
                "house_artifact_classification_rate[i1_cka_q4_k_m]",
                house_rate,
                f"== {V2_HOUSE_CLASSIFICATION_TARGET}",
                house_rate is not None and float(house_rate) == V2_HOUSE_CLASSIFICATION_TARGET,
            )
            hp = i1_behavioral.get("arena_house_hp")
            _add_gate(
                "arena_house_hp[i1_cka_q4_k_m]",
                hp,
                f">= {V2_ARENA_HOUSE_HP_FLOOR}",
                hp is not None and hp >= V2_ARENA_HOUSE_HP_FLOOR,
            )
            collapse = _find_metric(i1_behavioral, "length_collapse_rate")
            _add_gate(
                "length_collapse_rate[i1_cka_q4_k_m]",
                collapse,
                f"< {V2_LENGTH_COLLAPSE_LIMIT}",
                collapse is not None and collapse < V2_LENGTH_COLLAPSE_LIMIT,
            )
            slop = _find_metric(i1_behavioral, "slop_and_refusal_rate")
            _add_gate(
                "slop_and_refusal_rate[i1_cka_q4_k_m]",
                slop,
                f"<= {V2_GGUF_SLOP_REFUSAL_LIMIT}",
                slop is not None and slop <= V2_GGUF_SLOP_REFUSAL_LIMIT,
            )
            sql_rate = (i1_behavioral.get("sql_readonly") or {}).get("rate")
            _add_gate(
                "sql_readonly_rate[i1_cka_q4_k_m]",
                sql_rate,
                f"== {V2_SQL_READONLY_TARGET}",
                sql_rate is not None and float(sql_rate) == V2_SQL_READONLY_TARGET,
            )

        # --- Unslop pass-rate must survive quantization (i1 is best) -------
        def _unslop_rate(variant: str) -> Any | None:
            behavioral = _variant_behavioral(stage3_metrics, variant)
            return _find_metric(behavioral, "unslop_pass_rate") if behavioral else None

        i1_unslop = _unslop_rate("i1_cka_q4_k_m")
        cka_unslop = _unslop_rate("cka_q4_k_m")
        f16_unslop = _unslop_rate("f16")
        unslop_ok = (
            i1_unslop is not None
            and cka_unslop is not None
            and f16_unslop is not None
            and i1_unslop >= cka_unslop
            and i1_unslop >= f16_unslop
        )
        _add_gate(
            "unslop_pass_rate[i1_cka_q4_k_m]",
            i1_unslop,
            ">= cka_q4_k_m and >= f16",
            unslop_ok,
            detail=f"i1={i1_unslop} cka={cka_unslop} f16={f16_unslop}",
        )
    else:
        _add_gate(
            "gguf_audit",
            None,
            "available",
            False,
            detail="stage 3 skipped or audit missing variants",
        )

    failed_gates = [g["gate"] for g in gates if not g["passed"]]
    overall_pass = not failed_gates
    for gate in gates:
        log_step(
            f"gate {gate['gate']}: value={gate['value']} "
            f"({gate['limit']}) -> {'PASS' if gate['passed'] else 'FAIL'}"
        )
    log_step(
        f"{len(gates) - len(failed_gates)}/{len(gates)} gates passed; "
        f"overall={'PASS' if overall_pass else 'FAIL'}"
    )

    # --- Collect release artifacts ----------------------------------------
    artifact_names = (
        "lancet_recipe.json",
        "lancet_basis.safetensors",
        "cka_quant_recipe.json",
        "quantize_cka_gguf.sh",
        "Modelfile",
        "tenet_domain_imatrix.txt",
        "tenet_domain.imatrix",
        f"{args.release_name}-f16.gguf",
        f"{args.release_name}-cka-q4_k_m.gguf",
        f"{args.release_name}-i1-cka-q4_k_m.gguf",
        "gguf_audit/final_metrics.json",
    )
    artifacts = {
        name: str(release_dir / name) for name in artifact_names if (release_dir / name).is_file()
    }

    # --- release_manifest.json (full provenance) ---------------------------
    manifest: dict[str, Any] = {
        "release_name": args.release_name,
        "model_name": args.model_name,
        "release_dir": str(release_dir),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "skip_gguf": bool(getattr(args, "skip_gguf", False)),
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "stage3": stage3_metrics,
        "gates": gates,
        "failed_gates": failed_gates,
        "overall_pass": overall_pass,
        "artifacts": artifacts,
    }
    manifest_path = release_dir / "release_manifest.json"
    _json_write(manifest_path, manifest)
    log_step(f"wrote {manifest_path}")

    # --- foundry_summary.json (compact) ------------------------------------
    key_metrics: dict[str, Any] = {
        "max_row_crest_factor": stage2_metrics.get("max_row_crest_factor"),
        "fp8_subspace_releakage_pct": stage2_metrics.get("fp8_subspace_releakage_pct"),
        "v2_eval": stage2_metrics.get("v2_eval"),
        "cka_recipe_summary": stage2_metrics.get("cka_recipe_summary"),
    }
    if stage3_metrics and variants:
        key_metrics["gguf_variants"] = {
            name: {
                "cot_tag_integrity_rate": _variant_behavioral(stage3_metrics, name)
                and _variant_behavioral(stage3_metrics, name).get("cot_tag_integrity_rate"),
                "arena_house_hp": _variant_behavioral(stage3_metrics, name)
                and _variant_behavioral(stage3_metrics, name).get("arena_house_hp"),
                "unslop_pass_rate": _find_metric(
                    _variant_behavioral(stage3_metrics, name), "unslop_pass_rate"
                ),
            }
            for name in sorted(variants)
        }
    summary: dict[str, Any] = {
        "release_name": args.release_name,
        "model_name": args.model_name,
        "overall_pass": overall_pass,
        "gate_count": len(gates),
        "failed_gates": failed_gates,
        "key_metrics": key_metrics,
        "artifacts": artifacts,
        "manifest_path": str(manifest_path),
    }
    summary_path = release_dir / "foundry_summary.json"
    _json_write(summary_path, summary)
    log_step(f"wrote {summary_path}")

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse foundry CLI arguments (stages 1-4)."""
    parser = argparse.ArgumentParser(
        description="Tenet v2 local foundry pipeline (Lancet -> GRPO -> GGUF -> gates)"
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B", help="HF base model id")
    parser.add_argument(
        "--release-name",
        default="Tenet-4B-Qwen3-Unslop-v2",
        help="Release name (also the release_dir subfolder)",
    )
    parser.add_argument(
        "--releases-dir",
        type=Path,
        default=REPO_ROOT / "results" / "tenet_releases",
        help="Parent directory for release folders",
    )
    parser.add_argument(
        "--top-k-layers", type=int, default=6, help="Lancet surgical layers (top-k SCR)"
    )
    parser.add_argument("--grpo-steps", type=int, default=20, help="Stage 2 GRPO optimizer steps")
    parser.add_argument(
        "--ppl-chunks",
        type=int,
        default=40,
        help="llama-perplexity chunks in the stage 3 audit (-1 = all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Generation budget for the stage 3 behavioral audit",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Skip stage 3 (GGUF build + audit); stage 4 records a failed audit gate",
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload release bundle to Hugging Face Hub after passing gates",
    )
    parser.add_argument(
        "--hf-namespace",
        default=os.environ.get("HF_NAMESPACE", "mr-andrey-isachenko"),
        help="Hugging Face Hub namespace",
    )
    parser.add_argument(
        "--drop-local-weights",
        action="store_true",
        help="Delete heavy model*.safetensors and *.gguf after successful run/upload",
    )
    parser.add_argument(
        "--allow-local-mac-weights",
        action="store_true",
        help="Override macOS Zero-Local-Weights guard",
    )
    parser.add_argument(
        "--allow-gate-fail",
        action="store_true",
        help="Upload and exit 0 even if gates fail (keeps overall_pass=false in manifests)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Stage 1 -> Stage 2 -> Stage 3 (unless --skip-gguf) -> Stage 4."""
    args = parse_args(argv)
    started = time.time()
    torch.manual_seed(SEED)

    release_dir = Path(args.releases_dir) / args.release_name
    release_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"release_dir={release_dir}")

    device, dtype = resolve_device()
    log_step(f"loading {args.model_name} on {device} ({dtype})")
    enforce_mac_zero_local_weights_guard(args.model_name, args.allow_local_mac_weights)
    from transformers import AutoModelForCausalLM, AutoTokenizer  # heavy import

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype).to(device)

    # Stage 1 snapshots pre-surgery weights internally (via
    # capture_pre_surgery_weights) but does not return the snapshot, while
    # stage 2 needs it for the post-merge re-verification. Intercept the one
    # capture call to keep a reference, then restore the original function.
    pre_surgery: dict[int, dict[str, torch.Tensor]] = {}
    original_capture = capture_pre_surgery_weights

    def _capture_and_remember(
        snap_model: torch.nn.Module, selected_layers: list[int]
    ) -> dict[int, dict[str, torch.Tensor]]:
        nonlocal pre_surgery
        pre_surgery = original_capture(snap_model, selected_layers)
        return pre_surgery

    globals()["capture_pre_surgery_weights"] = _capture_and_remember
    try:
        recipe, stage1_metrics, scr_profile, forbidden_dir_map = stage1_h15_tri_anchor_lancet(
            model,
            tokenizer,
            args.release_name,
            args.model_name,
            args.top_k_layers,
            release_dir,
        )
    finally:
        globals()["capture_pre_surgery_weights"] = original_capture

    stage2_metrics = stage2_h16_grpo_merge(
        model,
        tokenizer,
        recipe,
        scr_profile,
        pre_surgery,
        forbidden_dir_map,
        args.grpo_steps,
        args.release_name,
        release_dir,
    )
    mps_cleanup()

    if args.skip_gguf:
        stage3_metrics: dict[str, Any] | None = None
        log_step("--skip-gguf set: skipping stage 3 (GGUF build + audit)")
    else:
        stage3_metrics = stage3_domain_gguf_and_audit(
            release_dir, args.release_name, args.ppl_chunks, args.max_tokens
        )

    summary = stage4_gates_and_manifest(
        release_dir, args, stage1_metrics, stage2_metrics, stage3_metrics
    )
    if (summary["overall_pass"] or args.allow_gate_fail) and args.upload_to_hf:
        upload_info = upload_and_drop_local_weights(
            release_dir, args.release_name, args.hf_namespace, args.drop_local_weights
        )
        log_step(f"uploaded {upload_info['repo_id']} ({upload_info['commit_url']})")
    elif args.drop_local_weights:
        dropped = 0
        for pattern in ("model*.safetensors", "*.gguf"):
            for path in sorted(release_dir.glob(pattern)):
                path.unlink()
                dropped += 1
        log_step(f"dropped {dropped} local weight file(s) from {release_dir}")
    log_step(f"foundry finished in {time.time() - started:.1f}s")
    if not summary["overall_pass"] and not args.allow_gate_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
