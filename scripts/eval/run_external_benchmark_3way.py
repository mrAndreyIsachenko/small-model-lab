#!/usr/bin/env python3
"""Run an honest 3-way external benchmark (Base vs Naive Abliteration vs Tenet v1).

Evaluates:
1. IFEval (strict prompt-level & instruction-level) via ``lm-evaluation-harness``
2. GSM8K CoT (strict-match & flexible-extract) via ``lm-evaluation-harness``
3. SQL-Incident tool-calling agent split (``StatefulToolEnv``, real multi-turn model loop)
4. Direct-Compliance / Unslop Audit (``TENET_UNSLOP_SCENARIOS``)

Outputs a canonical JSON under ``results/external_evals/<release_id>.json`` and
copies it to ``results/tenet_releases/<release_id>/external_benchmarks.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from sml.evals.harness import (
    _run_sql_incident_eval,
    _run_unslop_audit,
    build_hf_chat_responder,
    run_lm_eval_suite,
)
from sml.surgery.subspace import apply_lancet_biprojected_surgery

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _free_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_naive_target_dirs(basis_path: Path, num_layers: int) -> dict[int, torch.Tensor]:
    raw = load_file(str(basis_path))
    parsed: dict[int, torch.Tensor] = {}
    for key, tensor in raw.items():
        if key.startswith("layer_"):
            idx = int(key.split("_", 1)[1])
            parsed[idx] = tensor.detach().float()
    if not parsed:
        raise ValueError(f"No layer_* tensors found in {basis_path}")
    # Fill remaining layers with the normalized mean direction so naive ablation
    # acts across the full residual stream without anchor bi-projection.
    stacked = torch.stack(list(parsed.values()), dim=0)
    mean_dir = stacked.mean(dim=0)
    if mean_dir.ndim > 1:
        mean_dir = mean_dir[0]
    norm = torch.linalg.norm(mean_dir)
    if norm > 0:
        mean_dir = mean_dir / norm
    full_map: dict[int, torch.Tensor] = {}
    for idx in range(num_layers):
        t = parsed.get(idx, mean_dir)
        if t.ndim > 1:
            t = t[0]
        full_map[idx] = t
    return full_map


def _eval_single_arm(
    arm_name: str,
    model: Any,
    tokenizer: Any,
    *,
    limit: int | None,
    batch_size: int | str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    t0 = time.time()
    print(f"[{arm_name}] Running lm-eval (ifeval, gsm8k_cot), limit={limit}...", flush=True)
    lm_scores = run_lm_eval_suite(
        model,
        tokenizer,
        tasks=("ifeval", "gsm8k_cot"),
        limit=limit,
        batch_size=batch_size,
        device=device,
    )
    responder = build_hf_chat_responder(model, tokenizer, max_new_tokens=512)
    print(f"[{arm_name}] Running SQL-Incident real-model tool-calling eval...", flush=True)
    sql_eval = _run_sql_incident_eval("after", seed=seed, answer_fn=responder)
    print(f"[{arm_name}] Running Direct-Compliance / Unslop Audit...", flush=True)
    unslop = _run_unslop_audit(responder)
    elapsed = round(time.time() - t0, 1)
    result = {
        "ifeval_strict": lm_scores.get("ifeval_strict", 0.0),
        "ifeval_inst_strict": lm_scores.get("ifeval_inst_strict", 0.0),
        "gsm8k_cot_strict": lm_scores.get("gsm8k_cot_strict", 0.0),
        "gsm8k_cot_flexible": lm_scores.get("gsm8k_cot_flexible", 0.0),
        "gsm8k_math_acc": lm_scores.get("gsm8k_cot_strict", 0.0),
        "sql_incident_acc": sql_eval["acc"],
        "sql_incident_reward": sql_eval["mean_reward"],
        "direct_compliance_rate": unslop["direct_compliance_rate"],
        "unslop_pass_rate": unslop["unslop_pass_rate"],
        "slop_and_refusal_rate": unslop["slop_and_refusal_rate"],
        "elapsed_seconds": elapsed,
    }
    print(f"[{arm_name}] Done in {elapsed}s: {json.dumps(result)}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3-way external eval for a Tenet release.")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--tenet-repo", required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    _load_dotenv()
    token = os.environ.get("HF_TOKEN")
    batch_size: int | str = (
        int(args.batch_size) if str(args.batch_size).isdigit() else args.batch_size
    )

    release_dir = REPO_ROOT / "results" / "tenet_releases" / args.release_id
    basis_path = release_dir / "lancet_basis.safetensors"
    recipe_path = release_dir / "lancet_recipe.json"
    if not basis_path.is_file() or not recipe_path.is_file():
        release_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=args.tenet_repo,
            filename="lancet_basis.safetensors",
            local_dir=str(release_dir),
            token=token,
        )
        hf_hub_download(
            repo_id=args.tenet_repo,
            filename="lancet_recipe.json",
            local_dir=str(release_dir),
            token=token,
        )
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    scale = float(recipe.get("scale", 1.0))

    print(f"=== Loading Base model: {args.base_model} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        token=token,
    )
    base_metrics = _eval_single_arm(
        "base",
        base_model,
        tokenizer,
        limit=args.limit,
        batch_size=batch_size,
        device=args.device,
        seed=args.seed,
    )

    print(f"=== Constructing Naive Abliteration control from {basis_path} ===", flush=True)
    num_layers = int(getattr(base_model.config, "num_hidden_layers", recipe.get("num_layers", 32)))
    naive_dirs = _load_naive_target_dirs(basis_path, num_layers)
    # Move base_model to CPU briefly so deep-copy + surgery doesn't double MPS memory.
    base_model.to("cpu")
    _free_memory()
    naive_model = apply_lancet_biprojected_surgery(
        base_model,
        target_dirs=naive_dirs,
        anchor_dirs=None,
        layer_indices=None,
        scale=scale,
        preserve_row_norm=False,
    )
    del base_model
    _free_memory()
    naive_model.to(args.device)
    naive_metrics = _eval_single_arm(
        "naive",
        naive_model,
        tokenizer,
        limit=args.limit,
        batch_size=batch_size,
        device=args.device,
        seed=args.seed,
    )
    del naive_model
    _free_memory()

    print(f"=== Loading Tenet v1 model: {args.tenet_repo} ===", flush=True)
    tenet_tokenizer = AutoTokenizer.from_pretrained(args.tenet_repo, token=token)
    tenet_model = AutoModelForCausalLM.from_pretrained(
        args.tenet_repo,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        token=token,
    )
    tenet_metrics = _eval_single_arm(
        "tenet",
        tenet_model,
        tenet_tokenizer,
        limit=args.limit,
        batch_size=batch_size,
        device=args.device,
        seed=args.seed,
    )
    del tenet_model
    _free_memory()

    metric_keys = [
        "ifeval_strict",
        "ifeval_inst_strict",
        "gsm8k_cot_strict",
        "gsm8k_cot_flexible",
        "gsm8k_math_acc",
        "sql_incident_acc",
        "sql_incident_reward",
        "direct_compliance_rate",
        "unslop_pass_rate",
        "slop_and_refusal_rate",
    ]
    deltas_vs_base = {
        k: round(float(tenet_metrics[k]) - float(base_metrics[k]), 6) for k in metric_keys
    }
    deltas_vs_naive = {
        k: round(float(tenet_metrics[k]) - float(naive_metrics[k]), 6) for k in metric_keys
    }

    payload = {
        "model_id": args.release_id,
        "base_model": args.base_model,
        "tenet_repo": args.tenet_repo,
        "seed": args.seed,
        "lm_eval_limit": args.limit,
        "tasks": ["ifeval", "gsm8k_cot", "sql_incident_val_6", "tenet_unslop_audit_12"],
        "arms": {
            "base": base_metrics,
            "naive": naive_metrics,
            "tenet": tenet_metrics,
        },
        "before": base_metrics,
        "after": tenet_metrics,
        "deltas": deltas_vs_base,
        "deltas_vs_naive": deltas_vs_naive,
        "external_benchmarks": tenet_metrics,
    }

    out_dir = REPO_ROOT / "results" / "external_evals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.release_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (release_dir / "external_benchmarks.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved external evals to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
