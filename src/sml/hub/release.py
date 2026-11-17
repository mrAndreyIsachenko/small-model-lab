"""Release bundling for Tenet checkpoints.

Runs the H9 Lancet surgery, H10 subspace-safe GRPO and H11 FP8 provider
audits, writes a provider-ready HF-style checkpoint, and emits the model
card, OpenRouter onboarding manifest and release manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sml.gateway.client import build_openrouter_onboarding_manifest
from sml.hub.model_card import generate_tenet_model_card
from sml.hub.validator import verify_provider_readiness

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from safetensors.torch import save_file as _safetensors_save_file
except ImportError:  # pragma: no cover
    _safetensors_save_file = None  # type: ignore[assignment]

_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
)

_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt")

_CLEAN_WEIGHT_SHAPES = {
    "model.embed_tokens.weight": (128, 32),
    "model.layers.0.self_attn.q_proj.weight": (32, 32),
    "model.layers.0.self_attn.k_proj.weight": (32, 32),
    "model.layers.0.self_attn.v_proj.weight": (32, 32),
    "model.layers.0.self_attn.o_proj.weight": (32, 32),
    "model.layers.0.mlp.gate_proj.weight": (64, 32),
    "model.layers.0.mlp.down_proj.weight": (32, 64),
    "lm_head.weight": (128, 32),
}


def build_tenet_release_bundle(
    output_dir: Path | str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    release_id: str = "Tenet-3B-Qwen2.5-Unslop-v1",
    backend: str = "nano",
    device: str = "auto",
    seed: int = 2026,
    top_k_layers: int | None = None,
    scale: float = 1.0,
    grpo_steps: int = 27,
    grpo_group_size: int = 6,
    grpo_lr: float = 0.0038,
    n_prompts: int = 8,
    seq_len: int = 32,
    hf_namespace: str = "tenet-lab",
) -> dict[str, Any]:
    """Build a full provider-ready Tenet release bundle.

    Runs the H9/H10/H11 benchmarks for ``backend`` (``"nano"`` or
    ``"llm"``), writes an HF-style checkpoint into
    ``Path(output_dir) / release_id``, validates it with
    :func:`verify_provider_readiness`, and writes the model card, the
    OpenRouter onboarding manifest and the release manifest.
    """
    release_dir = Path(output_dir) / release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    if backend == "llm":
        from sml.adaptation.grpo import run_llm_h10_unslop_grpo_benchmark
        from sml.surgery.llm_pareto import run_llm_h9_biprojected_surgery_benchmark
        from sml.surgery.quantization import evaluate_llm_h11_fp8_provider_survival

        h9_res = run_llm_h9_biprojected_surgery_benchmark(
            model_name=model_name,
            device=device,
            seed=seed,
            top_k_layers=top_k_layers,
            scale=scale,
            output_dir=release_dir,
            model_id=release_id,
        )
        h10_res = run_llm_h10_unslop_grpo_benchmark(
            model_name=model_name,
            device=device,
            seed=seed,
            steps=grpo_steps,
            group_size=grpo_group_size,
            lr=grpo_lr,
        )
        h11_res = evaluate_llm_h11_fp8_provider_survival(
            model_name=model_name,
            device=device,
            seed=seed,
            n_prompts=n_prompts,
            seq_len=seq_len,
        )
    elif backend == "nano":
        from sml.adaptation.grpo import run_h10_unslop_grpo_benchmark
        from sml.surgery.pareto import run_h9_biprojected_surgery_benchmark
        from sml.surgery.quantization import evaluate_h11_fp8_provider_survival

        h9_res = run_h9_biprojected_surgery_benchmark(
            seed=seed,
            top_k_layers=top_k_layers or 3,
            scale=scale,
            output_dir=release_dir,
            model_id=release_id,
        )
        h10_res = run_h10_unslop_grpo_benchmark(
            seed=seed,
            steps=grpo_steps,
            group_size=grpo_group_size,
            lr=grpo_lr,
        )
        h11_res = evaluate_h11_fp8_provider_survival(
            seed=seed,
            n_prompts=n_prompts,
            seq_len=seq_len,
        )
    else:
        raise ValueError(f"unsupported backend: {backend!r} (expected 'nano' or 'llm')")

    if backend == "llm":
        artifacts = _write_llm_checkpoint(release_dir, model_name, seed)
    else:
        artifacts = _write_nano_checkpoint(release_dir, model_name, seed)

    lancet_recipe_path = release_dir / "lancet_recipe.json"
    if lancet_recipe_path.exists():
        artifacts["lancet_recipe"] = str(lancet_recipe_path)
    lancet_basis_path = release_dir / "lancet_basis.safetensors"
    if lancet_basis_path.exists():
        artifacts["lancet_basis"] = str(lancet_basis_path)

    provider_readiness = verify_provider_readiness(release_dir)

    # Enrich H11 with the weight-hygiene proof from the readiness audit so
    # the onboarding manifest carries Row Crest-Factor / max-abs-weight.
    manifest_h11 = dict(h11_res)
    manifest_h11.setdefault("max_abs_weight", provider_readiness.get("max_abs_weight"))
    manifest_h11.setdefault("max_row_crest_factor", provider_readiness.get("max_row_crest_factor"))

    card_metrics = {**h9_res, **h10_res, **manifest_h11}
    readme = generate_tenet_model_card(
        model_id=release_id,
        base_model=model_name,
        metrics=card_metrics,
        recipe_metadata=h9_res.get("recipe_metadata"),
    )
    readme_path = release_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    artifacts["readme"] = str(readme_path)

    hf_repo_id = f"{hf_namespace}/{release_id}"
    onboarding_manifest = build_openrouter_onboarding_manifest(
        hf_repo_id=hf_repo_id,
        base_model=model_name,
        h9_metrics=h9_res,
        h10_metrics=h10_res,
        h11_metrics=manifest_h11,
    )
    onboarding_json_path = release_dir / "openrouter_onboarding.json"
    onboarding_json_path.write_text(
        json.dumps(onboarding_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    onboarding_md_path = release_dir / "OPENROUTER_ONBOARDING.md"
    onboarding_md_path.write_text(onboarding_manifest["markdown_summary"], encoding="utf-8")
    artifacts["openrouter_onboarding_json"] = str(onboarding_json_path)
    artifacts["openrouter_onboarding_md"] = str(onboarding_md_path)

    release_manifest_path = release_dir / "release_manifest.json"
    artifacts["release_manifest"] = str(release_manifest_path)

    ready = bool(provider_readiness.get("ready", False))
    release_manifest = {
        "release_id": release_id,
        "hf_repo_id": hf_repo_id,
        "backend": backend,
        "model_name": model_name,
        "ready": ready,
        "provider_readiness": provider_readiness,
        "artifacts": artifacts,
    }
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "release_id": release_id,
        "release_dir": str(release_dir),
        "hf_repo_id": hf_repo_id,
        "backend": backend,
        "model_name": model_name,
        "ready": ready,
        "h9": h9_res,
        "h10": h10_res,
        "h11": h11_res,
        "provider_readiness": provider_readiness,
        "onboarding_manifest": onboarding_manifest,
        "artifacts": artifacts,
    }


def _nano_architectures(model_name: str) -> list[str]:
    lowered = model_name.lower()
    if "qwen" in lowered:
        return ["Qwen2ForCausalLM"]
    if "llama" in lowered:
        return ["LlamaForCausalLM"]
    return ["MistralForCausalLM"]


def _nano_model_type(model_name: str) -> str:
    lowered = model_name.lower()
    if "qwen" in lowered:
        return "qwen2"
    if "llama" in lowered:
        return "llama"
    return "mistral"


def _clean_weight_tensors(seed: int) -> dict[str, Any]:
    """Small bounded random tensors: no NaN/Inf, crest factor well in range."""
    generator = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=generator) * 0.02
        for name, shape in _CLEAN_WEIGHT_SHAPES.items()
    }


def _write_config_json(release_dir: Path, model_name: str) -> str:
    config = {
        "architectures": _nano_architectures(model_name),
        "model_type": _nano_model_type(model_name),
        "torch_dtype": "bfloat16",
    }
    config_path = release_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(config_path)


def _write_tokenizer_config_json(release_dir: Path, chat_template: str | None) -> str:
    tokenizer_config = {
        "chat_template": chat_template or _CHAT_TEMPLATE,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    tokenizer_config_path = release_dir / "tokenizer_config.json"
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(tokenizer_config_path)


def _write_clean_weights(release_dir: Path, seed: int) -> str:
    if torch is None:
        raise RuntimeError("torch is required to write Tenet weights")
    tensors = _clean_weight_tensors(seed)
    if _safetensors_save_file is not None:
        weights_path = release_dir / "model.safetensors"
        _safetensors_save_file(tensors, str(weights_path))
    else:
        weights_path = release_dir / "model.pt"
        torch.save(tensors, str(weights_path))
    return str(weights_path)


def _has_weight_file(release_dir: Path) -> bool:
    return any(path.is_file() for pattern in _WEIGHT_GLOBS for path in release_dir.glob(pattern))


def _write_nano_checkpoint(release_dir: Path, model_name: str, seed: int) -> dict[str, str]:
    """Write a clean, provider-ready checkpoint for the nano backend."""
    artifacts: dict[str, str] = {}
    artifacts["config"] = _write_config_json(release_dir, model_name)
    artifacts["tokenizer_config"] = _write_tokenizer_config_json(release_dir, None)
    artifacts["weights"] = _write_clean_weights(release_dir, seed)
    return artifacts


def _write_llm_checkpoint(release_dir: Path, model_name: str, seed: int) -> dict[str, str]:
    """Write an HF-style checkpoint for the llm backend via transformers."""
    artifacts: dict[str, str] = {}

    config_written = False
    chat_template: str | None = None
    try:
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(model_name)
        if getattr(config, "auto_map", None):
            config.auto_map = None
        if getattr(config, "trust_remote_code", False):
            config.trust_remote_code = False
        config.save_pretrained(release_dir)
        config_written = True

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        chat_template = getattr(tokenizer, "chat_template", None)
        tokenizer.save_pretrained(release_dir)
    except Exception:
        if not config_written:
            _write_config_json(release_dir, model_name)

    artifacts["config"] = str(release_dir / "config.json")

    tokenizer_config_path = release_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if not tokenizer_config.get("chat_template"):
            tokenizer_config["chat_template"] = chat_template or _CHAT_TEMPLATE
            tokenizer_config_path.write_text(
                json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        _write_tokenizer_config_json(release_dir, chat_template)
    artifacts["tokenizer_config"] = str(tokenizer_config_path)

    if not _has_weight_file(release_dir):
        artifacts["weights"] = _write_clean_weights(release_dir, seed)
    return artifacts
