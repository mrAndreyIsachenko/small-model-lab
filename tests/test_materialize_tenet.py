"""End-to-end tests for ADR-006 Tier 3 Tenet checkpoint materialization."""

import json
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from sml.config import load_model_config
from sml.hub.release import (
    cleanup_hf_model_cache,
    materialize_tenet_checkpoint,
    verify_provider_readiness,
    verify_recipe_provenance,
)
from sml.model.transformer import Transformer
from sml.surgery.subspace import build_lancet_recipe, save_lancet_recipe

RELEASE_ID = "Tenet-Nano-Test-v1"
HF_NAMESPACE = "mr-andrey-isachenko"
REPO_ROOT = Path(__file__).resolve().parents[1]
NANO_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "sml_nano.yaml"

READINESS_CHECKS = (
    "recipe_provenance",
    "no_nan_inf",
    "no_unmerged_lora",
    "weight_outliers_ok",
    "acceptable_row_crest_factor",
    "standard_architecture",
    "no_trust_remote_code",
    "valid_chat_template",
)

LORA_MODULES = ("layers.0.self_attn.o_proj", "layers.0.mlp.down_proj")
BASIS_LAYER = 0


def _build_nano_recipe(output_dir: Path) -> tuple[torch.Tensor, object, Path]:
    """Load the nano config, build a 2-layer Lancet recipe, and save it."""
    cfg = load_model_config(NANO_CONFIG_PATH)
    torch.manual_seed(2026)
    model = Transformer(cfg)

    torch.manual_seed(7)
    target_dirs: dict[int, list[torch.Tensor]] = {}
    for layer_idx in (0, 1):
        raw = torch.randn(4, cfg.d_model)
        q, _ = torch.linalg.qr(raw.T)  # columns of q are orthonormal
        target_dirs[layer_idx] = [q[:, i].contiguous() for i in range(4)]

    anchor_input_ids = torch.zeros((2, 8), dtype=torch.long)
    recipe = build_lancet_recipe(
        model,
        target_dirs,
        None,
        anchor_input_ids,
        model_id=RELEASE_ID,
        base_model="sml-nano",
    )
    assert BASIS_LAYER in recipe.basis_per_layer
    recipe_dir = save_lancet_recipe(recipe, output_dir / "recipe")
    return cfg, recipe, recipe_dir


def _seeded_base_state(cfg: torch.Tensor) -> dict[str, torch.Tensor]:
    """State dict of the seeded base nano Transformer used by materialize_tenet_checkpoint."""
    torch.manual_seed(2026)
    model = Transformer(cfg)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def _write_lora_adapter(
    path: Path,
    base_state: dict[str, torch.Tensor],
    basis: torch.Tensor,
    magnitude: float,
    seed: int,
) -> Path:
    """Write a synthetic SML-format LoRA adapter with deliberate basis components."""
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for module_name in LORA_MODULES:
        out_features, in_features = base_state[f"{module_name}.weight"].shape
        lora_a = torch.randn(8, in_features, generator=generator)
        lora_b = 0.01 * torch.randn(out_features, 8, generator=generator)
        num_basis = min(8, basis.shape[0])
        # Rows of lora_b seed the delta with components along the basis.
        lora_b[:, :num_basis] = magnitude * basis[:num_basis].T
        tensors[f"{module_name}.lora_a"] = lora_a.contiguous()
        tensors[f"{module_name}.lora_b"] = lora_b.contiguous()
    save_file(tensors, str(path))
    return path


def _assert_projection_near_zero(
    weights: dict[str, torch.Tensor], basis: torch.Tensor, threshold: float = 0.05
) -> None:
    """Forbidden-subspace projection of target rows must be near zero."""
    basis_f = basis.to(torch.float32)
    for module_name in LORA_MODULES:
        weight = weights[f"{module_name}.weight"].to(torch.float32)
        projection = basis_f @ weight
        max_coeff = projection.abs().max().item()
        assert max_coeff < threshold, f"{module_name}: projection {max_coeff:.4f} >= {threshold}"


def _assert_readiness(result: dict) -> None:
    assert result["ready"] is True
    readiness = result["provider_readiness"]
    assert readiness["ready"] is True
    assert set(readiness["checks"]) == set(READINESS_CHECKS)
    for name, ok in readiness["checks"].items():
        assert ok is True, f"readiness check failed: {name}"
    provenance = result["recipe_provenance"]
    assert provenance["valid"] is True


@pytest.fixture(scope="module")
def nano_recipe(tmp_path_factory: pytest.TempPathFactory):
    cfg, recipe, recipe_dir = _build_nano_recipe(tmp_path_factory.mktemp("tenet_recipe"))
    base_state = _seeded_base_state(cfg)
    return cfg, recipe, recipe_dir, base_state


def test_materialize_tenet_nano_end_to_end(tmp_path: Path) -> None:
    cfg, recipe, recipe_dir = _build_nano_recipe(tmp_path)
    result = materialize_tenet_checkpoint(
        release_id=RELEASE_ID,
        recipe_dir=recipe_dir,
        output_dir=tmp_path / "mat",
        backend="nano",
        device="cpu",
    )
    _assert_readiness(result)

    target_dir = tmp_path / "mat" / RELEASE_ID
    weights = load_file(str(target_dir / "model.safetensors"))
    basis = recipe.basis_per_layer[BASIS_LAYER]
    _assert_projection_near_zero(weights, basis)

    base_state = _seeded_base_state(cfg)
    for module_name in LORA_MODULES:
        weight = weights[f"{module_name}.weight"].to(torch.float32)
        base_weight = base_state[f"{module_name}.weight"].to(torch.float32)
        assert torch.allclose(weight.norm(dim=1), base_weight.norm(dim=1), rtol=0.05, atol=0.05), (
            f"{module_name}: row norms drifted after Lancet surgery"
        )


def test_materialize_tenet_nano_with_lora_and_ties_merge(tmp_path: Path, nano_recipe) -> None:
    cfg, recipe, recipe_dir, base_state = nano_recipe
    basis = recipe.basis_per_layer[BASIS_LAYER]
    unslop_path = _write_lora_adapter(
        tmp_path / "adapter_unslop.safetensors", base_state, basis, magnitude=2.0, seed=11
    )
    sql_path = _write_lora_adapter(
        tmp_path / "adapter_sql.safetensors", base_state, basis, magnitude=1.0, seed=23
    )

    solo = materialize_tenet_checkpoint(
        release_id=RELEASE_ID,
        recipe_dir=recipe_dir,
        output_dir=tmp_path / "solo",
        backend="nano",
        device="cpu",
        lora_path=unslop_path,
    )
    assert solo["lora_applied"] is True
    _assert_readiness(solo)
    solo_weights = load_file(str(tmp_path / "solo" / RELEASE_ID / "model.safetensors"))
    _assert_projection_near_zero(solo_weights, basis)
    assert verify_provider_readiness(tmp_path / "solo" / RELEASE_ID)["ready"] is True
    assert verify_recipe_provenance(tmp_path / "solo" / RELEASE_ID)["valid"] is True

    ties = materialize_tenet_checkpoint(
        release_id=RELEASE_ID,
        recipe_dir=recipe_dir,
        output_dir=tmp_path / "ties",
        backend="nano",
        device="cpu",
        lora_path=unslop_path,
        sql_lora_path=sql_path,
    )
    assert ties["lora_applied"] is True
    _assert_readiness(ties)
    ties_weights = load_file(str(tmp_path / "ties" / RELEASE_ID / "model.safetensors"))
    _assert_projection_near_zero(ties_weights, basis)
    assert verify_provider_readiness(tmp_path / "ties" / RELEASE_ID)["ready"] is True
    assert verify_recipe_provenance(tmp_path / "ties" / RELEASE_ID)["valid"] is True


def test_materialize_tenet_nano_fp8_export(tmp_path: Path, nano_recipe) -> None:
    _, recipe, recipe_dir, _ = nano_recipe
    result = materialize_tenet_checkpoint(
        release_id=RELEASE_ID,
        recipe_dir=recipe_dir,
        output_dir=tmp_path / "fp8",
        backend="nano",
        device="cpu",
        fp8_export=True,
    )
    assert result["fp8_export"] is True
    _assert_readiness(result)
    assert verify_provider_readiness(tmp_path / "fp8" / RELEASE_ID)["ready"] is True
    assert verify_recipe_provenance(tmp_path / "fp8" / RELEASE_ID)["valid"] is True


def test_cleanup_hf_model_cache_and_push_to_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nano_recipe
) -> None:
    _, recipe, recipe_dir, _ = nano_recipe

    hub_cache = tmp_path / "hub_cache"
    qwen_dir = hub_cache / "models--Qwen--Qwen2.5-3B-Instruct"
    (qwen_dir / "snapshots" / "abc123").mkdir(parents=True)
    (qwen_dir / "blobs").mkdir()
    (qwen_dir / "blobs" / "def456").write_bytes(b"0" * 4096)
    (qwen_dir / "snapshots" / "abc123" / "config.json").write_text("{}")
    other_dir = hub_cache / "models--Other--KeepMe"
    other_dir.mkdir(parents=True)
    (other_dir / "keep.bin").write_bytes(b"1" * 512)

    info = cleanup_hf_model_cache("Qwen/Qwen2.5-3B-Instruct", cache_dir=hub_cache)
    assert info["removed"] is True
    assert info["freed_bytes"] > 0
    assert not qwen_dir.exists()
    assert (other_dir / "keep.bin").is_file()

    calls: list[tuple[str, str]] = []

    class _FakeHfApi:
        def create_repo(self, **kwargs):
            calls.append(("create_repo", kwargs["repo_id"]))
            return "https://fake-hub"

        def upload_folder(self, **kwargs):
            calls.append(("upload_folder", kwargs["repo_id"]))
            return None

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeHfApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = materialize_tenet_checkpoint(
        release_id=RELEASE_ID,
        recipe_dir=recipe_dir,
        output_dir=tmp_path / "push",
        backend="nano",
        device="cpu",
        push_to_hub=True,
        keep_local_weights=False,
        hf_namespace=HF_NAMESPACE,
    )
    assert result["pushed"] is True
    expected_repo = f"{HF_NAMESPACE}/{RELEASE_ID}"
    assert ("create_repo", expected_repo) in calls
    assert ("upload_folder", expected_repo) in calls
    target_dir = tmp_path / "push" / RELEASE_ID
    assert list(target_dir.glob("model*.safetensors")) == []


def test_materialize_tenet_copies_release_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nano_recipe
) -> None:
    """Release artifacts staged in the recipe dir must land in the checkpoint and upload."""
    _, _, recipe_dir, _ = nano_recipe

    recipe_copy = tmp_path / "recipe_copy"
    shutil.copytree(recipe_dir, recipe_copy)
    artifacts = {
        "release_manifest.json": json.dumps({"release_id": RELEASE_ID}, indent=2) + "\n",
        "Modelfile": "FROM ./model.safetensors\n",
        "quantize_cka_gguf.sh": "#!/usr/bin/env bash\nset -euo pipefail\n",
    }
    for name, content in artifacts.items():
        (recipe_copy / name).write_text(content, encoding="utf-8")

    upload_snapshots: list[set[str]] = []

    class _FakeHfApi:
        def create_repo(self, **kwargs):
            return "https://fake-hub"

        def upload_folder(self, **kwargs):
            folder = Path(kwargs["folder_path"])
            upload_snapshots.append(
                {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            )
            return None

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeHfApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = materialize_tenet_checkpoint(
        release_id=RELEASE_ID,
        recipe_dir=recipe_copy,
        output_dir=tmp_path / "artifacts",
        backend="nano",
        device="cpu",
        push_to_hub=True,
        hf_namespace=HF_NAMESPACE,
    )
    assert result["pushed"] is True
    _assert_readiness(result)

    target_dir = tmp_path / "artifacts" / RELEASE_ID
    for name, content in artifacts.items():
        copied = target_dir / name
        assert copied.is_file(), f"missing release artifact in target_dir: {name}"
        assert copied.read_text(encoding="utf-8") == content, f"content mismatch: {name}"

    assert len(upload_snapshots) == 1
    for name in artifacts:
        assert name in upload_snapshots[0], f"{name} absent from upload_folder snapshot"


def test_cli_materialize_tenet_nano(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], nano_recipe
) -> None:
    from sml.cli import main

    _, _, recipe_dir, _ = nano_recipe
    return_code = main(
        [
            "materialize-tenet",
            "--release-id",
            RELEASE_ID,
            "--recipe-dir",
            str(recipe_dir),
            "--output-dir",
            str(tmp_path / "cli_out"),
            "--backend",
            "nano",
            "--device",
            "cpu",
        ]
    )
    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is True
