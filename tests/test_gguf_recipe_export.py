"""Tests for CKA-guided mixed-precision GGUF recipe export and CLI."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from sml.cli import main as cli_main
from sml.config import ModelConfig
from sml.hub.release import (
    build_cka_quant_recipe,
    build_tenet_release_bundle,
    export_gguf_recipe,
    render_ollama_modelfile,
    render_quantize_cka_gguf_script,
)
from sml.model.transformer import Transformer
from sml.surgery.quantization import (
    build_llama_quantize_command,
    estimate_cka_recipe_avg_bpw,
    evaluate_cka_gguf_kl_vs_uniform,
    generate_llama_cpp_tensor_type_args,
    validate_cka_quant_recipe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TENET_RELEASES_DIR = REPO_ROOT / "results" / "tenet_releases"
RELEASE_IDS = (
    "Tenet-3B-Qwen2.5-Unslop-v1",
    "Tenet-8B-Llama3.1-Unslop-v1",
    "Tenet-12B-Nemo-Unslop-v1",
)


def _sample_recipe() -> dict:
    return build_cka_quant_recipe(
        selected_layers=[1, 2, 4, 5],
        num_hidden_layers=8,
        layer_scr_profile=[
            {"layer": 1, "scr": 0.9},
            {"layer": 2, "scr": 0.8},
            {"layer": 4, "scr": 0.3},
            {"layer": 5, "scr": 0.2},
        ],
    )


def test_tensor_type_args_and_bpw_validation():
    recipe = _sample_recipe()
    args = generate_llama_cpp_tensor_type_args(recipe)
    # Boundary layers 0, 7 + top surgical 1, 2 -> Q8_0; remaining 4, 5 -> Q6_K.
    assert "--tensor-type" in args
    assert "blk.0.attn_output.weight=q8_0" in args
    assert "blk.0.ffn_down.weight=q8_0" in args
    assert "blk.1.attn_output.weight=q8_0" in args
    assert "blk.4.attn_output.weight=q6_k" in args
    assert "blk.5.ffn_down.weight=q6_k" in args
    assert len(args) == 2 * (len(recipe["q8_0_layers"]) + len(recipe["q6_k_layers"])) * 2

    # Q8_0 takes precedence when a layer appears in both q8_0_layers and q6_k_layers.
    overlap = {
        "num_hidden_layers": 4,
        "default_quant": "Q4_K_M",
        "boundary_layers": [0, 3],
        "q8_0_layers": [0, 1, 3],
        "q6_k_layers": [1, 2],
    }
    overlap_args = generate_llama_cpp_tensor_type_args(overlap)
    assert "blk.1.attn_output.weight=q8_0" in overlap_args
    assert "blk.1.attn_output.weight=q6_k" not in overlap_args
    assert "blk.2.attn_output.weight=q6_k" in overlap_args

    bpw = estimate_cka_recipe_avg_bpw(recipe)
    assert abs(bpw - recipe["estimated_avg_bpw"]) < 1e-3

    validation = validate_cka_quant_recipe(recipe)
    assert validation["valid"] is True
    assert validation["errors"] == []

    # Tampered estimated_avg_bpw must fail validation.
    tampered = {**recipe, "estimated_avg_bpw": 4.0}
    bad_validation = validate_cka_quant_recipe(tampered)
    assert bad_validation["valid"] is False
    assert bad_validation["errors"]


def test_build_llama_quantize_command():
    recipe = _sample_recipe()
    cmd = build_llama_quantize_command(
        recipe,
        input_gguf="in-f16.gguf",
        output_gguf="out-cka.gguf",
        llama_quantize_bin="/usr/local/bin/llama-quantize",
        nthreads=8,
    )
    assert cmd[0] == "/usr/local/bin/llama-quantize"
    assert "--tensor-type" in cmd
    assert cmd[-4:] == ["in-f16.gguf", "out-cka.gguf", "Q4_K_M", "8"]


def test_evaluate_cka_gguf_kl_vs_uniform():
    cfg = ModelConfig(
        name="test-gguf",
        vocab_size=64,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=32,
        rope_theta=10000.0,
    )
    model = Transformer(cfg)
    model.eval()
    recipe = build_cka_quant_recipe(selected_layers=[1, 2], num_hidden_layers=4)
    metrics = evaluate_cka_gguf_kl_vs_uniform(model, recipe, seed=2026, n_prompts=2, seq_len=16)

    assert metrics["kl_uniform_q4_k_m"] >= 0.0
    assert metrics["kl_cka_mixed"] >= 0.0
    assert metrics["kl_cka_mixed"] <= metrics["kl_uniform_q4_k_m"]
    assert metrics["cka_beats_uniform"] is True


def test_render_modelfile_and_quantize_script(tmp_path: Path):
    recipe = _sample_recipe()
    modelfile = render_ollama_modelfile(
        "Tenet-Test-v1",
        recipe,
        system_prompt="You are Tenet.",
    )
    assert "FROM ./Tenet-Test-v1-cka-q4_k_m.gguf" in modelfile
    assert "PARAMETER temperature" in modelfile
    assert "SYSTEM" in modelfile

    script = render_quantize_cka_gguf_script("Tenet-Test-v1", recipe)
    script_path = tmp_path / "quantize_cka_gguf.sh"
    script_path.write_text(script, encoding="utf-8")
    syntax = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    assert "--tensor-type" in script
    assert "Q4_K_M" in script
    assert "--imatrix" in script

    # Fake llama-quantize: append every argument it receives to a log file.
    stub_path = tmp_path / "fake_llama_quantize.sh"
    stub_path.write_text(
        "#!/usr/bin/env bash\n"
        'for arg in "$@"; do\n'
        "  printf '%s\\n' \"$arg\" >> \"$QUANTIZE_STUB_LOG\"\n"
        "done\n",
        encoding="utf-8",
    )
    stub_path.chmod(stub_path.stat().st_mode | stat.S_IXUSR)

    input_gguf = tmp_path / "fake-input-f16.gguf"
    input_gguf.write_bytes(b"fake f16 gguf")
    imatrix = tmp_path / "fake-imatrix.gguf"
    imatrix.write_bytes(b"fake imatrix")
    base_env = {
        **os.environ,
        "LLAMA_QUANTIZE_BIN": str(stub_path),
        "INPUT_GGUF": str(input_gguf),
        "OUTPUT_GGUF": str(tmp_path / "fake-output-q4_k_m.gguf"),
    }

    def _run_quantize_script(log_name: str, *cli_args: str) -> tuple[list[str], str]:
        env = {**base_env, "QUANTIZE_STUB_LOG": str(tmp_path / log_name)}
        run = subprocess.run(
            ["bash", str(script_path), *cli_args],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        logged = (tmp_path / log_name).read_text(encoding="utf-8").splitlines()
        return logged, run.stdout

    # Without --imatrix: no imatrix args reach llama-quantize.
    args_no_imatrix, stdout = _run_quantize_script("stub_args_no_imatrix.log")
    assert args_no_imatrix[0] == "--tensor-type"
    assert "--imatrix" not in args_no_imatrix
    assert "" not in args_no_imatrix  # empty IMATRIX_ARGS must expand to nothing
    assert str(input_gguf) in args_no_imatrix
    assert args_no_imatrix[-1] == "Q4_K_M"
    assert "Quantizing" in stdout

    # With --imatrix: it is passed first, ahead of the --tensor-type flags.
    args_with_imatrix, stdout = _run_quantize_script(
        "stub_args_with_imatrix.log", "--imatrix", str(imatrix)
    )
    assert args_with_imatrix[:3] == ["--imatrix", str(imatrix), "--tensor-type"]
    assert str(input_gguf) in args_with_imatrix
    assert args_with_imatrix[-1] == "Q4_K_M"
    assert f"imatrix: {imatrix}" in stdout


def test_build_tenet_release_bundle_writes_gguf_files(tmp_path: Path):
    result = build_tenet_release_bundle(
        output_dir=tmp_path,
        release_id="Tenet-3B-Qwen2.5-Unslop-v1",
        backend="nano",
        grpo_steps=2,
        n_prompts=2,
        seq_len=16,
    )
    release_dir = Path(result["release_dir"])
    modelfile = release_dir / "Modelfile"
    script = release_dir / "quantize_cka_gguf.sh"
    assert modelfile.is_file()
    assert script.is_file()
    assert os.access(script, os.X_OK)

    manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["modelfile"] == str(modelfile)
    assert manifest["artifacts"]["quantize_cka_gguf_sh"] == str(script)


def test_export_gguf_recipe_without_and_with_llama_quantize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = build_tenet_release_bundle(
        output_dir=tmp_path,
        release_id="Tenet-3B-Qwen2.5-Unslop-v1",
        backend="nano",
        grpo_steps=2,
        n_prompts=2,
        seq_len=16,
    )
    release_dir = Path(bundle["release_dir"])

    # Without llama-quantize in PATH -> validates and emits command without building GGUF.
    monkeypatch.setattr("shutil.which", lambda _name: None)
    dry_res = export_gguf_recipe(release_dir)
    assert dry_res["validation"]["valid"] is True
    assert dry_res["llama_quantize_available"] is False
    assert dry_res["gguf_built"] is False
    assert "--tensor-type" in dry_res["llama_quantize_command"]

    # With a mock llama-quantize binary + dummy f16.gguf -> builds GGUF & runs KL check.
    fake_bin = tmp_path / "llama-quantize"
    fake_bin.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "# Positional args are: [..., input_gguf, output_gguf, quant_type]\n"
        "out_path = pathlib.Path(sys.argv[-2])\n"
        "out_path.write_bytes(b'GGUF-MOCK')\n",
        encoding="utf-8",
    )
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IEXEC)

    in_gguf = release_dir / "Tenet-3B-Qwen2.5-Unslop-v1-f16.gguf"
    in_gguf.write_bytes(b"GGUF-F16-MOCK")

    built_res = export_gguf_recipe(release_dir, llama_quantize_bin=str(fake_bin))
    assert built_res["llama_quantize_available"] is True
    assert built_res["gguf_built"] is True
    assert Path(built_res["output_gguf"]).is_file()
    assert Path(built_res["uniform_gguf"]).is_file()
    assert built_res["kl_check"] is not None
    assert built_res["kl_check"]["cka_beats_uniform"] is True


def test_export_gguf_recipe_rejects_tampered_bpw(tmp_path: Path):
    bundle = build_tenet_release_bundle(
        output_dir=tmp_path,
        release_id="Tenet-3B-Qwen2.5-Unslop-v1",
        backend="nano",
        grpo_steps=2,
        n_prompts=2,
        seq_len=16,
    )
    release_dir = Path(bundle["release_dir"])
    recipe_path = release_dir / "cka_quant_recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["estimated_avg_bpw"] = 1.234
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="validation"):
        export_gguf_recipe(release_dir)

    assert cli_main(["export-gguf-recipe", "--release-dir", str(release_dir)]) == 1


def test_cli_export_gguf_recipe_and_build_tenet_integration(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    exit_code = cli_main(
        [
            "build-tenet",
            "--output-dir",
            str(tmp_path),
            "--release-id",
            "Tenet-8B-Llama3.1-Unslop-v1",
            "--grpo-steps",
            "2",
            "--n-prompts",
            "2",
            "--seq-len",
            "16",
            "--export-gguf-recipe",
        ]
    )
    assert exit_code == 0
    build_summary = json.loads(capsys.readouterr().out)
    assert "gguf_recipe" in build_summary
    assert build_summary["gguf_recipe"]["validation"]["valid"] is True

    release_dir = build_summary["release_dir"]
    assert cli_main(["export-gguf-recipe", "--release-dir", release_dir]) == 0
    export_summary = json.loads(capsys.readouterr().out)
    assert export_summary["release_id"] == "Tenet-8B-Llama3.1-Unslop-v1"
    assert Path(export_summary["modelfile_path"]).is_file()
    assert Path(export_summary["quantize_script_path"]).is_file()


@pytest.mark.parametrize("release_id", RELEASE_IDS)
@pytest.mark.skipif(not TENET_RELEASES_DIR.is_dir(), reason="tenet_releases not present")
def test_real_tenet_release_bundles_have_valid_gguf_artifacts(release_id: str):
    release_dir = TENET_RELEASES_DIR / release_id
    if not release_dir.is_dir():
        pytest.skip(f"{release_id} not present")

    recipe = json.loads((release_dir / "cka_quant_recipe.json").read_text(encoding="utf-8"))
    validation = validate_cka_quant_recipe(recipe)
    assert validation["valid"] is True, validation["errors"]

    modelfile = release_dir / "Modelfile"
    script = release_dir / "quantize_cka_gguf.sh"
    assert modelfile.is_file()
    assert script.is_file()
    assert os.access(script, os.X_OK)

    manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    assert "modelfile" in manifest["artifacts"]
    assert "quantize_cka_gguf_sh" in manifest["artifacts"]
