"""End-to-end tests for M4 and M5 pipelines (`prepare-data`, `profile-h2`, and `train`)."""

from __future__ import annotations

import json
from pathlib import Path

from sml.cli import main
from sml.config import DataConfig, ExperimentSpec, HardwareProfile, ModelConfig, TrainConfig
from sml.data import ShardManifest, prepare_dataset_shards
from sml.training import WaterfallStageResult, run_mfu_waterfall, run_pretraining_experiment


def _make_mini_spec(
    experiment_id: str = "test_m4_m5",
    branch_cooldown_tokens: int | None = 64,
) -> ExperimentSpec:
    model_cfg = ModelConfig(
        name="sml-mini-test",
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=64,
    )
    data_cfg = DataConfig(
        name="mini-synthetic",
        hf_dataset="roneneldan/TinyStories",
        hf_subset="default",
        vocab_size=256,
        train_tokens=4096,
        val_tokens=512,
        test_tokens=512,
        shard_size_tokens=1024,
        hf_repo_id="mrAndreyIsachenko/sml-mini-test",
    )
    hw_cfg = HardwareProfile(
        name="local-test",
        accelerator="cpu",
        vram_gb=16.0,
        peak_bf16_tflops=10.0,
        hourly_rate_usd=0.0,
        price_as_of="2026-11-08",
    )
    train_cfg = TrainConfig(
        batch_size_tokens=32,
        micro_batch_size=2,
        seq_len=16,
        max_steps=6,
        lr=2.0e-3,
        min_lr_ratio=0.1,
        weight_decay=0.01,
        grad_clip=1.0,
        schedule="wsd",
        warmup_steps=2,
        decay_fraction=0.33,
        branch_cooldown_tokens=branch_cooldown_tokens,
    )
    return ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id="H3",
        hypothesis="WSD cooldown branching matches standalone schedule.",
        prediction="Branched cooldown achieves comparable val_bpb.",
        acceptance_criterion="branch_val_bpb recorded cleanly.",
        seed=1337,
        model=model_cfg,
        data=data_cfg,
        hardware=hw_cfg,
        train=train_cfg,
        model_config_path="configs/model/sml_nano.yaml",
        data_config_path="configs/data/tinystories.yaml",
        hardware_config_path="configs/hardware/m4_pro_mps.yaml",
    )


def test_prepare_dataset_shards_and_decontamination(tmp_path: Path) -> None:
    docs = [
        (
            f"doc-{i:03d}",
            (
                f"Story {i}: The little language model read clean educational paragraphs and "
                "learned how attention heads route information across residual streams."
            ),
        )
        for i in range(24)
    ]
    contaminated_quote = (
        "When the airflow scheduler failed on upstream schema migration "
        "the pipeline halted immediately"
    )
    docs.append(("doc-contaminated", f"Incident report: {contaminated_quote} during nightly run."))

    data_cfg = DataConfig(
        name="test-decontam-ds",
        hf_dataset="roneneldan/TinyStories",
        hf_subset="default",
        vocab_size=300,
        train_tokens=2048,
        val_tokens=256,
        test_tokens=256,
        shard_size_tokens=512,
        hf_repo_id="mrAndreyIsachenko/test-decontam-ds",
    )

    manifest = prepare_dataset_shards(
        data_config=data_cfg,
        output_dir=tmp_path,
        documents=docs,
        eval_tasks={"piqa_like": [contaminated_quote]},
    )

    assert isinstance(manifest, ShardManifest)
    assert (tmp_path / "tokenizer.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert manifest.decontamination is not None
    assert manifest.decontamination["contaminated_doc_count"] >= 1
    splits = {s.split for s in manifest.shards}
    assert {"train", "val", "test"}.issubset(splits)


def test_run_mfu_waterfall_stages() -> None:
    spec = _make_mini_spec(experiment_id="test_h2_waterfall")
    stages = run_mfu_waterfall(
        spec=spec,
        device="cpu",
        warmup_steps=1,
        measure_steps=1,
        seq_len_override=16,
        micro_batch_override=2,
    )
    assert len(stages) == 5
    assert all(isinstance(s, WaterfallStageResult) for s in stages)
    assert stages[0].stage_name == "1_fp32_naive_eager"
    assert stages[-1].stage_name == "5_bf16_sdpa_compiled"
    assert all(s.tokens_per_sec > 0.0 for s in stages)
    assert stages[0].speedup_vs_baseline == 1.0


def test_run_pretraining_experiment_wsd_branch_and_h4_checkpoints(tmp_path: Path) -> None:
    spec = _make_mini_spec(experiment_id="test_h3_h4_run", branch_cooldown_tokens=64)
    docs = [
        (
            f"train-doc-{i:03d}",
            (
                "Once upon a time a tiny transformer learned induction heads and syntax from "
                "clean stories and structured SQL telemetry logs. " * 3
            ),
        )
        for i in range(30)
    ]
    shard_dir = tmp_path / "shards"
    prepare_dataset_shards(data_config=spec.data, output_dir=shard_dir, documents=docs)

    run_dir = run_pretraining_experiment(
        spec=spec,
        manifest_path=shard_dir / "manifest.json",
        output_dir=tmp_path / "results",
        device="cpu",
        num_checkpoints=3,
        val_batches=1,
    )

    assert (run_dir / "final_metrics.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoints" / "wsd_branch_start.pt").exists()

    final_metrics = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
    assert "val_bpb" in final_metrics
    assert "branch_val_bpb" in final_metrics
    assert "final_in_context_loss_delta" in final_metrics
    assert "final_synthetic_copy_accuracy" in final_metrics
    assert final_metrics["checkpoints_saved"] >= 3


def test_cli_prepare_data_profile_h2_and_train(tmp_path: Path) -> None:
    shard_dir = tmp_path / "cli_shards"
    results_dir = tmp_path / "cli_results"

    rc_prep = main(
        [
            "prepare-data",
            "--data-config",
            "configs/data/tinystories.yaml",
            "--output-dir",
            str(shard_dir),
            "--max-docs",
            "24",
            "--synthetic",
        ]
    )
    assert rc_prep == 0
    assert (shard_dir / "manifest.json").exists()

    rc_h2 = main(
        [
            "profile-h2",
            "--experiment",
            "experiments/h2_mfu_waterfall.yaml",
            "--output-dir",
            str(results_dir),
            "--run-id",
            "cli_h2_test",
            "--device",
            "cpu",
            "--warmup-steps",
            "1",
            "--measure-steps",
            "1",
            "--seq-len",
            "16",
            "--micro-batch-size",
            "1",
        ]
    )
    assert rc_h2 == 0
    assert (results_dir / "cli_h2_test" / "stage_waterfall.json").exists()

    rc_train = main(
        [
            "train",
            "--experiment",
            "experiments/h3_wsd_100m.yaml",
            "--manifest",
            str(shard_dir / "manifest.json"),
            "--output-dir",
            str(results_dir),
            "--run-id",
            "cli_train_test",
            "--device",
            "cpu",
            "--max-steps",
            "3",
            "--seq-len",
            "16",
            "--micro-batch-size",
            "1",
            "--num-checkpoints",
            "2",
        ]
    )
    assert rc_train == 0
    assert (results_dir / "cli_train_test" / "final_metrics.json").exists()
