"""Unit tests for M3: WSD/cosine schedules, grad accumulation, deterministic resume, and evals."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from sml.config import ModelConfig, TrainConfig
from sml.data import ShardedDataset, StreamingShardWriter
from sml.evals import (
    bootstrap_confidence_interval,
    compute_bpb,
    evaluate_induction_metrics,
    evaluate_validation_bpb,
    score_multiple_choice_example,
)
from sml.model import Transformer
from sml.tokenizer import BPETokenizer
from sml.training import (
    Trainer,
    build_log_spaced_steps,
    compute_branched_cooldown_lr,
    compute_lr,
)


def _make_tiny_model_and_config(
    grad_accum_steps: int = 1,
    micro_batch_size: int = 2,
    seq_len: int = 16,
    schedule: str = "wsd",
) -> tuple[Transformer, TrainConfig]:
    model_cfg = ModelConfig(
        name="sml-test-tiny",
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=64,
    )
    train_cfg = TrainConfig(
        batch_size_tokens=micro_batch_size * seq_len * grad_accum_steps,
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        max_steps=1000,
        lr=1.0e-3,
        min_lr_ratio=0.1,
        weight_decay=0.1,
        grad_clip=1.0,
        schedule=schedule,  # type: ignore[arg-type]
        warmup_steps=100,
        decay_fraction=0.1,
    )
    return Transformer(model_cfg), train_cfg


def test_compute_lr_wsd_and_cosine_and_branched_cooldown() -> None:
    """Verify WSD, cosine, branched cooldown schedules, and log-spaced checkpoint steps."""
    _, wsd_cfg = _make_tiny_model_and_config(schedule="wsd")
    assert compute_lr(0, wsd_cfg) == pytest.approx(1e-3 * (1 / 100))
    assert compute_lr(49, wsd_cfg) == pytest.approx(1e-3 * (50 / 100))
    assert compute_lr(99, wsd_cfg) == pytest.approx(1e-3)
    assert compute_lr(500, wsd_cfg) == pytest.approx(1e-3)
    assert compute_lr(900, wsd_cfg) == pytest.approx(1e-3)
    assert compute_lr(950, wsd_cfg) == pytest.approx(1e-3 * 0.55)
    assert compute_lr(1000, wsd_cfg) == pytest.approx(1e-3 * 0.1)

    _, cos_cfg = _make_tiny_model_and_config(schedule="cosine")
    assert compute_lr(0, cos_cfg) == pytest.approx(1e-3 * 0.01)
    assert compute_lr(100, cos_cfg) == pytest.approx(1e-3)
    assert compute_lr(550, cos_cfg) == pytest.approx(1e-3 * 0.55)
    assert compute_lr(1000, cos_cfg) == pytest.approx(1e-3 * 0.1)

    assert compute_branched_cooldown_lr(500, 1000, 100, 1.0) == pytest.approx(1.0)
    assert compute_branched_cooldown_lr(1050, 1000, 100, 1.0) == pytest.approx(0.5)
    assert compute_branched_cooldown_lr(1100, 1000, 100, 1.0) == pytest.approx(0.0)

    steps = build_log_spaced_steps(7630, num_checkpoints=15)
    assert steps[0] == 1
    assert steps[-1] == 7630
    assert len(steps) >= 15
    assert steps == sorted(set(steps))


def test_gradient_accumulation_equivalence() -> None:
    """Verify 4 micro-steps of B=2 match 1 full step of B=8 to within 1e-5."""
    torch.manual_seed(42)
    model_accum, cfg_accum = _make_tiny_model_and_config(
        grad_accum_steps=4, micro_batch_size=2, seq_len=16
    )
    model_full, cfg_full = _make_tiny_model_and_config(
        grad_accum_steps=1, micro_batch_size=8, seq_len=16
    )
    model_full.load_state_dict(model_accum.state_dict())

    micro_batches = [
        (
            torch.randint(0, 256, (2, 16)),
            torch.randint(0, 256, (2, 16)),
            torch.zeros((2, 16), dtype=torch.long),
        )
        for _ in range(4)
    ]
    idx = 0

    def micro_fn(_bsz: int, dev: str | torch.device):
        nonlocal idx
        x, y, d = micro_batches[idx]
        idx += 1
        return x.to(dev), y.to(dev), d.to(dev)

    full_x = torch.cat([b[0] for b in micro_batches], dim=0)
    full_y = torch.cat([b[1] for b in micro_batches], dim=0)
    full_d = torch.cat([b[2] for b in micro_batches], dim=0)

    def full_fn(_bsz: int, dev: str | torch.device):
        return full_x.to(dev), full_y.to(dev), full_d.to(dev)

    trainer_accum = Trainer(model_accum, cfg_accum, device="cpu")
    trainer_full = Trainer(model_full, cfg_full, device="cpu")

    res_accum = trainer_accum.train_step(micro_fn, step=0)
    res_full = trainer_full.train_step(full_fn, step=0)

    assert res_accum["loss"] == pytest.approx(res_full["loss"], abs=1e-5)
    for p1, p2 in zip(model_accum.parameters(), model_full.parameters(), strict=True):
        assert torch.allclose(p1, p2, atol=1e-5)


def test_checkpoint_save_and_deterministic_resume(tmp_path: Path) -> None:
    """Verify resuming from a checkpoint reproduces uninterrupted training loss and weights."""
    corpus = ["Deterministic resume test sentence with multiple words. " * 4] * 15
    tok = BPETokenizer.train_from_iterator(corpus, vocab_size=256, min_frequency=2)
    shard_dir = tmp_path / "shards"
    writer = StreamingShardWriter(
        output_dir=shard_dir,
        dataset_name="resume-test",
        tokenizer=tok,
        shard_size_tokens=200,
    )
    for i in range(20):
        writer.add_document(f"d-{i}", f"Sample training text {i} for resume test.", split="train")
    writer.finalize()
    manifest_path = shard_dir / "manifest.json"

    # Run 1: Uninterrupted 4 steps
    torch.manual_seed(100)
    model_a, cfg_a = _make_tiny_model_and_config(grad_accum_steps=2, micro_batch_size=2, seq_len=8)
    ds_a = ShardedDataset(manifest_path, split="train", seq_len=8)
    trainer_a = Trainer(model_a, cfg_a, device="cpu")

    losses_a = [trainer_a.train_step(ds_a.next_batch, step=s)["loss"] for s in range(4)]

    # Run 2: 2 steps -> save checkpoint -> corrupt state -> load checkpoint -> 2 more steps
    torch.manual_seed(100)
    model_b, cfg_b = _make_tiny_model_and_config(grad_accum_steps=2, micro_batch_size=2, seq_len=8)
    ds_b = ShardedDataset(manifest_path, split="train", seq_len=8)
    trainer_b = Trainer(model_b, cfg_b, device="cpu")

    losses_b = [trainer_b.train_step(ds_b.next_batch, step=s)["loss"] for s in range(2)]
    ckpt_path = tmp_path / "ckpt_step1.pt"
    trainer_b.save_checkpoint(ckpt_path, dataset_state=ds_b.state_dict())

    # Perturb model_b and ds_b before restoring
    _ = trainer_b.train_step(ds_b.next_batch, step=99)

    trainer_b.load_checkpoint(ckpt_path, dataset=ds_b)
    for s in range(2, 4):
        losses_b.append(trainer_b.train_step(ds_b.next_batch, step=s)["loss"])

    assert losses_a == pytest.approx(losses_b, abs=1e-6)
    for pa, pb in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_evals_bpb_multiple_choice_bootstrap_and_induction(tmp_path: Path) -> None:
    """Verify bpb, cursor-preserving val eval, MC scoring, bootstrap CI, and H4 induction."""
    assert compute_bpb(math.log(2.0), bytes_per_token=1.0) == pytest.approx(1.0)

    corpus = ["Evaluation harness validation text sample. " * 4] * 15
    tok = BPETokenizer.train_from_iterator(corpus, vocab_size=256, min_frequency=2)
    shard_dir = tmp_path / "val_shards"
    writer = StreamingShardWriter(
        output_dir=shard_dir,
        dataset_name="val-test",
        tokenizer=tok,
        shard_size_tokens=200,
    )
    for i in range(10):
        writer.add_document(f"v-{i}", f"Validation sample {i} text.", split="val")
    manifest = writer.finalize()

    ds_val = ShardedDataset(shard_dir / "manifest.json", split="val", seq_len=8)
    _ = ds_val.next_batch(batch_size=2)
    cursor_before = ds_val.state_dict()

    model, _ = _make_tiny_model_and_config()
    val_metrics = evaluate_validation_bpb(
        model=model,
        dataset=ds_val,
        num_batches=2,
        batch_size=2,
        bytes_per_token=manifest.bytes_per_token_by_split["val"],
    )
    assert ds_val.state_dict() == cursor_before
    assert val_metrics["val_bpb"] > 0.0

    raw_idx, norm_idx = score_multiple_choice_example(
        model=model,
        prompt_ids=[1, 2, 3, 4],
        continuations=[[5, 6], [7, 8, 9, 10]],
        continuation_byte_lengths=[4, 12],
    )
    assert raw_idx in {0, 1}
    assert norm_idx in {0, 1}

    mean, low, high = bootstrap_confidence_interval([0.0, 1.0, 1.0, 0.0, 1.0], n_resamples=200)
    assert low <= mean <= high

    long_doc = torch.randint(0, 256, (48,), dtype=torch.long)
    ind = evaluate_induction_metrics(
        model=model,
        long_doc_input_ids=long_doc,
        early_pos=4,
        late_pos=20,
        prefix_len=6,
    )
    assert "in_context_loss_delta" in ind
    assert 0.0 <= ind["synthetic_copy_accuracy"] <= 1.0
    assert 0.0 <= ind["synthetic_copy_prob"] <= 1.0
