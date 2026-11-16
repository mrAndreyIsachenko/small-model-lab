"""Pretraining runner with WSD cooldown branching (`H3`) and log-spaced checkpoints (`H4`)."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import torch

from sml.config import ExperimentSpec, ModelConfig, TrainConfig
from sml.data.sharding import ShardedDataset, ShardManifest
from sml.evals.harness import evaluate_induction_metrics, evaluate_validation_bpb
from sml.model.transformer import Transformer
from sml.tracking.manifest import RunRecorder
from sml.training.profiler import resolve_device
from sml.training.trainer import Trainer, build_log_spaced_steps, compute_branched_cooldown_lr


def run_pretraining_experiment(
    spec: ExperimentSpec,
    manifest_path: str | Path,
    output_dir: str | Path = "results",
    run_id: str | None = None,
    device: str = "auto",
    max_steps_override: int | None = None,
    seq_len_override: int | None = None,
    micro_batch_override: int | None = None,
    num_checkpoints: int = 5,
    val_batches: int = 2,
) -> Path:
    """Execute WSD pretraining with optional cooldown branching and induction tracking."""
    dev = resolve_device(device)
    manifest = ShardManifest.load(manifest_path)

    seq_len = min(seq_len_override or spec.train.seq_len, spec.model.max_seq_len)
    micro_batch = micro_batch_override or spec.train.micro_batch_size
    max_steps = max_steps_override or spec.train.max_steps

    adjusted_model: ModelConfig = replace(
        spec.model,
        vocab_size=manifest.vocab_size,
        max_seq_len=max(spec.model.max_seq_len, seq_len),
    )

    if (
        seq_len_override is not None
        or micro_batch_override is not None
        or max_steps_override is not None
    ):
        orig_accum = max(
            1,
            spec.train.batch_size_tokens
            // max(1, spec.train.micro_batch_size * spec.train.seq_len),
        )
        # Keep local/override runs nimble by capping grad_accum_steps when overrides are supplied
        effective_accum = min(orig_accum, 2)
        adjusted_batch_tokens = micro_batch * seq_len * effective_accum
    else:
        adjusted_batch_tokens = spec.train.batch_size_tokens

    adjusted_train: TrainConfig = replace(
        spec.train,
        batch_size_tokens=adjusted_batch_tokens,
        micro_batch_size=micro_batch,
        seq_len=seq_len,
        max_steps=max_steps,
        warmup_steps=min(spec.train.warmup_steps, max(1, max_steps // 4)),
    )

    train_ds = ShardedDataset(manifest_path, split="train", seq_len=seq_len)
    val_ds = ShardedDataset(manifest_path, split="val", seq_len=seq_len)

    recorder = RunRecorder(
        run_id=run_id or spec.experiment_id,
        spec=spec,
        output_root=Path(output_dir),
        mode="measured",
    )

    torch.manual_seed(spec.seed)
    model = Transformer(adjusted_model)
    trainer = Trainer(model, adjusted_train, device=dev)

    log_steps = set(build_log_spaced_steps(max_steps, num_checkpoints=num_checkpoints))
    val_bpt = max(manifest.bytes_per_token_by_split.get("val", 4.0), 1e-3)

    sample_x, _, _ = val_ds.next_batch(1, device=dev)
    long_doc_input_ids = sample_x[0]

    branch_start_step: int | None = None
    branch_ckpt: Path | None = None
    if spec.train.branch_cooldown_tokens is not None and max_steps >= 2:
        raw_branch = spec.train.branch_cooldown_tokens // adjusted_train.batch_size_tokens
        if raw_branch <= 0 or raw_branch >= max_steps:
            branch_start_step = max(1, max_steps // 2)
        else:
            branch_start_step = raw_branch

    initial_val = evaluate_validation_bpb(
        model,
        val_ds,
        num_batches=val_batches,
        batch_size=micro_batch,
        bytes_per_token=val_bpt,
        device=dev,
    )
    initial_val_bpb = float(initial_val["val_bpb"])
    last_val_bpb = initial_val_bpb
    last_val_loss = float(initial_val["val_loss_nats"])
    last_induction: dict[str, float] = {
        "in_context_loss_delta": 0.0,
        "synthetic_copy_accuracy": 0.0,
        "synthetic_copy_prob": 0.0,
    }
    checkpoints_saved = 0
    induction_history: list[dict[str, float | int]] = []

    t0 = time.perf_counter()
    for step_idx in range(1, max_steps + 1):
        step_metrics = trainer.train_step(train_ds.next_batch, step=step_idx - 1)

        if branch_start_step is not None and step_idx == branch_start_step:
            branch_ckpt = recorder.run_dir / "checkpoints" / "wsd_branch_start.pt"
            trainer.save_checkpoint(branch_ckpt, dataset_state=train_ds.state_dict())

        if step_idx in log_steps or step_idx == max_steps:
            val_res = evaluate_validation_bpb(
                model,
                val_ds,
                num_batches=val_batches,
                batch_size=micro_batch,
                bytes_per_token=val_bpt,
                device=dev,
            )
            last_val_bpb = float(val_res["val_bpb"])
            last_val_loss = float(val_res["val_loss_nats"])

            early_pos = max(2, seq_len // 8)
            late_pos = max(early_pos + 2, seq_len // 2)
            prefix_len = max(2, seq_len // 8)
            last_induction = evaluate_induction_metrics(
                model,
                long_doc_input_ids,
                early_pos=early_pos,
                late_pos=late_pos,
                prefix_len=prefix_len,
                device=dev,
                seed=spec.seed,
            )

            ckpt_path = recorder.run_dir / "checkpoints" / f"step_{step_idx:06d}.pt"
            trainer.save_checkpoint(ckpt_path, dataset_state=train_ds.state_dict())
            checkpoints_saved += 1

            recorder.log_step(
                step=step_idx,
                metrics={
                    "train_loss": round(step_metrics["loss"], 6),
                    "lr": round(step_metrics["lr"], 8),
                    "grad_norm": round(step_metrics["grad_norm"], 4),
                    "tokens_processed": int(step_metrics["tokens_processed"]),
                    "val_loss_nats": round(last_val_loss, 6),
                    "val_bpb": round(last_val_bpb, 6),
                    "in_context_loss_delta": round(last_induction["in_context_loss_delta"], 6),
                    "synthetic_copy_accuracy": round(last_induction["synthetic_copy_accuracy"], 6),
                    "synthetic_copy_prob": round(last_induction["synthetic_copy_prob"], 6),
                },
            )

            induction_history.append(
                {
                    "step": int(step_idx),
                    "tokens_processed": int(step_metrics["tokens_processed"]),
                    "val_bpb": round(last_val_bpb, 6),
                    "in_context_loss_delta": round(last_induction["in_context_loss_delta"], 6),
                    "synthetic_copy_accuracy": round(last_induction["synthetic_copy_accuracy"], 6),
                    "synthetic_copy_prob": round(last_induction["synthetic_copy_prob"], 6),
                }
            )

    branch_val_bpb: float | None = None
    if branch_ckpt is not None and branch_ckpt.exists() and branch_start_step is not None:
        branch_model = Transformer(adjusted_model)
        branch_trainer = Trainer(branch_model, adjusted_train, device=dev)
        branch_trainer.load_checkpoint(branch_ckpt, dataset=train_ds)

        cooldown_steps = max(1, int(round(branch_start_step * adjusted_train.decay_fraction)))
        for idx in range(cooldown_steps):
            lr_override = compute_branched_cooldown_lr(
                step=branch_start_step + idx,
                branch_start_step=branch_start_step,
                cooldown_steps=cooldown_steps,
                base_lr=adjusted_train.lr,
                min_lr_ratio=adjusted_train.min_lr_ratio,
            )
            branch_trainer.train_step(
                train_ds.next_batch,
                step=branch_start_step + idx,
                lr_override=lr_override,
            )

        branch_eval = evaluate_validation_bpb(
            branch_model,
            val_ds,
            num_batches=val_batches,
            batch_size=micro_batch,
            bytes_per_token=val_bpt,
            device=dev,
        )
        branch_val_bpb = round(float(branch_eval["val_bpb"]), 6)

    elapsed = max(time.perf_counter() - t0, 1e-6)
    tokens_per_sec = trainer.tokens_processed / elapsed
    mfu = (tokens_per_sec * adjusted_model.flops_per_token(seq_len=seq_len)) / (
        spec.hardware.peak_bf16_tflops * 1e12
    )

    final_metrics: dict[str, float | int | bool | str] = {
        "initial_val_bpb": round(initial_val_bpb, 6),
        "val_bpb": round(last_val_bpb, 6),
        "val_loss_nats": round(last_val_loss, 6),
        "final_in_context_loss_delta": round(last_induction["in_context_loss_delta"], 6),
        "final_synthetic_copy_accuracy": round(last_induction["synthetic_copy_accuracy"], 6),
        "final_synthetic_copy_prob": round(last_induction["synthetic_copy_prob"], 6),
        "tokens_processed": int(trainer.tokens_processed),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "mfu": round(mfu, 6),
        "checkpoints_saved": checkpoints_saved,
    }
    if branch_val_bpb is not None:
        final_metrics["branch_val_bpb"] = branch_val_bpb
        final_metrics["branch_start_step"] = int(branch_start_step or 0)

    max_window_gain_share = 0.0
    if len(induction_history) >= 2:
        copy_accs = [entry["synthetic_copy_accuracy"] for entry in induction_history]
        total_copy_gain = copy_accs[-1] - copy_accs[0]
        if total_copy_gain > 0:
            max_window_gain_share = max(
                (next_acc - prev_acc) / total_copy_gain
                for prev_acc, next_acc in zip(copy_accs, copy_accs[1:], strict=False)
            )
        else:
            induction_gains = [abs(entry["in_context_loss_delta"]) for entry in induction_history]
            total_delta_gain = induction_gains[-1] - induction_gains[0]
            if total_delta_gain > 0:
                max_window_gain_share = max(
                    (next_gain - prev_gain) / total_delta_gain
                    for prev_gain, next_gain in zip(
                        induction_gains, induction_gains[1:], strict=False
                    )
                )

    final_metrics["max_window_gain_share"] = round(max_window_gain_share, 6)
    final_metrics["delta_val_bpb"] = round(
        last_val_bpb - (branch_val_bpb if branch_val_bpb is not None else initial_val_bpb),
        6,
    )

    if spec.hypothesis_id == "H3" and checkpoints_saved >= 2:
        effective_run_id = run_id or spec.experiment_id
        h4_run_id = (
            "h4_induction_dynamics"
            if effective_run_id == "h3_wsd_100m"
            else f"{effective_run_id}_h4"
        )
        h4_spec = replace(
            spec,
            experiment_id="h4_induction_dynamics",
            hypothesis_id="H4",
            hypothesis=(
                "In-context induction capability emerges via a sharp early phase transition "
                "across log-spaced checkpoints."
            ),
            prediction=(
                "> 60% of the total improvement in synthetic copying accuracy and intra-document "
                "in-context loss delta occurs within a narrow window spanning < 15% of total "
                "training steps."
            ),
            acceptance_criterion="max_window_gain_share >= 0.60 and checkpoints_saved >= 15",
        )
        h4_recorder = RunRecorder(
            run_id=h4_run_id,
            spec=h4_spec,
            output_root=Path(output_dir),
            mode="measured",
        )
        for entry in induction_history:
            h4_recorder.log_step(
                step=int(entry["step"]),
                metrics={
                    "tokens_processed": entry["tokens_processed"],
                    "val_bpb": entry["val_bpb"],
                    "in_context_loss_delta": entry["in_context_loss_delta"],
                    "synthetic_copy_accuracy": entry["synthetic_copy_accuracy"],
                    "synthetic_copy_prob": entry["synthetic_copy_prob"],
                },
            )
        h4_recorder.finalize(
            final_metrics=final_metrics,
            active_seconds=elapsed,
            billed_seconds=0.0,
            status="completed",
            conclusion=(
                f"Companion H4 induction-dynamics bundle for {effective_run_id}: "
                f"max_window_gain_share={max_window_gain_share:.6f} across "
                f"{len(induction_history)} log-spaced checkpoints "
                f"(checkpoints_saved={checkpoints_saved}), "
                f"delta_val_bpb={final_metrics['delta_val_bpb']:.6f}."
            ),
        )

    return recorder.finalize(
        final_metrics=final_metrics,
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed",
        conclusion=(
            f"Completed WSD pretraining for {spec.experiment_id}: "
            f"val_bpb={last_val_bpb:.4f} (initial={initial_val_bpb:.4f}), "
            f"branch_val_bpb={branch_val_bpb}, checkpoints_saved={checkpoints_saved}."
        ),
    )
