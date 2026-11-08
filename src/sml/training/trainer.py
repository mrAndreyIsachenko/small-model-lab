"""Trainer loop, WSD/cosine schedules, cooldown branching, and deterministic checkpointing."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from sml.config import TrainConfig


def compute_lr(step: int, config: TrainConfig) -> float:
    """Compute learning rate for step under Warmup-Stable-Decay (WSD) or Cosine schedule."""
    if step < config.warmup_steps:
        return config.lr * ((step + 1) / max(1, config.warmup_steps))

    if config.schedule == "wsd":
        decay_steps = max(1, int(config.max_steps * config.decay_fraction))
        decay_start = config.max_steps - decay_steps
        if step < decay_start:
            return config.lr
        decay_progress = min(1.0, max(0.0, (step - decay_start) / decay_steps))
        return config.lr * (1.0 - decay_progress * (1.0 - config.min_lr_ratio))

    if config.schedule == "cosine":
        progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.lr * (config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine_factor)

    return config.lr


def compute_branched_cooldown_lr(
    step: int,
    branch_start_step: int,
    cooldown_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.0,
) -> float:
    """Compute linear cooldown LR for an intermediate WSD branch (`H3`)."""
    if step < branch_start_step:
        return base_lr
    cooldown_progress = (step - branch_start_step) / max(1, cooldown_steps)
    if cooldown_progress >= 1.0:
        return base_lr * min_lr_ratio
    return base_lr * (1.0 - cooldown_progress * (1.0 - min_lr_ratio))


def build_log_spaced_steps(max_steps: int, num_checkpoints: int = 15) -> list[int]:
    """Build logarithmically spaced checkpoint step indices in `[1, max_steps]` (`H4`)."""
    if max_steps <= 0 or num_checkpoints <= 0:
        return []
    if num_checkpoints >= max_steps:
        return list(range(1, max_steps + 1))

    steps = {1, max_steps}
    if num_checkpoints > 2:
        raw = np.geomspace(1, max_steps, num=num_checkpoints * 2)
        for val in raw:
            steps.add(int(round(float(val))))
            if len(steps) >= num_checkpoints:
                break

    # If rounding collapsed steps below num_checkpoints, fill gaps deterministically
    candidate = 2
    while len(steps) < num_checkpoints and candidate < max_steps:
        steps.add(candidate)
        candidate += 1

    sorted_steps = sorted(s for s in steps if 1 <= s <= max_steps)
    return sorted_steps


class Trainer:
    """Pretraining loop with gradient accumulation, bf16 autocast, and deterministic resume."""

    def __init__(
        self,
        model: nn.Module,
        train_config: TrainConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = train_config
        self.step = 0
        self.tokens_processed = 0

        decay_params = []
        nodecay_params = []
        for _name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": train_config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=train_config.lr,
            betas=(train_config.beta1, train_config.beta2),
        )

    def train_step(
        self,
        batch_fn: Callable[
            [int, str | torch.device], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        step: int,
        lr_override: float | None = None,
    ) -> dict[str, float]:
        """Execute one optimization step over `grad_accum_steps` micro-batches."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        accum_steps = self.config.grad_accum_steps

        for _ in range(accum_steps):
            x, y, doc_ids = batch_fn(self.config.micro_batch_size, self.device)
            out = self.model(x, targets=y, doc_ids=doc_ids)
            loss = out.loss / accum_steps
            total_loss += float(loss.item())
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        lr = lr_override if lr_override is not None else compute_lr(step, self.config)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.optimizer.step()
        self.step = step
        self.tokens_processed += self.config.batch_size_tokens

        return {
            "loss": total_loss,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "tokens_processed": float(self.tokens_processed),
        }

    def save_checkpoint(
        self, path: str | Path, dataset_state: dict[str, Any] | None = None
    ) -> Path:
        """Save model, optimizer, step counters, dataloader cursor, and CPU RNG state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "tokens_processed": self.tokens_processed,
            "dataset_state": dataset_state,
            "rng_state": torch.get_rng_state(),
        }
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str | Path, dataset: Any | None = None) -> dict[str, Any]:
        """Restore model, optimizer, step counters, dataloader cursor, and CPU RNG state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = int(checkpoint["step"])
        self.tokens_processed = int(checkpoint["tokens_processed"])

        if dataset is not None and checkpoint.get("dataset_state") is not None:
            dataset.load_state_dict(checkpoint["dataset_state"])

        if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
            rng_state = checkpoint["rng_state"]
            if isinstance(rng_state, torch.Tensor):
                torch.set_rng_state(rng_state.cpu())

        return checkpoint
