import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def compute_bpb(loss_nats: float, bytes_per_token: float) -> float:
    """Compute bits per byte from loss in nats."""
    return loss_nats / (math.log(2.0) * bytes_per_token)


@torch.no_grad()
def evaluate_validation_bpb(
    model: torch.nn.Module,
    dataset: Any,
    num_batches: int,
    batch_size: int,
    bytes_per_token: float,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    """Evaluate model on validation dataset and compute BPB."""
    model.eval()

    # Save dataset state
    original_state = dataset.state_dict()

    total_loss = 0.0
    total_tokens = 0

    try:
        for _ in range(num_batches):
            # Support both ShardedDataset.next_batch and custom get_batch
            batch_loader = getattr(dataset, "next_batch", None) or dataset.get_batch
            x, y, doc_ids = batch_loader(batch_size, device)

            # Forward pass
            out = model(x, targets=y, doc_ids=doc_ids)

            # Accumulate loss and token count
            total_loss += out.loss.item() * y.numel()
            total_tokens += y.numel()

        # Calculate averages
        mean_loss = total_loss / max(1, total_tokens)
        bpb = compute_bpb(mean_loss, bytes_per_token)

        return {"val_loss_nats": mean_loss, "val_bpb": bpb}
    finally:
        # Restore dataset state
        dataset.load_state_dict(original_state)


@torch.no_grad()
def score_multiple_choice_example(
    model: torch.nn.Module,
    prompt_ids: list[int],
    continuations: list[list[int]],
    continuation_byte_lengths: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[int, int]:
    """Score multiple choice example with raw and byte-normalized scores."""
    model.eval()

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    prompt_len = prompt_tensor.size(0)

    raw_scores = []
    byte_normalized_scores = []

    for i, cont_ids in enumerate(continuations):
        # Combine prompt and continuation
        full_seq = torch.cat(
            [prompt_tensor, torch.tensor(cont_ids, dtype=torch.long, device=device)]
        )

        # Get logits for the entire sequence
        logits = model(full_seq.unsqueeze(0)).logits.squeeze(0)

        # Calculate log probabilities for continuation tokens
        cont_logits = logits[prompt_len - 1 : -1]  # -1 because we predict next token
        cont_targets = full_seq[prompt_len:]

        # Compute log probabilities
        log_probs = -F.cross_entropy(cont_logits, cont_targets, reduction="sum")

        raw_scores.append(log_probs.item())

        # Byte-normalized score
        byte_length = continuation_byte_lengths[i] if continuation_byte_lengths else len(cont_ids)
        byte_normalized_scores.append(log_probs.item() / max(1, byte_length))

    # Find best indices
    best_raw_idx = int(torch.argmax(torch.tensor(raw_scores)).item())
    best_byte_norm_idx = int(torch.argmax(torch.tensor(byte_normalized_scores)).item())

    return (best_raw_idx, best_byte_norm_idx)


def bootstrap_confidence_interval(
    scores: Sequence[float], n_resamples: int = 1000, ci: float = 0.95, seed: int = 1337
) -> tuple[float, float, float]:
    """Calculate bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    scores_array = np.array(scores)
    n_scores = len(scores_array)

    if n_scores == 0:
        return (0.0, 0.0, 0.0)

    # Sample means
    sample_means = []
    for _ in range(n_resamples):
        sample = rng.choice(scores_array, size=n_scores, replace=True)
        sample_means.append(np.mean(sample))

    sample_means = np.array(sample_means)

    # Calculate percentiles
    alpha = 1.0 - ci
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1.0 - alpha / 2.0) * 100

    lower_bound = np.percentile(sample_means, lower_percentile)
    upper_bound = np.percentile(sample_means, upper_percentile)
    mean_score = np.mean(scores_array)

    return (float(mean_score), float(lower_bound), float(upper_bound))


@torch.no_grad()
def evaluate_induction_metrics(
    model: torch.nn.Module,
    long_doc_input_ids: torch.Tensor,
    early_pos: int = 8,
    late_pos: int = 24,
    prefix_len: int = 8,
    device: str | torch.device = "cpu",
    seed: int = 1337,
) -> dict[str, float]:
    """Evaluate induction metrics on long document."""
    model.eval()
    torch.manual_seed(seed)

    # Move tensor to device
    long_doc_input_ids = long_doc_input_ids.to(device)
    seq_len = long_doc_input_ids.size(0)

    # Calculate in-context loss difference
    with torch.no_grad():
        logits = model(long_doc_input_ids.unsqueeze(0)).logits.squeeze(0)

        # Loss at early position
        early_start = early_pos
        early_end = min(early_start + prefix_len, seq_len - 1)
        early_loss = F.cross_entropy(
            logits[early_start - 1 : early_end - 1],
            long_doc_input_ids[early_start:early_end],
            reduction="mean",
        )

        # Loss at late position
        late_start = late_pos
        late_end = min(late_start + prefix_len, seq_len - 1)
        late_loss = F.cross_entropy(
            logits[late_start - 1 : late_end - 1],
            long_doc_input_ids[late_start:late_end],
            reduction="mean",
        )

        in_context_loss_delta = float(late_loss - early_loss)

    # Synthetic copy accuracy
    # Create repeated pattern: [p1...pL, p1...pL]
    pattern_len = 16
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 256))
    pattern = torch.randint(0, vocab_size, (pattern_len,), device=device)
    repeated_pattern = torch.cat([pattern, pattern])

    # Get model predictions for the repeated pattern
    with torch.no_grad():
        pattern_logits = model(repeated_pattern.unsqueeze(0)).logits.squeeze(0)
        # Predictions for second half based on first half
        predictions = torch.argmax(pattern_logits[pattern_len - 1 : -1], dim=-1)
        targets = repeated_pattern[pattern_len:]

        # Accuracy of copying
        copy_accuracy = (predictions == targets).float().mean().item()

        # Average probability of correct tokens
        correct_log_probs = -F.cross_entropy(
            pattern_logits[pattern_len - 1 : -1], targets, reduction="none"
        )
        copy_prob = torch.exp(correct_log_probs).mean().item()

    return {
        "in_context_loss_delta": in_context_loss_delta,
        "synthetic_copy_accuracy": copy_accuracy,
        "synthetic_copy_prob": copy_prob,
    }
