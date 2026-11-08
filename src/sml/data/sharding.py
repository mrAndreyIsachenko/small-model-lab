import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from sml.model.attention import compute_doc_ids_from_eos

SplitName = Literal["train", "val", "test", "tokenizer_fit"]


def assign_split_by_doc_id(
    doc_id: str,
    val_ratio: float = 0.002,
    test_ratio: float = 0.002,
    tokenizer_fit_ratio: float = 0.02,
) -> SplitName:
    """Assign a document to a split based on its ID hash."""
    # Hash the doc_id and convert first 8 bytes to a float in [0.0, 1.0)
    hash_bytes = hashlib.sha256(doc_id.encode("utf-8")).digest()[:8]
    hash_value = int.from_bytes(hash_bytes, byteorder="big") / (2**64)

    # Assign to splits based on cumulative probabilities
    if hash_value < tokenizer_fit_ratio:
        return "tokenizer_fit"
    elif hash_value < tokenizer_fit_ratio + val_ratio:
        return "val"
    elif hash_value < tokenizer_fit_ratio + val_ratio + test_ratio:
        return "test"
    else:
        return "train"


@dataclass(frozen=True)
class ShardInfo:
    filename: str
    split: SplitName
    num_tokens: int
    num_bytes_utf8: int
    sha256: str


@dataclass(frozen=True)
class ShardManifest:
    dataset_name: str
    vocab_size: int
    eos_token_id: int
    tokenizer_sha256: str
    bytes_per_token_by_split: dict[str, float]
    shards: list[ShardInfo]
    decontamination: dict[str, Any] | None = None

    def save(self, path: str | Path) -> Path:
        """Save the manifest to a JSON file."""
        path = Path(path)
        manifest_dict = {
            "dataset_name": self.dataset_name,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "bytes_per_token_by_split": self.bytes_per_token_by_split,
            "shards": [
                {
                    "filename": shard.filename,
                    "split": shard.split,
                    "num_tokens": shard.num_tokens,
                    "num_bytes_utf8": shard.num_bytes_utf8,
                    "sha256": shard.sha256,
                }
                for shard in self.shards
            ],
            "decontamination": self.decontamination,
        }

        with open(path, "w") as f:
            json.dump(manifest_dict, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ShardManifest":
        """Load a manifest from a JSON file."""
        path = Path(path)
        with open(path) as f:
            manifest_dict = json.load(f)

        shards = [
            ShardInfo(
                filename=shard["filename"],
                split=shard["split"],
                num_tokens=shard["num_tokens"],
                num_bytes_utf8=shard["num_bytes_utf8"],
                sha256=shard["sha256"],
            )
            for shard in manifest_dict["shards"]
        ]

        return cls(
            dataset_name=manifest_dict["dataset_name"],
            vocab_size=manifest_dict["vocab_size"],
            eos_token_id=manifest_dict["eos_token_id"],
            tokenizer_sha256=manifest_dict["tokenizer_sha256"],
            bytes_per_token_by_split=manifest_dict["bytes_per_token_by_split"],
            shards=shards,
            decontamination=manifest_dict["decontamination"],
        )


class StreamingShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        dataset_name: str,
        tokenizer,
        shard_size_tokens: int = 100_000_000,
        tokenizer_sha256: str = "",
    ):
        """Initialize the streaming shard writer."""
        if tokenizer.vocab_size > 65535:
            raise ValueError("Tokenizer vocab size must be <= 65535")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.shard_size_tokens = shard_size_tokens
        self.tokenizer_sha256 = tokenizer_sha256

        # Buffers for each split
        self.buffers: dict[SplitName, list[int]] = {
            "train": [],
            "val": [],
            "test": [],
            "tokenizer_fit": [],
        }

        # Track statistics for each split
        self.split_stats: dict[SplitName, dict[str, int]] = {
            "train": {"tokens": 0, "bytes": 0},
            "val": {"tokens": 0, "bytes": 0},
            "test": {"tokens": 0, "bytes": 0},
            "tokenizer_fit": {"tokens": 0, "bytes": 0},
        }

        # Shard counters for naming
        self.shard_counters: dict[SplitName, int] = {
            "train": 0,
            "val": 0,
            "test": 0,
            "tokenizer_fit": 0,
        }

        # Store shard info for manifest
        self.shard_info_list: list[ShardInfo] = []

    def add_document(self, doc_id: str, text: str, split: SplitName | None = None) -> SplitName:
        """Add a document to the appropriate split buffer."""
        if split is None:
            split = assign_split_by_doc_id(doc_id)

        # For tokenizer_fit split, we just track stats but don't tokenize
        if split == "tokenizer_fit":
            utf8_bytes = len(text.encode("utf-8"))
            self.split_stats[split]["tokens"] += 0  # No tokens for tokenizer fit
            self.split_stats[split]["bytes"] += utf8_bytes
            return split

        # Encode the text with EOS token
        tokens = self.tokenizer.encode(text, add_eos=True)

        # Update stats
        utf8_bytes = len(text.encode("utf-8"))
        self.split_stats[split]["tokens"] += len(tokens)
        self.split_stats[split]["bytes"] += utf8_bytes

        # Add tokens to buffer
        self.buffers[split].extend(tokens)

        # Flush shard if buffer is full
        while len(self.buffers[split]) >= self.shard_size_tokens:
            self._flush_shard(split)

        return split

    def _flush_shard(self, split: SplitName):
        """Flush a full shard to disk."""
        # Take exactly shard_size_tokens from buffer
        tokens_to_write = self.buffers[split][: self.shard_size_tokens]
        self.buffers[split] = self.buffers[split][self.shard_size_tokens :]

        # Generate filename
        filename = f"shard_{split}_{self.shard_counters[split]:05d}.bin"
        filepath = self.output_dir / filename
        self.shard_counters[split] += 1

        # Convert to numpy array and write to file
        token_array = np.asarray(tokens_to_write, dtype=np.uint16)
        token_array.tofile(filepath)

        # Calculate SHA-256 of the file
        with open(filepath, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Create ShardInfo
        shard_info = ShardInfo(
            filename=filename,
            split=split,
            num_tokens=len(tokens_to_write),
            num_bytes_utf8=0,  # This would need to be tracked separately
            sha256=file_hash,
        )

        self.shard_info_list.append(shard_info)

    def finalize(self, decontamination: dict[str, Any] | None = None) -> ShardManifest:
        """Finalize writing by flushing remaining buffers and creating manifest."""
        # Flush remaining buffers
        for split in ["train", "val", "test"]:
            if self.buffers[split]:
                self._flush_remaining_shard(split)

        # Calculate bytes per token for each split
        bytes_per_token_by_split = {}
        for split in ["train", "val", "test", "tokenizer_fit"]:
            tokens = self.split_stats[split]["tokens"]
            bytes_count = self.split_stats[split]["bytes"]
            if tokens > 0:
                bytes_per_token_by_split[split] = bytes_count / tokens
            else:
                bytes_per_token_by_split[split] = 0.0

        # Create manifest
        manifest = ShardManifest(
            dataset_name=self.dataset_name,
            vocab_size=self.tokenizer.vocab_size,
            eos_token_id=self.tokenizer.eos_token_id,
            tokenizer_sha256=self.tokenizer_sha256,
            bytes_per_token_by_split=bytes_per_token_by_split,
            shards=self.shard_info_list,
            decontamination=decontamination,
        )

        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest.save(manifest_path)

        return manifest

    def _flush_remaining_shard(self, split: SplitName):
        """Flush remaining tokens in buffer to a shard."""
        if not self.buffers[split]:
            return

        tokens_to_write = self.buffers[split]
        self.buffers[split] = []

        # Generate filename
        filename = f"shard_{split}_{self.shard_counters[split]:05d}.bin"
        filepath = self.output_dir / filename
        self.shard_counters[split] += 1

        # Convert to numpy array and write to file
        token_array = np.asarray(tokens_to_write, dtype=np.uint16)
        token_array.tofile(filepath)

        # Calculate SHA-256 of the file
        with open(filepath, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Create ShardInfo
        shard_info = ShardInfo(
            filename=filename,
            split=split,
            num_tokens=len(tokens_to_write),
            num_bytes_utf8=0,  # This would need to be tracked separately
            sha256=file_hash,
        )

        self.shard_info_list.append(shard_info)


class ShardedDataset:
    def __init__(
        self,
        manifest_path: str | Path,
        split: SplitName = "train",
        seq_len: int = 2048,
        verify_checksums: bool = True,
    ):
        """Initialize the sharded dataset."""
        self.manifest_path = Path(manifest_path)
        self.manifest = ShardManifest.load(self.manifest_path)
        self.split = split
        self.seq_len = seq_len

        # Filter shards for the requested split
        self.shards = [s for s in self.manifest.shards if s.split == split]
        if not self.shards:
            raise ValueError(f"No shards found for split '{split}'")

        # Verify checksums if requested
        if verify_checksums:
            for shard in self.shards:
                filepath = self.manifest_path.parent / shard.filename
                with open(filepath, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                if file_hash != shard.sha256:
                    raise ValueError(
                        f"SHA-256 mismatch for shard {shard.filename}: "
                        f"expected {shard.sha256}, got {file_hash}"
                    )

        # Load tokens into memory-mapped/concatenated array
        self.tokens = self._load_tokens()
        if len(self.tokens) < 2:
            raise ValueError(f"Split '{split}' has fewer than 2 tokens ({len(self.tokens)})")
        self._cursor = 0

    def _load_tokens(self) -> np.ndarray:
        """Load all tokens from shards into a contiguous uint16 array."""
        token_arrays = []
        for shard in self.shards:
            filepath = self.manifest_path.parent / shard.filename
            token_array = np.memmap(filepath, dtype=np.uint16, mode="r")
            token_arrays.append(np.asarray(token_array, dtype=np.uint16))
        return np.concatenate(token_arrays) if token_arrays else np.array([], dtype=np.uint16)

    def next_batch(
        self, batch_size: int, device: str | torch.device = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the next batch of sequences `(input_ids, targets, doc_ids)` of shape `(B, T)`."""
        stride = self.seq_len + 1
        total_tokens_needed = batch_size * stride
        num_tokens = len(self.tokens)

        if self._cursor + total_tokens_needed <= num_tokens:
            batch_tokens = self.tokens[self._cursor : self._cursor + total_tokens_needed]
        else:
            indices = (np.arange(total_tokens_needed, dtype=np.int64) + self._cursor) % num_tokens
            batch_tokens = self.tokens[indices]

        self._cursor = (self._cursor + total_tokens_needed) % num_tokens

        sequences = batch_tokens.reshape(batch_size, stride)
        input_ids_np = np.asarray(sequences[:, :-1], dtype=np.int64)
        targets_np = np.asarray(sequences[:, 1:], dtype=np.int64)

        input_ids_tensor = torch.from_numpy(input_ids_np).to(device=device, dtype=torch.long)
        targets_tensor = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)
        doc_ids_tensor = compute_doc_ids_from_eos(input_ids_tensor, self.manifest.eos_token_id)

        return input_ids_tensor, targets_tensor, doc_ids_tensor

    def state_dict(self) -> dict[str, int]:
        """Return the current state of the dataset."""
        return {"cursor": self._cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        """Load the state of the dataset."""
        self._cursor = int(state["cursor"])
