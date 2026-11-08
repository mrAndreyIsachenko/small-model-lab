"""Unit tests for M2: BPE tokenizer, hash splits, 13-gram decontamination, and uint16 sharding."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sml.data import (
    DecontaminationIndex,
    ShardedDataset,
    StreamingShardWriter,
    assign_split_by_doc_id,
)
from sml.tokenizer import SPECIAL_TOKENS, BPETokenizer


def test_bpe_tokenizer_roundtrip_and_compression_ratio(tmp_path: Path) -> None:
    """Verify byte-level BPE training, save/load, lossless UTF-8 round-trip, and compression."""
    corpus = [
        "Transformer models use Grouped-Query Attention and Rotary Position Embeddings.",
        "Data engineering pipelines in dbt and DuckDB require deterministic testing.",
        "Unicode round-trip check: café, naïve, résumé, Привет, こんにちは, 🚀!",
    ] * 25

    tok = BPETokenizer.train_from_iterator(corpus, vocab_size=512, min_frequency=2)
    assert tok.vocab_size <= 512
    for sp in SPECIAL_TOKENS:
        assert tok.token_to_id(sp) >= 0

    tok_path = tmp_path / "tokenizer.json"
    tok.save(tok_path)
    loaded = BPETokenizer.load(tok_path)

    samples = [
        "Hello, Small Model Lab! 12345",
        "Unicode: café, naïve, Привет мир, 🚀",
        "SELECT count(*) FROM mart_appointments WHERE status = 'COMPLETED';",
    ]
    for sample in samples:
        ids = loaded.encode(sample, add_bos=True, add_eos=True)
        assert ids[0] == loaded.bos_token_id
        assert ids[-1] == loaded.eos_token_id
        decoded = loaded.decode(ids[1:-1], skip_special_tokens=False)
        assert decoded == sample

    ratio = loaded.compression_ratio_bytes_per_token(samples)
    assert ratio > 1.0


def test_assign_split_by_doc_id_determinism_and_disjoint_buckets() -> None:
    """Verify deterministic hash splitting and zero document ID overlap across splits."""
    buckets: dict[str, set[str]] = {
        "train": set(),
        "val": set(),
        "test": set(),
        "tokenizer_fit": set(),
    }
    for i in range(3000):
        doc_id = f"fineweb-doc-{i:06d}"
        s1 = assign_split_by_doc_id(
            doc_id, val_ratio=0.02, test_ratio=0.02, tokenizer_fit_ratio=0.05
        )
        s2 = assign_split_by_doc_id(
            doc_id, val_ratio=0.02, test_ratio=0.02, tokenizer_fit_ratio=0.05
        )
        assert s1 == s2
        buckets[s1].add(doc_id)

    for k, v in buckets.items():
        assert len(v) > 0, f"Split {k} received 0 documents"

    # Pairwise intersections must be strictly empty
    keys = list(buckets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert buckets[keys[i]].isdisjoint(buckets[keys[j]])


def test_13gram_decontamination_index() -> None:
    """Verify 13-gram benchmark overlap detection and contamination report metrics."""
    eval_tasks = {
        "hellaswag": [
            "A person is standing in front of a chalkboard explaining "
            "a complex mathematical proof to students."
        ],
        "arc_easy": [
            "Photosynthesis is the biochemical process by which green plants "
            "convert sunlight into chemical energy."
        ],
    }
    index = DecontaminationIndex(eval_tasks=eval_tasks, n=13)

    train_docs = [
        "Completely unrelated training document about distributed database "
        "transactions and consensus.",
        "In biology class we learned that photosynthesis is the biochemical "
        "process by which green plants convert sunlight into chemical energy for growth.",
        "Another clean article discussing graphics processors and high "
        "bandwidth memory architectures.",
    ]
    report = index.audit_stream(train_docs)
    assert report.total_train_docs_scanned == 3
    assert report.contaminated_doc_count == 1
    assert report.overlap_by_task["arc_easy"] == 1
    assert report.overlap_by_task.get("hellaswag", 0) == 0
    assert report.contamination_rate == pytest.approx(1.0 / 3.0)


def test_streaming_shard_writer_checksums_and_deterministic_resume(tmp_path: Path) -> None:
    """Verify uint16 sharding, SHA-256 validation, doc_ids masking, and deterministic resume."""
    corpus = ["The quick brown fox jumps over the lazy dog. " * 4] * 20
    tok = BPETokenizer.train_from_iterator(corpus, vocab_size=400, min_frequency=2)

    shard_dir = tmp_path / "shards"
    writer = StreamingShardWriter(
        output_dir=shard_dir,
        dataset_name="test-fineweb",
        tokenizer=tok,
        shard_size_tokens=120,
        tokenizer_sha256="deadbeef",
    )

    for i in range(25):
        writer.add_document(
            f"doc-{i}", f"Document number {i} has several words for testing shards.", split="train"
        )
    for i in range(4):
        writer.add_document(f"val-{i}", f"Validation document {i} text.", split="val")

    manifest = writer.finalize(decontamination={"contamination_rate": 0.0})
    manifest_path = shard_dir / "manifest.json"
    assert manifest_path.exists()
    assert manifest.bytes_per_token_by_split["train"] > 0.0

    dataset = ShardedDataset(
        manifest_path=manifest_path, split="train", seq_len=16, verify_checksums=True
    )
    x1, y1, d1 = dataset.next_batch(batch_size=2)
    assert x1.shape == (2, 16)
    assert y1.shape == (2, 16)
    assert d1.shape == (2, 16)
    assert x1.dtype == torch.long

    # Save dataloader state after batch 1, draw batch 2, then restore and verify exact match
    saved_state = dataset.state_dict()
    x2_expected, y2_expected, d2_expected = dataset.next_batch(batch_size=2)

    dataset.load_state_dict(saved_state)
    x2_resumed, y2_resumed, d2_resumed = dataset.next_batch(batch_size=2)
    assert torch.equal(x2_expected, x2_resumed)
    assert torch.equal(y2_expected, y2_resumed)
    assert torch.equal(d2_expected, d2_resumed)

    # Corrupting a shard byte must trigger SHA-256 ValueError on load
    first_train_shard = next(s for s in manifest.shards if s.split == "train")
    corrupt_path = shard_dir / first_train_shard.filename
    raw = bytearray(corrupt_path.read_bytes())
    raw[0] ^= 0xFF
    corrupt_path.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ShardedDataset(
            manifest_path=manifest_path, split="train", seq_len=16, verify_checksums=True
        )
