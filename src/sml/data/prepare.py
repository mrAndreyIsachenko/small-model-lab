"""Streaming dataset preparation: BPE training, 13-gram decontamination, and uint16 sharding."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from sml.config import DataConfig
from sml.data.decontamination import DecontaminationIndex
from sml.data.sharding import ShardManifest, StreamingShardWriter, assign_split_by_doc_id
from sml.tokenizer.bpe import BPETokenizer


def prepare_dataset_shards(
    data_config: DataConfig,
    output_dir: str | Path,
    documents: Iterable[tuple[str, str]] | None = None,
    eval_tasks: dict[str, Iterable[str]] | None = None,
    max_docs: int | None = None,
    min_frequency: int = 2,
) -> ShardManifest:
    """Train BPE tokenizer, filter contaminated docs, and write `.bin` shards + `manifest.json`."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if documents is None:
        from datasets import load_dataset

        ds = load_dataset(
            data_config.hf_dataset,
            data_config.hf_subset,
            split="train",
            streaming=True,
        )

        def _hf_stream() -> Iterable[tuple[str, str]]:
            for idx, row in enumerate(ds):
                if max_docs is not None and idx >= max_docs:
                    break
                doc_id = str(row.get("id") or f"{data_config.name}-{idx}")
                text = str(row.get("text") or "")
                if text.strip():
                    yield (doc_id, text)

        doc_iter = _hf_stream()
    else:
        doc_iter = documents

    buffered_docs: list[tuple[str, str]] = []
    for idx, (doc_id, text) in enumerate(doc_iter):
        if max_docs is not None and idx >= max_docs:
            break
        if text.strip():
            buffered_docs.append((str(doc_id), str(text)))

    if not buffered_docs:
        raise ValueError("No non-empty documents provided to prepare_dataset_shards")

    # Fit BPE tokenizer on tokenizer_fit split documents if enough exist, else full sample
    fit_texts = [
        text for doc_id, text in buffered_docs if assign_split_by_doc_id(doc_id) == "tokenizer_fit"
    ]
    if len(fit_texts) < 8:
        fit_texts = [text for _, text in buffered_docs]

    tokenizer = BPETokenizer.train_from_iterator(
        fit_texts,
        vocab_size=data_config.vocab_size,
        min_frequency=min_frequency,
    )
    tokenizer_path = tokenizer.save(out_path / "tokenizer.json")
    tok_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()

    decontam_index = DecontaminationIndex(eval_tasks) if eval_tasks else None
    train_texts_for_audit: list[str] = []

    writer = StreamingShardWriter(
        output_dir=out_path,
        dataset_name=data_config.name,
        tokenizer=tokenizer,
        shard_size_tokens=data_config.shard_size_tokens,
        tokenizer_sha256=tok_sha256,
    )

    # Guarantee that small corpora still populate train, val, and test splits deterministically
    natural_splits = [assign_split_by_doc_id(doc_id) for doc_id, _ in buffered_docs]
    if len(buffered_docs) >= 3:
        if "val" not in natural_splits:
            natural_splits[0] = "val"
        if "test" not in natural_splits:
            natural_splits[1] = "test"

    for (_, (doc_id, text)), split in zip(enumerate(buffered_docs), natural_splits, strict=True):
        if split == "train":
            train_texts_for_audit.append(text)
            if decontam_index is not None and decontam_index.matched_tasks(text):
                continue

        writer.add_document(doc_id, text, split=split)

    decontam_dict = (
        decontam_index.audit_stream(train_texts_for_audit).to_dict()
        if decontam_index is not None
        else None
    )
    return writer.finalize(decontamination=decontam_dict)
