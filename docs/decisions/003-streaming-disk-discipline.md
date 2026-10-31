# ADR-003: Streaming Data Shards & `< 15 GB` Local Disk Discipline

* **Status:** Accepted
* **Date:** 2026-10-31

## Context

Local development takes place on a MacBook Pro (Apple M4 Pro, 48 GB Unified Memory) with ~126 GB of free SSD space, while heavy training runs execute on ephemeral cloud GPU instances (`RTX 4090` / `L4` for pilots, `1x H100 80GB SXM` for main runs). Downloading the full `FineWeb-Edu sample-10BT` dataset (`28.5 GB` parquet) alongside Hugging Face cache duplicates (`+28.5 GB`), tokenized `.bin` shards (`~8.5 GB`), local checkpoints, and CUDA container images would exhaust local storage.

## Decision

1. **Streaming Parquet-to-Shard Pipeline (`< 15 GB` Peak Disk):** Process `FineWeb-Edu` parquet files sequentially, writing deterministic `100M`-token `uint16` `.bin` shards (`~200 MB` per shard) with SHA-256 digests recorded in `manifest.json`, uploading completed shards to a Hugging Face Dataset repository, and purging transient local parquet caches immediately.
2. **Zero Local CUDA Image Builds:** Ephemeral GPU instances boot official `pytorch/pytorch` CUDA base images and install locked Python 3.12 dependencies directly via `uv sync --frozen` in `scripts/remote/bootstrap.sh`.
3. **File-Based Tracking + Remote Weights:** Keep only lightweight, human- and machine-readable JSON/JSONL/Markdown records (`run.json`, `config.yaml`, `environment.json`, `metrics.jsonl`, `final_metrics.json`, `cost.json`, `summary.md`) in git under `results/<run_id>/`. Stream model checkpoints directly to Hugging Face Hub during training.

## Consequences

* Local disk usage remains safely below `15 GB` throughout all four sprints.
* New ephemeral GPU instances start training within 2–3 minutes by pulling pre-tokenized `uint16` shards directly from Hugging Face Hub at gigabit speeds instead of burning paid GPU time on tokenization.
