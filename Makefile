.PHONY: help sync test lint format verify-h1 verify-env compare-adaptation run-surgery verify-h9 verify-h10 verify-h11 build-tenet train-smoke profile-h2-local gpu-smoke smoke report clean

help:
	@echo "Small Model Lab — Available Commands:"
	@echo "  make sync               Install dependencies via uv (Python 3.12)"
	@echo "  make test               Run full unit & integration test suite via pytest"
	@echo "  make lint               Run ruff linter and format check"
	@echo "  make format             Auto-format codebase via ruff"
	@echo "  make verify-h1          Run H1 GPT-2 124M mathematical & causal equivalence verification"
	@echo "  make verify-env         Run H5 StatefulToolEnv & Gated Rubric anti-hacking verification"
	@echo "  make compare-adaptation Run H6 Context vs. Weights adaptation comparison (7 strategies)"
	@echo "  make run-surgery        Run H7/H8 CKA mixed-precision, rank-1 subspace & Pareto verification"
	@echo "  make verify-h9          Run H9 Lancet bi-projected surgery verification (nano/LLM)"
	@echo "  make verify-h10         Run H10 unslop GRPO verification (gated reward shaping)"
	@echo "  make verify-h11         Run H11 OpenRouter FP8 provider readiness & HF Hub verification"
	@echo "  make build-tenet        Build Tenet release bundle (GRPO-tuned weights + readiness report)"
	@echo "  make train-smoke        Run end-to-end M2+M3 local pretraining smoke test (MPS/CUDA/CPU)"
	@echo "  make profile-h2-local   Run 5-stage MFU throughput waterfall locally (MPS/CPU)"
	@echo "  make gpu-smoke          Dry-run the full M4-M5 GPU script in fast smoke mode"
	@echo "  make report             Compile markdown report tables from results/"

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

verify-h1:
	uv run sml verify-h1 --experiment experiments/h1_gpt2_parity.yaml --output-dir results --run-id h1_gpt2_parity

verify-env:
	uv run sml verify-env --experiment experiments/h5_env_verification.yaml --output-dir results --run-id h5_env_verification

compare-adaptation:
	uv run sml compare-adaptation --experiment experiments/h6_adaptation_comparison.yaml --output-dir results --run-id h6_adaptation_comparison

run-surgery:
	uv run sml run-surgery --experiment experiments/h7_h8_surgery_pareto.yaml --output-dir results --run-id h7_h8_surgery_pareto

verify-h9:
	uv run sml verify-h9 --experiment experiments/h9_biprojected_surgery.yaml --output-dir results --run-id h9_biprojected_surgery

verify-h10:
	uv run sml verify-h10 --experiment experiments/h10_unslop_grpo.yaml --output-dir results --run-id h10_unslop_grpo

verify-h11:
	uv run sml verify-h11 --experiment experiments/h11_provider_fp8_readiness.yaml --output-dir results --run-id h11_provider_fp8_readiness

build-tenet:
	uv run sml build-tenet --output-dir results/tenet_releases --release-id Tenet-3B-Qwen2.5-Unslop-v1

train-smoke:
	uv run sml train-smoke --steps 20 --device auto

profile-h2-local:
	uv run sml profile-h2 --device auto --seq-len 256 --micro-batch-size 2 --warmup-steps 1 --measure-steps 2 --run-id smoke_h2_local

gpu-smoke:
	SMOKE_ONLY=1 START_WATCHDOG=0 SHARD_DIR=shards/smoke_gpu RESULTS_DIR=results/smoke_gpu REPORT_PATH=reports/sprint1/smoke_gpu.md bash ./scripts/remote/run_m4_m5_gpu.sh

smoke:
	uv run sml dry-run --experiment experiments/h3_wsd_100m.yaml --output-dir results/dry_run_m0
	uv run sml report --results-dir results/dry_run_m0 --output reports/sprint1/dry_run_preview.md --include-dry-runs

report:
	uv run sml report --results-dir results --output reports/sprint1/summary.md

clean:
	rm -rf results/dry_run_* results/smoke_* shards/smoke_* reports/sprint1/dry_run_preview.md reports/sprint1/smoke_gpu.md .pytest_cache .ruff_cache
