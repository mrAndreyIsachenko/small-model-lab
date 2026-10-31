.PHONY: help sync test lint format smoke report clean

help:
	@echo "Small Model Lab — Available Commands:"
	@echo "  make sync    Install dependencies via uv (Python 3.12)"
	@echo "  make test    Run unit test suite via pytest"
	@echo "  make lint    Run ruff linter and format check"
	@echo "  make format  Auto-format codebase via ruff"
	@echo "  make smoke   Run M0 synthetic dry-run lifecycle and compile report"
	@echo "  make report  Compile markdown report tables from results/"

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

smoke:
	uv run sml dry-run --experiment experiments/h3_wsd_100m.yaml --output-dir results/dry_run_m0
	uv run sml report --results-dir results/dry_run_m0 --output reports/sprint1/dry_run_preview.md --include-dry-runs

report:
	uv run sml report --results-dir results --output reports/sprint1/summary.md

clean:
	rm -rf results/dry_run_* reports/sprint1/dry_run_preview.md .pytest_cache .ruff_cache
