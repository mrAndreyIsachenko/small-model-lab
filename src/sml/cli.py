"""Command-line interface for Small Model Lab (`sml`)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from sml.config import load_experiment_spec, load_model_config
from sml.tracking.cost import estimate_experiment_compute
from sml.tracking.manifest import RunRecorder
from sml.tracking.report import compile_markdown_report


def _cmd_inspect_config(args: argparse.Namespace) -> int:
    cfg = load_model_config(args.model)
    summary = {
        "name": cfg.name,
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "d_ff": cfg.d_ff,
        "max_seq_len": cfg.max_seq_len,
        "embedding_params": cfg.embedding_params,
        "non_embedding_params": cfg.non_embedding_params,
        "total_params": cfg.total_params,
        "embedding_share_pct": round(100.0 * cfg.embedding_params / cfg.total_params, 2),
        "kv_cache_bytes_per_token_bf16": cfg.kv_cache_bytes_per_token(2),
        "flops_per_token": cfg.flops_per_token(),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    spec = load_experiment_spec(args.experiment)
    output_dir = Path(args.output_dir)
    run_id = args.run_id or f"dry_run_{spec.experiment_id}"

    t0 = time.perf_counter()
    recorder = RunRecorder(
        run_id=run_id,
        spec=spec,
        output_root=output_dir,
        mode="dry_run",
    )

    init_loss = math.log(spec.model.vocab_size)
    bytes_per_token = 4.15
    for step in range(1, 4):
        sim_loss = init_loss - 0.12 * step
        sim_bpb = sim_loss / (math.log(2.0) * bytes_per_token)
        recorder.log_step(
            step=step,
            metrics={
                "train_loss": round(sim_loss, 4),
                "val_bpb": round(sim_bpb, 4),
                "lr": spec.train.lr,
                "tokens_processed": step * spec.train.batch_size_tokens,
            },
        )

    elapsed = time.perf_counter() - t0
    final_loss = init_loss - 0.36
    final_bpb = final_loss / (math.log(2.0) * bytes_per_token)
    est = estimate_experiment_compute(spec)

    run_path = recorder.finalize(
        final_metrics={
            "initial_loss_nats": round(init_loss, 4),
            "final_train_loss_nats": round(final_loss, 4),
            "val_bpb": round(final_bpb, 4),
            "tokens_processed": 3 * spec.train.batch_size_tokens,
            "tokens_per_sec": round((3 * spec.train.batch_size_tokens) / max(elapsed, 1e-3), 1),
            "mfu": 0.0,
            "planned_full_run_tokens": est["planned_tokens"],
            "estimated_full_run_gpu_hours": est["estimated_clean_gpu_hours"],
        },
        active_seconds=elapsed,
        billed_seconds=elapsed,
        status="completed",
        conclusion=(
            "M0 dry-run pipeline verified: run.json, config.yaml, environment.json, "
            "metrics.jsonl, final_metrics.json, cost.json, and summary.md written."
        ),
    )
    print(f"Dry-run bundle written to: {run_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    compile_markdown_report(
        results_dir=results_dir,
        output_path=output_path,
        include_dry_runs=bool(args.include_dry_runs),
    )
    print(f"Compiled report to: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sml",
        description="Small Model Lab CLI — experiment runner, evaluator, and report compiler.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser(
        "inspect-config",
        help="Print exact parameter, KV-cache, and FLOP math for a model YAML config.",
    )
    inspect_p.add_argument("--model", required=True, help="Path to model config YAML.")
    inspect_p.set_defaults(func=_cmd_inspect_config)

    dry_p = sub.add_parser(
        "dry-run",
        help="Execute an M0 synthetic dry-run lifecycle creating a full results/<run_id>/ bundle.",
    )
    dry_p.add_argument("--experiment", required=True, help="Path to experiment YAML.")
    dry_p.add_argument(
        "--output-dir",
        default="results",
        help="Parent directory for run output bundle (default: results).",
    )
    dry_p.add_argument("--run-id", default=None, help="Optional explicit run_id override.")
    dry_p.set_defaults(func=_cmd_dry_run)

    rep_p = sub.add_parser(
        "report",
        help="Compile Markdown tables strictly from results/<run_id>/ JSON artifacts.",
    )
    rep_p.add_argument("--results-dir", default="results", help="Directory containing run bundles.")
    rep_p.add_argument(
        "--output",
        default="reports/sprint1/summary.md",
        help="Output Markdown file path.",
    )
    rep_p.add_argument(
        "--include-dry-runs",
        action="store_true",
        help="Include runs marked with mode='dry_run' (excluded by default).",
    )
    rep_p.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
