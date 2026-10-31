"""Report compiler that generates markdown tables strictly from results/<run_id>/ artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_RUN_FILES = ("run.json", "final_metrics.json", "cost.json", "environment.json")


def load_run_bundle(run_dir: Path) -> dict[str, Any] | None:
    """Load a completed run bundle if all required JSON files exist."""
    if not run_dir.is_dir():
        return None
    for filename in REQUIRED_RUN_FILES:
        if not (run_dir / filename).exists():
            return None

    with (run_dir / "run.json").open("r", encoding="utf-8") as handle:
        run_meta = json.load(handle)
    with (run_dir / "final_metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (run_dir / "cost.json").open("r", encoding="utf-8") as handle:
        cost = json.load(handle)
    with (run_dir / "environment.json").open("r", encoding="utf-8") as handle:
        env = json.load(handle)

    return {
        "run": run_meta,
        "metrics": metrics,
        "cost": cost,
        "environment": env,
    }


def discover_run_bundles(
    results_dir: Path,
    include_dry_runs: bool = False,
) -> list[dict[str, Any]]:
    """Scan results_dir (or a single run dir) for valid run bundles."""
    if not results_dir.exists():
        return []

    direct = load_run_bundle(results_dir)
    candidates = (
        [direct]
        if direct is not None
        else [load_run_bundle(child) for child in sorted(results_dir.iterdir())]
    )

    bundles: list[dict[str, Any]] = []
    for bundle in candidates:
        if bundle is None:
            continue
        mode = bundle["run"].get("mode", "measured")
        if mode == "dry_run" and not include_dry_runs:
            continue
        bundles.append(bundle)
    return bundles


def compile_markdown_report(
    results_dir: Path,
    output_path: Path,
    include_dry_runs: bool = False,
) -> str:
    """Compile a Markdown summary table strictly from machine-readable run artifacts."""
    bundles = discover_run_bundles(results_dir, include_dry_runs=include_dry_runs)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Sprint 1 Experimental Report (Auto-Generated)",
        "",
        "> Generated strictly from machine-readable artifacts in "
        f"`{results_dir}` (`include_dry_runs={include_dry_runs}`).",
        "",
    ]

    if not bundles:
        lines.extend(
            [
                "No matching completed run bundles found.",
                "",
            ]
        )
        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")
        return content

    header_cols = (
        "| Run ID | Hyp. | Mode | Model | Non-Emb Params | Tokens "
        "| Val bpb | Tok/s | MFU | Hardware | Billed Cost ($) | Git SHA |"
    )
    header_sep = (
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |"
    )
    lines.extend(
        [
            "## Summary Table",
            "",
            header_cols,
            header_sep,
        ]
    )

    for item in bundles:
        run = item["run"]
        metrics = item["metrics"]
        cost = item["cost"]

        val_bpb = metrics.get("val_bpb")
        val_bpb_str = f"{val_bpb:.4f}" if isinstance(val_bpb, (int, float)) else "n/a"

        tok_s = metrics.get("tokens_per_sec")
        tok_s_str = f"{tok_s:,.0f}" if isinstance(tok_s, (int, float)) else "n/a"

        mfu = metrics.get("mfu")
        mfu_str = f"{mfu * 100:.1f}%" if isinstance(mfu, (int, float)) else "n/a"

        tokens_proc = metrics.get("tokens_processed", 0)
        tokens_str = f"{tokens_proc:,}" if isinstance(tokens_proc, int) else str(tokens_proc)

        git_sha_short = str(run.get("git_sha", "unknown"))[:8]
        lines.append(
            f"| `{run['run_id']}` | `{run['hypothesis_id']}` | `{run['mode']}` | "
            f"`{run['model_name']}` | `{run['non_embedding_params']:,}` | "
            f"`{tokens_str}` | `{val_bpb_str}` | `{tok_s_str}` | `{mfu_str}` | "
            f"`{cost['hardware_name']}` | `${cost['billed_cost_usd']:.4f}` | `{git_sha_short}` |"
        )

    lines.append("")
    lines.append("## Run Conclusions")
    lines.append("")
    for item in bundles:
        run = item["run"]
        lines.append(f"### `{run['run_id']}` (`{run['hypothesis_id']}` — `{run['mode']}`)")
        lines.append(f"- **Prediction:** {run['prediction']}")
        lines.append(f"- **Conclusion:** {run.get('conclusion') or 'Pending.'}")
        lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    return content
