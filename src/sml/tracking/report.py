"""Report compiler that generates markdown tables strictly from results/<run_id>/ artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REQUIRED_RUN_FILES = ("run.json", "final_metrics.json", "cost.json", "environment.json")

SURGERY_HYPOTHESES = ("H9", "H10", "H11")

COMPANION_RUN_FILES = {
    "stage_waterfall": "stage_waterfall.json",
    "adaptation_comparison": "adaptation_comparison.json",
    "surgery_pareto": "surgery_pareto.json",
    "unslop_grpo": "unslop_grpo.json",
    "fp8_survival": "fp8_survival.json",
    "provider_readiness": "provider_readiness.json",
    "subspace_safe_merge": "subspace_safe_merge.json",
}

RELEASE_BUNDLE_FILES = (
    "release_manifest.json",
    "cka_quant_recipe.json",
    "lancet_recipe.json",
    "provider_readiness.json",
    "openrouter_onboarding.json",
)


def _format_metric(value: Any, fmt: str = ".4f") -> str:
    """Format a numeric metric for a markdown table cell."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:{fmt}}"


def _format_flag(value: Any) -> str:
    """Format a boolean verification flag for a markdown table cell."""
    if value is None:
        return "n/a"
    return "✅" if bool(value) else "❌"


def _format_readiness_checks(checks: Any) -> str:
    """Summarise provider readiness checks as passed/total."""
    if isinstance(checks, dict):
        values = list(checks.values())
    elif isinstance(checks, (list, tuple)):
        values = list(checks)
    else:
        return "n/a"
    passed = sum(1 for value in values if bool(value))
    return f"{passed}/{len(values)}"


def _surgery_section_lines(bundles: list[dict[str, Any]]) -> list[str]:
    """Build the H9-H11 surgery / GRPO / provider-readiness section, if applicable."""
    surgery_bundles = [
        bundle for bundle in bundles if bundle["run"].get("hypothesis_id") in SURGERY_HYPOTHESES
    ]
    if not surgery_bundles:
        return []

    lines = [
        "## H9–H11 Surgery, GRPO & Provider Readiness",
        "",
        "| Run ID | Hyp. | Verified | Target Supp. | KL Red. (H9) | Anchor Ortho | Tool Ortho "
        "| Row Drift | Provider Ready | Ready Checks | FP8 Retention "
        "| KL Red. (FP8) | KL Naive (FP8) | KL Lancet (FP8) | Re-leakage (Naive) "
        "| Re-leakage (Lancet) | Max Crest | Slop & Refusal "
        "| Gated vs Naive | Post-Surgery | Merged LoRA |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- | "
        "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for bundle in surgery_bundles:
        run = bundle["run"]
        metrics = bundle["metrics"]
        hypothesis_id = run["hypothesis_id"]
        verified = metrics.get(f"{hypothesis_id.lower()}_verified")
        lines.append(
            f"| `{run['run_id']}` | `{hypothesis_id}` | {_format_flag(verified)} "
            f"| {_format_metric(metrics.get('target_suppression_ratio'))} "
            f"| {_format_metric(metrics.get('h9_kl_reduction_ratio'))} "
            f"| {_format_metric(metrics.get('anchor_orthogonality_error'), '.2e')} "
            f"| {_format_metric(metrics.get('tool_sql_orthogonality_error'), '.2e')} "
            f"| {_format_metric(metrics.get('max_row_norm_drift'), '.2e')} "
            f"| {_format_flag(metrics.get('provider_ready'))} "
            f"| {_format_readiness_checks(metrics.get('readiness_checks'))} "
            f"| {_format_metric(metrics.get('fp8_retention_ratio'))} "
            f"| {_format_metric(metrics.get('h11_fp8_kl_reduction_ratio'))} "
            f"| {_format_metric(metrics.get('kl_naive_fp8'))} "
            f"| {_format_metric(metrics.get('kl_lancet_fp8'))} "
            f"| {_format_metric(metrics.get('fp8_subspace_releakage_naive'))} "
            f"| {_format_metric(metrics.get('fp8_subspace_releakage_lancet'))} "
            f"| {_format_metric(metrics.get('max_row_crest_factor'))} "
            f"| {_format_metric(metrics.get('slop_and_refusal_rate'))} "
            f"| {_format_metric(metrics.get('h10_gated_vs_naive_gain'))} "
            f"| {_format_metric(metrics.get('h10_post_surgery_gain'))} "
            f"| {_format_metric(metrics.get('merged_lora_count'), ',d')} |"
        )

    lines.append("")
    return lines


def _load_json_if_exists(path: Path) -> Any | None:
    """Load a companion JSON artifact, returning None when missing or malformed."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


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

    bundle: dict[str, Any] = {
        "run": run_meta,
        "metrics": metrics,
        "cost": cost,
        "environment": env,
    }
    for key, filename in COMPANION_RUN_FILES.items():
        companion = _load_json_if_exists(run_dir / filename)
        if companion is not None:
            bundle[key] = companion
    return bundle


def discover_release_bundles(releases_dir: Path) -> list[dict[str, Any]]:
    """Load release manifest + companion JSONs from each subdirectory of releases_dir."""
    bundles: list[dict[str, Any]] = []
    if not releases_dir.is_dir():
        return bundles
    for child in sorted(releases_dir.iterdir()):
        if not child.is_dir():
            continue
        bundle: dict[str, Any] = {"release_dir": child}
        for filename in RELEASE_BUNDLE_FILES:
            doc = _load_json_if_exists(child / filename)
            if doc is not None:
                bundle[filename[: -len(".json")]] = doc
        bundles.append(bundle)
    return bundles


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
    lines.extend(_surgery_section_lines(bundles))
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


# ---------------------------------------------------------------------------
# AUTO-TABLE machinery for the long-form technical report.
# ---------------------------------------------------------------------------

AUTO_TABLE_START_TEMPLATE = "<!-- AUTO-TABLE:{key}:START -->"
AUTO_TABLE_END_TEMPLATE = "<!-- AUTO-TABLE:{key}:END -->"

AUTO_TABLE_BLOCK_RE = re.compile(
    r"(?P<start><!-- AUTO-TABLE:(?P<key>[A-Z0-9_]+):START -->\n)"
    r"(?P<body>.*?)"
    r"(?P<end><!-- AUTO-TABLE:(?P=key):END -->)",
    re.DOTALL,
)

NO_DATA_LINE = "_No matching machine-readable artifacts found for this table._"


def _table_lines(
    headers: list[str],
    aligns: list[str],
    rows: list[list[str]],
) -> list[str]:
    """Render headers/rows as a GitHub-flavoured Markdown table (trailing blank line)."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(aligns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _resolved_value(bundle: dict[str, Any], key: str) -> Any:
    """Resolve a metric value from final_metrics, run metadata, then companion JSONs."""
    metrics = bundle["metrics"]
    if metrics.get(key) is not None:
        return metrics[key]
    run = bundle["run"]
    if run.get(key) is not None:
        return run[key]
    for companion_key in COMPANION_RUN_FILES:
        doc = bundle.get(companion_key)
        if isinstance(doc, dict) and doc.get(key) is not None:
            return doc[key]
    return None


def _bundle_model_name(bundle: dict[str, Any]) -> str | None:
    value = _resolved_value(bundle, "model_name")
    return str(value) if value is not None else None


def _bundle_hardware(bundle: dict[str, Any]) -> str | None:
    env = bundle["environment"]
    value = env.get("accelerator_name")
    if value is None:
        value = bundle["cost"].get("hardware_name")
    return str(value) if value is not None else None


def _bundles_with_hypothesis(
    bundles: list[dict[str, Any]],
    hypothesis_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [b for b in bundles if b["run"].get("hypothesis_id") in hypothesis_ids]


def update_technical_report_tables(
    results_dir: Path,
    report_path: Path,
    include_dry_runs: bool = False,
) -> str:
    """Replace AUTO-TABLE blocks in report_path with tables rebuilt from JSON artifacts."""
    text = report_path.read_text(encoding="utf-8")
    bundles = discover_run_bundles(results_dir, include_dry_runs=include_dry_runs)
    releases = discover_release_bundles(results_dir / "tenet_releases")
    tables = build_auto_tables(bundles, releases)

    def _replace_block(match: re.Match[str]) -> str:
        key = match.group("key")
        lines = tables.get(key)
        if lines is None:
            return match.group(0)
        return match.group("start") + "\n".join(lines) + match.group("end")

    updated = AUTO_TABLE_BLOCK_RE.sub(_replace_block, text)
    report_path.write_text(updated, encoding="utf-8")
    return updated


def build_auto_tables(
    bundles: list[dict[str, Any]],
    releases: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Build every AUTO-TABLE body (list of Markdown lines) from loaded bundles."""
    return {
        "H1_PARITY": _h1_parity_lines(bundles),
        "H2_WATERFALL": _h2_waterfall_lines(bundles),
        "H3_H4_PRETRAIN": _h3_h4_pretrain_lines(bundles),
        "H5_ENV_REWARD": _h5_env_reward_lines(bundles),
        "H6_ADAPTATION": _h6_adaptation_lines(bundles),
        "H7_H8_PARETO": _h7_h8_pareto_lines(bundles),
        "H9_LANCET": _h9_lancet_lines(bundles),
        "H10_GRPO": _h10_grpo_lines(bundles),
        "H11_FP8": _h11_fp8_lines(bundles),
        "H12_MERGE": _h12_merge_lines(bundles),
        "TENET_RELEASES": _tenet_release_lines(releases),
    }


def _h1_parity_lines(bundles: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for bundle in _bundles_with_hypothesis(bundles, ("H1",)):
        metrics = bundle["metrics"]
        rows.append(
            [
                f"`{bundle['run'].get('run_id')}`",
                _format_metric(metrics.get("max_abs_logit_diff_sdpa"), ".2e"),
                _format_metric(metrics.get("max_abs_logit_diff_naive"), ".2e"),
                _format_metric(metrics.get("max_diff_sdpa_vs_naive"), ".2e"),
                _format_metric(metrics.get("max_diff_kv_cache"), ".2e"),
                _format_metric(metrics.get("cross_doc_leakage"), ".2e"),
            ]
        )
    if not rows:
        return [NO_DATA_LINE, ""]
    return _table_lines(
        ["Run ID", "SDPA Diff", "Naive Diff", "SDPA vs Naive", "KV-Cache Diff", "Cross-Doc Leak"],
        [":---", "---:", "---:", "---:", "---:", "---:"],
        rows,
    )


def _h2_waterfall_lines(bundles: list[dict[str, Any]]) -> list[str]:
    for bundle in _bundles_with_hypothesis(bundles, ("H2",)):
        waterfall = bundle.get("stage_waterfall")
        if isinstance(waterfall, dict):
            waterfall = waterfall.get("stages")
        if not isinstance(waterfall, list):
            continue
        rows = []
        for stage in waterfall:
            if not isinstance(stage, dict):
                continue
            stage_name = (
                stage.get("stage")
                or stage.get("stage_name")
                or stage.get("name")
                or stage.get("config_name")
            )
            mfu = stage.get("mfu")
            mfu_str = f"{mfu * 100:.1f}%" if isinstance(mfu, (int, float)) else "n/a"
            tok_s = stage.get("tokens_per_sec")
            tok_s_str = f"{tok_s:,.1f}" if isinstance(tok_s, (int, float)) else "n/a"
            speedup = stage.get("speedup_vs_stage1", stage.get("speedup_vs_baseline"))
            peak_mem = stage.get("peak_memory_mb")
            peak_str = f"{peak_mem:,.1f}" if isinstance(peak_mem, (int, float)) else "n/a"
            rows.append(
                [
                    f"`{stage_name}`",
                    tok_s_str,
                    mfu_str,
                    _format_metric(speedup, ".2f"),
                    peak_str,
                ]
            )
        if rows:
            return _table_lines(
                ["Stage", "Tokens/s", "MFU", "Speedup vs Stage 1", "Peak Mem (MB)"],
                [":---", "---:", "---:", "---:", "---:"],
                rows,
            )
    return [NO_DATA_LINE, ""]


def _h3_h4_pretrain_lines(bundles: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for bundle in _bundles_with_hypothesis(bundles, ("H3", "H4")):
        metrics = bundle["metrics"]
        mfu = metrics.get("mfu")
        mfu_str = f"{mfu * 100:.1f}%" if isinstance(mfu, (int, float)) else "n/a"
        tok_s = metrics.get("tokens_per_sec")
        tok_s_str = f"{tok_s:,.1f}" if isinstance(tok_s, (int, float)) else "n/a"
        tokens = metrics.get("tokens_processed")
        tokens_str = f"{tokens:,}" if isinstance(tokens, int) else "n/a"
        rows.append(
            [
                f"`{bundle['run'].get('run_id')}`",
                f"`{bundle['run'].get('hypothesis_id')}`",
                tokens_str,
                tok_s_str,
                mfu_str,
                _format_metric(metrics.get("val_bpb")),
                _format_metric(metrics.get("branch_val_bpb")),
                _format_metric(metrics.get("delta_val_bpb")),
                _format_metric(metrics.get("final_in_context_loss_delta")),
                _format_metric(metrics.get("final_synthetic_copy_accuracy")),
            ]
        )
    if not rows:
        return [NO_DATA_LINE, ""]
    return _table_lines(
        [
            "Run ID",
            "Hyp.",
            "Tokens",
            "Tokens/s",
            "MFU",
            "Val bpb",
            "Branch Val bpb",
            "Δ Val bpb",
            "ICL Δ",
            "Copy Acc",
        ],
        [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        rows,
    )


def _h5_env_reward_lines(bundles: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for bundle in _bundles_with_hypothesis(bundles, ("H5",)):
        metrics = bundle["metrics"]
        rows.append(
            [
                f"`{bundle['run'].get('run_id')}`",
                _format_metric(metrics.get("oracle_gated_mean")),
                _format_metric(metrics.get("oracle_naive_mean")),
                _format_metric(metrics.get("lazy_gated_mean")),
                _format_metric(metrics.get("lazy_naive_mean")),
                _format_metric(metrics.get("unsafe_gated_mean")),
                _format_metric(metrics.get("unsafe_naive_mean")),
            ]
        )
    if not rows:
        return [NO_DATA_LINE, ""]
    return _table_lines(
        [
            "Run ID",
            "Oracle Gated",
            "Oracle Naive",
            "Lazy Gated",
            "Lazy Naive",
            "Unsafe Gated",
            "Unsafe Naive",
        ],
        [":---", "---:", "---:", "---:", "---:", "---:", "---:"],
        rows,
    )


def _h6_adaptation_lines(bundles: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for bundle in _bundles_with_hypothesis(bundles, ("H6",)):
        comp = bundle.get("adaptation_comparison")
        comp = comp if isinstance(comp, dict) else {}
        results = comp.get("results")
        if not isinstance(results, list):
            continue
        hypotheses = comp.get("hypotheses") if isinstance(comp.get("hypotheses"), dict) else {}
        llm_backend = comp.get("llm_backend") if isinstance(comp.get("llm_backend"), dict) else {}
        default_model = (
            comp.get("model_name")
            or llm_backend.get("model_name")
            or bundle.get("model_name")
            or _bundle_model_name(bundle)
            or "unknown"
        )
        for item in results:
            if not isinstance(item, dict):
                continue
            model = item.get("model_name") or item.get("model") or default_model
            h5_diff = item.get("h5_skill_difference", hypotheses.get("H5_skill_difference"))
            h6_diff = item.get("h6_skill_difference", hypotheses.get("H6_skill_difference"))
            skill = item.get("skill_bound_score", item.get("skill_score"))
            rag_knowledge = item.get("knowledge_bound_score", item.get("rag_knowledge_score"))
            rows.append(
                [
                    str(model),
                    f"`{item.get('strategy_name', item.get('strategy', 'unknown'))}`",
                    _format_metric(skill),
                    _format_metric(item.get("mean_gated_reward")),
                    _format_metric(rag_knowledge),
                    _format_metric(h5_diff),
                    _format_metric(h6_diff),
                ]
            )
    if not rows:
        return [NO_DATA_LINE, ""]
    return _table_lines(
        [
            "Model",
            "Strategy",
            "Skill Score",
            "Mean Gated Reward",
            "RAG Knowledge",
            "H5 Skill Δ",
            "H6 Skill Δ",
        ],
        [":---", ":---", "---:", "---:", "---:", "---:", "---:"],
        rows,
    )


def _h7_h8_pareto_lines(bundles: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for bundle in _bundles_with_hypothesis(bundles, ("H7", "H8")):
        pareto = bundle.get("surgery_pareto")
        pareto = pareto if isinstance(pareto, dict) else {}
        kls = pareto.get("kl_divergences") if isinstance(pareto.get("kl_divergences"), dict) else {}
        cka_guided = kls.get("cka_guided")
        if cka_guided is None:
            cka_guided = _resolved_value(bundle, "cka_guided_kl")
        bottom_k = kls.get("bottom_k_control")
        if bottom_k is None:
            bottom_k = _resolved_value(bundle, "bottom_k_control_kl")
        h8_diag = pareto.get("h8_diagnostics")
        if not isinstance(h8_diag, dict):
            h8_diag = {}
        before = pareto.get("write_proj_before", h8_diag.get("write_proj_before"))
        if before is None:
            before = h8_diag.get("layer5_write_proj_before")
        after = pareto.get("write_proj_after", h8_diag.get("write_proj_after"))
        if after is None:
            after = h8_diag.get("layer5_write_proj_after")
        if before is None or after is None:
            proj_str = "n/a"
        else:
            proj_str = f"{_format_metric(before)} → {_format_metric(after)}"
        max_ortho = pareto.get("max_ortho_residual", h8_diag.get("max_ortho_residual"))
        kl_collat = pareto.get("kl_ablation_collateral", h8_diag.get("kl_ablation_collateral"))
        rows.append(
            [
                f"`{bundle['run'].get('run_id')}`",
                f"`{bundle['run'].get('hypothesis_id')}`",
                _format_metric(cka_guided),
                _format_metric(bottom_k),
                _format_metric(pareto.get("h7_kl_reduction_ratio")),
                _format_metric(max_ortho, ".2e"),
                proj_str,
                _format_metric(kl_collat),
            ]
        )
    if not rows:
        return [NO_DATA_LINE, ""]
    return _table_lines(
        [
            "Run ID",
            "Hyp.",
            "CKA-Guided KL",
            "Bottom-k KL",
            "H7 KL Red.",
            "Max Ortho Res.",
            "Write Proj",
            "KL Ablation Collateral",
        ],
        [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"],
        rows,
    )


def _model_cell(bundle: dict[str, Any]) -> str:
    value = _bundle_model_name(bundle)
    return f"`{value}`" if value else "n/a"


def _hardware_cell(bundle: dict[str, Any]) -> str:
    value = _bundle_hardware(bundle)
    return f"`{value}`" if value else "n/a"


def _metric_cell(bundle: dict[str, Any], key: str, fmt: str = ".4f") -> str:
    return _format_metric(_resolved_value(bundle, key), fmt)


def _flag_cell(bundle: dict[str, Any], key: str) -> str:
    return _format_flag(_resolved_value(bundle, key))


def _surgery_style_table(
    bundles: list[dict[str, Any]],
    hypothesis_ids: tuple[str, ...],
    headers: list[str],
    aligns: list[str],
    cell_builders: list,
) -> list[str]:
    rows = []
    for bundle in _bundles_with_hypothesis(bundles, hypothesis_ids):
        rows.append([builder(bundle) for builder in cell_builders])
    if not rows:
        return [NO_DATA_LINE, ""]
    return _table_lines(headers, aligns, rows)


def _h9_lancet_lines(bundles: list[dict[str, Any]]) -> list[str]:
    return _surgery_style_table(
        bundles,
        ("H9",),
        [
            "Model",
            "Hardware",
            "Target Supp.",
            "H9 KL Red.",
            "Naive KL Collateral",
            "Lancet KL Collateral",
            "Anchor Ortho",
            "Max Row Drift",
        ],
        [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"],
        [
            _model_cell,
            _hardware_cell,
            lambda b: _metric_cell(b, "target_suppression_ratio"),
            lambda b: _metric_cell(b, "h9_kl_reduction_ratio"),
            lambda b: _metric_cell(b, "naive_kl_collateral"),
            lambda b: _metric_cell(b, "lancet_kl_collateral"),
            lambda b: _metric_cell(b, "anchor_orthogonality_error", ".2e"),
            lambda b: _metric_cell(b, "max_row_norm_drift", ".2e"),
        ],
    )


def _h10_grpo_lines(bundles: list[dict[str, Any]]) -> list[str]:
    return _surgery_style_table(
        bundles,
        ("H10",),
        [
            "Model",
            "Hardware",
            "Slop & Refusal",
            "Gated vs Naive",
            "Post-Surgery",
            "Unprojected Re-leak",
            "Subspace-Safe Re-leak",
            "Suppression Retained",
        ],
        [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"],
        [
            _model_cell,
            _hardware_cell,
            lambda b: _metric_cell(b, "slop_and_refusal_rate"),
            lambda b: _metric_cell(b, "h10_gated_vs_naive_gain"),
            lambda b: _metric_cell(b, "h10_post_surgery_gain"),
            lambda b: _metric_cell(b, "unprojected_lora_releakage", ".2e"),
            lambda b: _metric_cell(b, "subspace_safe_lora_releakage", ".2e"),
            lambda b: _metric_cell(b, "post_merge_suppression_retention"),
        ],
    )


def _h11_fp8_lines(bundles: list[dict[str, Any]]) -> list[str]:
    return _surgery_style_table(
        bundles,
        ("H11",),
        [
            "Model",
            "Hardware",
            "FP8 Retention",
            "H11 KL Red.",
            "Re-leak (Naive)",
            "Re-leak (Lancet)",
            "Max Crest",
            "Provider Ready",
        ],
        [":---", ":---", "---:", "---:", "---:", "---:", "---:", ":---:"],
        [
            _model_cell,
            _hardware_cell,
            lambda b: _metric_cell(b, "fp8_retention_ratio"),
            lambda b: _metric_cell(b, "h11_fp8_kl_reduction_ratio"),
            lambda b: _metric_cell(b, "fp8_subspace_releakage_naive", ".2e"),
            lambda b: _metric_cell(b, "fp8_subspace_releakage_lancet", ".2e"),
            lambda b: _metric_cell(b, "max_row_crest_factor"),
            lambda b: _flag_cell(b, "provider_ready"),
        ],
    )


def _h12_merge_lines(bundles: list[dict[str, Any]]) -> list[str]:
    return _surgery_style_table(
        bundles,
        ("H12",),
        [
            "Model",
            "Hardware",
            "SQL Retention",
            "Unslop Retention",
            "Slop & Refusal",
            "Naive Merge Re-leak",
            "Subspace-Safe Re-leak",
            "TIES vs Naive Gain",
            "Max Crest",
            "Provider Ready",
        ],
        [":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", ":---:"],
        [
            _model_cell,
            _hardware_cell,
            lambda b: _metric_cell(b, "sql_incident_retention_ratio"),
            lambda b: _metric_cell(b, "unslop_retention_ratio"),
            lambda b: _metric_cell(b, "slop_and_refusal_rate"),
            lambda b: _metric_cell(b, "naive_merge_releakage", ".2e"),
            lambda b: _metric_cell(b, "subspace_safe_merge_releakage", ".2e"),
            lambda b: _metric_cell(b, "ties_biprojected_vs_naive_merge_gain"),
            lambda b: _metric_cell(b, "max_row_crest_factor"),
            lambda b: _flag_cell(b, "provider_ready"),
        ],
    )


def _release_lookup(
    bundle: dict[str, Any],
    key: str,
    sources: tuple[str, ...],
) -> Any:
    """Resolve a release value from the named bundle JSON docs (top level or metrics)."""
    for source in sources:
        doc = bundle.get(source)
        if not isinstance(doc, dict):
            continue
        if doc.get(key) is not None:
            return doc[key]
        metrics = doc.get("metrics")
        if isinstance(metrics, dict) and metrics.get(key) is not None:
            return metrics[key]
    return None


def _tenet_release_lines(releases: list[dict[str, Any]]) -> list[str]:
    rows: list[list[str]] = []
    for bundle in releases:
        manifest = bundle.get("release_manifest")
        if not isinstance(manifest, dict):
            continue
        release_id = (
            _release_lookup(
                bundle, "release_id", ("release_manifest", "cka_quant_recipe", "lancet_recipe")
            )
            or bundle["release_dir"].name
        )
        base_model = _release_lookup(
            bundle, "base_model", ("release_manifest", "lancet_recipe", "cka_quant_recipe")
        )
        default_quant = _release_lookup(
            bundle, "default_quant", ("release_manifest", "cka_quant_recipe")
        ) or _release_lookup(bundle, "quant", ("cka_quant_recipe",))
        estimated_bpw = _release_lookup(
            bundle,
            "estimated_avg_bpw",
            ("cka_quant_recipe", "release_manifest", "lancet_recipe"),
        )
        top_k = _release_lookup(
            bundle, "lancet_top_k_layers", ("lancet_recipe", "release_manifest")
        ) or _release_lookup(bundle, "top_k_layers", ("lancet_recipe", "release_manifest"))
        onboarding = bundle.get("openrouter_onboarding")
        h11_audit = (
            onboarding.get("fp8_and_crest_proof", {}).get("h11_fp8_provider_audit", {})
            if isinstance(onboarding, dict)
            else {}
        )
        if not isinstance(h11_audit, dict):
            h11_audit = {}
        fp8_retention = _release_lookup(
            bundle,
            "fp8_retention_ratio",
            ("release_manifest", "cka_quant_recipe", "lancet_recipe", "provider_readiness"),
        )
        if fp8_retention is None:
            fp8_retention = h11_audit.get("fp8_retention_ratio")
        h11_kl = _release_lookup(
            bundle,
            "h11_fp8_kl_reduction_ratio",
            ("release_manifest", "cka_quant_recipe", "lancet_recipe", "provider_readiness"),
        )
        if h11_kl is None:
            h11_kl = h11_audit.get("h11_fp8_kl_reduction_ratio")
        crest = _release_lookup(
            bundle,
            "max_row_crest_factor",
            ("provider_readiness", "release_manifest", "cka_quant_recipe", "lancet_recipe"),
        )
        provider_ready = _release_lookup(
            bundle, "provider_ready", ("release_manifest", "provider_readiness")
        ) or _release_lookup(bundle, "ready", ("release_manifest", "provider_readiness"))
        rows.append(
            [
                f"`{release_id}`",
                f"`{base_model}`" if base_model else "n/a",
                f"`{default_quant}`" if default_quant else "n/a",
                _format_metric(estimated_bpw, ".2f"),
                f"{int(top_k):d}" if isinstance(top_k, (int, float)) else "n/a",
                _format_metric(fp8_retention),
                _format_metric(h11_kl),
                _format_metric(crest),
                _format_flag(provider_ready),
            ]
        )
    if not rows:
        return [NO_DATA_LINE, ""]
    return _table_lines(
        [
            "Release",
            "Base Model",
            "Default Quant",
            "Est. Avg bpw",
            "Lancet Top-k",
            "FP8 Retention",
            "H11 KL Red.",
            "Max Crest",
            "Provider Ready",
        ],
        [":---", ":---", ":---", "---:", "---:", "---:", "---:", "---:", ":---:"],
        rows,
    )
