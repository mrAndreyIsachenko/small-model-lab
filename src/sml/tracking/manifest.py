"""Machine-readable run manifest recorder for results/<run_id>/."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from sml.config import ExperimentSpec
from sml.tracking.cost import compute_run_cost, estimate_experiment_compute
from sml.tracking.environment import capture_environment

RunMode = Literal["measured", "dry_run"]
RunStatus = Literal["running", "completed", "failed"]


class RunRecorder:
    """Manages the lifecycle and artifacts of a single run in results/<run_id>/."""

    def __init__(
        self,
        run_id: str,
        spec: ExperimentSpec,
        output_root: Path,
        mode: RunMode = "measured",
        repo_root: Path | None = None,
    ) -> None:
        self.run_id = run_id
        self.spec = spec
        self.mode = mode
        self.repo_root = repo_root if repo_root is not None else Path.cwd()
        self.run_dir = output_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = datetime.now(UTC)
        self.env = capture_environment(self.repo_root)
        self.pre_run_estimate = estimate_experiment_compute(self.spec)
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self._write_initial_artifacts()

    def _write_initial_artifacts(self) -> None:
        with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.spec.to_dict(), handle, sort_keys=False)

        with (self.run_dir / "environment.json").open("w", encoding="utf-8") as handle:
            json.dump(self.env, handle, indent=2)

        run_meta = {
            "run_id": self.run_id,
            "experiment_id": self.spec.experiment_id,
            "hypothesis_id": self.spec.hypothesis_id,
            "hypothesis": self.spec.hypothesis,
            "prediction": self.spec.prediction,
            "acceptance_criterion": self.spec.acceptance_criterion,
            "mode": self.mode,
            "status": "running",
            "seed": self.spec.seed,
            "model_name": self.spec.model.name,
            "total_params": self.spec.model.total_params,
            "non_embedding_params": self.spec.model.non_embedding_params,
            "dataset_name": self.spec.data.name,
            "git_sha": self.env["git_sha"],
            "git_dirty": self.env["git_dirty"],
            "started_at_utc": self.started_at.isoformat(),
            "completed_at_utc": None,
            "pre_run_estimate": self.pre_run_estimate,
            "conclusion": None,
        }
        with (self.run_dir / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, indent=2)

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        record = {"step": step, "timestamp_utc": datetime.now(UTC).isoformat(), **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def finalize(
        self,
        final_metrics: dict[str, Any],
        active_seconds: float,
        billed_seconds: float | None = None,
        status: RunStatus = "completed",
        conclusion: str = "",
    ) -> Path:
        completed_at = datetime.now(UTC)

        with (self.run_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(final_metrics, handle, indent=2)

        cost_record = compute_run_cost(
            hardware=self.spec.hardware,
            active_seconds=active_seconds,
            billed_seconds=billed_seconds,
            estimated_pre_run_cost_usd=float(self.pre_run_estimate["estimated_cost_usd"]),
        )
        with (self.run_dir / "cost.json").open("w", encoding="utf-8") as handle:
            json.dump(cost_record.to_dict(), handle, indent=2)

        with (self.run_dir / "run.json").open("r", encoding="utf-8") as handle:
            run_meta = json.load(handle)

        run_meta["status"] = status
        run_meta["completed_at_utc"] = completed_at.isoformat()
        run_meta["conclusion"] = conclusion

        with (self.run_dir / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, indent=2)

        summary_md = self._render_summary_markdown(run_meta, final_metrics, cost_record.to_dict())
        summary_path = self.run_dir / "summary.md"
        summary_path.write_text(summary_md, encoding="utf-8")
        return self.run_dir

    def _render_summary_markdown(
        self,
        run_meta: dict[str, Any],
        final_metrics: dict[str, Any],
        cost_meta: dict[str, Any],
    ) -> str:
        total_p = run_meta["total_params"]
        non_emb_p = run_meta["non_embedding_params"]
        hw_name = cost_meta["hardware_name"]
        hw_rate = cost_meta["hourly_rate_usd"]
        lines = [
            f"# Run Summary: `{self.run_id}`",
            "",
            f"- **Experiment:** `{run_meta['experiment_id']}` (`{run_meta['hypothesis_id']}`)",
            f"- **Mode:** `{run_meta['mode']}`",
            f"- **Status:** `{run_meta['status']}`",
            f"- **Model:** `{run_meta['model_name']}` "
            f"(`{total_p:,}` total / `{non_emb_p:,}` non-emb)",
            f"- **Dataset:** `{run_meta['dataset_name']}`",
            f"- **Hardware:** `{hw_name}` (`${hw_rate:.2f}/hr`)",
            f"- **Git SHA:** `{run_meta['git_sha']}` (dirty: `{run_meta['git_dirty']}`)",
            "",
            "## Hypothesis & Prediction",
            f"- **Hypothesis:** {run_meta['hypothesis']}",
            f"- **Prediction:** {run_meta['prediction']}",
            f"- **Acceptance Criterion:** `{run_meta['acceptance_criterion']}`",
            "",
            "## Final Metrics",
        ]
        for key, value in final_metrics.items():
            lines.append(f"- **`{key}`:** `{value}`")
        lines.extend(
            [
                "",
                "## Compute & Cost",
                f"- **Active GPU-Hours:** `{cost_meta['active_gpu_hours']}`",
                f"- **Billed GPU-Hours:** `{cost_meta['billed_gpu_hours']}`",
                f"- **Billed Cost (USD):** `${cost_meta['billed_cost_usd']:.4f}` "
                f"(pre-run estimate: `${cost_meta['estimated_pre_run_cost_usd']:.4f}`)",
                "",
                "## Conclusion",
                run_meta.get("conclusion") or "No conclusion recorded.",
                "",
            ]
        )
        return "\n".join(lines)
