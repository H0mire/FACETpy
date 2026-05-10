"""Standardized storage for model evaluation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

EVALUATION_SCHEMA_VERSION = "facetpy.model_evaluation.v1"


def _default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_default(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None

    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
            return f"{value:.6e}"
        return f"{value:.6f}"
    return str(value)


def _flatten_numeric_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float | int | bool]:
    flat: dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        metric_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            flat[metric_key] = value
        elif isinstance(value, (int, float)):
            flat[metric_key] = value
        elif isinstance(value, dict):
            flat.update(_flatten_numeric_metrics(value, metric_key))
    return flat


@dataclass(frozen=True)
class ModelEvaluationRun:
    """Filesystem locations for one standardized model evaluation run."""

    model_id: str
    run_id: str
    run_dir: Path
    docs_dir: Path
    manifest_path: Path
    metrics_path: Path
    summary_path: Path


@dataclass
class ModelEvaluationWriter:
    """Write comparable model evaluation outputs.

    Large artifacts such as plots are stored in ``output_root``. Stable,
    versionable model notes live in ``docs_root/<model_id>/documentation``.
    The same manifest schema is used for every model so evaluation runs can be
    compared without reverse-engineering model-specific scripts.
    """

    model_id: str
    model_name: str
    model_description: str
    output_root: Path = Path("output/model_evaluations")
    docs_root: Path = Path("src/facet/models")
    run_id: str = field(default_factory=_default_run_id)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")

    @property
    def run(self) -> ModelEvaluationRun:
        run_dir = Path(self.output_root) / self.model_id / self.run_id
        docs_dir = Path(self.docs_root) / self.model_id / "documentation"
        return ModelEvaluationRun(
            model_id=self.model_id,
            run_id=self.run_id,
            run_dir=run_dir,
            docs_dir=docs_dir,
            manifest_path=run_dir / "evaluation_manifest.json",
            metrics_path=run_dir / "metrics.json",
            summary_path=run_dir / "evaluation_summary.md",
        )

    def write(
        self,
        *,
        metrics: dict[str, Any],
        config: dict[str, Any],
        artifacts: dict[str, str | Path] | None = None,
        interpretation: str = "",
        limitations: list[str] | None = None,
    ) -> ModelEvaluationRun:
        run = self.run
        run.run_dir.mkdir(parents=True, exist_ok=True)
        run.docs_dir.mkdir(parents=True, exist_ok=True)

        artifacts = artifacts or {}
        limitations = limitations or []
        flat_metrics = _flatten_numeric_metrics(metrics)

        _write_json(run.metrics_path, {"metrics": metrics, "flat_metrics": flat_metrics})

        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_description": self.model_description,
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run.run_dir),
            "docs_dir": str(run.docs_dir),
            "metrics_path": str(run.metrics_path),
            "summary_path": str(run.summary_path),
            "artifacts": {key: str(path) for key, path in artifacts.items()},
            "config": config,
            "primary_comparison_contract": {
                "metrics_json": "metrics.json",
                "flat_metrics_key": "flat_metrics",
                "summary_markdown": "evaluation_summary.md",
            },
        }
        _write_json(run.manifest_path, manifest)
        run.summary_path.write_text(
            self._format_summary(
                metrics=metrics,
                flat_metrics=flat_metrics,
                config=config,
                artifacts=artifacts,
                interpretation=interpretation,
                limitations=limitations,
            ),
            encoding="utf-8",
        )
        self._ensure_docs_index(run)
        return run

    def _format_summary(
        self,
        *,
        metrics: dict[str, Any],
        flat_metrics: dict[str, float | int | bool],
        config: dict[str, Any],
        artifacts: dict[str, str | Path],
        interpretation: str,
        limitations: list[str],
    ) -> str:
        metric_rows = "\n".join(
            f"| `{key}` | {_format_metric_value(value)} |" for key, value in sorted(flat_metrics.items())
        )
        if not metric_rows:
            metric_rows = "| _none_ | _n/a_ |"

        artifact_rows = "\n".join(f"- `{name}`: `{path}`" for name, path in sorted(artifacts.items()))
        if not artifact_rows:
            artifact_rows = "- No external artifacts recorded."

        limitation_rows = "\n".join(f"- {item}" for item in limitations)
        if not limitation_rows:
            limitation_rows = "- No limitations recorded."

        config_block = json.dumps(config, indent=2, default=_json_default)
        metrics_block = json.dumps(metrics, indent=2, default=_json_default)
        interpretation_text = interpretation.strip() or "No interpretation recorded."

        return f"""# Evaluation Run: {self.model_name}

## Identity

- model id: `{self.model_id}`
- run id: `{self.run_id}`
- schema: `{EVALUATION_SCHEMA_VERSION}`

## Metric Summary

| Metric | Value |
| --- | ---: |
{metric_rows}

## Interpretation

{interpretation_text}

## Artifacts

{artifact_rows}

## Limitations

{limitation_rows}

## Configuration

```json
{config_block}
```

## Raw Metrics

```json
{metrics_block}
```
"""

    def _ensure_docs_index(self, run: ModelEvaluationRun) -> None:
        index_path = run.docs_dir / "evaluations.md"
        if index_path.exists():
            return
        index_path.write_text(
            f"""# {self.model_name} Evaluations

This file indexes standardized evaluation runs for `{self.model_id}`.

Large generated plots and JSON artifacts are written to:

```text
output/model_evaluations/{self.model_id}/<run_id>/
```

Each run must contain:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`

Use the schema `{EVALUATION_SCHEMA_VERSION}` for comparable runs.
""",
            encoding="utf-8",
        )
