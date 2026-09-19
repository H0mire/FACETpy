"""Build per-model result views from explicit thesis-catalog associations."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import yaml

from masterthesis_guide.reproduce import ROOT, load_catalog, repository_path


class Sources:
    """Cache original records without changing their bytes."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = {}

    def read(self, relative: str):
        if relative not in self.cache:
            data = repository_path(relative, self.root).read_bytes()
            text = data.decode("utf-8")
            suffix = Path(relative).suffix
            if suffix == ".jsonl":
                value = [json.loads(line) for line in text.splitlines() if line.strip()]
            elif suffix == ".csv":
                value = list(csv.DictReader(io.StringIO(text)))
            elif suffix in {".yaml", ".yml"}:
                value = yaml.safe_load(text)
            elif suffix == ".json":
                value = json.loads(text)
            else:
                value = text
            self.cache[relative] = ({"path": relative, "sha256": hashlib.sha256(data).hexdigest()}, value)
        return self.cache[relative]


def finite_values(value):
    """Keep absent or non-finite measurements out of numeric comparisons."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_values(item) for item in value]
    return value


def json_text(value) -> str:
    return json.dumps(finite_values(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def hyperparameters(experiment: dict, sources: Sources) -> dict:
    """Prefer the original resolved config; retain reconstruction provenance."""
    result = {
        "status": "unavailable",
        "source": None,
        "reproduction_config": experiment.get("config"),
        "configuration_provenance": experiment.get("configuration_provenance"),
        "parameters": {},
    }
    source = experiment.get("original_config") or experiment.get("config")
    if source:
        info, config = sources.read(source)
        result.update(
            status="original_resolved" if experiment.get("original_config") else "reproduction_config",
            source=info,
            parameters={key: config[key] for key in ("model", "data", "training") if key in config},
        )
        if experiment.get("configuration_provenance", {}).get("status") == "reconstructed":
            result["status"] = "reconstructed"
    else:
        for path in experiment.get("evidence", []):
            if path.endswith("_training.json"):
                info, record = sources.read(path)
                if "hyperparameters" in record:
                    result.update(status="partial_training_record", source=info, parameters=record["hyperparameters"])
    return result


def training_history(experiment: dict, sources: Sources) -> dict:
    paths = [path for path in experiment.get("evidence", []) if path.endswith("training.jsonl")]
    if len(paths) > 1:
        raise ValueError("Multiple histories require an explicit run association")
    if not paths:
        return {"status": "unavailable", "source": None, "records": []}
    info, rows = sources.read(paths[0])
    epochs = [row.get("epoch") for row in rows]
    if any(not isinstance(epoch, (int, float)) or not math.isfinite(epoch) for epoch in epochs):
        raise ValueError(f"Invalid epoch in {paths[0]}")
    if any(right <= left for left, right in zip(epochs, epochs[1:], strict=False)):
        raise ValueError(f"Repeated or restarted epochs need separate curves: {paths[0]}")
    recorded = any(
        isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
        for row in rows
        for key in ("train_loss", "val_loss")
    )
    return {"status": "available" if recorded else "no_finite_losses", "source": info, "records": rows}


def training_curve(rows: list[dict], *, phase: int, model_id: str, experiment_id: str) -> str:
    """Plot recorded epoch losses; preserve gaps and use a run-specific scale."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with plt.rc_context({"svg.hashsalt": "facetpy-model-results-v1", "font.family": "DejaVu Sans", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(7.0, 3.5), layout="constrained")
        for key, label, color in (("train_loss", "Training", "#2f6d94"), ("val_loss", "Validation", "#c27431")):
            values = [row.get(key) for row in rows]
            values = [v if isinstance(v, (int, float)) and math.isfinite(v) else math.nan for v in values]
            if any(math.isfinite(v) for v in values):
                ax.plot(
                    [row["epoch"] for row in rows],
                    values,
                    label=label,
                    color=color,
                    linewidth=1.5,
                    marker=".",
                    markersize=3,
                )
        ax.set(xlabel="Epoch", ylabel="Recorded loss", title=f"Phase {phase} · {model_id}")
        fig.suptitle(textwrap.fill(experiment_id, 75), fontsize=8)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.text(0.99, -0.015, "Objective and scale belong to this run; no smoothing.", ha="right", fontsize=8)
        stream = io.StringIO()
        fig.savefig(stream, format="svg", metadata={"Date": None}, bbox_inches="tight")
        plt.close(fig)
    return "\n".join(line.rstrip() for line in stream.getvalue().splitlines()) + "\n"


def evaluation(experiment_id: str, experiment: dict, catalog: dict, sources: Sources) -> dict:
    """Extract only explicit run records; shared tables remain shared tables."""
    records = []
    source_records = []
    for path in experiment.get("evidence", []):
        info, value = sources.read(path)
        source_records.append(info)
        name = Path(path).name
        if name.endswith("_training.json"):
            if "agreement_with_aas" in value:
                records.append(
                    {"source": info, "selection": "agreement_with_aas", "values": value["agreement_with_aas"]}
                )
        elif any(
            word in name
            for word in (
                "metrics",
                "eval",
                "comparison",
                "preservation",
                "reproduction",
                "anc_contribution",
                "legacy_vs_current",
            )
        ):
            # Some run-owned comparison files contain several named arms. Keep them together.
            records.append({"source": info, "selection": "run-owned evidence (arm names retained)", "values": value})
    if selection := experiment.get("result"):
        collection = catalog["results"][selection["collection"]]
        if experiment_id not in collection.get("experiments", []):
            raise ValueError(f"{experiment_id}: grid row has no catalog association")
        info, value = sources.read(collection["path"])
        row = selection["row"]
        if not isinstance(row, int) or row < 0 or row >= len(value["zeilen"]):
            raise ValueError(f"{experiment_id}: invalid explicit grid row {row}")
        records.append(
            {
                "source": info,
                "selection": {"key": "zeilen", "row": row},
                "values": value["zeilen"][row],
                "context": {k: v for k, v in value.items() if k != "zeilen"},
            }
        )
    shared = []
    for rid, collection in catalog["results"].items():
        if experiment_id not in collection.get("experiments", []):
            continue
        info, value = sources.read(collection["path"])
        shared.append(
            {
                "collection": rid,
                "source": info,
                "scope": collection.get("scope"),
                "association": "catalog collection; not every row belongs to this run",
            }
        )
        if isinstance(value, list) and value and isinstance(value[0], dict) and "experiment" in value[0]:
            matched = [row for row in value if row["experiment"] == experiment_id]
            if matched:
                records.append(
                    {
                        "source": info,
                        "selection": {"experiment": experiment_id},
                        "values": matched,
                        "scope": collection.get("scope"),
                    }
                )
    status = "recorded_evidence" if records else "shared_collections_only" if shared else "unavailable"
    return {
        "status": status,
        "catalog_outcome": experiment["outcome"],
        "verification": experiment["verification"],
        "protocol": catalog["protocols"][experiment["protocol"]],
        "notes": {
            key: experiment[key]
            for key in ("identity_note", "selection_note", "outcome_reason", "verification_reason")
            if key in experiment
        },
        "nonfinite_values": "Represented as null; see unchanged source files for the original values.",
        "records": records,
        "shared_collections": shared,
        "evidence_sources": source_records,
    }


def rst_index(model_id: str, runs: list[dict], directory: Path, root: Path) -> str:
    """Use source-root paths so the same index can be included by a family page."""

    def download(path: Path, label: str) -> str:
        target = Path(os.path.relpath(path, root / "docs/source")).as_posix()
        return f":download:`{label} </{target}>`"

    lines = [
        ".. Generated by python -m masterthesis_guide.reproduce model-results; do not edit.",
        "",
        f"Recorded results for ``{model_id}``.",
        "",
        "These files summarize existing records. No training or evaluation was rerun.",
        "Each experiment keeps its own phase, dataset, objective and verification status.",
        "A recorded curve does not establish successful correction. Missing curves stay missing.",
        "",
        download(directory / "manifest.json", "Manifest and source associations") + ".",
        "",
    ]
    if not runs:
        lines += [
            "No experiment is assigned to this exact variant in the thesis catalog.",
            "Hyperparameters, training curves and evaluation records are not assigned here.",
            "This does not establish that no other material exists.",
            "",
        ]
        return "\n".join(lines)
    lines += [
        ".. list-table:: Recorded experiments",
        "   :header-rows: 1",
        "   :widths: 32 22 28 18",
        "",
        "   * - Experiment",
        "     - Phase and dataset",
        "     - Files",
        "     - Recorded status",
    ]
    for run in runs:
        folder = directory / run["experiment"]
        links = [
            download(folder / "hyperparameters.yaml", "Hyperparameters"),
            download(folder / "evaluation.json", "Evaluation"),
        ]
        if run["training"]["status"] == "available":
            links.append(download(folder / "training_curve.svg", "Training curve (SVG)"))
        else:
            links.append("No recorded epoch curve")
        lines += [
            f"   * - ``{run['experiment']}``",
            f"     - Phase {run['phase']}; training: ``{run.get('training_dataset') or 'not recorded'}``; evaluation: ``{run['dataset']}``",
            "     - " + "; ".join(links),
            f"     - Outcome: ``{run['outcome']}``; verification: ``{run['verification']}``. Parameters: ``{run['hyperparameters_status']}``. Evaluation: ``{run['evaluation_status']}``.",
        ]
    lines += [
        "",
        "Hyperparameter files preserve the original resolved settings where available.",
        "The reproduction configuration records current module paths. Partial and reconstructed",
        "settings are labelled. Evaluation files retain source paths, hashes, comparison arms",
        "and known limits. Shared collections are links, not inferred per-run measurements.",
        "",
    ]
    return "\n".join(lines)


def generate_model_results(catalog=None, *, root: Path = ROOT, check: bool = False) -> dict[str, int]:
    catalog = catalog or load_catalog(root / "masterthesis_guide/catalog.yaml")
    sources = Sources(root)
    owners = defaultdict(list)
    for eid, experiment in catalog["experiments"].items():
        if experiment["model"] is not None:
            if experiment["model"] not in catalog["models"]:
                raise ValueError(f"Unknown model for {eid}")
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", eid):
                raise ValueError(f"Unsafe experiment ID {eid}")
            owners[experiment["model"]].append((eid, experiment))
    errors = []
    totals = {"models": 0, "experiments": 0, "curves": 0, "models_without_assigned_runs": 0}

    def emit(path: Path, content: str):
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                errors.append(str(path.relative_to(root)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    for mid, model in catalog["models"].items():
        directory = repository_path(model["readme"], root).parent / "results"
        runs = []
        for eid, experiment in sorted(owners[mid], key=lambda pair: (pair[1]["phase"], pair[0])):
            context = {
                "experiment": eid,
                "model": mid,
                "phase": experiment["phase"],
                "dataset": experiment["dataset"],
                "training_dataset": experiment.get("training_dataset"),
                "protocol": experiment["protocol"],
            }
            parameters = hyperparameters(experiment, sources)
            history = training_history(experiment, sources)
            metrics = evaluation(eid, experiment, catalog, sources)
            emit(
                directory / eid / "hyperparameters.yaml",
                yaml.safe_dump(finite_values({**context, **parameters}), sort_keys=False, allow_unicode=True),
            )
            emit(directory / eid / "evaluation.json", json_text({**context, **metrics}))
            if history["status"] == "available":
                emit(
                    directory / eid / "training_curve.svg",
                    training_curve(history["records"], phase=experiment["phase"], model_id=mid, experiment_id=eid),
                )
                totals["curves"] += 1
            runs.append(
                {
                    **context,
                    "outcome": experiment["outcome"],
                    "verification": experiment["verification"],
                    "hyperparameters_status": parameters["status"],
                    "evaluation_status": metrics["status"],
                    "training": {k: v for k, v in history.items() if k != "records"},
                    "recorded_epochs": len(history["records"]),
                    "artifacts": {aid: catalog["artifacts"][aid] for aid in experiment["artifacts"]},
                    "commands": experiment["commands"],
                }
            )
        emit(
            directory / "manifest.json",
            json_text(
                {
                    "schema_version": 1,
                    "model": mid,
                    "catalog": "masterthesis_guide/catalog.yaml",
                    "source_snapshot": catalog["source_snapshot"],
                    "path_convention": "Source and artifact paths are relative to the repository root.",
                    "experiments": runs,
                }
            ),
        )
        emit(directory / "index.rst", rst_index(mid, runs, directory, root))
        totals["models"] += 1
        totals["experiments"] += len(runs)
        totals["models_without_assigned_runs"] += not bool(runs)
    if errors:
        raise ValueError("Outdated model results:\n" + "\n".join(errors))
    return totals
