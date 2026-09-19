"""Resolve the thesis catalog for repository-based reproduction commands."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import re
import shlex
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "masterthesis_guide/catalog.yaml"


class UniqueKeysLoader(yaml.SafeLoader):
    """Reject repeated IDs instead of silently replacing their records."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate catalog key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeysLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_catalog(path: Path = CATALOG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=UniqueKeysLoader)


def repository_path(relative: str, root: Path = ROOT) -> Path:
    """Accept only paths within this checkout."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected a repository-relative path: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the supplied root: {relative}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_path(dataset_id: str, catalog=None, *, data_root: Path | None = None) -> Path:
    """Resolve an external dataset without fetching or changing its contents."""
    catalog = catalog or load_catalog()
    record = catalog["datasets"][dataset_id]
    directory = data_root or os.environ.get("FACETPY_ARTIFACT_DIR")
    if directory is None:
        raise FileNotFoundError(
            "Set FACETPY_ARTIFACT_DIR or pass --data-root. The directory must contain "
            f"the recorded relative path {record['external_paths'][0]}."
        )
    path = repository_path(record["external_paths"][0], Path(directory).expanduser())
    if not path.is_file():
        raise FileNotFoundError(f"Dataset {dataset_id} is missing: {path}")
    return path


def load_holdout(dataset_path: Path, indices):
    """Read the recorded proof-fit fields using the saved window indices."""
    import numpy as np

    with np.load(dataset_path, allow_pickle=True) as data:
        return {
            **{
                key: data[key][indices].astype(np.float32, copy=False)
                for key in ("noisy_context", "noisy_center", "clean_center", "artifact_center")
            },
            "sfreq": float(np.asarray(data["sfreq"]).reshape(-1)[0]),
        }


def _artifact_record(experiment_id: str, catalog=None, *, device="cpu") -> tuple[str, dict]:
    """Select the recorded artifact without reading its local file."""
    catalog = catalog or load_catalog()
    experiments = catalog["experiments"]
    if not isinstance(experiment_id, str) or experiment_id not in experiments:
        matches = get_close_matches(experiment_id, experiments, n=3) if isinstance(experiment_id, str) else []
        suggestion = "\nDid you mean:\n" + "\n".join(f"  {match}" for match in matches) if matches else ""
        raise ValueError(
            f"Unknown experiment ID: {experiment_id!r}.{suggestion}\n"
            "Choose an ID from the 'experiments' section in masterthesis_guide/catalog.yaml."
        )
    experiment = experiments[experiment_id]
    selected = experiment.get("inference_artifact")
    candidates = [(aid, catalog["artifacts"][aid]) for aid in ([selected] if selected else experiment["artifacts"])]
    if not candidates:
        raise FileNotFoundError(f"{experiment_id} has no available model artifact; see its catalog gap.")
    source_loader = experiment.get("model") in {"d4pm", "denoise_mamba"}
    if source_loader:
        candidates = [(aid, item) for aid, item in candidates if item["kind"] == "checkpoint"]
    if not candidates:
        raise FileNotFoundError(f"{experiment_id} requires its source checkpoint, not the traced export.")

    def priority(candidate):
        _, item = candidate
        cpu = item["role"] == "cpu_export" and not device.startswith("cuda")
        return (not cpu, item["kind"] != "export", item["role"] != "evaluated")

    return sorted(candidates, key=priority)[0]


def check_downloaded(experiment_id: str, *, device="cpu") -> None:
    """Check local weights and explain how to fetch them when missing; download nothing."""
    _, record = _artifact_record(experiment_id, device=device)
    from facet.models.masterthesis.adapters import require_artifact

    path = repository_path(record["path"])
    try:
        require_artifact(path)
    except FileNotFoundError:
        include = shlex.quote(record["path"])
        raise FileNotFoundError(
            f"The model weights for '{experiment_id}' have not been downloaded.\n"
            f"Required file: {record['path']}\n\n"
            "Install Git LFS from https://git-lfs.com, then run these commands\n"
            "in the repository root:\n\n"
            "  git lfs install\n"
            f'  git lfs pull --include={include} --exclude=""\n\n'
            "Then run the example again. The download requires the file to be\n"
            "available on the remote and your account to have access."
        ) from None


def selected_artifact(experiment_id: str, catalog=None, *, device="cpu") -> tuple[str, Path]:
    """Select a recorded artifact; prefer an explicit CPU export on non-CUDA devices."""
    aid, record = _artifact_record(experiment_id, catalog, device=device)
    from facet.models.masterthesis.adapters import require_artifact

    path = require_artifact(repository_path(record["path"]))
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Artifact size does not match the catalog: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Artifact SHA-256 does not match the catalog: {path}")
    return aid, path


def adapter(experiment_id: str, catalog=None, *, device="cpu"):
    """Build the installed-library adapter with explicit model inputs."""
    catalog = catalog or load_catalog()
    _, path = selected_artifact(experiment_id, catalog, device=device)
    experiment = catalog["experiments"][experiment_id]
    if experiment["model"] == "legacy_cascaded_dae":
        from facet.models.masterthesis.legacy_cascaded_dae import LegacyDLAdapter

        return LegacyDLAdapter(path, device=device)
    from facet.models.masterthesis.adapters import FamilyAdapter

    model_id = experiment["model"].replace("_deployment_edition", "_deployment")
    factory = None
    kwargs = {}
    if path.suffix != ".ts" and model_id not in {"d4pm", "denoise_mamba"}:
        if "config" not in experiment:
            raise FileNotFoundError(f"{experiment_id} lacks its resolved model configuration.")
        config = yaml.safe_load(repository_path(experiment["config"]).read_text())
        factory = config["model"]["factory"]
        kwargs = config["model"].get("kwargs", {})
    from dataclasses import replace

    from facet.models.masterthesis.adapters import DEPLOYMENT_SPECS, FAMILY_SPECS

    spec = {**FAMILY_SPECS, **DEPLOYMENT_SPECS}[model_id]
    if experiment.get("config") and "_deployment" in model_id:
        config = yaml.safe_load(repository_path(experiment["config"]).read_text())
        data_kwargs = config.get("data", {}).get("kwargs", {})
        packing = data_kwargs.get("packing")
        axis = data_kwargs.get("axis")
        if axis:
            packing = "b1ts" if axis == "epochs" else "bcs"
        if packing:
            spec = replace(
                spec,
                packing=packing,
                context="single" if packing in {"b1s", "bcs"} else "stack",
                multichannel=packing in {"bcs", "bcts", "btcs"},
            )
    return FamilyAdapter(
        model_id, checkpoint=path, model_factory=factory, model_kwargs=kwargs, packing_spec=spec, device=device
    )


def executable_config(experiment_id: str, catalog=None, *, data_root=None, output_dir: Path, device="cpu") -> dict:
    """Resolve recorded dataset paths and redirect generated outputs explicitly."""
    catalog = catalog or load_catalog()
    experiment = catalog["experiments"][experiment_id]
    if "config" not in experiment:
        raise FileNotFoundError(f"{experiment_id} has no executable training configuration.")
    config = copy.deepcopy(yaml.safe_load(repository_path(experiment["config"]).read_text()))

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/").removeprefix("./")
            for dataset_id, dataset in catalog["datasets"].items():
                for relative in dataset["external_paths"]:
                    if normalized == relative or normalized.endswith("/" + relative):
                        return str(data_path(dataset_id, catalog, data_root=data_root))
        return value

    config = replace(config)
    # Output settings belong to this invocation, not to the immutable provenance.
    section = config.setdefault("training", {})
    section["output_dir"] = str(output_dir.expanduser().resolve())
    config["model"]["device"] = device
    return config


def validate(catalog: dict, *, root=ROOT, hashes=False) -> list[str]:
    """Validate identities, associations and files; weight bytes are optional for CI."""
    errors = []
    required = {"models", "datasets", "artifacts", "experiments", "results", "thesis_items", "protocols", "tools"}
    missing = required - catalog.keys()
    if missing:
        return [f"Missing catalog sections: {sorted(missing)}"]
    if catalog.get("schema_version") != 1:
        errors.append("Unsupported catalog schema version")

    def file(relative, owner):
        try:
            path = repository_path(relative, root)
            if not path.is_file():
                errors.append(f"{owner}: missing file {relative}")
            return path
        except ValueError as exc:
            errors.append(f"{owner}: {exc}")
            return None

    for eid, experiment in catalog["experiments"].items():
        if experiment.get("phase") not in (0, 1, 2, 3):
            errors.append(f"{eid}: invalid phase")
        for field, records in (("model", "models"), ("dataset", "datasets")):
            if experiment.get(field) is not None and experiment[field] not in catalog[records]:
                errors.append(f"{eid}: unknown {field} {experiment[field]}")
        if experiment.get("outcome") not in {"valid", "invalid", "unavailable"}:
            errors.append(f"{eid}: invalid scientific outcome")
        if experiment.get("verification") not in {"not_run", "verified", "blocked"}:
            errors.append(f"{eid}: invalid verification state")
        if experiment.get("protocol") not in catalog["protocols"]:
            errors.append(f"{eid}: unknown metric protocol")
        if experiment.get("verification") == "blocked" and not experiment.get("verification_reason"):
            errors.append(f"{eid}: blocked verification lacks a reason")
        if experiment.get("inference_artifact") and experiment["inference_artifact"] not in experiment.get(
            "artifacts", []
        ):
            errors.append(f"{eid}: selected inference artifact is not associated with this experiment")
        if baseline := experiment.get("comparison_baseline_artifact"):
            if baseline not in experiment.get("artifacts", []) or baseline not in catalog["artifacts"]:
                errors.append(f"{eid}: comparison baseline is not associated with this experiment")
            if not experiment.get("inference_artifact") or baseline == experiment["inference_artifact"]:
                errors.append(f"{eid}: comparison baseline needs a separate explicit inference artifact")
        for aid in experiment.get("artifacts", []):
            if aid not in catalog["artifacts"]:
                errors.append(f"{eid}: unknown artifact {aid}")
        for relative in experiment.get("evidence", []):
            file(relative, eid)
        for field in ("config", "original_config"):
            if field in experiment:
                file(experiment[field], eid)
        result = experiment.get("result")
        if result and result["collection"] not in catalog["results"]:
            errors.append(f"{eid}: unknown result collection")
    paths = set()
    for aid, record in catalog["artifacts"].items():
        if record["path"] in paths:
            errors.append(f"{aid}: duplicate artifact path")
        paths.add(record["path"])
        if record["owner"] not in catalog["experiments"]:
            errors.append(f"{aid}: unknown owning experiment")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            errors.append(f"{aid}: invalid artifact size")

        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            errors.append(f"{aid}: invalid SHA-256")
        path = file(record["path"], aid)
        if path and path.is_file() and path.stat().st_size < 1024:
            pointer = path.read_text(errors="replace")
            expected = (
                f"version https://git-lfs.github.com/spec/v1\noid sha256:{record['sha256']}\nsize {record['bytes']}\n"
            )
            if pointer != expected:
                errors.append(f"{aid}: invalid LFS pointer")
        if (
            hashes
            and path
            and path.is_file()
            and (path.stat().st_size != record["bytes"] or sha256(path) != record.get("sha256"))
        ):
            errors.append(f"{aid}: binary size or SHA-256 does not match")
    for mid, record in catalog["models"].items():
        if "readme" in record:
            file(record["readme"], mid)
    for did, record in catalog["datasets"].items():
        for field in ("metadata", "split"):
            if field in record:
                file(record[field], did)
    for rid, record in catalog["results"].items():
        file(record["path"], rid)
    for pid, protocol in catalog["protocols"].items():
        for relative in protocol.get("code", []):
            file(relative, pid)
    for tid, tool in catalog["tools"].items():
        file(tool["path"], tid)
        if not tool.get("role"):
            errors.append(f"{tid}: missing reproduction role")
    for record in catalog.get("verification_records", []):
        file(record["path"], "verification record")
    for record in catalog.get("external_predictions", []):
        if record["experiment"] not in catalog["experiments"]:
            errors.append("Prediction record refers to an unknown experiment")
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            errors.append("Prediction record lacks a valid SHA-256")
    item_ids = set()
    for item in catalog.get("thesis_items", []):
        if item["id"] in item_ids:
            errors.append(f"Duplicate thesis item {item['id']}")
        item_ids.add(item["id"])
        if item.get("figure"):
            path = file(item["figure"]["path"], item["id"])
            if path and path.is_file() and sha256(path) != item["figure"]["sha256"]:
                errors.append(f"{item['id']}: original figure hash changed")
        if item.get("reproduction"):
            file(item["reproduction"]["generator"], item["id"])
        for field, section in (("models", "models"), ("results", "results"), ("datasets", "datasets")):
            for value in item.get(field, []):
                if value not in catalog[section]:
                    errors.append(f"{item['id']}: unknown {field} {value}")
        for relative in item.get("reference_pages", []) + item.get("code", []):
            file(relative, item["id"])
        for eid in item.get("experiments", []):
            if eid not in catalog["experiments"]:
                errors.append(f"{item['id']}: unknown experiment {eid}")
    return errors


def index_text(catalog: dict, *, sphinx=False) -> str:
    """Render both navigation views from the same catalog and template."""
    import os

    def link(path, label):
        if sphinx:
            return f":download:`{label} <../../../{path}>`"
        return f"`{label} <{os.path.relpath(path, 'masterthesis_guide')}>`_"

    lines = [
        "Thesis evidence index",
        "=====================",
        "",
        "Generated from ``catalog.yaml``. Source records retain their original fields.",
        "",
        "Find a thesis figure or table",
        "-----------------------------",
        "",
    ]
    for item in catalog["thesis_items"]:
        lines += [f".. _{item['id'].replace('_', '-')}:", "", f"* **{item['title'].strip()}**"]
        for field in ("experiments", "results", "datasets"):
            values = item.get(field, [])
            if values:
                lines += [f"  {field.capitalize()}: " + ", ".join(f"``{value}``" for value in values) + "."]
        for path in item.get("reference_pages", []):
            docname = path.removeprefix("docs/source/").removesuffix(".rst")
            lines += [f"  :doc:`/{docname}`." if sphinx else "  " + link(path, "Reference") + "."]
        if item.get("figure"):
            lines += ["  " + link(item["figure"]["path"], "Original thesis figure") + "."]
        if item.get("reproduction"):
            lines += [
                "  Generator: " + link(item["reproduction"]["generator"], "source") + ".",
                "  Command: ``" + item["reproduction"]["command"] + "``.",
            ]
        if item.get("note"):
            lines += ["  " + item["note"]]
        lines += [""]
    lines += ["Datasets and saved splits", "-------------------------", ""]
    for did, record in catalog["datasets"].items():
        lines += [f"* ``{did}``: external ``{record['external_paths'][0]}``."]
        for field in ("metadata", "split"):
            if field in record:
                lines += ["  " + link(record[field], field) + "."]
    lines += ["", "Recorded comparison tables", "--------------------------", ""]
    for rid, record in catalog["results"].items():
        lines += [
            "* " + link(record["path"], rid) + ".",
            "  "
            + record.get("scope", record.get("selection_note", "Recorded measurements; see the associated protocol.")),
        ]
    for phase in range(4):
        title = f"Phase {phase} experiments"
        lines += ["", title, "-" * len(title), ""]
        for eid, experiment in catalog["experiments"].items():
            if experiment["phase"] != phase:
                continue
            lines += [
                f"* ``{eid}``: {experiment.get('model') or 'engineering/baseline'}; outcome **{experiment['outcome']}**; verification ``{experiment['verification']}``."
            ]
            for role in ("config", "original_config"):
                if role in experiment:
                    lines += ["  " + link(experiment[role], role) + "."]
            for number, path in enumerate(experiment.get("evidence", [])):
                lines += ["  " + link(path, f"Record {number + 1}") + "."]
            if experiment.get("artifacts"):
                lines += ["  Artifacts: " + ", ".join(f"``{a}``" for a in experiment["artifacts"]) + "."]
            if experiment.get("verification_reason"):
                lines += ["  " + experiment["verification_reason"]]
            lines += [""]
    lines += ["Reproduction tools", "------------------", ""]
    for tool in catalog["tools"].values():
        lines += ["* " + link(tool["path"], tool["path"]) + ": " + tool["role"]]
    lines += ["", "Availability gaps", "-----------------", ""]
    for gap in catalog.get("gaps", []):
        lines += ["* " + str(gap)]
    return "\n".join(lines) + "\n"


def generate_indexes(catalog=None, *, root=ROOT, check=False) -> None:
    catalog = catalog or load_catalog(root / "masterthesis_guide/catalog.yaml")
    for relative, sphinx in (
        ("masterthesis_guide/INDEX.rst", False),
        ("docs/source/masterthesis_guide/catalog.rst", True),
    ):
        path = root / relative
        text = index_text(catalog, sphinx=sphinx)
        if check:
            if not path.is_file() or path.read_text() != text:
                raise ValueError(f"Generated index is stale: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    validation = actions.add_parser("validate", help="Check catalog associations and local files")
    validation.add_argument("--hashes", action="store_true", help="Also read and hash all selected weight files")
    index = actions.add_parser("index", help="Generate the repository and Sphinx experiment indexes")
    index.add_argument("--check", action="store_true")
    results = actions.add_parser("model-results", help="Build per-model parameter, curve and evaluation views")
    results.add_argument("--check", action="store_true")
    configuration = actions.add_parser("config", help="Write a resolved training configuration")
    configuration.add_argument("experiment")
    configuration.add_argument("--data-root", type=Path)
    configuration.add_argument("--output-dir", type=Path, required=True)
    configuration.add_argument("--device", default="cpu")
    configuration.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    catalog = load_catalog()
    if args.action == "validate":
        errors = validate(catalog, hashes=args.hashes)
        for error in errors:
            print(error)
        return int(bool(errors))
    if args.action == "index":
        generate_indexes(catalog, check=args.check)
    if args.action == "model-results":
        from masterthesis_guide.model_results import generate_model_results

        print(generate_model_results(catalog, check=args.check))
    if args.action == "config":
        config = executable_config(
            args.experiment, catalog, data_root=args.data_root, output_dir=args.output_dir, device=args.device
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
