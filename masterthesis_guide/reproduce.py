"""Resolve the thesis catalog for repository-based reproduction commands."""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
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
    return root / path


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
            f"the recorded relative path {record['external_paths'][0]}.")
    path = repository_path(record["external_paths"][0], Path(directory).expanduser())
    if not path.is_file():
        raise FileNotFoundError(f"Dataset {dataset_id} is missing: {path}")
    return path


def selected_artifact(experiment_id: str, catalog=None, *, device="cpu") -> tuple[str, Path]:
    """Select a recorded artifact; prefer an explicit CPU export on non-CUDA devices."""
    catalog = catalog or load_catalog()
    experiment = catalog["experiments"][experiment_id]
    candidates = [(aid, catalog["artifacts"][aid]) for aid in experiment["artifacts"]]
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
    aid, record = sorted(candidates, key=priority)[0]
    from facet.models.masterthesis.adapters import require_artifact
    path = require_artifact(repository_path(record["path"]))
    if path.stat().st_size != record["bytes"]:
        raise ValueError(f"Artifact size does not match the catalog: {path}")
    return aid, path


def adapter(experiment_id: str, catalog=None, *, device="cpu"):
    """Build the installed-library adapter with explicit model inputs."""
    catalog = catalog or load_catalog()
    experiment = catalog["experiments"][experiment_id]
    _, path = selected_artifact(experiment_id, catalog, device=device)
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
    return FamilyAdapter(model_id, checkpoint=path, model_factory=factory,
                         model_kwargs=kwargs, device=device)


def executable_config(experiment_id: str, catalog=None, *, data_root=None,
                      output_dir: Path, device="cpu") -> dict:
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
        path = file(record["path"], aid)
        if hashes and path and path.is_file():
            if path.stat().st_size != record["bytes"] or sha256(path) != record.get("sha256"):
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
    for item in catalog.get("thesis_items", []):
        for eid in item.get("experiments", []):
            if eid not in catalog["experiments"]:
                errors.append(f"{item['id']}: unknown experiment {eid}")
    return errors


def index_text(catalog: dict, *, sphinx=False) -> str:
    """Render both navigation views from the same catalog and template."""
    lines = ["Experiment index", "================", "", "Generated from ``catalog.yaml``. Do not edit this index by hand.", ""]
    for phase in range(4):
        title = f"Phase {phase}"
        lines += [title, "-" * len(title), ""]
        for eid, experiment in catalog["experiments"].items():
            if experiment["phase"] != phase:
                continue
            lines += [f"* ``{eid}``: {experiment.get('model') or 'engineering/baseline'}; "
                      f"outcome **{experiment['outcome']}**; verification ``{experiment['verification']}``."]
            for role in ("config", "original_config"):
                if role not in experiment:
                    continue
                path = experiment[role]
                link = f":download:`{role} <../../../{path}>`" if sphinx else f"`{role} <{Path(path).relative_to('masterthesis_guide').as_posix()}>`_"
                lines += [f"  {link}."]
        lines += [""]
    lines += ["Availability gaps", "-----------------", ""]
    for gap in catalog.get("gaps", []):
        lines += [f"* {str(gap)}"]
    if not catalog.get("gaps"):
        lines += ["No unresolved source gaps are recorded."]
    return "\n".join(lines) + "\n"


def generate_indexes(catalog=None, *, root=ROOT, check=False) -> None:
    catalog = catalog or load_catalog(root / "masterthesis_guide/catalog.yaml")
    for relative, sphinx in (("masterthesis_guide/INDEX.rst", False),
                             ("docs/source/masterthesis_guide/catalog.rst", True)):
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
    if args.action == "config":
        config = executable_config(args.experiment, catalog, data_root=args.data_root,
                                   output_dir=args.output_dir, device=args.device)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
