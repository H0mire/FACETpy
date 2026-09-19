import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("thesis_reproduce", ROOT / "masterthesis_guide/reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


def test_catalog_associations_are_valid():
    assert reproduce.validate(reproduce.load_catalog()) == []


def test_catalog_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "catalog.yaml"
    path.write_text("models: {}\nmodels: {}\n")
    with pytest.raises(ValueError, match="Duplicate"):
        reproduce.load_catalog(path)


def test_catalog_rejects_dangling_artifact_reference():
    catalog = copy.deepcopy(reproduce.load_catalog())
    catalog["experiments"]["holdout_demucs"]["artifacts"].append("nonexistent")
    assert any("unknown artifact" in error for error in reproduce.validate(catalog))


def test_resolver_rejects_symlink_escape(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        reproduce.repository_path("outside/file", root)


def test_protected_reference_titles_are_preserved():
    import json

    fixture = json.loads((ROOT / "tests/fixtures/thesis_references.json").read_text())
    for reference in fixture["references"]:
        source = ROOT / "docs/source" / (reference["docname"] + ".rst")
        assert source.read_text().splitlines()[0] == reference["title"]


def test_explicit_evaluated_checkpoint_precedes_another_export(tmp_path, monkeypatch):
    """The final training export must not displace the recorded best checkpoint."""
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"recorded evaluated weights")
    export = tmp_path / "last.ts"
    export.write_bytes(b"different last-epoch weights")
    monkeypatch.setattr(reproduce, "repository_path", lambda path: tmp_path / path)
    catalog = {
        "experiments": {
            "run": {"model": "demucs_deployment_edition", "artifacts": ["last", "best"], "inference_artifact": "best"}
        },
        "artifacts": {
            name: {
                "path": path.name,
                "kind": kind,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": reproduce.sha256(path),
            }
            for name, path, kind, role in [
                ("last", export, "export", "cpu_export"),
                ("best", checkpoint, "checkpoint", "evaluated_best_checkpoint"),
            ]
        },
    }
    assert reproduce.selected_artifact("run", catalog) == ("best", checkpoint)


def test_guide_preserves_the_selected_dhct_context_axis(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "selected_artifact", lambda *args, **kwargs: ("fixture", tmp_path / "fixture.pt"))
    catalog = reproduce.load_catalog()
    primary = reproduce.adapter("deployment_dhct_gan", catalog)
    selected = reproduce.adapter("run8_dhct_gan_lr0_0001_bc8_sisdr0_s42", catalog)
    assert primary.packing.packing == "b1s"
    assert primary.packing.context == "single"
    assert selected.packing.packing == "b1ts"
    assert selected.packing.context == "stack"


def test_lfs_pointer_fails_before_model_loading(tmp_path):
    from facet.models.masterthesis.adapters import require_artifact

    pointer = tmp_path / "model.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 10\n")
    with pytest.raises(FileNotFoundError, match="Git LFS pointer"):
        require_artifact(pointer)


@pytest.mark.parametrize("state", ["missing", "pointer", "downloaded", "empty"])
def test_download_check_for_selected_wega_checkpoint(tmp_path, monkeypatch, state):
    experiment = "wega_demucs_lr0_0001_ic96_sisdr3_s42"
    catalog = reproduce.load_catalog()
    aid, record = reproduce._artifact_record(experiment, catalog)
    assert aid == f"{experiment}_epoch0053_val_loss1_5065_pt"
    checkpoint = tmp_path / record["path"]
    checkpoint.parent.mkdir(parents=True)
    if state == "pointer":
        checkpoint.write_text("version https://git-lfs.github.com/spec/v1\n")
    elif state == "downloaded":
        checkpoint.write_bytes(b"local weights; the adapter checks size and hash")
    elif state == "empty":
        checkpoint.touch()
    monkeypatch.setattr(reproduce, "repository_path", lambda path: tmp_path / path)
    if state in {"missing", "pointer"}:
        with pytest.raises(FileNotFoundError) as error:
            reproduce.check_downloaded(experiment)
        message = str(error.value)
        assert "https://git-lfs.com" in message
        assert "git lfs install" in message
        assert f'git lfs pull --include={record["path"]} --exclude=""' in message
        assert "repository root" in message
        assert checkpoint.exists() == (state == "pointer")
    elif state == "empty":
        with pytest.raises(ValueError, match="empty"):
            reproduce.check_downloaded(experiment)
    else:
        assert reproduce.check_downloaded(experiment) is None


@pytest.mark.parametrize("resolve", [reproduce.check_downloaded, reproduce.adapter])
@pytest.mark.parametrize("experiment", ["deployment_demuc", "blödsinn", "", None, []])
def test_unknown_experiment_has_actionable_error(resolve, experiment, monkeypatch):
    def unexpected_artifact_access(path):
        pytest.fail("An unknown experiment must fail before accessing model files")

    monkeypatch.setattr(reproduce, "repository_path", unexpected_artifact_access)
    with pytest.raises(ValueError) as error:
        resolve(experiment)
    message = str(error.value)
    assert f"Unknown experiment ID: {experiment!r}" in message
    assert "masterthesis_guide/catalog.yaml" in message
    assert "git lfs pull" not in message
    if experiment == "deployment_demuc":
        assert "Did you mean:" in message
        assert "  deployment_demucs" in message
    else:
        assert "Did you mean:" not in message


@pytest.mark.parametrize("family", ["demucs", "nested_gan", "vit_spectrogram"])
def test_comparison_inputs_match_the_recorded_hashes(family):
    catalog = reproduce.load_catalog()
    experiment = catalog["experiments"][f"spike_aware_{family}"]
    source = next(path for path in experiment["evidence"] if path.endswith(f"/{family}/comparison.json"))
    comparison = json.loads((ROOT / source).read_text())
    baseline = experiment["comparison_baseline_artifact"]
    checkpoint = experiment["inference_artifact"]
    assert baseline != checkpoint
    assert catalog["artifacts"][baseline]["sha256"] == comparison["baseline_sha256"]
    assert catalog["artifacts"][checkpoint]["sha256"] == comparison["checkpoint_sha256"]
    assert catalog["datasets"][experiment["dataset"]]["sha256"] == comparison["dataset_sha256"]


def test_comparison_baseline_requires_a_distinct_inference_selection():
    catalog = copy.deepcopy(reproduce.load_catalog())
    experiment = catalog["experiments"]["spike_aware_demucs"]
    experiment["inference_artifact"] = experiment["comparison_baseline_artifact"]
    assert any("separate explicit inference artifact" in error for error in reproduce.validate(catalog))
