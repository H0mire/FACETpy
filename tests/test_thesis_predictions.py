"""Original predictions must work from current checkout files, including LFS checks."""

import hashlib

import numpy as np
import pytest
from masterthesis_guide.reproduce import load_catalog, prediction_path, validate
from tools.plotting.plot_phase1_holdout_signal_comparison import MODELS


@pytest.fixture
def prediction_catalog(tmp_path):
    path = tmp_path / "artifacts/predictions/phase_1/demucs/predicted_artifact.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((2, 3, 512), dtype=np.float32))
    record = {
        "experiment": "holdout_demucs",
        "path": path.relative_to(tmp_path).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    catalog = {
        "schema_version": 1,
        "models": {},
        "datasets": {},
        "artifacts": {},
        "predictions": [record],
        "experiments": {
            "holdout_demucs": {"phase": 1, "outcome": "valid", "verification": "not_run", "protocol": "fixture"}
        },
        "results": {},
        "thesis_items": [],
        "protocols": {"fixture": {}},
        "tools": {},
    }
    return catalog, path


def pointer(record):
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{record['sha256']}\nsize {record['bytes']}\n"


def test_all_plotted_models_have_versioned_predictions():
    records = load_catalog()["predictions"]
    assert {r["experiment"] for r in records} == {"holdout_" + model for model, _ in MODELS}
    assert len(records) == len(MODELS)
    assert all(r["path"].startswith("artifacts/predictions/phase_1/") for r in records)


def test_predictions_need_only_current_files(tmp_path, prediction_catalog):
    catalog, path = prediction_catalog
    # No historical source metadata or Git repository is present in this fixture.
    assert not (tmp_path / ".git").exists()
    assert prediction_path("holdout_demucs", catalog, root=tmp_path) == path
    assert validate(catalog, root=tmp_path, hashes=True) == []


@pytest.mark.parametrize("state", ["missing", "pointer"])
def test_missing_prediction_explains_targeted_download(tmp_path, prediction_catalog, state):
    catalog, path = prediction_catalog
    record = catalog["predictions"][0]
    if state == "missing":
        path.unlink()
    else:
        path.write_text(pointer(record))
    with pytest.raises(FileNotFoundError) as error:
        prediction_path("holdout_demucs", catalog, root=tmp_path)
    assert "git lfs install" in str(error.value)
    assert f'git lfs pull --include={record["path"]} --exclude=""' in str(error.value)
    if state == "missing":
        assert any("missing file" in error for error in validate(catalog, root=tmp_path))
    else:
        assert validate(catalog, root=tmp_path) == []
        assert any("binary size or SHA-256" in error for error in validate(catalog, root=tmp_path, hashes=True))


def test_corrupted_prediction_is_rejected(tmp_path, prediction_catalog):
    catalog, path = prediction_catalog
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="bytes do not match"):
        prediction_path("holdout_demucs", catalog, root=tmp_path)
    assert any("SHA-256" in error for error in validate(catalog, root=tmp_path, hashes=True))


def test_wrong_prediction_lfs_pointer_fails_ci_validation(tmp_path, prediction_catalog):
    catalog, path = prediction_catalog
    record = catalog["predictions"][0]
    path.write_text(pointer(record).replace(record["sha256"], "0" * 64))
    assert any("invalid LFS pointer" in error for error in validate(catalog, root=tmp_path))


def test_unknown_prediction_experiment(tmp_path, prediction_catalog):
    catalog, _ = prediction_catalog
    with pytest.raises(ValueError, match="No recorded predictions"):
        prediction_path("unknown", catalog, root=tmp_path)
