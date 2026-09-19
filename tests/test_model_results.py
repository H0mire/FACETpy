"""Keep per-model result views tied to exact experiments and original evidence."""

import json
from pathlib import Path

import pytest
from masterthesis_guide.model_results import Sources, evaluation, hyperparameters, training_history
from masterthesis_guide.reproduce import ROOT, load_catalog


def test_every_catalog_variant_owns_only_its_assigned_runs():
    catalog = load_catalog()
    found = set()
    for model_id, model in catalog["models"].items():
        folder = ROOT / Path(model["readme"]).parent / "results"
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["model"] == model_id
        expected = {eid for eid, run in catalog["experiments"].items() if run["model"] == model_id}
        assert {run["experiment"] for run in manifest["experiments"]} == expected
        assert not found & expected
        found |= expected
        for run in manifest["experiments"]:
            curve = folder / run["experiment"] / "training_curve.svg"
            assert curve.is_file() == (run["training"]["status"] == "available")
    assert found == {eid for eid, run in catalog["experiments"].items() if run["model"] is not None}


def test_shared_family_table_is_not_assigned_as_a_run_measurement(tmp_path):
    (tmp_path / "table.csv").write_text("model,score\nDemucs,0.5\n")
    experiment = {"evidence": [], "protocol": "test", "outcome": "unavailable", "verification": "not_run"}
    catalog = {
        "protocols": {"test": {}},
        "results": {
            "shared": {"path": "table.csv", "experiments": ["run_a", "run_b"]},
        },
    }
    result = evaluation("run_a", experiment, catalog, Sources(tmp_path))
    assert result["status"] == "shared_collections_only"
    assert result["records"] == []
    assert len(result["shared_collections"]) == 1


def test_explicit_csv_experiment_does_not_copy_other_runs(tmp_path):
    (tmp_path / "table.csv").write_text("experiment,score\nrun_a,0.5\nrun_b,9.0\n")
    experiment = {"evidence": [], "protocol": "test", "outcome": "valid", "verification": "not_run"}
    catalog = {
        "protocols": {"test": {}},
        "results": {
            "shared": {"path": "table.csv", "experiments": ["run_a", "run_b"], "scope": "new replay"},
        },
    }
    result = evaluation("run_a", experiment, catalog, Sources(tmp_path))
    assert result["records"][0]["values"] == [{"experiment": "run_a", "score": "0.5"}]
    assert result["records"][0]["scope"] == "new replay"


def test_reconstructed_settings_remain_labelled(tmp_path):
    (tmp_path / "config.yaml").write_text("training:\n  learning_rate: 0.1\n")
    result = hyperparameters(
        {
            "config": "config.yaml",
            "configuration_provenance": {
                "status": "reconstructed",
                "reason": "other attempt supplied model settings",
            },
        },
        Sources(tmp_path),
    )
    assert result["status"] == "reconstructed"
    assert result["configuration_provenance"]["reason"]
    assert result["source"]["sha256"]


def test_missing_history_is_not_replaced_with_final_loss(tmp_path):
    (tmp_path / "summary.json").write_text('{"loss": 0.1}')
    result = training_history({"evidence": ["summary.json"]}, Sources(tmp_path))
    assert result["status"] == "unavailable"
    assert result["records"] == []


def test_restarted_history_requires_separate_curves(tmp_path):
    (tmp_path / "training.jsonl").write_text(
        '{"epoch": 1, "train_loss": 1}\n{"epoch": 2, "train_loss": 0.5}\n{"epoch": 1, "train_loss": 2}\n'
    )
    with pytest.raises(ValueError, match="separate curves"):
        training_history({"evidence": ["training.jsonl"]}, Sources(tmp_path))
