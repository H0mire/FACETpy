from __future__ import annotations

import json

import pytest

from facet.evaluation import EVALUATION_SCHEMA_VERSION, ModelEvaluationWriter


def test_model_evaluation_writer_creates_standard_run_files(tmp_path):
    writer = ModelEvaluationWriter(
        model_id="toy_model",
        model_name="Toy Model",
        model_description="Small test model.",
        output_root=tmp_path / "output",
        docs_root=tmp_path / "models",
        run_id="run001",
    )

    run = writer.write(
        metrics={
            "synthetic": {
                "clean_snr_improvement_db": 1.25,
                "n_examples": 4,
            },
            "real_proxy": {
                "template_rms_reduction_pct": -3.0,
            },
        },
        config={"checkpoint": "checkpoint.ts"},
        artifacts={"plot": "plot.png"},
        interpretation="Synthetic improved, real proxy did not.",
        limitations=["No clean real EEG reference."],
    )

    assert run.run_dir == tmp_path / "output" / "toy_model" / "run001"
    assert run.manifest_path.exists()
    assert run.metrics_path.exists()
    assert run.summary_path.exists()
    assert (tmp_path / "models" / "toy_model" / "documentation" / "evaluations.md").exists()

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert manifest["model_id"] == "toy_model"
    assert manifest["artifacts"]["plot"] == "plot.png"

    metrics = json.loads(run.metrics_path.read_text(encoding="utf-8"))
    assert metrics["flat_metrics"]["synthetic.clean_snr_improvement_db"] == 1.25
    assert metrics["flat_metrics"]["synthetic.n_examples"] == 4
    assert metrics["flat_metrics"]["real_proxy.template_rms_reduction_pct"] == -3.0

    summary = run.summary_path.read_text(encoding="utf-8")
    assert "# Evaluation Run: Toy Model" in summary
    assert "`synthetic.clean_snr_improvement_db`" in summary
    assert "No clean real EEG reference." in summary


@pytest.mark.parametrize("bad_id", ["../etc", "a/b", "..", ".", "foo/../bar", ""])
def test_model_evaluation_writer_rejects_path_traversal_model_id(tmp_path, bad_id):
    """L4: model_id must be a single safe path segment (no traversal/separators)."""
    with pytest.raises(ValueError):
        ModelEvaluationWriter(
            model_id=bad_id,
            model_name="X",
            model_description="",
            output_root=tmp_path / "o",
            docs_root=tmp_path / "d",
        )


def test_model_evaluation_writer_rejects_path_traversal_run_id(tmp_path):
    """L4: a malicious run_id cannot escape the output directory either."""
    with pytest.raises(ValueError):
        ModelEvaluationWriter(
            model_id="ok",
            model_name="X",
            model_description="",
            output_root=tmp_path / "o",
            docs_root=tmp_path / "d",
            run_id="../escape",
        )


def test_model_evaluation_writer_emits_strict_json_for_non_finite(tmp_path):
    """L6: non-finite metric values become null so the emitted JSON is strict
    (no bare NaN/Infinity tokens that strict parsers reject)."""
    writer = ModelEvaluationWriter(
        model_id="m",
        model_name="M",
        model_description="",
        output_root=tmp_path / "o",
        docs_root=tmp_path / "d",
        run_id="r",
    )
    run = writer.write(
        metrics={"snr": float("nan"), "gain": float("inf"), "ok": 1.5},
        config={"bad": float("-inf")},
    )
    raw = run.metrics_path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    parsed = json.loads(raw)
    assert parsed["flat_metrics"]["snr"] is None
    assert parsed["flat_metrics"]["gain"] is None
    assert parsed["flat_metrics"]["ok"] == 1.5
