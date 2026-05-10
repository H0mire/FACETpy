from __future__ import annotations

import json

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
