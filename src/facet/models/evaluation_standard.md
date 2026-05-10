# Model Evaluation Standard

This standard defines how FACETpy deep-learning model evaluations are stored so different model families remain comparable.

## Goals

- Every model has a dedicated documentation area.
- Every evaluation run has a machine-readable manifest.
- Metrics are stored in a flattened form for direct comparison.
- Plots and large artifacts are kept out of source-controlled model code by default.
- Reports remain understandable for researchers reading the repository later.

## Directory Contract

Each model package should contain:

```text
src/facet/models/<model_id>/
├── README.md
├── documentation/
│   ├── model_card.md
│   └── evaluations.md
├── training.py
├── processor.py
└── training.yaml
```

Generated evaluation outputs should be written to:

```text
output/model_evaluations/<model_id>/<run_id>/
├── evaluation_manifest.json
├── metrics.json
├── evaluation_summary.md
└── plots_or_other_artifacts...
```

The `documentation/evaluations.md` file should describe what was evaluated and point to important runs. Large generated artifacts should stay in `output/` unless a small figure is explicitly useful enough to version.

## Required Run Files

`evaluation_manifest.json` records identity and reproducibility information:

- schema version
- model id
- model name
- run id
- checkpoint path
- dataset paths
- evaluation script/config
- artifact file paths
- links to metrics and summary files

`metrics.json` stores:

- nested metric groups for readability
- `flat_metrics` for direct model-to-model comparison

`evaluation_summary.md` stores:

- short model/run description
- metric table
- interpretation
- limitations
- plot/artifact list
- raw config and raw metrics

## Minimum Metric Groups

For synthetic supervised datasets:

- clean reconstruction error before correction
- clean reconstruction error after correction
- clean SNR before correction
- clean SNR after correction
- artifact prediction error
- artifact correlation
- residual RMS ratio

For real EEG-fMRI datasets without clean ground truth:

- trigger-locked RMS before correction
- trigger-locked RMS after correction
- median-template RMS before correction
- median-template RMS after correction
- median-template peak-to-peak before/after
- predicted artifact RMS
- number of corrected epochs
- native epoch length min/median/max

For FACET framework metrics when available:

- SNR
- legacy SNR
- RMS ratio
- RMS residual
- median artifact
- FFT Allen
- FFT Niazy
- spectral coherence
- spike detection rate

## Naming

Use stable model ids:

- `demo01`
- `cascaded_dae`
- `cascaded_context_dae`

Use timestamp-like run ids unless there is a named experiment:

```text
20260502_112643
```

## Programmatic Writer

Use `facet.evaluation.ModelEvaluationWriter` for new evaluation scripts:

```python
from facet.evaluation import ModelEvaluationWriter

writer = ModelEvaluationWriter(
    model_id="cascaded_context_dae",
    model_name="Cascaded Context DAE",
    model_description="Seven-epoch channel-wise denoising autoencoder.",
)

run = writer.write(
    metrics={
        "synthetic": {
            "clean_snr_improvement_db": 3.1,
            "artifact_corr": 0.82,
        },
        "real_proxy": {
            "template_rms_reduction_pct": 41.0,
        },
    },
    config={
        "checkpoint": "training_output/<run>/exports/cascaded_context_dae.ts",
        "dataset": "output/synthetic_spike_artifact_context_512/synthetic_spike_artifact_context_dataset.npz",
    },
    artifacts={
        "synthetic_examples": "synthetic_cleaning_examples.png",
    },
    interpretation="Short, thesis-usable interpretation of the run.",
    limitations=[
        "Synthetic clean reference only covers the generated data distribution.",
    ],
)
```

## Comparison Rule

A model should not be considered better only because one metric improves. Compare at least:

- supervised synthetic performance
- real-data trigger-locked proxy metrics
- visual inspection plots
- spike-preservation behavior when spike labels are available
- runtime/memory behavior
- compatibility assumptions: channel count, context length, artifact length

## Current Policy

The closed-beta phase may keep helper scripts in `examples/`, but any new evaluation should still emit the standard run files.
