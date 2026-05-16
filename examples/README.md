# Examples

This directory contains runnable examples and scripts that demonstrate FACETpy
usage. After the May 2026 reorganization the layout is:

```
examples/
├── README.md                          ← you are here
├── quickstart.py                       Minimal hello-world entry point
├── complete_pipeline_example.py        Full AAS correction pipeline (recommended start)
├── complete_pipeline_example_bcg.py    Same plus BCG correction
├── complete_pipeline_example_large_dataset.py
├── complete_pipeline_example_large_dataset_bcg.py
├── convert_types.py                    Small helper (MNE event-array conversions)
│
├── datasets/                           Sample EEG data used by the examples (EDF, BIDS, GDF)
├── notebooks/                          Legacy Jupyter notebooks (provenance)
│
├── pipelines/                          General pipeline + processing patterns
│   ├── advanced_workflows.py           ConditionalProcessor, SwitchProcessor
│   ├── batch_processing.py             Run the same pipeline over many files
│   ├── channelwise_execution.py        Parallel per-channel processing
│   ├── inline_steps.py                 Compose processors inline
│   ├── memory_efficient_pipeline.py    Stream-friendly long-recording patterns
│   ├── farm_volume_pipeline_example.py FARM volume-correction reference
│   ├── cleanexjanik_parity_pipeline.py Replicates the cleanExJanik MATLAB pipeline
│   ├── new_processors_compact_example.py
│   ├── evaluation.py                   How to wire MetricsReport + calculators
│   └── eeg_generation_visualization_example.py
│
├── dataset_building/                   Build training/eval datasets from raw EEG
│   ├── build_niazy_proof_fit_context_dataset.py   ★ Core proof-fit dataset builder
│   ├── build_synthetic_spike_artifact_context_dataset.py
│   ├── build_epoch_context_dataset.py
│   ├── extract_niazy_artifact_signal.py
│   ├── generate_synthetic_spike_source_dataset.py
│   │
│   │   --- legacy / pre-proof-fit variants kept for provenance ---
│   ├── build_niazy_artifact_windows.py
│   ├── build_synthetic_spike_artifact_dataset.py
│   ├── extract_large_mff_aas_artifact_signal.py
│   └── generate_synthetic_fmri_artifact_source.py
│
├── model_training/                     Training-CLI demos and PyTorch hooks
│   ├── facet_train_demo.py + .yaml     facet-train fit demo
│   ├── pytorch_training_example.py     Bare-PyTorch training loop
│   ├── pytorch_inference_example.py    Inference with a trained TorchScript
│   ├── context_artifact_training_demo.py + .yaml   Legacy context-CNN demo
│   └── apply_context_artifact_model.py
│
└── model_evaluation/                   Per-model evaluators (Niazy proof-fit val split)
    ├── evaluate_conv_tasnet.py
    ├── evaluate_demucs.py
    ├── evaluate_ic_unet.py
    ├── evaluate_sepformer_niazy_proof_fit.py
    ├── evaluate_st_gnn.py
    ├── evaluate_cascaded_daes_niazy_proof_fit.py
    │
    │   --- legacy synthetic-spike evaluators kept for provenance ---
    ├── evaluate_context_artifact_model.py
    └── evaluate_context_artifact_model_full_metrics.py
```

## How to run

All scripts assume the working directory is the repository root:

```bash
uv run python examples/quickstart.py
uv run python examples/complete_pipeline_example.py
uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py --help
uv run python examples/model_evaluation/evaluate_demucs.py --checkpoint <path>
```

## See also

- [`docs/source/`](../docs/source/) for the rendered user guide
- [`docs/research/thesis_results_report.md`](../docs/research/thesis_results_report.md) for the cross-model study using the per-model evaluators
- [`output/model_evaluations/INDEX.md`](../output/model_evaluations/INDEX.md) for the latest evaluation results
