# Examples

This directory contains runnable examples and scripts that demonstrate FACETpy
usage. The layout is:

```
examples/
├── README.md                          ← you are here
├── quickstart.py                       Minimal hello-world entry point
├── complete_pipeline_example.py        Full AAS correction pipeline (recommended start)
├── complete_pipeline_example_bcg.py    Same plus BCG correction
├── complete_pipeline_example_large_dataset.py
├── complete_pipeline_example_large_dataset_bcg.py
├── complete_pipeline_example_large_dataset_volume_to_slice.py
├── convert_types.py                    Small helper (MNE event-array conversions)
│
├── datasets/                           Sample EEG data used by the examples (EDF, BIDS, GDF)
├── notebooks/                          Legacy Jupyter notebooks (provenance)
│
└── pipelines/                          General pipeline + processing patterns
    ├── advanced_workflows.py           ConditionalProcessor, SwitchProcessor
    ├── batch_processing.py             Run the same pipeline over many files
    ├── channelwise_execution.py        Parallel per-channel processing
    ├── inline_steps.py                 Compose processors inline
    ├── memory_efficient_pipeline.py    Stream-friendly long-recording patterns
    ├── farm_volume_pipeline_example.py FARM volume-correction reference
    ├── cleanexjanik_parity_pipeline.py Replicates the cleanExJanik MATLAB pipeline
    ├── new_processors_compact_example.py
    ├── evaluation.py                   How to wire MetricsReport + calculators
    └── eeg_generation_visualization_example.py
```

## How to run

All scripts assume the working directory is the repository root:

```bash
uv run python examples/quickstart.py
uv run python examples/complete_pipeline_example.py
uv run python examples/pipelines/evaluation.py
```

## See also

- [`docs/source/`](../docs/source/) for the rendered user guide
