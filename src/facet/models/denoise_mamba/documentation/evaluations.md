# DenoiseMamba Evaluations

This file indexes standardized evaluation runs for `denoise_mamba`.

Large generated plots and JSON artifacts are written to:

```text
output/model_evaluations/denoise_mamba/<run_id>/
```

Each run produces:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`
- plots / artifacts referenced from the manifest

## Runs

This index is updated by `ModelEvaluationWriter` after each evaluation run.
