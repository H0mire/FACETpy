# Conv-TasNet Evaluations

Standardised evaluation runs for `conv_tasnet` are stored under:

```text
output/model_evaluations/conv_tasnet/<run_id>/
```

Each run must contain:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`
- `plots/`

The first evaluation should compare:

- supervised synthetic correction on the 512-sample Niazy proof-fit
  context dataset
- Niazy real-data trigger-locked proxy metrics
- visual plots showing noisy mixture, predicted artifact, predicted
  clean, and reference clean for at least one representative epoch
  per channel
- runtime and VRAM use against `cascaded_context_dae` and
  `cascaded_dae` baselines

When a run completes, update this file with:

- the run id and run directory
- a one-paragraph interpretation
- the metric table emitted by `ModelEvaluationWriter`
