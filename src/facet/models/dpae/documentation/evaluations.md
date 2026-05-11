# DPAE Evaluations

Standardised evaluation runs for `dpae` should be stored under:

```text
output/model_evaluations/dpae/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`
- plots (e.g., predicted-vs-true artifact, residual spectra)

The first evaluation should compare:

- supervised proof-fit reconstruction on the Niazy 512-sample context dataset
  (clean reconstruction error before/after, artifact correlation, residual RMS
  ratio)
- side-by-side metrics against `cascaded_dae` and `cascaded_context_dae`
- visual inspection plots for noisy, predicted artifact, corrected, and the
  AAS-corrected surrogate clean signal

Use `facet.evaluation.ModelEvaluationWriter` so the run files match the
schema enforced by `src/facet/models/evaluation_standard.md`.
