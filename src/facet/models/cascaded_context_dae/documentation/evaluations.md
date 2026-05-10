# Cascaded Context DAE Evaluations

Standardized evaluation runs for `cascaded_context_dae` should be stored under:

```text
output/model_evaluations/cascaded_context_dae/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`

The first evaluation should compare:

- synthetic supervised correction on the 512-sample context dataset
- Niazy real-data proxy metrics
- large-MFF-derived artifact generalization if a real target pipeline is available
- visual plots for noisy, predicted artifact, corrected, and reference clean where available
