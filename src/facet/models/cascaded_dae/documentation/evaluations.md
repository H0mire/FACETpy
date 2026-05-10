# Cascaded DAE Evaluations

Standardized evaluation runs for `cascaded_dae` should be stored under:

```text
output/model_evaluations/cascaded_dae/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`

Compare this model against context-aware variants to quantify how much information is gained by neighboring trigger epochs.
