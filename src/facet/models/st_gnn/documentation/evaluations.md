# ST-GNN Evaluations

Standardised evaluation runs for `st_gnn` should be stored under:

```text
output/model_evaluations/st_gnn/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`

The first evaluation should compare:

- supervised reconstruction error on the Niazy proof-fit dataset
- artifact correlation against the AAS-derived target
- `cascaded_context_dae` baseline on the same dataset
- `cascaded_dae` baseline on the same dataset
- spatial-consistency plots (per-electrode RMSE topomap) to highlight
  any improvements expected from the graph constraint

Runs are produced by following the standard
`facet.evaluation.ModelEvaluationWriter` flow. Do not invent a custom
metrics format.
