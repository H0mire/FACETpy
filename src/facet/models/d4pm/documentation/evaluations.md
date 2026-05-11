# D4PM Evaluations

Standardized evaluation runs for `d4pm` should be stored under:

```text
output/model_evaluations/d4pm/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`

Inference is iterative (one forward pass per sampling step per channel
per epoch). Record `sample_steps`, `data_consistency_weight`, and
wall-clock time in the run's `config` block so different runs are
comparable.

The first evaluation should compare:

- supervised synthetic correction on the 512-sample Niazy proof-fit
  dataset
- residual RMS reduction vs `cascaded_dae` and `cascaded_context_dae`
  baselines
- visual plots for noisy, predicted artifact, corrected, and reference
  clean
- inference wall-clock per epoch, since diffusion is the slowest model
  family in the catalog
