# SepFormer Evaluations

Standardized evaluation runs for `sepformer` should be stored under:

```text
output/model_evaluations/sepformer/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`
- plots referenced from `evaluation_manifest.json`

## Required comparisons

- Synthetic supervised correction on the Niazy proof-fit context
  dataset (clean target is the AAS-corrected Niazy surrogate; the
  dataset metadata flags this is not a generalization claim).
- Trigger-locked RMS before / after on the same dataset.
- Visual plots: noisy, predicted artifact, corrected, reference clean.
- Side-by-side comparison vs. `cascaded_dae` and `cascaded_context_dae`
  best metrics. Pull their `metrics.json` from
  `output/model_evaluations/<model_id>/<latest_run>/`.

## What to look for

SepFormer is the first dual-path attention architecture for FACETpy.
The architectural argument is that intra/inter attention should pick
up periodic gradient artifact structure more reliably than the pure
FC autoencoder baselines. Look in particular at:

- artifact prediction error vs `cascaded_context_dae` on identical
  channel splits,
- residual RMS ratio (does the predicted artifact actually subtract
  to less than the FC baselines?),
- visual smoothness of the predicted artifact across epoch boundaries
  (SepFormer should produce continuous predictions because the
  encoder/decoder operates on the concatenated waveform).
