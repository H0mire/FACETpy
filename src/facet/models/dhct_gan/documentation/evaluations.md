# DHCT-GAN Evaluations

This file is the human-readable index of evaluation runs for the
`dhct_gan` model. Each run's machine-readable artifacts live under
`output/model_evaluations/dhct_gan/<run_id>/`.

## Runs

Populated after the first GPU dispatch completes. See `HANDOFF.md` in the
worktree root for the latest smoke and full run identifiers, and follow
the link to the corresponding `evaluation_manifest.json`.

## Metric groups reported

Per `src/facet/models/evaluation_standard.md`, the supervised synthetic-style
group is used because the Niazy proof-fit dataset provides paired
clean / artifact / noisy windows:

- clean reconstruction error before correction
- clean reconstruction error after correction
- artifact prediction error
- artifact correlation (Pearson)
- residual RMS ratio

Real-data trigger-locked proxies and FACET framework metrics are emitted
when applicable.

## Comparison anchors

- `cascaded_dae`
- `cascaded_context_dae`
