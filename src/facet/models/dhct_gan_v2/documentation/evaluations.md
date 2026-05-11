# DHCT-GAN v2 Evaluations

This file is the human-readable index of evaluation runs for the
`dhct_gan_v2` model. Each run's machine-readable artifacts live under
`output/model_evaluations/dhct_gan_v2/<run_id>/`.

## Runs

To be populated after the v2 full-training run completes. See
`HANDOFF.md` in the worktree root for the most recent numeric summary
and the v1 ↔ v2 comparison.

## Metric groups reported

Supervised synthetic-style metrics on the Niazy proof-fit dataset
(paired noisy / clean / artifact). Real-data trigger-locked proxies and
the FACET framework metric battery are not yet computed for this model.

## Comparison anchors

- `dhct_gan` (v1, single-epoch input) — the model this v2 is iterating on.
- `cascaded_context_dae` (7-epoch context autoencoder).
- `dpae` (current strongest discriminative baseline on this benchmark).
- `vit_spectrogram` — current best overall on this benchmark.
