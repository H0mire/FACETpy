# Nested-GAN Evaluations

Standardized evaluation runs for `nested_gan` are stored under:

```text
output/model_evaluations/nested_gan/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`
- `plots/`

Required comparisons:

- supervised synthetic metrics on the 512-sample Niazy proof-fit context dataset
- Niazy real-data trigger-locked proxy metrics
- visual plots: noisy, predicted artifact, corrected, reference clean
- side-by-side metric table vs `cascaded_dae`, `cascaded_context_dae`,
  `dhct_gan`, and (if available) `dhct_gan_v2`

The first evaluation should explicitly test the hypothesis:
**nested time-frequency decomposition outperforms single-domain GAN
refinement on this dataset.** If Nested-GAN performs worse than
`dhct_gan`'s -7.13 dB result, the hypothesis is unsupported and the
HANDOFF should document that.
