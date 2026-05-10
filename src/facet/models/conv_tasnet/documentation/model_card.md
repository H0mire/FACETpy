# Conv-TasNet Model Card

## Summary

Conv-TasNet is a time-domain source-separation network. In FACETpy it
takes a single noisy EEG channel of fixed length and predicts two
ordered sources: clean EEG (source 0) and gradient artifact (source
1). The pipeline subtracts the predicted artifact from the noisy
signal via `DeepLearningCorrection`.

## Intended Role

This is the first audio-inspired source-separation baseline in the
deep-learning catalog. It pairs with the autoencoder family
(`cascaded_dae`, `cascaded_context_dae`) and the diffusion/sequence
families to compare time-domain mask-based separation against direct
artifact regression.

## Input And Output

- Input: `(batch, 1, samples)`, single EEG channel mixture.
- Default samples: `512`.
- Output: `(batch, 2, samples)` ordered `(clean, artifact)`.
- Correction: `ConvTasNetCorrection` extracts the artifact source and
  hands it to `DeepLearningCorrection`, which subtracts it from raw
  data and stores the estimated noise.

## Compatibility Notes

- Compatible with arbitrary EEG channel counts (channel-wise inference).
- Does **not** require trigger metadata at inference time. Chunks are
  walked by sample index, not trigger index.
- The checkpoint is coupled to `chunk_size_samples` (encoder kernel
  size and stride determine the latent frame count). Inference
  defaults to the training value.
- Source-additivity (`clean + artifact = noisy`) is not enforced
  during training. At inference time, `DeepLearningCorrection` only
  consumes the artifact source, so any residual that the network
  splits between the two sources stays out of the corrected signal.

## Training Reference

Default Niazy proof-fit configuration:

```bash
uv run facet-train fit --config src/facet/models/conv_tasnet/training_niazy_proof_fit.yaml
```

## Hyperparameters

| Parameter | Default |
|---|---|
| Encoder filters `N` | 256 |
| Encoder kernel `L` (stride `L/2`) | 16 |
| TCN bottleneck `B` | 128 |
| TCN block hidden `H` | 256 |
| TCN block kernel `P` | 3 |
| Blocks per repeat `X` | 8 |
| Repeats `R` | 2 |
| Mask activation | sigmoid |
| Loss | MSE on stacked sources |
| Optimizer | Adam (`facet-train` default) |
| Learning rate | 1e-3 |

Hardware envelope: ~1.3 M parameters, well below the 24 GB VRAM of an
RTX 5090. The model is *not* a budget-reduced variant — the published
recipe was sized for 4-s mixtures and is not appropriate for our
512-sample epochs. Justification is recorded in
`documentation/research_notes.md`.

## Evaluation Notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md`. Output run directory:

```text
output/model_evaluations/conv_tasnet/<run_id>/
```

When the smoke run lands, attach the run id and best-validation
metric to `HANDOFF.md`. After the full run, update
`documentation/evaluations.md` with the comparison against
`cascaded_dae` and `cascaded_context_dae`.
