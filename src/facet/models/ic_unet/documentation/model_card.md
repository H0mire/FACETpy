# IC-U-Net Model Card

## Summary

`ic_unet` is a multichannel 1-D U-Net adapted from Chuang et al. 2022. A
frozen ICA unmixing matrix transforms the 30-channel input into IC space, a
U-Net denoises in IC space, and a frozen pseudoinverse maps back to channel
space. The center-epoch artifact is returned by subtracting the predicted
clean center from the input's center segment.

## Intended role

First U-Net-based discriminative model for the Niazy proof-fit experiment.
Compared to `cascaded_context_dae`, IC-U-Net exposes the network to a
statistical-independence prior; compared to `dpae` (DPAE, +7.48 dB clean-SNR
improvement reference from the technical report), IC-U-Net replaces the
dual-pathway encoder with an ICA prior plus a vanilla U-Net.

## Input and output

- Input: `(batch, 30, 3584)`, where `3584 = 7 context epochs × 512 samples`.
- Output: `(batch, 30, 512)` — predicted artifact for the center epoch.
- Correction: predicted artifact is subtracted by `DeepLearningCorrection`.

## Compatibility notes

- **Coupled to the 30-channel Niazy montage** (multichannel checkpoint).
  Retraining is required for other channel counts or montages.
- **Requires trigger metadata at inference time.**
- **Native epoch length variation** (584 – 605 samples on this dataset) is
  resampled to the canonical 512-sample model input length; the predicted
  artifact is resampled back to the native length before subtraction.
- **ICA is frozen.** A failed FastICA fit falls back to identity at
  build_model time; this is logged and recorded in the run summary.
- **Sampling rate:** trained on 4096 Hz Niazy proof-fit data. Other sampling
  rates require retraining.

## Current training reference

Smoke and full training are launched via the GPU fleet:

```bash
uv run python tools/gpu_fleet/fleet.py submit \
  --name ic_unet_niazy_smoke \
  --worktree . \
  --training-config src/facet/models/ic_unet/training_niazy_proof_fit_smoke.yaml \
  --prepare-command "uv run python examples/dataset_building/build_niazy_proof_fit_context_dataset.py --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz --target-epoch-samples 512 --context-epochs 7 --output-dir output/niazy_proof_fit_context_512"
```

## Evaluation notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md`. Required comparisons:

- vs `cascaded_context_dae` (same 7-epoch context, channel-wise DAE)
- vs `cascaded_dae` (single-channel windowed DAE baseline)
- vs DPAE reference (+7.48 dB clean-SNR improvement from the technical
  report, when the `dpae` model agent's evaluation is available)
