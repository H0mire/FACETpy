# Demucs Model Card

## Summary

`demucs` is the FACETpy adaptation of the time-domain Demucs U-Net (Défossez
et al. 2019) for fMRI gradient-artifact prediction. It runs per channel on a
flattened 7-epoch trigger-defined context, treating the artifact as the
"percussion track" of the recording in the music-source-separation analogy.

## Intended Role

The first audio-source-separation U-Net baseline in FACETpy. Paired with
`conv_tasnet` (the dilated-TCN sibling) it answers whether a U-Net + BiLSTM
inductive bias beats the TCN approach on this dataset.

## Input And Output

- Input: `(batch, 1, total_samples)` with `total_samples = context_epochs * epoch_samples`.
- Default: `context_epochs=7`, `epoch_samples=512`, so `total_samples=3584`.
- Output: `(batch, 1, total_samples)` — predicted artifact across all 7 epochs.
- The inference adapter slices the center epoch and resamples it back to the
  native trigger-to-trigger length before `DeepLearningCorrection` subtracts it.

## Compatibility Notes

- Channel-wise inference; checkpoint works with any EEG channel count.
- Requires trigger metadata.
- Native trigger-to-trigger epoch lengths may vary; the adapter resamples each
  epoch to the model length and resamples the predicted center back to the
  native length.
- Checkpoint is coupled to `context_epochs` and `epoch_samples` (the
  model-domain length).

## Architecture Knobs

| Knob | Default | Notes |
|---|---|---|
| `depth` | 4 | Largest depth that keeps `3584` samples above 1 at the bottleneck under stride 4. |
| `initial_channels` | 64 (full) / 32 (smoke) | Original paper's best initial channel count. |
| `kernel_size` | 8 | Demucs default. |
| `stride` | 4 | Demucs default. |
| `lstm_layers` | 2 | Demucs default; bidirectional. |
| `rescale` | 0.1 | Weight-rescale-at-init target std (paper default). |

## Loss

L1 over the full 3584-sample predicted artifact (matches the paper's default
reconstruction loss; the paper compares L1 vs L2 and chooses L1).
`build_loss` accepts `name="l1"` (default), `"mse"`, `"smooth_l1"`, or
`"huber"`. A center-weighted L1 variant is a documented follow-up.

## Current Training Reference

```bash
uv run facet-train fit \
  --config src/facet/models/demucs/training_niazy_proof_fit.yaml
```

## Evaluation Notes

Use the standard evaluation structure described in
`src/facet/models/evaluation_standard.md`. Compare against `cascaded_dae`,
`cascaded_context_dae`, and `conv_tasnet`.

## Author & License

- Author: Müller Janik Michael
- Reference paper license: see <https://github.com/facebookresearch/demucs>
  (MIT for the reference implementation; the paper itself is the authoritative
  source for the architecture used here).
