# Demucs (Time-Domain), Paper-Accurate Edition — fMRI Gradient Artifact Removal

A more faithful re-implementation of the time-domain Demucs
(Défossez, Usunier, Bottou, Bach 2019, *Music Source Separation in the Waveform
Domain*, [arXiv:1911.13254](https://arxiv.org/abs/1911.13254)) adapted to
channel-wise EEG gradient-artifact prediction on the Niazy proof-fit context
dataset.

This package lives alongside the original `facet.models.demucs` and **does not
modify it**. It keeps the same factory contract, dataset format, and inference
adapter structure, but closes four gaps between the original edition and the
source paper while staying CPU-runnable and sensible for single-channel
EEG-fMRI.

## What changed vs. the original edition (and why)

| # | Change | Paper section | Why |
|---|--------|---------------|-----|
| 1 | **Length-agnostic forward** — `valid_length(length)` rounds the input up to the next multiple of `stride ** depth`, `F.pad`s it, center-crops every encoder skip to the decoder feature length, and center-crops the output back to the original length. | Sec 4 / official reference `valid_length` | The original `forward` summed skips with no length alignment and **crashed** on any input length not divisible by `stride ** depth` (verified: `depth=2`, length 700 → `RuntimeError: size of tensor a (172) must match b (175)`). The proof-fit length 3584 = 4⁴·14 happened to be divisible, masking the bug for any other `epoch_samples`. |
| 2 | **2x resampling trick** — optional `resample` factor (default 2; 1 disables). The waveform is upsampled before the encoder and downsampled after the decoder, inside the forward pass and therefore inside the end-to-end loss. Implemented with a dependency-free sinc/Kaiser FIR via `F.conv1d` (TorchScript-traceable, CPU-cheap). | Sec 4.1 *Resampling* | Ablation Table 4 shows removing it costs SDR (6.03 vs 6.28). It also buys one extra usable U-Net level on the short 3584-sample EEG context. The original did not implement it. |
| 3 | **Paper-exact init weight rescaling** — keeps `α = std(w)/a`, `w' = w/√α` (`a = 0.1`) but **no longer zeroes conv biases**. | Sec 4.3 | Sec 4.3 rescales weights only and is silent about biases; the original additionally called `bias.zero_()` on every conv, which the paper does not describe. |
| 4 | **Principled depth** — `auto_depth` caps `depth` at the maximum that keeps the (optionally upsampled) bottleneck length ≥ 1 sample, instead of a hard-coded `depth=4`. `initial_channels` stays at the paper-best **64**. | Sec 4 / Table 5 | Makes the necessary EEG depth-reduction principled and self-documenting rather than a magic constant; the paper's `L=6` collapses the 3584-sample context below 1 sample under stride 4. |

### Inference-time addition

- **Test-time shift trick** (Sec 4.4): the new processor exposes `n_shifts`
  (default 1). With `n_shifts > 1` the concatenated context waveform is rolled
  by evenly-spaced circular offsets, the model is run on each, predictions are
  inverse-rolled and averaged before the center epoch is sliced. This is the
  paper's only inference-time trick (+0.3 SDR) and transfers directly to EEG.
  `n_shifts=1` reproduces the original single-pass behaviour.

### Verified matches kept from the original (no change needed)

- Encoder block `Conv1d(K=8,S=4)+ReLU` then `Conv1d(K=1)→2C` + GLU.
- Decoder block `Conv1d(K=3)+GLU` then `ConvTranspose1d(K=8,S=4)+ReLU`, with the
  final block linear (no activation) so the artifact can be signed.
- 2-layer bidirectional LSTM bottleneck (hidden = bottleneck channels) +
  `Linear(2C→C)`; LSTM output summed with the deepest encoder skip.
- Summed (not concatenated) U-Net skips.
- L1 reconstruction loss default (Sec 4.2), with `mse`/`smooth_l1` ablation
  options retained.

## Documented EEG deviations (kept on purpose)

- **Single channel in / single artifact source out** (`in_channels=1`,
  `n_sources=1`): the paper uses stereo I/O and S=4 instrument sources. EEG is
  processed channel-wise with one artifact source so the checkpoint stays
  decoupled from the EEG channel count (consistent with the sibling
  `cascaded_*_dae` models).
- **Reduced depth** (`~4` vs the paper's `6`): see change #4.
- **No music augmentation**: pitch/tempo shift, 4-instrument source remixing,
  and stereo channel-swap are music/multi-source specific and do not transfer to
  single-channel gradient artifacts. A FACETpy-appropriate substitute (optional
  small random gain + small circular time-shift) is available via the
  `augment*` dataset flags and is **off by default** (off for the proof fit).

See `documentation/paper_accuracy_review.md` for the full discrepancy table and
the per-method EEG-fMRI applicability assessment.

## Scope

- Input shape: `(batch, 1, context_epochs * epoch_samples)`. Default `(1, 3584)`.
- Output shape: `(batch, 1, context_epochs * epoch_samples)` — predicted
  artifact across all context epochs.
- The pipeline adapter slices the **center epoch** of the prediction and
  resamples it back to the native trigger-to-trigger length before subtraction.
- Channel-wise inference; requires trigger metadata at inference.

## Training

```bash
uv run facet-train fit \
  --config src/facet/models/demucs_paper_accurate_edition/training_niazy_proof_fit_smoke.yaml
```

(The smoke YAML is illustrative and tiny; for a real run raise `depth`,
`initial_channels`, `lstm_layers`, `max_examples`, set `resample: 2`, and point
`data.kwargs.path` at the full proof-fit context bundle.)

## Inference

```python
from facet.models.demucs_paper_accurate_edition import DemucsPaperAccurateCorrection

context = context | DemucsPaperAccurateCorrection(
    checkpoint_path="training_output/<run>/exports/demucs_paper_accurate.ts",
    context_epochs=7,
    epoch_samples=512,
    n_shifts=4,   # test-time shift trick (Sec 4.4); 1 = single pass
)
```

The processor registers under the unique name
`demucs_paper_accurate_correction` (the original keeps `demucs_correction`).

## Status

- Author: Müller Janik Michael (FACETpy thesis)
- Source paper: Défossez et al. 2019, arXiv:1911.13254 (Sec 4.1 / 4.2 / 4.3 / 4.4)
- Original edition: `src/facet/models/demucs/`
