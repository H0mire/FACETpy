# Conv-TasNet Research Notes

## Source paper

- Yi Luo and Nima Mesgarani.
  *Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for
  Speech Separation*. IEEE/ACM Transactions on Audio, Speech, and Language
  Processing, 27(8):1256–1266, 2019.
- arXiv: <https://arxiv.org/abs/1809.07454>

Cited from the FACETpy report `dl_eeg_gradient_artifacts.pdf`,
section 7.1.1 (Conv-TasNet: Time-Domain Audio Separation Network).

## Reference implementations consulted

- Asteroid: `asteroid-team/asteroid` —
  `asteroid/models/conv_tasnet.py` for default hyperparameters.
  Defaults observed: `n_filters=512`, `kernel_size=16`, `bn_chan=128`,
  `hid_chan=512`, `conv_kernel_size=3`, `n_blocks=8`, `n_repeats=3`,
  `norm_type="gLN"`, `mask_act="sigmoid"`, `stride=8`.
- SpeechBrain: `speechbrain/lobes/models/conv_tasnet.py` (cross-checked
  for layer ordering inside a TCN block: 1×1 conv → PReLU → norm → DConv
  → PReLU → norm → 1×1 res/skip).
- Kaituoxu/Conv-TasNet (PyTorch reference reproducing the original
  paper) for sanity check on the cumulative layer-norm definition and
  the dilation schedule `dilation = 2^x` per block within a repeat.

These are reference only; this implementation is written from scratch
following the paper notation.

## One-paragraph plain-language description

Conv-TasNet treats source separation as a regression problem on raw
waveforms, without ever computing a spectrogram. A short 1D convolution
("encoder") slides over the noisy mixture and produces a learned,
non-negative time-frequency-like representation. A stack of dilated
1D convolutions (the "TCN separator") then predicts one mask per
source in that learned feature space. Each masked feature map is fed
through a 1D transposed convolution ("decoder") that maps it back to
the time domain, yielding one waveform per source. Adapted to
EEG-fMRI gradient artifact correction, source 1 is the clean EEG and
source 2 is the gradient artifact, and the two are constrained to
sum back to the noisy input by the learned mask geometry.

## Key architectural components

| Component | Purpose |
|---|---|
| Encoder (Conv1D, kernel `L`, stride `L/2`, `N` filters, ReLU) | Maps a single-channel noisy waveform `(1, T)` to a non-negative latent `(N, T')` with `T' ≈ 2T/L`. |
| Pre-separator norm + bottleneck | Global layer norm over the latent, then a 1×1 conv that projects from `N` channels to a smaller bottleneck `B`. |
| Temporal Convolutional Network | `R` repeats of `X` 1D conv blocks. Block `x` uses dilation `2^x`. Each block is `(B → H by 1×1) → PReLU → gLN → depthwise dilated conv kernel P → PReLU → gLN → (H → B by 1×1) residual + (H → B by 1×1) skip`. The skip connections are summed across all blocks. |
| Mask predictor | PReLU on the summed skip tensor, then a 1×1 conv `B → n_sources × N`, reshaped to `(n_sources, N, T')`, then a sigmoid mask activation (clean and artifact share the encoder feature space). |
| Decoder (ConvTranspose1D, kernel `L`, stride `L/2`, `N → 1` filters) | Maps each masked latent back to a 1-channel waveform `(1, T)`. Applied independently to each masked source so the output is `(n_sources, 1, T)`. |

A *source-additivity* projection step (paper section III.D, sometimes
called *consistency*): in the EEG/GA setting the noisy input is
exactly the sum of clean EEG and gradient artifact, so we may
optionally constrain `clean + artifact = noisy` at inference by
distributing the residual evenly between the two predicted sources.
This implementation exposes that as a configuration flag rather than
forcing it during training (the paper trains without that constraint).

## Inputs the original paper expects vs. our dataset

| Aspect | Conv-TasNet paper (WSJ0-2mix) | Our Niazy proof-fit dataset |
|---|---|---|
| Sampling rate | 8 kHz | typically 5 kHz (Niazy proof-fit, after upsampling factor 10) |
| Mixture length | 4-second segments (≈32 000 samples) | trigger-aligned center epoch, default 512 samples (≈100 ms) |
| Channels | 1 (mono mixture) | 1 per inference instance (channel-wise) |
| Sources | 2 unknown speakers | 2 known sources: clean EEG, gradient artifact |
| Permutation invariance loss (PIT) | required | **not** required: sources are ordered |

Mapping decisions:
- We work on a single EEG channel at a time. The mixture is
  `noisy_center[example, channel]` of shape `(1, samples)`. The target
  is `(clean_center, artifact_center)` stacked into `(2, samples)`.
- The smoke and full configs both use 512 samples per epoch (matches
  `cascaded_context_dae` and the default Niazy proof-fit dataset).
- We keep encoder kernel `L=16`, stride `L/2 = 8`. With `T = 512` this
  produces `T' = 63` encoder frames, large enough for a TCN with
  dilations up to 2^7 = 128 (receptive field exceeds 63, so the
  separator sees the full mixture in every position).
- Single channel per inference call means encoder input channels = 1.

## Loss function

Original paper: scale-invariant signal-to-distortion ratio (SI-SDR)
maximization with utterance-level permutation-invariant training (uPIT).

Adapted choice for this implementation:
- Default: `mse` over both predicted sources (clean and artifact).
  The dataset comes from a deterministic AAS-cleaned reference, so
  the absolute scale of clean and artifact is meaningful and SI-SDR's
  scale invariance would actually hide useful information.
- Optional: `si_sdr_neg` (negative SI-SDR averaged over the two
  sources), exposed via the loss factory for ablation. SI-SDR is
  defined per source and is *not* permutation-invariant in this
  implementation, since the source order is fixed (clean first,
  artifact second).
- Optional: a weighted combination
  `α * mse_clean + β * mse_artifact` lets us bias the model toward
  artifact reconstruction (which is the quantity actually subtracted
  by `DeepLearningCorrection`). Default is `α = β = 1.0`.

## Non-obvious training tricks (from paper and reference repos)

- **Global layer norm (gLN)** normalises across both feature and time
  dimensions; this matters for non-causal Conv-TasNet and is what we
  use. Cumulative layer norm (cLN) is only used for causal models.
- **Skip-connection sum** across all TCN blocks before the mask
  predictor matters for gradient flow.
- **PReLU** (not ReLU) inside the TCN blocks.
- **No batch normalisation** on the encoder/decoder side.
- **Mask activation = sigmoid** for non-negative encoder output. ReLU
  works too but converges more slowly.
- **Stride `L/2`** (50 % overlap of encoder windows) is important;
  non-overlapping encoders degrade SDR.
- **Weight initialisation**: PyTorch defaults are sufficient. The
  asteroid implementation does not override them.

## Hardware feasibility note

| Quantity | Estimate |
|---|---|
| Filters `N` | 256 |
| Encoder kernel `L`, stride `L/2` | 16, 8 |
| Bottleneck `B` | 128 |
| Conv-block channels `H` | 256 |
| Block kernel `P`, blocks per repeat `X`, repeats `R` | 3, 8, 2 |
| Approx parameter count | ≈1.3 M |
| FP32 model state | ≈5 MB |
| Activations per `(batch=64, T=512)` mini-batch | small (<200 MB) |
| Single-epoch wall-clock on RTX 5090 (Niazy 512-sample dataset) | well under 1 minute (smoke) |

This is far below the 24 GB VRAM envelope. Even pushing toward the
asteroid defaults (`N=512, H=512, R=3`) would stay under 4 M
parameters and well under 1 GB of activation memory at our batch
size.

We are *not* reducing a published model to fit. The published
recipe is significantly larger because it targets 4-second 8-kHz
mixtures, but the EEG-fMRI proof-fit segment is 100 ms at 5 kHz, so
the receptive-field requirement is much smaller. The model size
above is a deliberate match to the input length, not a budget cut.

## Open questions (recorded for hand-off)

- **Loss choice on this dataset**: MSE is a safe default but it is
  not what Conv-TasNet was originally trained against. If the
  smoke + full run does not converge on artifact prediction, the
  first follow-up experiment should be `si_sdr_neg` or the weighted
  MSE variant.
- **Source-additivity projection at inference**: if the predicted
  sources do not sum to the input, `DeepLearningCorrection` only uses
  the artifact source anyway, so the residual silently goes into the
  "clean" output. This is acceptable but means the model's "clean"
  output is not directly comparable to a Wiener-style baseline. The
  open question is whether enforcing `clean + artifact = noisy` at
  *training* time (by adding a consistency penalty) materially
  improves artifact reconstruction.
- **Per-channel vs. multichannel**: this implementation is strictly
  per-channel for compatibility with arbitrary EEG montages. A
  multichannel variant could exploit cross-channel artifact
  correlation but would couple the checkpoint to a specific montage.
- **Causal variant**: not implemented. Online correction would need
  the causal Conv-TasNet (cLN, no future-looking skip connections);
  the offline use case in FACETpy does not require it.
