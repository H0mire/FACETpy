# Nested-GAN Research Notes

## Source Material

Primary paper:

- Anonymous (2025). *End-to-End EEG Artifact Removal Method via Nested
  Generative Adversarial Network*. **Biomed. Phys. Eng. Express**, published
  online 2025-11-25. PMID 41183389. DOI:
  [10.1088/2057-1976/ae1a8c](https://doi.org/10.1088/2057-1976/ae1a8c).
  PubMed entry: https://pubmed.ncbi.nlm.nih.gov/41183389/

The paper is paywalled (IOP) and not on arXiv. Only the PubMed abstract is
freely accessible. Architectural details below come from (a) the abstract,
(b) the report `docs/research/dl_eeg_gradient_artifacts.pdf` Section 4.3,
(c) the orchestrator's PRIMARY_PAPER_HINT, and (d) the closely related
public work below. The orchestrator explicitly flagged the dataset-context
strategy lesson learned from DHCT-GAN (-7.13 dB on center-only input).

Closely related public work consulted for technical details that the
abstract omits:

- Zamir, S. W. et al. (2022). *Restormer: Efficient Transformer for
  High-Resolution Image Restoration.* CVPR.
  https://github.com/swz30/Restormer — source of MDTA (multi-DConv head
  transposed attention) and GDFN (gated DConv feed-forward) blocks.
- Lee et al. (2025). *TF-Restormer: Complex Spectral Prediction for Speech
  Restoration.* arXiv 2509.21003 — adaptation of Restormer to complex
  STFT with time/frequency dual-path blocks. Used as template for the
  inner branch.
- Cao et al. (2022). *CMGAN: Conformer-based Metric GAN for Speech
  Enhancement.* Interspeech. — origin of the "metric discriminator"
  pattern (predicts perceptual quality score, e.g. PESQ).
- Kong et al. (2020). *HiFi-GAN.* NeurIPS. — origin of the
  multi-resolution discriminator (multiple STFT/multi-period heads).

## Plain-language description

Nested-GAN attacks the gradient artifact in two complementary domains in
sequence. The **inner GAN** operates on the short-time Fourier transform
(STFT) of the noisy EEG. Its generator — a light-weighted complex-valued
Restormer transformer — removes specific harmonic spectral content of the
gradient artifact while leaving the residual EEG spectrum intact. Two
*metric discriminators* score the inner generator's output against a
target spectrogram, so the inner generator does not just minimise MSE but
also matches a perceptual/quality metric. The cleaned spectrogram is
inverted back to time. The **outer GAN** then refines that time-domain
waveform: it sees the inner-branch reconstruction *plus the surrounding
context epochs* and corrects residual phase discontinuities at trigger
boundaries. Two *multi-resolution discriminators* compare the refined
waveform against ground truth at multiple time-frequency resolutions,
which is what gives the high-frequency texture preservation GAN-based
methods are known for. A *gradient-balance* scheme keeps the four
discriminators contributing comparable magnitudes during training so
that none dominates the gradient signal that flows into the generators.

Design intuition (also from the report, §4.3): the gradient artifact is
**sparse in frequency** (concentrated at TR-locked harmonics) but
**dense in time** (continuous across every TR-bounded epoch). The
inner-GAN attacks the sparse-in-frequency view directly. The outer-GAN
attacks the dense-in-time view including inter-epoch boundary continuity.

## Architectural components

| Component | Domain | Job |
|---|---|---|
| Inner generator (complex-valued Restormer) | STFT (complex) | Predict cleaned complex spectrogram of the center epoch |
| Inner metric discriminators (×2) | STFT magnitude | Score predicted spectrogram against target on a perceptual scalar |
| Outer generator (time-domain CNN/U-Net) | Time, 1D | Refine waveform; close inter-epoch phase discontinuities |
| Outer multi-resolution discriminators (×2) | Time → multi-STFT | Adversarial loss at several window sizes |
| Gradient balance | training | Rescale the four adversarial losses to comparable magnitudes |

Inputs the original paper expects: realistic and semi-synthetic EEG
datasets. The abstract reports MSE 0.098, Pearson r 0.892, RRMSE 0.065,
71.6 % temporal artifact reduction, 76.9 % spectral artifact reduction.
Exact sampling rate, segment length, channel layout, optimiser, batch
size, and epoch count are **not disclosed in the publicly accessible
abstract**. This is an open question (see below) that I resolve by
following the FACETpy proof-fit dataset shape and the related-work
defaults.

## Mapping to the Niazy proof-fit dataset

The dataset bundle produced by
`examples/build_niazy_proof_fit_context_dataset.py` (already validated in
the cascaded_context_dae pipeline) contains:

```text
noisy_context        (N, 7, 30, 512)   float32
artifact_context     (N, 7, 30, 512)   float32
clean_context        (N, 7, 30, 512)   float32
noisy_center         (N, 30, 512)      float32
artifact_center      (N, 30, 512)      float32
clean_center         (N, 30, 512)      float32
artifact_epoch_lengths_samples         int
trigger_phase_linear                   float
trigger_phase_sincos                   float
sfreq                                  float
ch_names                               object
```

The center-epoch artifact (`artifact_center`) is the supervision target.
`noisy_context` carries the seven-epoch context the orchestrator demands.

### Context-handling strategy (mandated)

> Project lesson learned from DHCT-GAN: use the full 7-epoch noisy_context
> input (shape 7, 30, 512), NOT just the center epoch.

I follow this rule. The inner-GAN may compute its STFT on the center
epoch alone, but the outer-GAN sees all seven context epochs so its
time-domain refinement has phase-continuity boundary information at
trigger boundaries. The 7-channel × 512-sample tensor that feeds the
outer branch is built by replacing the center channel of the
noisy-context stack with the inner-GAN time-domain output and leaving
the surrounding six epochs unchanged. This preserves boundary
information exactly the way DHCT-GAN-v2 should have (and DHCT-GAN
should have but did not).

### Channel layout

The model trains **per channel**. Each training example is one channel
of one window, shape `(7, 1, 512)` → target `(1, 512)`. This matches the
`cascaded_context_dae` ChannelWiseContextArtifactDataset pattern. Two
practical consequences:

1. The TorchScript checkpoint is independent of the channel count at
   inference time. The 30-channel Niazy dataset and a 64-channel BIDS
   dataset use the same checkpoint.
2. The batch dimension is `n_examples × n_channels = 833 × 30 ≈ 25 000`
   per epoch of training, which gives a healthy number of gradient
   updates even though only 833 windows exist.

### Sampling rate and segment length

The bundle is at the dataset's native sfreq (Niazy proof-fit is
upsampled to ~5 kHz before AAS; the `sfreq` field in the bundle records
the resolved value). Each epoch is 512 samples. The model is therefore
designed for input length 512 and arbitrary sfreq within the validated
range; the bundle's reported `sfreq` is stored in the run metadata for
reproducibility.

## Loss functions

The original paper uses four adversarial losses plus a reconstruction
loss with gradient balance. For the FACETpy proof-fit run I use a
**generator-only training recipe** that approximates the same multi-
resolution spectral fidelity without requiring four discriminators or a
custom alternating GAN training loop (see scope reduction below). The
loss is

```
L_total = lambda_time * L1(time)
        + lambda_mrstft * MultiResolutionSTFTLoss(pred, target)
```

where `MultiResolutionSTFTLoss` is the standard sum of L1 errors over
log-magnitude STFT at four window sizes `[32, 64, 128, 256]` with
hop=window/4. This is the same loss family that the multi-resolution
discriminators in the published Nested-GAN are effectively learning to
score. It is widely used in modern neural speech enhancement (HiFi-GAN,
DEMUCS, BigVGAN) and gives much better high-frequency fidelity than
plain MSE/L1.

## Non-obvious training tricks

- **Demean center epoch before STFT**. The center epoch can carry a
  large DC offset from the artifact baseline; removing it before STFT
  keeps the spectrogram dynamics in the network's input range and
  matches the pattern already used in cascaded_dae and
  cascaded_context_dae (`demean_input=true` in their YAML).
- **Residual injection at the outer branch**. The outer branch receives
  the inner-GAN inverse-STFT output in the center-epoch slot of the
  7-epoch context and predicts a residual artifact correction. Avoids
  re-learning the bulk artifact the inner branch already removed.
- **Per-channel mean removal on the prediction**. The model output is
  demeaned channel-wise before being returned by the adapter, matching
  the `remove_prediction_mean=true` convention used by the other
  channel-wise context DAEs. This prevents a constant baseline drift
  from leaking through the subtraction step.

## Hardware feasibility

| Component | Parameter count (estimate) | Activation memory at B=128 |
|---|---|---|
| Inner generator (4 transformer blocks @ C=48 over 33×33 complex bins) | ~1.2 M | ~10 MB |
| Outer generator (1D U-Net, 4 levels, base width 32, 7→1 channels, 512 samples) | ~0.7 M | ~6 MB |
| Total trainable | **~2.0 M** | **~16 MB / batch** |

With batch_size 128 and an RTX 5090 (32 GB VRAM on the 5090 D, 24 GB on
the consumer 5090), the model fits with two-digit gigabytes of headroom.
A full pass over the 833-example bundle expanded to channels takes
`833 × 30 / 128 ≈ 196` steps. An A-grade RTX 5090 step on a model this
size lands around 30-40 ms, so one epoch is well below a minute and a
50-epoch full run finishes inside an hour.

The smoke run uses `max_epochs=1`, no data augmentation, the same
batch size, so it must finish in well under one minute.

## Scope reduction vs the published paper

I deliberately reduce the published Nested-GAN recipe in three places.
The reasons are recorded here so the orchestrator can decide whether
the reduction is acceptable.

1. **No separate discriminator networks.** Reason: the FACETpy
   `PyTorchModelWrapper` accepts a single `(prediction, target) → scalar`
   loss and a single optimizer over `model.parameters()`. A full GAN
   training loop with alternating generator/discriminator updates and
   four discriminators would require a custom `TrainableModelWrapper`
   plus changes to the training CLI, which is prohibited by the
   Phase-3 reuse rules ("Do not roll your own training loop unless your
   architecture genuinely cannot be expressed through this CLI plus a
   custom wrapper"). The multi-resolution STFT loss captures the
   spectral-fidelity behaviour the multi-resolution discriminators are
   trained to enforce.
2. **Generator-only metric supervision.** Reason: the metric
   discriminators in the paper predict a single perceptual scalar.
   Without a defined perceptual metric for EEG (PESQ is speech-only),
   the metric discriminator would learn an ill-defined target. The
   per-frequency multi-resolution STFT L1 loss is a defensible
   substitute that does not require an extra trainable network.
3. **No explicit gradient-balance scheme.** Reason: gradient balance
   only matters when the four discriminator losses are present.
   Without them the question disappears. The two loss weights
   `lambda_time` and `lambda_mrstft` are tuned manually.

If the orchestrator wants the full GAN training recipe, the right
follow-up is a model-specific `TrainableModelWrapper` subclass plus
either a CLI extension or a stand-alone training script under
`src/facet/models/nested_gan/`. The HANDOFF.md will flag this.

## Open questions

- Original-paper STFT parameters (window length, hop, n_fft) are not
  in the public abstract. I default to `n_fft=64, hop=16, win=64` for
  the inner branch operating on 512-sample epochs, giving a 33-bin
  spectrogram with 33 time frames — a square grid that Restormer-style
  transformers handle naturally.
- Original-paper number of Restormer blocks and feature channels are
  not disclosed. I default to four blocks at C=48 to stay well under
  the 24 GB VRAM envelope.
- Original-paper outer-branch architecture is described only as
  "time-domain refinement". I implement a 1D residual U-Net with four
  encoder/decoder levels because that matches the most common
  refinement choice in dual-domain speech enhancement.
- Original-paper loss weights and optimiser schedule are not
  disclosed. I default to AdamW with `lr=1e-3`, `weight_decay=1e-4`,
  no scheduler — same defaults as cascaded_context_dae and
  cascaded_dae for cross-model comparability.

## Comparison baselines

The HANDOFF.md must benchmark Nested-GAN against:

- `cascaded_dae` — channel-wise DAE baseline (single epoch input).
- `cascaded_context_dae` — channel-wise DAE with the same 7-epoch
  context input. This is the most direct architectural comparison
  because the dataset, channel layout, and target are identical.
- `dhct_gan` — known to have failed at -7.13 dB on this dataset
  (orchestrator note). Nested-GAN should beat this clearly or the
  hypothesis "nested time-frequency decomposition outperforms single-
  domain GAN" is unsupported.
- `dhct_gan_v2` — if a result is available by evaluation time, the
  comparison directly isolates time-frequency decomposition vs single
  domain.
