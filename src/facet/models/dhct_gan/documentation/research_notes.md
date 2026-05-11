# DHCT-GAN Research Notes

## Source

- Cai, M.; Zeng, H.; et al. *DHCT-GAN: Improving EEG Signal Quality with a Dual-Branch
  Hybrid CNN-Transformer Network.* MDPI Sensors **25**(1), 231 (2025).
  DOI: 10.3390/s25010231. <https://www.mdpi.com/1424-8220/25/1/231>
  PMC mirror: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11723461/>
- Related GAN-style EEG denoising background: *GAN-Guided Parallel CNN-Transformer
  Network for EEG Denoising* (PubMed 37220036), and the pix2pix conditional GAN
  setup (Isola et al., 2017) for general training-stability tricks.

## Plain-language description

DHCT-GAN is a conditional generative adversarial network for EEG denoising. Its
generator is a U-shaped hybrid CNN-Transformer encoder followed by two parallel
decoders: one decoder predicts the clean EEG (`Y1`), the other predicts the noise
component (`Y2`). A learned gating network produces per-sample mixing weights
`(Ymask1, Ymask2)` that fuse the two branches into a final clean prediction:

```
Ypre = Ymask1 * Y1 + Ymask2 * (Xraw - Y2)
```

The intuition is that a dedicated "clean" decoder and a dedicated "noise"
decoder learn complementary representations, and the gate decides per-sample
which branch is more trustworthy. The published model uses three discriminators
operating on `Y1` (clean), `Y2` (noise) and `Ypre` (fused output) respectively.
Each component has a reconstruction + perceptual + adversarial loss, summed into
a single generator objective.

## Architectural components (published model)

| Component | What it does |
|---|---|
| Preprocessing | Two 1D conv layers expand the 1-channel input to 32-dim features, followed by average pooling. |
| Encoder | 5 CNN-LGTB blocks. CNN sub-block: 2x conv (k=3) + BN + LeakyReLU + downsample. LGTB: 8x local self-attention + 1x global self-attention + FFN. Channel widths double per stage: 64 → 128 → 256 → 512 → 1024. |
| Decoder (x2) | Two parallel decoder branches with symmetric upsampling + skip connections from the encoder. One branch outputs the clean estimate `Y1`, the other outputs the noise estimate `Y2`. |
| Gating network | Two FC layers + tanh, produces per-sample `(Ymask1, Ymask2)`. |
| Discriminator | 3 sibling discriminators (D1/D2/D3). Each is an 8-layer 1D-CNN: channels 64→64→128→128→256→256→512→512, k=3, stride=2, padding=1. |
| Loss | `Lmse + λ1 * Lfeat + λ2 * Ladv` per output branch; total generator loss = sum over the three branches. |

## Published training contract

- Input: 1-channel time-domain segments, **2 s @ 512 Hz = 1024 samples**.
- Datasets: EEGdenoiseNet (clean EEG / EOG / EMG segments), MIT-BIH ECG, and
  semi-simulated EEG+EOG recordings, all resampled to 512 Hz.
- Train / val / test: 80 / 10 / 10.
- Synthetic SNR levels: −7 dB ... +2 dB during training, 10 discrete levels for
  evaluation.
- Optimizer: Adam. Generator betas `(0.5, 0.9)`. Discriminator betas
  `(0.9, 0.999)`. Learning rate `0.001` (initial), warmed down to `0.0001`.
- Batch size 40. Max 1000 epochs. Early stopping on validation MSE.
- λ1, λ2 not numerically specified in the paper — typical values for similar
  pix2pix-style adversarial denoising are `λ_recon = 100`, `λ_adv = 1`.
- Reported metrics: RRMSE (time), RRMSE (spectrum), correlation coefficient,
  artifact reduction ratio η, SSIM, mutual information.

## Mapping to the FACETpy Niazy proof-fit dataset

The Niazy bundle in `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`
exposes per-trigger context windows:

| Array | Shape | Meaning |
|---|---|---|
| `noisy_context` | `(833, 7, 30, 512)` | 7-epoch noisy context per example, 30 channels, 512 samples |
| `noisy_center` | `(833, 30, 512)` | center epoch only |
| `clean_center` | `(833, 30, 512)` | AAS-corrected (clean) center |
| `artifact_center` | `(833, 30, 512)` | AAS-estimated artifact (= `noisy_center - clean_center`) |
| `sfreq` | `4096.0` | native sfreq, but each epoch is resampled to 512 samples |

DHCT-GAN's input is single-channel time-domain. We therefore train per channel
with each example being `(1, 512)`: 833 examples × 30 channels = **24990 training
windows**. This matches the strategy used by `cascaded_dae` /
`cascaded_context_dae`. The center epoch is enough — DHCT-GAN as published is
single-epoch and does not consume trigger context.

The dataset's `artifact_center` becomes the training target; the network's
predicted noise branch `Y2` is the artifact estimate, so at inference time we
subtract `Y2` from the noisy input to obtain the clean signal. We expose
`Y2` (artifact) as the model's output for the FACETpy
`DeepLearningCorrection` pipeline (`output_type=ARTIFACT`).

## Reductions vs. the published model (justified)

The published architecture has 5 stages with 64→1024 channel widths and uses
three discriminators on 2-second segments at 512 Hz (= 1024 samples). For our
proof-fit dataset we make the following reductions:

| Aspect | Published | Adopted | Reason |
|---|---|---|---|
| Segment length | 1024 samples | 512 samples | Niazy proof-fit windows are 512 samples per epoch. |
| Encoder depth | 5 stages | 4 stages | 512-sample windows do not survive 5x downsampling. |
| Channel widths | 64,128,256,512,1024 | 16,32,64,128 | Fits 24 GB VRAM and matches the much smaller dataset size (~25 k windows vs. tens of thousands of EEGdenoiseNet windows). |
| Transformer | local-only (8 heads) + global (1 head) | local + global with 4 heads, depth 1 | Same idea, smaller. |
| Discriminators | 3 (clean, noise, fused) | 1 (PatchGAN on the noise/artifact output) | Single optimizer constraint in `facet-train` (see below). |
| Loss heads | `L1+L2+L3` summed over Y1, Y2, Ypre | reconstruction on Y2 (artifact) + consistency `Ypre vs clean` + adversarial on Y2 | Same objective spirit. |

### Parameter and memory budget

Rough estimate with the reduced widths:

- Encoder: ~4 × (2 × 3 × C_in × C_out) conv + transformer FFN ≈ 1.5 M parameters.
- Two decoders + gate: ~1.2 M parameters.
- Generator total: ~2.7 M parameters.
- PatchGAN discriminator (channels 16,32,64,128, k=4, stride=2): ~0.3 M parameters.

At batch 64 with 1-channel × 512 samples, activations peak well under 1 GB on
RTX 5090. Tracing for TorchScript export fits comfortably.

### Wall-clock per epoch

24 990 windows / batch 64 = ~390 batches per epoch. On RTX 5090 a single
forward+backward of a 3 M-param transformer-light model at this batch size is
< 10 ms, so an epoch budget of 4-8 s is realistic. A full 100-epoch run is
expected to finish in well under 15 minutes. Smoke (1 epoch) finishes in
< 30 s including dataset load.

## Loss function used in this implementation

Let `x` = noisy_center, `a_t` = artifact_center target, `c_t` = clean_center
target. The generator outputs `(a_hat, c_hat, g)` where `g ∈ [0,1]` is the gate.

```
clean_pre  = g * c_hat + (1 - g) * (x - a_hat)
L_recon    = L1(a_hat, a_t) + α * L1(clean_pre, c_t)
L_adv      = BCEWithLogits(D(a_hat), 1)                   # generator side
L_disc     = 0.5 * BCEWithLogits(D(a_t), 1)
           + 0.5 * BCEWithLogits(D(a_hat.detach()), 0)    # discriminator side
L_total    = L_recon + β * L_adv
```

We use `α = 0.5` and `β = 0.1` as starting points (typical pix2pix scaling but
softened because the adversarial term tends to destabilize at high weight for
EEG-style data). These are exposed as YAML hyperparameters.

The discriminator and its private optimizer are encapsulated inside the loss
module rather than the generator. This keeps the standard `facet-train`
single-optimizer contract intact: the CLI's optimizer only sees generator
parameters, while the loss module performs the alternating discriminator step
internally whenever gradients are enabled.

## Non-obvious training tricks

- Generator output `a_hat` is *demeaned per window* on the noisy input branch
  inside the dataset wrapper so the model learns artifact morphology rather
  than baseline drift, matching the convention used by other models in this
  repo (`cascaded_dae`).
- LeakyReLU(0.2) throughout, matching the published model.
- Spectral / FFT loss can be added optionally; not enabled by default.
- We feed `a_hat.detach()` to the discriminator during the discriminator step
  to prevent gradient leakage into the generator on that pass.
- We skip the internal discriminator step under `torch.is_grad_enabled() == False`
  so validation runs cleanly under `torch.no_grad()`.

## Open questions

1. The paper does not give numerical λ1 / λ2 values. Starting values `α = 0.5`
   and `β = 0.1` are educated guesses; the gradient-reversal-style trick that
   we use keeps disc training independent from the generator's adversarial
   weight, so β is mostly a tradeoff knob between fidelity and crispness.
2. The published model targets EOG/EMG/ECG artifact patterns, not fMRI
   gradient artifacts. Gradient artifacts have a much more periodic, narrow-band
   structure. The CNN+Transformer architecture should still be appropriate
   because the global attention can capture the TR periodicity, but the
   adversarial component may end up redundant when reconstruction error is
   already very low.
3. The dual-branch (clean + noise) is somewhat redundant on this dataset
   because `clean = noisy - artifact` is an algebraic identity given perfect
   prediction. The gating network is intended to use whichever branch is more
   reliable per sample. With a single Niazy AAS source, we expect the noise
   branch to dominate (artifact estimation is the supervised signal).
