# IC-U-Net — paper-accurate edition

A more faithful re-implementation of **IC-U-Net** for FACETpy gradient-artifact
removal, staying compatible with the facet-train factory and inference contracts
and runnable on CPU.

**Source paper.** C.-H. Chuang, K.-Y. Chang, C.-S. Huang, T.-P. Jung,
*"IC-U-Net: A U-Net-based denoising autoencoder using mixtures of independent
components for automatic EEG artifact removal,"* NeuroImage 263:119586 (2022);
arXiv:2111.10026; reference repo `roseDwayane/AIEEG`. Section/equation references
below are to that paper.

## What this edition changes vs. the original `facet.models.ic_unet`

The original edition is **not** modified. This package lives beside it at the
same import depth and registers a **new, globally-unique** processor name
`ic_unet_paper_accurate_correction`.

| # | Change | Paper basis | Original behaviour |
|---|--------|-------------|--------------------|
| 1 | **Sensor-level network, no in-graph ICA** | Sec 2.3, Conclusion: ICA/ICLabel are used only to *synthesise training data*; the runtime network is a plain channel-space U-Net. | Original baked a frozen ICA `W` and pseudoinverse `W_pinv` as buffers, sandwiching the U-Net in IC space. |
| 2 | **ReLU CBR blocks** (default) | Sec 2.1: "Convolution, Batch normalization, and ReLU activation". | Original used `LeakyReLU(0.1)`. |
| 3 | **Transposed-convolution decoder** | Sec 2.1: decoding uses a 1-D *transposed convolution* (deconvolution). | Original used parameter-free `nn.Upsample(mode='linear')`. |
| 4 | **Normalised equal-weight ensemble loss** | Eq. 2, Fig 4A: `L_ens = (1/Σαᵢ)·Σ αᵢ Lᵢ`, best config `α=[1,1,1,1]`. | Original used weights `[1,1,1,0.5]` summed with **no** `1/Σαᵢ` normalisation. |
| 5 | **Paper-accurate frequency term** | Eq. 4: MSE of the **z-scored power spectral density** `\|rfft\|²` restricted to **1-50 Hz**. | Original used raw `\|rfft\|` magnitude over the **full** spectrum, no PSD, no band limit, no z-score. |
| 6 | **Per-time-series z-score normalisation** (default) | Sec 3.1/4.1: z-score each time series before training. | Original only per-channel demeaned (variance untouched). |
| 7 | **Clean-reconstruction target** (default) | Eqs 1-3: the DDAE reconstructs the clean signal `Y ≈ X`. | Original targeted `artifact_center` (artifact head). |

Items 1, 5 are the **high-severity** corrections; 2, 3, 4, 6 are medium; 7 is
low. Each is configurable so the original behaviour can be reproduced for
like-for-like comparison.

## Architecture

`IcUNetPaperAccurate` wraps `IcUNetCore`, a sensor-level 1-D U-Net:

- **Encoder**: an input CBR double-conv, then `depth-1` downsampling stages
  (`MaxPool1d(2)` + CBR double-conv). Filter count **doubles** after each
  downsample (`base, 2·base, 4·base, …`), as the paper specifies.
- **Decoder**: `depth-1` upsampling stages, each a learned
  `nn.ConvTranspose1d(stride=2)` (the paper's deconvolution) followed by
  concatenation of the encoder skip and a CBR double-conv. Filter count
  **halves** per stage. A pad/crop guard tolerates odd context lengths.
- **Head**: a `1×1` Conv1d back to `n_channels`. The center epoch is then sliced
  out, and the model returns either the clean center epoch (`output_type='clean'`)
  or `noisy_center − clean_center` (`output_type='artifact'`).

Defaults: `depth=4`, `base_channels=64`, `kernel_size=7`, `activation='relu'`.
The exact Fig 2A kernel sizes could not be transcribed from the PDF; the
doubling/halving ladder is paper-consistent and depth/base/kernel are exposed as
kwargs. This follows the documented `roseDwayane/AIEEG` configuration.

## Loss

`build_loss('ensemble', sfreq=…)` returns `IcUNetEnsembleLoss`:

```
L_ens = (1/Σαᵢ) · (α1·MSE(Y,X) + α2·MSE(ΔY,ΔX) + α3·MSE(Δ²Y,Δ²X) + α4·L_freq)
```

with `α=[1,1,1,1]`. `L_freq` is the MSE of the per-channel z-scored PSD
(`|rfft|²`) over the 1-50 Hz band; `sfreq` (injected by facet-train) resolves the
band, the hi edge is clamped below Nyquist, and short signals fall back to the
full positive-frequency band so the term is safe for arbitrary sample rates.
`'mse'`, `'l1'`, `'smooth_l1'` aliases remain available.

## Dataset

`build_dataset(path=…, target_type='clean', normalize='zscore')` wraps
`NPZContextArtifactDataset`, selecting `clean_center` (clean) or
`artifact_center` (artifact) as the target. The context `(epochs, channels,
samples)` is flattened to `(channels, epochs·samples)` for the 1-D U-Net.
Per-channel z-score normalisation is the default; the per-example `(mean, std)`
is recoverable via `scale_for(idx)` for denormalisation at inference. `'demean'`
(the original convention) and `'none'` remain opt-in.

## Inference

`IcUNetPaperAccurateCorrection` (processor name
`ic_unet_paper_accurate_correction`) loads a TorchScript checkpoint and runs the
sensor-level U-Net over the trigger-aligned epoch context. The adapter applies
the same training-time normalisation, denormalises the network output, and (for
the clean head) recovers the artifact as `noisy_center − predicted_clean` so the
standard FACET subtract-the-artifact pipeline works unchanged, while also
returning `clean_data`.

## Deviations kept for EEG-fMRI (documented, not blind copies)

- **In-graph frozen ICA** is *not* in the paper and is meaningless for
  single-/few-channel EEG-fMRI (too few channels for a stable decomposition; the
  original code itself falls back to identity). Dropped from the default; kept as
  an opt-in `use_frozen_ica=True` extension.
- **ICA + ICLabel mixB/mixBnB data synthesis** (Eqs 5-6) targets eye/muscle/
  heart/line/channel ICs — none of which is the fMRI gradient artifact — and
  needs many channels and a resting-state IC library we do not have. FACETpy's
  physically paired noisy/clean NPZ (AAS-referenced) is the appropriate domain
  substitute for the paper's synthetic pairs.
- **Published hyperparameters** (lr=0.01, batch=128, 150 epochs, 256 Hz,
  1024-sample 4-s segments) are tuned for 150k resting-state segments; they are
  documented in this folder but FACETpy-appropriate values are used for the
  proof-fit run, and the smoke run shrinks everything for CPU.

See `documentation/paper_accuracy_review.md` for the full discrepancy table and
EEG-fMRI applicability assessment.

## Smoke test

`tests/models/ic_unet_paper_accurate_edition/test_training_smoke.py` builds a
tiny CPU model (`n_channels=4`, `base_channels=4`, `depth=2`, context `3×32=96`),
runs a few AdamW steps asserting the loss decreases, checks the ensemble loss
runs and is finite with `sfreq=256`, and asserts the forward output shape is
`(batch, n_channels, epoch_samples)`. `training_niazy_proof_fit_smoke.yaml`
mirrors these tiny dims with `device: cpu` (illustrative; not executed by the
test).
