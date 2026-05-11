# IC-U-Net Research Notes

## Source Papers

- **Primary paper**:
  Chuang, C.-H., Chang, K.-Y., Huang, C.-S., & Jung, T.-P. (2022).
  *IC-U-Net: A U-Net-based Denoising Autoencoder Using Mixtures of Independent
  Components for Automatic EEG Artifact Removal.* NeuroImage, 263, 119586.
  - DOI: 10.1016/j.neuroimage.2022.119586
  - PubMed: <https://pubmed.ncbi.nlm.nih.gov/36031182/>
  - arXiv preprint: <https://arxiv.org/abs/2111.10026>
  - Reference implementation: <https://github.com/roseDwayane/AIEEG>
- **Family overview (internal)**:
  `docs/research/dl_eeg_gradient_artifacts.pdf`, Section 3.3
  (*The U-Net Variation: IC-U-Net*).

## Plain-language description

IC-U-Net trains a 1-D multichannel U-Net to denoise EEG by working in
independent-component (IC) space rather than raw channel space. The original
recipe is: decompose EEG into ICs via Independent Component Analysis (ICA),
synthesize training pairs where the *noisy* input is a mixture of brain ICs
plus non-brain (artifact) ICs and the *clean* target is the same mixture
without the artifact ICs, and train the U-Net to reconstruct the clean signal.
ICA contributes a statistical-independence prior; the U-Net contributes
non-linear, data-driven reconstruction that can preserve the morphology that
classical IC-rejection would discard.

For gradient-artifact removal we adapt the recipe: the *noisy* input is the
raw context window that contains the fMRI gradient artifact, the *clean*
target is the AAS-corrected reference signal in the Niazy proof-fit dataset.
ICA is fit once offline on a sample of the training data, the resulting
30×30 unmixing matrix `W` is stored as a frozen linear layer in front of the
U-Net, and its pseudoinverse `W⁺` is stored as the post-U-Net inverse map.

## Key architectural components

Reverse-engineered from `roseDwayane/AIEEG/model/cumbersome_model2.py`:

| Block | Stage | Channels | Kernel | Down/Up |
|---|---|---|---|---|
| `inc`  | input  | 30 → 64  | 7 | – |
| `down1` | enc 1 | 64 → 128 | 7 | maxpool ×2 |
| `down2` | enc 2 | 128 → 256 | 5 | maxpool ×2 |
| `down3` | enc 3 (bottleneck) | 256 → 512 | 3 | maxpool ×2 |
| `up1`  | dec 1 | 512 + 256 → 256 | 3 | bilinear ×2 |
| `up2`  | dec 2 | 256 + 128 → 128 | 3 | bilinear ×2 |
| `up3`  | dec 3 | 128 + 64 → 64  | 3 | bilinear ×2 |
| `outc` | output | 64 → 30 | 1 | – |

`DoubleConv` = `Conv1d → BN → activation → Conv1d → BN → activation`. The
reference repo uses `sigmoid`; we use `LeakyReLU(0.1)` because the EEG signal
in our dataset is unbounded micro-volt amplitude and a sigmoid output bounds
the prediction in a way that does not match the target range. The change is
recorded here as a deliberate divergence from the published code.

Around the U-Net core we wrap:

- **Frozen `W`**: 30×30 linear map (no bias), initialised from
  `sklearn.decomposition.FastICA` fit on a flat sample of the training
  `noisy_context`. Maps channel-space signal → IC-space.
- **Frozen `W⁺`**: pseudoinverse of `W`, recovers channel space after the
  U-Net.
- **Center extractor**: slices the middle epoch out of the 7-epoch context
  to match the FACETpy `artifact_center` target.
- **Artifact head**: returns `noisy_center − clean_center` as the prediction
  so the loss can be applied directly on the FACETpy `artifact_center` target
  used for direct comparison with `cascaded_context_dae` and `cascaded_dae`.

The ensemble loss from the paper is reimplemented in `build_loss` and
includes four MSE terms: amplitude, first temporal difference, second
temporal difference, and FFT magnitude.

## Input contract vs the Niazy proof-fit dataset

| Parameter | IC-U-Net paper | This implementation |
|---|---|---|
| Sampling rate | 256 Hz | 4096 Hz (Niazy proof-fit) |
| Segment length | 1024 samples (≈4 s) | 7 × 512 = 3584 samples (≈0.87 s) |
| Channels | 30 ICs (1 IC = 1 channel) | 30 channels (Niazy proof-fit) |
| Target | clean IC mixture | AAS-corrected center epoch (`artifact_center` head) |
| ICA decomposition | offline, EEGLAB | offline, `sklearn.decomposition.FastICA` (see "ICA choice" below) |

The Niazy proof-fit dataset NPZ provides (see
`examples/build_niazy_proof_fit_context_dataset.py`):

- `noisy_context` — `(N, 7, 30, 512)`
- `artifact_center` — `(N, 30, 512)` (training target)
- `clean_center` — `(N, 30, 512)`
- `clean_context`, `artifact_context`, `noisy_center`, …
- `sfreq` — `4096 Hz`

The dataset wrapper reshapes `noisy_context` from `(7, 30, 512)` into a single
multichannel time series `(30, 3584)` before feeding the model. The output is
the predicted center-epoch artifact, shape `(30, 512)`, which matches the
dataset's `artifact_center` head. **Use of the full 7-epoch context is
explicit** — this addresses the lesson learned from DHCT-GAN, where a single-
epoch input was the suspected cause of catastrophic failure.

## Loss function — ensemble

The paper combines amplitude, velocity, acceleration, and frequency terms.
We implement an ensemble that mirrors this:

```
L = w_a · MSE(y, ŷ)
  + w_v · MSE(Δy, Δŷ)
  + w_j · MSE(Δ²y, Δ²ŷ)
  + w_f · MSE(|FFT(y)|, |FFT(ŷ)|)
```

Default weights `(1, 1, 1, 0.5)`. `MSE` denotes elementwise mean. The
frequency term is computed with `torch.fft.rfft` on the last dim.

## Non-obvious training tricks

- The reference repo uses `Sigmoid` activations; we use `LeakyReLU(0.1)`
  because the unbounded EEG amplitudes do not fit a sigmoid output. This is
  the only documented divergence from the published architecture.
- The reference repo trains on a heavily preprocessed signal (FIR 1–50 Hz,
  resampled to 256 Hz). FACETpy intentionally keeps the input at native
  sampling rate so we can learn the GA morphology at its actual bandwidth.
- ICA is fit once at `build_model` time on a flat sample of training data,
  then frozen. Refitting ICA every epoch is impractical and is not what the
  paper does either.

## ICA choice — `sklearn.decomposition.FastICA`

Two viable options:

- `mne.preprocessing.ICA` — domain-aware, expects an MNE `Raw`. Adds a
  dependency on building a `Raw` from arbitrary arrays in the model factory.
- `sklearn.decomposition.FastICA` — pure NumPy, accepts an `(n_samples,
  n_channels)` matrix directly.

**Decision: `sklearn.decomposition.FastICA`.** Reasons:

1. We do not need MNE-specific channel metadata for the unmixing matrix.
2. The model factory operates on the NPZ bundle, not on raw recordings.
3. Reproducible numerics through `random_state`.

The same `W` is used at training and inference. Inference does **not** refit
ICA — the matrix is baked into the TorchScript checkpoint via
`nn.Linear(bias=False)` and `register_buffer`.

## Hardware feasibility (RTX 5090, 24 GB VRAM)

- Input per batch (batch 16, 30 channels, 3584 samples, float32): ~7 MB.
- U-Net activations across all encoder/decoder feature maps: ~50 – 100 MB.
- Parameters ≈ 4 M (encoder + decoder); ICA matrices add 30×30 = 900 floats.
- Estimated single-epoch wall-clock on Niazy proof-fit (833 examples,
  batch 16): ~5 – 10 s on RTX 5090.
- Total VRAM budget at batch 16: well below 4 GB even with optimizer state.

No reduction relative to the published model is required.

## Comparison context

The orchestrator prompt requested comparison against the DPAE discriminative
baseline (+7.48 dB clean-SNR improvement reported for that family). DPAE is
the only direct discriminative-CNN family comparator; ICA-augmented U-Net
trades parameter count for an explicit independence prior. We compare to:

- `cascaded_dae` (single-channel windowed DAE baseline)
- `cascaded_context_dae` (7-epoch channel-wise DAE; matches our context shape)
- DPAE (+7.48 dB reference, when its run is available in the fleet)

## Open questions

1. Does `FastICA(n_components=30)` converge reliably on the gradient-artifact-
   dominated noisy context? If not, we fall back to PCA-whitened identity.
2. Is the ensemble loss strictly better than plain MSE on this dataset? We
   default the smoke run to MSE and switch the full run to ensemble.
3. Native epoch length variation (584 – 605 samples on this dataset, see
   metadata) is normalised away by `_resample_1d` to 512 samples before
   training. At inference we re-stretch back to the native length, like
   `cascaded_context_dae`.
