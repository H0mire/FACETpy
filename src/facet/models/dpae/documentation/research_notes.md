# DPAE Research Notes

## Source paper

Xiong, W., Ma, L., & Li, H. (2023). "A general dual-pathway network for EEG denoising." *Frontiers in Neuroscience*, 17, 1258024. <https://doi.org/10.3389/fnins.2023.1258024>

- PMC mirror: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10847297/>
- Frontiers HTML: <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1258024/full>

The FACETpy report (`docs/research/dl_eeg_gradient_artifacts.pdf`, Section 3.2) summarises this model as the canonical Dual-Pathway Autoencoder (DPAE).

## One-paragraph description

DPAE is a discriminative supervised denoising autoencoder. Its core idea is that a single fixed convolutional receptive field cannot simultaneously resolve fast spike-like neural activity and slow rhythmic artifact structure. The encoder is therefore split into two parallel pathways operating on the same input: a *local* pathway with small kernels (and optionally dilation) that captures high-frequency / fine-temporal detail (spikes, gamma activity, sharp gradient edges), and a *global* pathway with large kernels and pooling that captures low-frequency / long-range structure (BCG drift, slow cortical potentials, periodic gradient ramp). The two latent streams are concatenated, batch-normalised, and decoded back to the original temporal resolution by transposed convolutions. A residual connection across the fusion module stabilises training. The published variants (MLP, 1D-CNN, 1D-RNN) all share this dual-pathway encoder template; we implement the 1D-CNN variant because it matches the FACETpy correction setting (single-channel time-domain input, per-epoch artifact prediction).

## Architectural components

| Component | Responsibility |
|---|---|
| Local pathway | Small-kernel 1D convolutions (k=3) with increasing dilation rates to capture spike-like high-frequency morphology with a moderate receptive field while keeping the parameter count low. SeLU activation. |
| Global pathway | Large-kernel 1D convolutions (k=15, 11, 7) with strided pooling to capture low-frequency slow drifts and the gradient-pulse envelope. SeLU activation. |
| Fusion block | Channel-wise concatenation of local and global latents at the bottleneck, batch-normalised, projected with a 1×1 conv. |
| Decoder | Mirror of the encoder using transposed 1D convolutions (k=4, stride=2) to upsample the fused latent back to the input length. SeLU activation. The final layer is a 1×1 conv that reduces to one output channel. |
| Residual connection | Adds the encoder input to the decoder output, gated by a learned per-channel scalar. Stabilises early training and lets the network start from an identity-like behaviour. |

## Inputs the original paper expects

- Sampling rate: 256 Hz
- Segment length: 2 s = 512 samples
- Single-channel 1D time-series input, target is the clean EEG estimate
- Trained on the EEGdenoiseNet benchmark (EEG mixed with EOG/EMG synthetic noise)

## Mapping to the Niazy proof-fit dataset

The bundle built by `examples/dataset_building/build_niazy_proof_fit_context_dataset.py` produces:

- `noisy_context`: `(n_examples, 7, n_channels, 512)`
- `noisy_center`:  `(n_examples, n_channels, 512)`
- `artifact_center`: `(n_examples, n_channels, 512)`
- `clean_center`: `(n_examples, n_channels, 512)`
- `sfreq`: scalar (Niazy original sampling rate, ~5 kHz, but the bundle resamples per-epoch to 512 samples; the per-sample rate inside the network is therefore not 256 Hz, it is the 512-sample resampled per-epoch rate)

To match DPAE's per-channel single-segment contract we follow the `cascaded_dae` pattern: the `ChannelWiseArtifactDataset` view expands `(examples × channels)` so each item is a single-channel `(1, 512)` window. The center epoch of the context is used as the noisy input, and `artifact_center` as the target. Output type is `artifact`; FACETpy's `DeepLearningCorrection` subtracts the predicted artifact from the noisy signal at inference time.

We do **not** use the surrounding 6 context epochs — DPAE is by design a single-segment encoder, not a context model. The 7-epoch context is only the form the cached dataset happens to ship in.

## Loss in original paper

Mean squared error between predicted clean (or denoised) waveform and ground-truth clean waveform. We use MSE with `target = artifact_center`, equivalent in our setting because the network is trained as an artifact predictor.

## Non-obvious training tricks

- All activations are SeLU (not ReLU/LeakyReLU). The paper does not justify this; SeLU is self-normalising and was likely chosen so that the network behaves well without explicit batch-normalisation in some variants. We keep SeLU to stay faithful.
- Adam optimiser, batch size 128, 200 epochs in the published experiments. Learning rate 1e-3 from the ablation study.
- No explicit dropout reported for the CNN variant.

## Hardware feasibility note

- Approximate parameter count: ~2.0 M (matches the paper's 1D-CNN variant).
- Memory: a single `(B=128, 1, 512)` minibatch is ~256 KB activation per layer pre-fusion. With ~10 conv stages and ~256 channels at the bottleneck, peak forward+backward activation memory is well under 1 GB on an RTX 5090 (24 GB VRAM).
- A single epoch on the Niazy proof-fit dataset (`n_examples` ~ 1300, `n_channels` ~ 32 → ~40 k channel-wise items) is on the order of a few seconds at batch=128. 50 epochs fit comfortably within minutes; 200 epochs are feasible but unnecessary for the proof-fit run.

No reduction from the published architecture is required.

## Open questions

- The paper does not tabulate the 1D-CNN layer-by-layer (kernel widths, channel counts, depth). The exact channel progression is reconstructed from the reported parameter budget (~2 M); we match the budget rather than the literal layout.
- The paper does not specify dilation in the local pathway. The orchestrator brief explicitly mentions dilation as part of the family-defining trait, so we adopt dilation = (1, 2, 4, 8) over four small-kernel conv layers in the local pathway.
- Native artifact epoch lengths in the Niazy data vary (median ~ several thousand samples at 5 kHz). The dataset builder already resamples each native epoch to 512 samples, so the network sees a fixed shape. The processor must mirror this resampling at inference time. We reuse the same `_resample_1d` pattern used by `cascaded_context_dae`.
