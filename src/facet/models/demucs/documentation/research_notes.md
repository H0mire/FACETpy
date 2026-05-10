# Demucs — Research Notes For FACETpy Adaptation

## Source Papers

- **Primary**: Défossez, A., Usunier, N., Bottou, L., & Bach, F. (2019).
  *Music Source Separation in the Waveform Domain.* arXiv:1911.13254.
  <https://arxiv.org/abs/1911.13254>
- **Follow-up (not adopted here)**: Rouard, S., Massa, F., & Défossez, A. (2022).
  *Hybrid Transformers for Music Source Separation* (Hybrid Demucs v4 / HT-Demucs).
  arXiv:2211.08553. <https://arxiv.org/abs/2211.08553>
- **Reference implementation**: <https://github.com/facebookresearch/demucs>
  (original v1/v2 in the `master` branch and earlier `v2` tag; v3/v4 introduce
  hybrid time/spectrogram and transformer additions).

This implementation targets the **original time-domain Demucs (v1)** because:

1. The waveform-domain U-Net + bidirectional LSTM is the clearest contrast to
   `conv_tasnet` (the sibling audio model already in the project — TCN with
   stacked dilated convolutions) and to `cascaded_context_dae` (a fully-connected
   context DAE). Sharing the same Niazy proof-fit dataset makes the comparison
   apples-to-apples.
2. Hybrid Demucs adds a spectrogram branch and a cross-domain transformer, which
   would muddy the audio-source-separation comparison the report calls for in
   Section 7.1.2.
3. v1 fits comfortably in one RTX 5090; v4 does not without significant
   downscaling.

## One-Paragraph Description

Demucs is a 1D U-Net for end-to-end waveform-to-waveform source separation. A
stack of strided 1D convolutions encodes the input waveform into a compressed
deep representation; a 2-layer bidirectional LSTM models long-range temporal
dependencies at the bottleneck; a symmetric stack of transposed convolutions
decodes back to the waveform domain. U-Net-style skip connections sum encoder
outputs into the matching decoder layer, preserving high-frequency detail that
would otherwise be lost through the strided downsampling. The final linear layer
emits `S` separated waveforms (one per source). In the original paper Demucs is
trained with an L1 reconstruction loss across all sources. We reuse this
inductive bias to separate gradient-artifact (rhythmic, harmonic, periodic at
multiples of the slice repetition) from EEG (stochastic, broadband, mostly
non-stationary), treating the artifact as the "drum track" of the recording.

## Key Architectural Components

| Component | Role |
|---|---|
| Strided 1D encoder (L blocks) | Compresses the waveform; each block downsamples by stride 4 with kernel 8, doubling channels. GLU after a 1×1 conv injects nonlinearity. |
| Bidirectional LSTM (2 layers) | Models long-range temporal structure at the bottleneck; output projected back to channel count by a linear layer. |
| Transposed-conv decoder (L blocks) | Symmetric upsampling; each block has a kernel-3 conv with GLU, then a kernel-8 stride-4 ConvTranspose1d back to half the channels. |
| U-Net skip connections | Encoder output `e_i` is summed into the decoder input at the matching level. Restores high-frequency content. |
| Weight rescaling at initialization | Custom rescaling so the output at init has the same magnitude across layers; the paper reports a ~20× difference would otherwise appear. |
| Final 1×1 linear layer | Maps the decoder output to `S × n_input_channels` sources. |

Original block shapes (paper, MusDB):

- `L = 6`, kernel `K = 8`, stride `S = 4`.
- Input channels `C_0 = 2` (stereo), `C_1 = 64`, `C_i = 2·C_{i-1}` → `C_L = 2048`.
- Encoder block: `Conv1d(C_{i-1}, C_i, K=8, S=4)` → ReLU → `Conv1d(C_i, 2C_i, K=1, S=1)` → GLU (output `C_i` channels).
- Decoder block: `Conv1d(C_i, 2C_i, K=3, S=1)` → GLU → `ConvTranspose1d(C_i, C_{i-1}, K=8, S=4)` → ReLU.
- Bottleneck: `BiLSTM(input=C_L, hidden=C_L, num_layers=2, bidirectional=True)` → `Linear(2·C_L, C_L)`.
- Final layer: `Linear(C_0, S·C_0)` per time step.
- Loss: L1 over the predicted waveforms vs ground-truth sources.

## Inputs The Original Paper Expects

- **Sample rate**: 44.1 kHz audio (stereo MusDB).
- **Segment length**: 11-second extracts (≈ 485k samples at 44.1 kHz) with random shift augmentation.
- **Channels**: 2 (stereo mixture in, 4 sources × 2 channels out).

## Mapping To The Niazy Proof-Fit Dataset

The shared NPZ bundle produced by
`examples/build_niazy_proof_fit_context_dataset.py` provides:

| Key | Shape | Meaning |
|---|---|---|
| `noisy_context` | `(N, 7, n_ch, 512)` | 7-epoch context windows of the gradient-corrupted EEG. |
| `artifact_context` | `(N, 7, n_ch, 512)` | 7-epoch AAS-estimated artifact (supervised target). |
| `clean_context` | `(N, 7, n_ch, 512)` | 7-epoch AAS-corrected EEG (the residual after artifact removal). |
| `noisy_center` / `artifact_center` / `clean_center` | `(N, n_ch, 512)` | Center-epoch slices of the above. |
| `sfreq` | `(1,)` | Sampling frequency (Hz). |
| `ch_names` | `(n_ch,)` | Channel labels. |

Adaptation choices (justified by the project-lesson note in the orchestrator
prompt and Demucs' inductive bias):

1. **Use the 7-epoch context, not just the center epoch.** This matches the
   project lesson from DHCT-GAN and gives Demucs the long receptive field it
   was designed for. Per-channel input is a single 1D signal of length
   `T = 7 · 512 = 3584` samples.
2. **Per-channel inference**, like `cascaded_dae` and `cascaded_context_dae`,
   so the checkpoint stays decoupled from the channel count of any target
   dataset.
3. **Predict the artifact across all 7 epochs** (`artifact_context`), not only
   the center, because (a) it matches the symmetric Demucs U-Net naturally
   (output length = input length) and (b) supervising on more samples uses the
   available labels efficiently. At inference, the adapter slices the center
   epoch of the predicted artifact for the subtract-from-context-center
   correction pattern.
4. **Single "source" output**: `S = 1`. We do not split clean + artifact at the
   output; we predict the artifact only and let `DeepLearningCorrection`
   subtract it. The alternative — predicting both clean and artifact with a
   consistency loss — is closer to D4PM territory and out of scope here.

Resulting tensor shapes for one batch element during training:

- Input: `(1, T)` = `(1, 3584)` (1 channel, 7-epoch concatenation).
- Target: `(1, T)` = `(1, 3584)` (artifact across all 7 epochs).
- Loss: L1 over the full 3584 samples (matches the paper). A center-weighted
  variant is documented in the model card but not used for the proof fit.

## Loss Function

Plain `torch.nn.L1Loss` over the full predicted artifact waveform — the same
mean-absolute-error reconstruction loss the original paper uses (the paper
explicitly compares L1 and L2 and selects L1 as default; Section 4.2). A
`build_loss` factory accepts `name="l1"` (default), `name="mse"`,
`name="smooth_l1"`, or `name="huber"` for ablations.

## Non-Obvious Training Tricks From The Paper

- **Weight rescaling at init**: each layer's weights are rescaled at init so
  that the output of every layer has the same standard deviation as its input.
  Without this, the output of the last layer would be roughly 20× smaller than
  the first. The paper does this without batch normalization. Implemented here
  as a post-init rescale on every `Conv1d` / `ConvTranspose1d`.
- **GLU after the 1×1 convolutions** inside both encoder and decoder blocks;
  the ablation in §6.2 of the paper shows GLU gives a measurable boost over
  plain ReLU there.
- **Linear projection after the BiLSTM**: the LSTM outputs `2·C_L` channels per
  time step (forward + backward); a linear layer reduces this back to `C_L`
  before the decoder. Without it, the channel count would not match the skip
  connections.
- **No batch norm.** The paper trains without batch normalization; the
  rescale-at-init trick is the substitute. We follow this.

## Hardware Feasibility On RTX 5090 (24 GB VRAM)

Original v1 with `C_1 = 64`, `L = 6` → final channels `2048` → ~30M-40M
parameters. The expensive layers are LSTM(2048, 2048) (per direction →
~33M params for the LSTM alone) and the deepest conv block (Conv1d(1024, 2048)
with kernel 8 → ~16M params). Memory at training time is dominated by
activations from the long input waveform.

Our adapted setting is much smaller:

- **L = 4** blocks (because input length is `3584`; `L = 6` with stride 4
  would collapse the time dimension below 1 sample).
- **C_1 = 64** → channel progression `64, 128, 256, 512`, `C_L = 512`.
- BiLSTM(input=512, hidden=512, 2 layers, bidirectional) ≈ 4.2M params.
- Encoder/decoder convolutions ≈ 4–5M params total.
- Final linear `(512 → 1)` per timestep, plus standard biases.
- **Total ~10M parameters.** Well within 24 GB at batch size 64 for 3584-sample
  waveforms.

Length flow under `L = 4`, kernel `8`, stride `4`, padding `2`:

```
encoder:   3584 → 896 → 224 → 56 → 14
decoder:    14 →  56 → 224 → 896 → 3584
```

(Convolution math: `out = floor((in + 2·pad - kernel) / stride) + 1`. With
`pad = 2`: `(3584 + 4 - 8)/4 + 1 = 896`, and so on. ConvTranspose with
`pad = 2, output_pad = 0`: `out = (in − 1)·stride − 2·pad + kernel = in·4`.)

Per the original paper one epoch on 8 V100s over MusDB takes ≈ 5 minutes. Our
dataset is 2–3 orders of magnitude smaller (single Niazy recording → a few
hundred N × 30 channels of 3584-sample examples). A full epoch on one RTX 5090
should be on the order of 5–15 seconds. 50 epochs ≈ 10 minutes wall-clock.

If the smoke run shows memory pressure we will halve `C_1` to 48 before
touching the depth or batch size.

## Open Questions

1. **Center-weighted vs uniform L1 loss.** The reported center-epoch artifact is
   what the correction pipeline actually uses; supervising uniformly across 7
   epochs is simpler but may waste capacity on the boundaries. The proof-fit
   run uses uniform L1; a center-weighted variant is documented in the model
   card as a follow-up.
2. **Single-source vs dual-output.** Predicting both clean and artifact with a
   consistency term (`noisy ≈ clean + artifact`) could help, but the original
   Demucs predicts independent sources without such a constraint. Skipping for
   the proof fit to keep the comparison clean.
3. **Per-channel vs cross-channel.** The 30 EEG channels are correlated; a
   variant where the input is `(30, 3584)` rather than `(1, 3584)` would
   exploit that, at the cost of locking the checkpoint to a channel count. The
   project's stated preference (`cascaded_*_dae`) is channel-wise; we keep that
   for now.
4. **LSTM vs no-LSTM ablation.** The paper shows the BiLSTM is critical for
   music. Whether it helps on the very periodic gradient artifact is open; the
   answer is part of why this model is worth running.
