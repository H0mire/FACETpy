# SepFormer Research Notes

Model id: `sepformer`
Model name: `SepFormer`
Architecture family: Audio-Inspired Source Separation
Report section: 7.1.3 (SepFormer: Separation Transformer)

## Source Paper

- Subakan, C., Ravanelli, M., Cornell, S., Bronzi, M., & Zhong, J. (2021).
  "Attention is All You Need in Speech Separation."
  ICASSP 2021. arXiv:2010.13154.
  https://arxiv.org/abs/2010.13154
- Reference implementation: SpeechBrain
  `speechbrain.lobes.models.dual_path.SepformerWrapper` and
  `Dual_Path_Model` / `SBTransformerBlock`.
  https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.dual_path.html

## One-paragraph plain-language description

SepFormer is an RNN-free Transformer-based time-domain source-separation
network that learns to split a single audio mixture into its component
sources. It uses a learned 1D-convolutional encoder to convert the
mixture waveform into a dense latent representation, then splits that
representation into fixed-size overlapping chunks. A stack of dual-path
Transformer blocks alternates **intra-chunk** self-attention (modelling
short-term, local structure inside each chunk) with **inter-chunk**
self-attention (modelling long-range dependencies between chunks). The
network predicts a soft mask per source which is multiplied with the
encoded mixture and decoded back to the time domain via a 1D
transposed-convolution. The dual-path structure breaks the quadratic
attention cost of full-sequence Transformers while still preserving long
context.

## Mapping the architecture to fMRI gradient artifact removal

The intra/inter split maps cleanly onto the slice-acquisition periodicity
of fMRI gradient artifacts: each TR contains repeated slice acquisitions
with strong intra-TR structure, and gradient amplitude drifts slowly
across consecutive TRs. The Niazy proof-fit dataset already provides
seven trigger-locked artifact epochs per example (`noisy_context` with
shape `(7, 1, 512)` after channel-wise expansion). Each of the seven
epochs is a natural "chunk" of length 512 in the SepFormer sense, so the
chunking step in the original SepFormer mask-net collapses into the
provided context structure:

- intra-chunk attention models morphology **within one slice/TR**,
- inter-chunk attention models drift / coupling **across the seven
  consecutive TRs**.

This is the explicit lesson from DHCT-GAN that the orchestrator flagged:
do not waste SepFormer on the center epoch alone. SepFormer's
inductive bias is "long sequence with periodic structure", which is what
the 7 × 512 input provides.

## Key architectural components

| Component | Role |
|---|---|
| `Encoder` (1D conv) | maps raw `(B, 1, 7·512)` waveform to `(B, C, T_enc)` features (in original paper: 256 filters, kernel 16, stride 8 → 50 % overlap). |
| `Chunking` | folds the feature sequence into chunks of length `K` with hop `K/2`. Here we choose `K = 64` features per chunk so that the seven feature-domain chunks align with the seven trigger-locked epochs. |
| `Dual-Path Block` (×N) | each block = stack of intra-chunk SBTransformer layers + stack of inter-chunk SBTransformer layers. Each SBTransformer layer = `MHSA -> LayerNorm -> FFN -> LayerNorm` (`d_model`, `d_ffn`, `n_heads`, dropout, sinusoidal positional encoding). |
| `Skip around intra` | a residual connection that keeps the encoder representation accessible to inter-chunk attention. |
| `Mask network head` | PReLU + 1×1 conv producing a mask in feature space; multiplied with the encoded mixture. |
| `Overlap-Add (decoder)` | folds chunks back into a contiguous feature sequence. |
| `Decoder` (1D conv-transpose) | maps masked features back to the time-domain `(B, 1, 7·512)`. |
| `Centre slice` | the centre epoch of the reconstructed waveform is returned as the artifact prediction `(B, 1, 512)`. |

This is a single-channel (per-EEG-channel) model, identical in
granularity to `cascaded_dae` and `cascaded_context_dae` so it remains
compatible with any EEG channel count at inference time.

## Inputs

| Aspect | Original SepFormer (speech) | This project (Niazy proof-fit) |
|---|---|---|
| Domain | Single-channel waveform mixture | Single EEG channel, gradient-artifact-contaminated |
| Sampling rate | 8 kHz (WSJ0-2mix) | 4096 Hz |
| Input length | ~4 s (32k samples) | 7 trigger epochs × 512 samples = 3584 samples (~0.87 s @ 4096 Hz) |
| Sources | 2 speakers | 1 "source" — artifact center epoch (single-output regression) |
| Number of channels | 1 | 1 (channel-wise) |

The Niazy bundle (`output/niazy_proof_fit_context_512/`) exposes
`noisy_context` shape `(833, 7, 30, 512)` and `artifact_center` shape
`(833, 30, 512)` at 4096 Hz. The channel-wise dataset wrapper expands
this to ~25 000 channel examples of shape `(7, 1, 512) → (1, 512)`.

## Loss function

The original SepFormer trains on **scale-invariant SI-SNR** loss with
permutation-invariant training (PIT) over the speaker permutations. We
do not have a permutation problem (one target only), so PIT is not
used. We default to a **negative SI-SNR loss** combined with a small
**MSE term** as a stability prior — pure SI-SNR is amplitude-scale
invariant which can let the network pick an arbitrary gain. The
training YAML exposes a `loss.kind` switch (`si_snr`, `si_snr_mse`,
`mse`) so we can fall back to MSE if SI-SNR is unstable on the proof-fit
dataset.

## Non-obvious training tricks (from the paper / SpeechBrain)

- LayerNorm **before** the attention/FFN sub-layer ("pre-norm") rather
  than the original post-norm Transformer scheme. Reproduced.
- Sinusoidal positional encodings added once at the input of every intra
  and inter block (not just the first one). Reproduced.
- `skip_around_intra = True`: residual connection that lets inter-chunk
  attention see the unprocessed encoder features.
- Heavy data augmentation in the original paper (speed perturbation,
  random gain). We disable these for the proof-fit run; the agent
  prompt is explicit that the question is "can the model overfit Niazy
  AAS artifact morphology" first.
- Half-precision training: original paper uses fp32; we keep fp32 too on
  the RTX 5090 since the compact config is well under VRAM.
- Gradient clipping `‖g‖₂ ≤ 5`. We use 1.0 to match the FACETpy
  trainer's default and existing models in this repo.

## Hardware feasibility

Original SepFormer (8 + 8 intra/inter layers per block, N = 2 blocks,
d_model = 256, d_ffn = 1024, 8 heads, encoder 256@kernel 16) lands
around **25.7 M parameters**. That fits a single RTX 5090 24 GB even at
batch 1 with the original 32k-sample input, but it is overkill for the
833-example proof-fit dataset and would over-fit immediately.

I reduce the model to a **compact SepFormer** appropriate for the
dataset size:

| Hyperparameter | Original | Compact (this work) |
|---|---:|---:|
| Encoder filters `C` | 256 | 128 |
| Encoder kernel | 16 | 16 |
| Encoder stride | 8 | 8 |
| Chunk size `K` (feature steps) | 250 | 64 |
| Chunk hop | 125 (50 %) | 32 (50 %) |
| Dual-path blocks `N` | 2 | 2 |
| Intra layers per block | 8 | 4 |
| Inter layers per block | 8 | 4 |
| `d_model` | 256 | 128 |
| `d_ffn` | 1024 | 256 |
| `n_heads` (intra & inter) | 8 | 4 |
| Dropout | 0.1 | 0.1 |
| Number of sources | 2 | 1 |

Rough parameter count for the compact config:

- one SBTransformer layer ≈ 4 · d_model² (QKV+proj) + 2 · d_model · d_ffn
  ≈ 4·128² + 2·128·256 ≈ 131 k
- 8 layers per dual-path block × 2 blocks = 16 transformer layers
  ≈ 2.1 M
- encoder + mask head + decoder ≈ 200 k
- **Total ≈ 2.3 M parameters.**

Expected memory at batch 64, fp32, input 3584 samples per channel-wise
example:

- activations per layer dominated by `(B, C, T_enc) ≈ (64, 128, 447)`
  ≈ 14 MB per layer × ~40 layers (incl. residual & FFN) ≈ 0.6 GB
- model + gradients + AdamW states ≈ 3 · 2.3 M · 4 B ≈ 30 MB
- forward/backward peak ≈ ~1–2 GB

Comfortable on 24 GB. Batch can be raised to 128 or 256 if needed.

Single-epoch wall-clock estimate on RTX 5090 (TensorCore fp32, ~80 TFLOPS
effective):

- ~25 000 channel examples / batch 64 ≈ 390 steps per epoch
- ~2.3 M params × ~3 forward+backward passes per step ≈ ~14 GFLOPS/step
  forward, ~40 GFLOPS/step with backward — utterly negligible
- attention FLOPs dominate: 16 layers × (B · n_heads · L²) where
  L ≈ 64 inside chunks and ≈ 14 across chunks. Still well under one
  second per step.
- **Expected wall-clock per epoch: ≈ 20–40 seconds.**
- Full run of `max_epochs = 50` with early stopping: ≤ 30 minutes.
- Smoke (`max_epochs = 1`, `max_examples = 256`): a few seconds of
  actual training plus model construction + export.

This easily fits the "smoke run must finish quickly" requirement.

## Open questions

1. **Mask vs. direct regression.** Original SepFormer multiplies a
   predicted mask with the encoded mixture; that is well motivated for
   strict source separation. For artifact prediction it might be more
   robust to predict the artifact representation **additively** and
   skip the mask. The first run uses the canonical multiplicative mask
   with ReLU activation to stay close to the paper.
2. **Center-only vs. full reconstruction loss.** We compute SI-SNR /
   MSE on the **centre epoch only** because the dataset only ships
   `artifact_center` as ground truth. The network still sees the full
   seven-epoch context. We could later train against
   `artifact_context` to give the network supervised signal on
   neighbours too — left as a follow-up.
3. **Positional encoding length.** With only 7 chunks, learned
   positional embeddings for the inter-chunk stream might be cheaper
   and more expressive than sinusoidal ones. The first run keeps
   sinusoidal to match the paper.
4. **Channel-wise vs. multichannel.** A future variant could pass all
   30 EEG channels jointly so the inter-chunk attention can exploit
   cross-channel co-occurrence of artifact. Left as a follow-up.

## Inductive bias summary

Compared to the existing baselines and the catalog:

- `cascaded_dae` / `cascaded_context_dae` are pure FC autoencoders with
  no notion of time order beyond a flat linear layer. They model
  morphology but cannot model temporal periodicity.
- `conv_tasnet` (TCN, same family) has long temporal context via
  stacked dilations but no explicit chunking.
- `demucs` (U-Net + LSTM) builds multi-scale features but is heavier.
- SepFormer's dual-path attention is the only model in the
  audio-source-separation family that mechanically reflects the
  intra-TR vs. inter-TR structure of the gradient artifact. That is
  the architectural argument for trying it on this dataset.
