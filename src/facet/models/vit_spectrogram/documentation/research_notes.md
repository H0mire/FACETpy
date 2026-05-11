# Research Notes — ViT Spectrogram Inpainter (`vit_spectrogram`)

## Source papers

1. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
   Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S.,
   Uszkoreit, J., & Houlsby, N. (2020).** *An Image Is Worth 16x16 Words:
   Transformers for Image Recognition at Scale.* arXiv:2010.11929.
   <https://arxiv.org/abs/2010.11929>
2. **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021).**
   *Masked Autoencoders Are Scalable Vision Learners.* arXiv:2111.06377.
   <https://arxiv.org/abs/2111.06377>
3. **Supporting / motivating section in our own report:**
   `docs/research/dl_eeg_gradient_artifacts.pdf`, Section 7.2.1
   "Vision Transformers (ViT) for Spectrograms".

## One-paragraph plain-language description

The fMRI gradient artifact (GA) leaves a characteristic image-like footprint
in the time-frequency domain: broadband vertical stripes locked to each
slice-acquisition trigger, plus a faint horizontal harmonic comb at the
slice frequency. We exploit that by reframing artifact removal as
**spectrogram inpainting**. Each per-channel EEG snippet from the 7-epoch
Niazy proof-fit context is converted to a magnitude spectrogram via STFT.
A Vision Transformer (ViT) splits this 2D "image" into non-overlapping
time-frequency patches, linearly projects each patch, adds learnable
positional embeddings, and processes the patch sequence through a stack
of self-attention blocks. The patches whose time index falls inside the
center-epoch window (where the GA is concentrated) are "masked" — in the
MAE sense, replaced with a learned mask token before being fed to the
encoder — and the decoder is trained to reconstruct the clean magnitude
spectrogram at those positions, conditioned on the surrounding clean
context patches. At inference, the predicted clean center-epoch magnitude
is combined with the original noisy phase via inverse STFT to obtain a
clean center-epoch time signal; the artifact estimate is the difference
between the original noisy center epoch and that reconstruction.

## Architectural components

| Component | Responsibility |
|---|---|
| **STFT front-end** | Convert per-channel EEG snippet to log-magnitude + phase spectrograms. Reuses `facet.correction.deep_learning.SpectrogramMixin` helpers (`stft_forward` / `istft_backward`) at inference time. During training, the loss is computed in the log-magnitude space directly, so the STFT helper lives inside `training.py` as a small `torch.stft`-based wrapper to keep autograd intact. |
| **Patch embedder** | `Conv2d` with kernel = stride = patch size projects each `(F_p, T_p)` patch to a `d_model`-dim token. Equivalent to ViT's linear patch projection. |
| **Positional embedding** | Learnable 2D positional embedding (separate freq and time axes, summed), as in ViT. CLS token omitted — we are doing dense regression, not classification. |
| **Transformer encoder** | Stack of pre-norm self-attention blocks (multi-head attention + MLP, GELU, residuals). Sees the unmasked patches and the mask tokens. We do NOT use MAE's asymmetric encoder-only-on-visible trick because (a) at our scale memory is not the binding constraint, and (b) the trick complicates the YAML-driven training factory pattern. We keep the MAE *objective* (predict clean magnitude at masked positions) but use a *symmetric* encoder that sees mask tokens directly. |
| **Decoder head** | A small `Linear` projection that maps each output token back into a `(F_p, T_p)` magnitude patch. Patches are unfolded into the full spectrogram. |
| **Mask scheduler** | Deterministic mask covering the center-epoch time window. For each training example we know which time frames overlap the center epoch (epoch index = `context_epochs // 2`), so the mask is **not random** — it is structurally aligned with where the GA actually is. |
| **iSTFT** | Reconstructs time-domain signal from the predicted center-epoch clean magnitude and the input's noisy phase. Lossy step — see "Phase handling" below. |

## Inputs and dataset mapping

Original ViT expects images of e.g. 224×224×3. Our spectrograms are much
smaller and single-channel.

We use the existing **Niazy proof-fit context dataset** built by
`examples/build_niazy_proof_fit_context_dataset.py`. Relevant arrays:

- `noisy_context`: `(N, 7, n_channels, 512)` — input EEG with GA
- `clean_context`: `(N, 7, n_channels, 512)` — AAS-corrected EEG surrogate
  (target for the inpainting loss)
- `artifact_context`: `(N, 7, n_channels, 512)` — AAS-estimated GA
- `sfreq`: 250 Hz typical
- `ch_names`: array of channel names

### How we feed it to ViT

Following the **channel-wise** pattern of `cascaded_context_dae` (so the
checkpoint stays channel-count-agnostic) we expand the dataset to
per-channel examples. Each training example is:

- Input: per-channel, **concatenated 7-epoch noisy signal**, length
  `7 × 512 = 3584` samples.
- Target: per-channel **concatenated 7-epoch clean signal**, length 3584.
- Mask: ones only over the center epoch time region (samples
  `3 × 512 : 4 × 512`, i.e. center-epoch indices) → drives where the
  reconstruction loss is computed.

STFT params (chosen for a clean ViT-friendly patch grid):

- `nperseg = 64`
- `noverlap = 48` → hop = 16
- → `n_freqs = 33`, `n_frames ≈ 224` for 3584 input samples
- We crop the freq axis to **32** and pad/crop time to **224** → final
  spectrogram shape `(32, 224)`, divisible by patch size `(4, 16)` →
  `(8, 14) = 112` patches per channel.

This gives a sequence length of 112 patches, well within attention's
quadratic budget on an RTX 5090.

### Per-channel inference shape contract

At inference, the per-channel adapter constructs the same 3584-sample
context (resampling native trigger-to-trigger epochs to 512 samples each,
mirroring `cascaded_context_dae`'s strategy), runs the model, takes the
center-epoch slice of the output (samples `1536:2048`), resamples back to
the native epoch length, and writes that as the estimated artifact for
the trigger interval. So the published checkpoint is independent of the
specific channel count of the target recording.

## Loss function

The original MAE loss is **per-pixel MSE on masked patches only**. We
inherit that. Concretely:

- `L = MSE(pred_log_magnitude[mask], target_log_magnitude[mask])`
- `mask` selects time frames overlapping the center epoch and **all**
  frequency bins (we want to recover full broadband content where the GA
  killed it).
- We compute the loss in **log-magnitude** space (`log1p` applied to
  magnitudes) because EEG spectrograms span several orders of magnitude
  and a linear MSE would be dominated by low-frequency drift.

This is consistent with `facet.training.losses` style — a small wrapper
class living in `training.py` exposes a `.loss(pred, target, mask)` API.

We optionally add a tiny waveform-MSE term on the iSTFT-reconstructed
center epoch (weight 0.1) so the optimizer also sees the time-domain
fidelity. This is gated by a config flag and off by default in the smoke
run for speed.

## Non-obvious training tricks (from the original papers, applied here)

- **High mask ratio is not required for us.** MAE found 75% random
  masking is a good self-supervised objective. Our masking is structural
  (only center-epoch patches), so the ratio is fixed by the data layout
  (≈ 1/7 of patches). The encoder still sees most of the spectrogram.
- **Learnable mask token + 2D positional embedding** as in ViT/MAE.
- **Pre-norm transformer blocks**, GELU activations, layer dropout 0.0
  (small dataset → little risk of overfit on attention layers, dropout
  hurts).
- **AdamW** with `lr=3e-4`, `weight_decay=0.05`, cosine schedule with
  10-epoch warmup. (MAE uses lr blow-up + warmup; we keep similar.)
- **Gradient clip** at 1.0 (matches the project's existing
  `cascaded_context_dae` config).
- **No pre-training on natural images.** The report mentions transfer
  from large pretrained ViTs is a benefit, but loading 224×224×3 ImageNet
  weights into a 1-channel 32×224 spectrogram encoder would require
  non-trivial weight surgery (patch-embedding resize, positional-embed
  interpolation, channel-conv averaging). The proof-fit dataset is too
  small for that to add scientific value at this milestone — we train
  from scratch and document the choice.

## Hardware feasibility (RTX 5090, 24 GB VRAM)

Model size (proposed):

- Patch size `(4, 16)`, embedding dim 192, depth 6, heads 6, MLP ratio 4,
  decoder = single `Linear(192 → 64)` projecting back to patch pixels.
- Parameters: ~1.7 M (encoder) + 12 k (decoder) ≈ 1.7 M total.
- Activations per batch of 64 examples × 112 patches × 192 dim ≈
  64 × 112 × 192 × 4 B ≈ 5.5 MB per layer; attention matrices ≈
  64 × 6 × 112² × 4 B ≈ 18 MB per layer; 6 layers → ~140 MB activations.
- Forward+backward with optimizer state (AdamW = 2x params for moments):
  well under 1 GB.

Expected single-epoch wall-clock on RTX 5090:

- Niazy proof-fit gives roughly `(N_examples × n_channels)` training
  pairs. For a typical Niazy run that is ≈ `(few hundred) × 31` ≈ 10k
  pairs. At batch 64 → ~160 steps/epoch × ~2 ms/step (small model) →
  ~0.3 s/epoch GPU time. We expect 50 epochs to finish in well under a
  minute of pure compute, dominated by dataloader setup and validation.

Conclusion: **no reduction relative to the published ViT papers is
needed; if anything we are training a much smaller model than ViT-B.**
We document this because the report flagged that ViT can be memory-heavy
— at our chosen patch grid and spectrogram size we are not memory-bound.

## Phase-handling decision

**We use log-magnitude + preserved-noisy-phase reconstruction** (Option A
from the orchestrator's brief). Reasoning:

1. The repository already provides this pattern via
   `SpectrogramMixin.istft_backward(magnitude, phase, ...)`, which
   inverts the STFT using the *input's* original phase. Reusing it keeps
   our adapter compact and consistent with the existing infrastructure.
2. The MAE objective is naturally framed as predicting masked content,
   not phase residuals. Magnitude-only output preserves the inpainting
   metaphor.
3. Lower output dimensionality (one real channel vs. two for complex)
   halves the decoder's regression target, which matters on the small
   Niazy proof-fit dataset.
4. The lossy nature of phase recovery from magnitude is **explicitly
   acknowledged in the report** as a known limitation; we record it here
   and in the model card so downstream readers don't mistake it for a
   bug.

**Caveat we are taking on:** at the GA-dominated time frames, the noisy
phase is not the clean phase. Reconstructing with the noisy phase
therefore retains some of the artifact's temporal sign structure even
after the magnitude is fully cleaned, which limits the maximum
achievable SNR improvement. If results are weak, the most promising next
experiment (and the one we explicitly flag in HANDOFF) would be to
re-train with a **two-channel real/imag output** so the model can also
correct phase. We do not do this in v1.

## Open questions

- **Edge effects of the structural mask.** The center-epoch time
  boundaries do not align exactly with STFT frame boundaries. We use a
  generous overlap (one extra patch on each side of the center-epoch
  time window is also included in the mask) but this is a deliberate
  bias-vs-variance tradeoff that may not be optimal. Open.
- **Per-channel vs. multichannel formulation.** Treating each channel
  independently makes the checkpoint channel-count-agnostic (matches
  `cascaded_context_dae`) but discards spatial information. The report
  also lists ST-GNN as the architecture that exploits topography. We
  stay per-channel.
- **Loss in linear vs. log-magnitude.** We picked log-magnitude. MAE used
  raw pixel intensity (sometimes normalized per patch). For EEG
  spectrograms the dynamic range is too wide for linear MSE to be well
  conditioned, but we have not done a controlled ablation.
- **Whether to include a small waveform-MSE term.** Off in smoke, gated
  by config in full.

These are flagged in HANDOFF as suggested next experiments if proof-fit
metrics are unsatisfactory.
