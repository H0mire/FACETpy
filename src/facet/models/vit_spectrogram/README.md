# ViT Spectrogram Inpainter (`vit_spectrogram`)

Channel-wise Vision Transformer that treats fMRI gradient-artifact removal as
**spectrogram inpainting**. For every trigger-defined center epoch, the model
ingests a 7-epoch context, takes a per-channel STFT, masks the center-epoch
time region, and reconstructs the clean center-epoch magnitude spectrogram
through a small ViT. An inverse STFT using the input's noisy phase produces
the cleaned center waveform; the corresponding artifact estimate is the
difference between the noisy and reconstructed center signals.

This model belongs to the **Vision-Inspired (Spectrogram Inpainting)** family
described in `docs/research/dl_eeg_gradient_artifacts.pdf`, Section 7.2.1
("Vision Transformers (ViT) for Spectrograms").

## Scope

- Input shape: `(batch, 7, 1, samples)` per channel.
- Output shape: `(batch, 1, samples)` reconstructed clean center epoch.
- Current training default: `samples = 512`.
- Inference granularity: per channel, so the checkpoint stays
  channel-count-agnostic at deployment time.
- Inference requires trigger metadata in the `ProcessingContext`.

## Architecture overview

1. Concatenate the 7 context epochs into a single per-channel signal of
   length `7 × samples`.
2. Compute `torch.stft` (Hann window, `n_fft = 64`, `hop_length = 16`,
   `center = True`). Crop to a clean patch grid of `(freq_bins, time_frames)
   = (32, 224)`.
3. Patchify the log-magnitude into `(patch_freq, patch_time) = (4, 16)`
   patches, linearly embed each patch into `embed_dim = 192`, and add a
   factorized 2D learnable positional embedding.
4. Replace tokens whose time index overlaps the center-epoch region with a
   learnable mask token (structural mask aligned with the GA's location).
5. Six pre-norm transformer encoder blocks (multi-head self-attention + MLP,
   GELU, `mlp_ratio = 4`).
6. Linear decoder head projects each output token back to its
   `(patch_freq × patch_time)` log-magnitude patch, and patches are unfolded
   into a full predicted log-magnitude spectrogram.
7. Reconstruct the complex spectrogram with the model's predicted magnitude
   and the **input's original (noisy) phase**, pad back to the full STFT
   shape, run `torch.istft`, and return the center-epoch slice.

Total parameters at default settings: **~2.7 M**. The model fits comfortably
in 24 GB RTX 5090 VRAM with the recommended batch size of 64.

See `documentation/research_notes.md` for the design rationale, the explicit
phase-handling decision, and the hardware-feasibility estimate.

## Training

The fleet workflow prepares the Niazy proof-fit context dataset remotely via
the job's `--prepare-command`. Then:

```bash
uv run facet-train fit --config src/facet/models/vit_spectrogram/training_niazy_proof_fit.yaml
```

A smoke variant — single epoch, smaller model, capped at 512 examples — is
provided at `training_niazy_proof_fit_smoke.yaml` for fleet verification.

Both configs export a TorchScript checkpoint at
`<run_dir>/exports/vit_spectrogram.ts`.

## Inference

```python
from facet.models.vit_spectrogram import ViTSpectrogramInpainterCorrection

context = context | ViTSpectrogramInpainterCorrection(
    checkpoint_path="training_output/<run>/exports/vit_spectrogram.ts",
    context_epochs=7,
    epoch_samples=512,
)
```

The processor builds 7-epoch contexts from the context's triggers, resamples
each native trigger-to-trigger epoch to 512 samples for the model's input
contract, predicts the clean center epoch, resamples back to the native
epoch length, and emits the artifact estimate
`artifact = noisy - predicted_clean`. The standard `DeepLearningCorrection`
machinery subtracts that estimate.

## Architectural decisions

- **Per-channel, channel-count-agnostic checkpoint** — matches
  `cascaded_context_dae`. Each channel is processed independently at
  inference; the deployed weights do not encode a fixed channel count.
- **Magnitude inpainting with preserved noisy phase** — the existing
  `SpectrogramMixin` and the report both follow this paradigm. Phase
  recovery from magnitude alone is acknowledged as lossy; see
  `documentation/research_notes.md` for the chosen trade-off and the
  follow-up experiment that would address it.
- **Structural (not random) mask** — center-epoch time patches are masked
  with a one-patch margin on each side, because we know where the GA is
  concentrated (around the trigger). This deviates from MAE's
  high-ratio random masking but is appropriate for the supervised
  artifact-removal task.
- **Hand-rolled self-attention** — `torch.nn.MultiheadAttention`'s
  fast/slow path dispatcher produces unstable `torch.jit.trace` graphs;
  an explicit Q/K/V projection traces cleanly and exports to TorchScript
  without manual `check_trace=False`.
