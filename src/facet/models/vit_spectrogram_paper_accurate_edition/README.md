# ViT/MAE Spectrogram Inpainter — Paper-Accurate Edition

`facet.models.vit_spectrogram_paper_accurate_edition`

A more faithful re-implementation of the two source papers behind the original
`facet.models.vit_spectrogram` package, adapted for single-/few-channel
EEG-fMRI gradient-artifact (GA) inpainting on the tiny Niazy proof-fit dataset
and runnable entirely on CPU.

## Source papers

1. **ViT** — A. Dosovitskiy *et al.*, *An Image is Worth 16x16 Words:
   Transformers for Image Recognition at Scale*, ICLR 2021, **arXiv:2010.11929**.
   Defines the pre-norm encoder block (Eqs. 2–3):
   `z' = MSA(LN(z)) + z`, then `z = MLP(LN(z')) + z'`, a two-layer GELU MLP of
   ratio 4, and a final LayerNorm (Eq. 4).

2. **MAE** — K. He *et al.*, *Masked Autoencoders Are Scalable Vision Learners*,
   CVPR 2022, **arXiv:2111.06377**. Defines the asymmetric encoder/decoder
   (Sec. 3, Table 1c), the masked-patch reconstruction loss (footnote 1,
   Table 1d), fixed 2D sin-cos position embeddings (Appendix A.1), and the
   ViT-official-code init (`xavier_uniform_` on Linear, `trunc_normal_(0.02)`
   on the mask token).

## What changed vs. the original (paper-accuracy fixes)

The original package faithfully matched the **ViT encoder block** but collapsed
the **MAE-specific machinery** into a single symmetric encoder + Linear head for
YAML-factory simplicity. This edition restores the MAE core designs:

| # | Change | Paper reference |
|---|--------|-----------------|
| 1 | **True asymmetric encoder/decoder.** The encoder now runs ONLY on the *visible* (non-masked, i.e. context-epoch) tokens; mask tokens NEVER enter the encoder. A separate lightweight decoder (`decoder_embed_dim`, `decoder_depth`, `decoder_heads`) re-inserts a single shared learnable mask token at the masked positions, adds its own position embeddings over the full token set, runs a few Transformer blocks, applies a decoder LayerNorm and a Linear head to patch pixels. | MAE Sec. 3, Table 1a/1b/1c |
| 2 | **Masked-patch magnitude reconstruction loss.** `build_loss` now returns `MaskedPatchMagnitudeLoss`, which computes per-patch MSE/L1/Huber over the MASKED center-epoch patches only, in log-magnitude space — not the original full-waveform time-domain MSE. | MAE footnote 1 |
| 3 | **Per-patch target normalisation** (subtract patch mean, divide by patch std, eps). On by default (`normalize_target=True`). | MAE Table 1d |
| 4 | **Fixed 2D sine-cosine position embeddings** as registered buffers in BOTH encoder and decoder, replacing the original factorised *learnable* 2D embeddings. Parameter-free and trace-stable. | MAE Appendix A.1 |
| 5 | **MAE/ViT reference init**: `xavier_uniform_` on all Linear weights (zero bias), `trunc_normal_(0.02)` on the mask token. The original left Linear layers at PyTorch's Kaiming-uniform default. | MAE / ViT official code |

The model now exposes **two clean forward paths**:

* **Training** (`model.train()`): returns
  `{"pred_masked_patches": (B, n_masked, patch_pixels), "mask": (n_patches,)}`,
  consumed by `MaskedPatchMagnitudeLoss`. The clean time-domain center epoch
  `y` from the dataset is turned into masked target patches *inside the loss*
  (the model never sees `y`, matching the `loss(model(x), y)` trainer contract).
* **Inference** (`model.eval()`): returns `(B, 1, epoch_samples)` — visible
  patches pass through, the decoder fills the masked patches, `expm1` magnitude
  is combined with the input's noisy phase, iSTFT, and the center-epoch slice is
  returned. This path is `torch.jit.trace`-stable (the visible/masked token
  splits come from a *static* mask buffer; the hand-rolled QKV attention avoids
  `nn.MultiheadAttention`'s dispatch nondeterminism).

## Deliberate EEG-fMRI deviations from MAE (documented, not blindly copied)

* **Structural center-epoch mask, not 75% uniform random masking.** MAE's random
  masking is for self-supervised representation learning where the masked region
  is unknown. Here the GA location is *known* (trigger-locked center epoch), so a
  deterministic structural mask is the correct supervised inductive bias. An
  optional `extra_random_mask_ratio` regulariser is left as a flagged follow-up.
* **Magnitude-only prediction with the input's noisy phase** (kept from the
  original). Halves the regression target on a tiny dataset and reuses the
  spectrogram convention; the phase limitation at GA-dominated bins is
  documented. A complex/2-channel head is a flagged follow-up, not the default.
* **From-scratch training** (no ImageNet ViT transfer): the dataset is far too
  small and a 1-channel 32×224 spectrogram is dimensionally incompatible.
* **Config-level recipe**: AdamW + `weight_decay 0.05` + (optional) cosine +
  warmup + no dropout transfer from MAE; the 4096 batch / RandomResizedCrop
  augmentation / massive-scale schedule do not fit trigger-aligned EEG epochs.
  FACETpy's `grad_clip_norm` is a stability addition not present in MAE.

See `documentation/paper_accuracy_review.md` for the full discrepancy table and
the per-technique EEG-fMRI applicability assessment.

## Inference

The processor `ViTSpectrogramMAECorrection` (registered globally-unique name
**`vit_spectrogram_paper_accurate_correction`**) and the adapter
`ViTSpectrogramMAEAdapter` mirror the original's per-channel, 7-epoch
trigger-context inference and emit `artifact = noisy_center − predicted_clean`.
The exported TorchScript graph is the model's `eval()` forward.

## Compatibility

`build_model` / `build_loss` / `build_dataset` keep facet-train-compatible
signatures (injected kwargs accepted via `**_`, never required-positional). The
masked-patch loss owns its own STFT/patch geometry, so the YAML `loss_kwargs`
must repeat the model geometry and supply `epoch_samples` — see
`training_niazy_proof_fit_smoke.yaml`.

## Files

* `training.py` — `ViTSpectrogramMAEInpainter`, `MaskedPatchMagnitudeLoss`,
  `build_model` / `build_loss` / `build_dataset`.
* `processor.py` — `ViTSpectrogramMAEAdapter` + `ViTSpectrogramMAECorrection`.
* `training_niazy_proof_fit_smoke.yaml` — illustrative tiny CPU config.
* `documentation/paper_accuracy_review.md` — discrepancy table + applicability.
