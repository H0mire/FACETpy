# Paper-Accuracy Review — ViT/MAE Spectrogram Inpainter

This review compares the original `facet.models.vit_spectrogram` package against
its two source papers and records what the paper-accurate edition
(`facet.models.vit_spectrogram_paper_accurate_edition`) changed, what it kept,
and why.

Source papers:

* **ViT** — Dosovitskiy et al., ICLR 2021, arXiv:2010.11929.
* **MAE** — He et al., CVPR 2022, arXiv:2111.06377.

## Discrepancy table

| Aspect | Paper specifies | Original impl | Severity | Resolution in this edition |
|--------|-----------------|---------------|----------|----------------------------|
| **Asymmetric encoder/decoder** (MAE core design #1) | Encoder processes ONLY visible tokens; mask tokens never enter it. A separate lightweight decoder (default 8 blocks, width 512) reconstructs masked patches (Table 1c: mask-tokens-in-encoder ≈ −14% acc, 3.3× slower). | Single SYMMETRIC encoder sees the mask token in-place; "decoder" is one Linear head. | **High** | **FIXED.** Encoder runs on visible tokens only via a static `visible_index` buffer; a real decoder Transformer stack (`decoder_embed_dim`/`decoder_depth`/`decoder_heads`) re-inserts a shared learnable mask token and reconstructs. |
| **Reconstruction loss** | Per-pixel MSE on MASKED patches only, in (optionally per-patch-normalised) target space (footnote 1). | Plain time-domain MSE/L1/SmoothL1 on the full reconstructed waveform. | **High** | **FIXED.** `MaskedPatchMagnitudeLoss` computes MSE/L1/Huber over masked patches only, in log-magnitude space. Model emits masked predicted patches in training mode; loss builds masked target patches from the clean center epoch. |
| **Per-patch target normalisation** | Normalise each target patch by its own mean/std before MSE (Table 1d). | None (log1p only). | Medium | **FIXED.** `normalize_target=True` by default; per-patch (mean, std, eps) on the target. |
| **Decoder depth/width** | Narrower/shallower decoder, Linear final layer. | No Transformer decoder; single Linear head. | **High** | **FIXED.** Configurable decoder stack, decoder LayerNorm, Linear-to-patch-pixels head; smaller width than the encoder. |
| **Positional embeddings** | FIXED 2D sin-cos in encoder and decoder (Appendix A.1). | Factorised LEARNABLE 2D (freq_pos + time_pos). | Low | **FIXED.** `_build_2d_sincos_pos_embed` registered as buffers in both encoder and decoder (parameter-free, trace-stable). |
| **Weight init** | `xavier_uniform_` Linear, `trunc_normal_(0.02)` mask/pos. | Only mask/pos got trunc_normal; Linear at Kaiming default. | Low | **FIXED.** `xavier_uniform_` on all Linear (zero bias); `trunc_normal_(0.02)` mask token. |
| **Masking strategy/ratio** | 75% UNIFORM RANDOM masking; random > block/grid (Table 1f). | Structural deterministic center-epoch mask (~1/7), low ratio. | Medium | **KEPT (documented deviation).** Structural mask is the correct inductive bias for supervised GA inpainting at a KNOWN location; encoder honours it (encodes context patches only). Optional random masked patches left as a flagged follow-up. |
| **Phase / reconstruction target** | Reconstruct raw target content (no phase shortcut). | Magnitude-only + input noisy phase via iSTFT. | Low | **KEPT (documented deviation).** Magnitude-only halves the target on a tiny dataset; complex/2-channel head is a flagged follow-up. |
| **Optimizer/lr recipe** | AdamW betas (0.9, 0.95), base lr 1.5e-4 (linear batch scaling), 40-epoch warmup, cosine, no grad clip. | Fixed lr 3e-4, wd 0.05, grad_clip 1.0; betas/warmup = trainer defaults. | Low | **CONFIG-LEVEL.** YAML documents the MAE base lr / linear scaling / betas / cosine+warmup intent and notes grad clip is a FACETpy stability addition. Does not affect the smoke test. |

## EEG-fMRI applicability assessment

| Paper method | Keep for EEG-fMRI? | Rationale |
|--------------|--------------------|-----------|
| ViT pre-norm encoder block (LN-MSA-residual, LN-MLP(GELU)-residual, final LN, mlp_ratio 4, no CLS) | **Yes** | Directly applicable, already correct. A small encoder runs cheaply on CPU for ≤112 spectrogram tokens; CLS correctly omitted for dense regression. |
| MAE asymmetric encoder/decoder (encoder on visible tokens only + lightweight decoder with re-inserted mask token) | **Yes** | The defining MAE contribution and the main faithfulness gap. Fully compatible with channel-wise single-/few-channel inference: encoder sees the surrounding context patches, decoder reconstructs the masked center-epoch patches. Cheap at tiny dims. |
| Masked-patch reconstruction loss in (per-patch normalised) target space | **Yes** | Exactly what inpainting needs — reconstruct only where the GA destroyed the signal. Per-patch normalisation stabilises the wide EEG spectral dynamic range. Trivial CPU cost. |
| Fixed 2D sin-cos positional embeddings | **Yes** | Cheap, parameter-free, trace-stable; the exact MAE choice for the small freq×time grid. |
| 75% uniform RANDOM masking | **No** | Random masking targets self-supervised learning with an unknown mask location. GA location is KNOWN, so the structural center-epoch mask is the correct supervised bias. Documented deviation; optional random masking behind a flag. |
| MAE pretraining recipe (4096 batch, RandomResizedCrop, 40-epoch warmup, no dropout/grad-clip) | **Partial** | Adopt AdamW + wd 0.05 + cosine + warmup + no dropout. RandomResizedCrop is meaningless for trigger-aligned EEG epochs; the 4096-batch/massive-scale schedule does not fit the tiny dataset; FACETpy's grad clip is a reasonable stability addition. |
| ImageNet pretraining / ViT-weight transfer | **No** | 224×224×3 ImageNet weights need patch-embed resize, pos-embed interpolation and channel averaging to fit a 1-channel 32×224 spectrogram; the proof-fit dataset is far too small to benefit. Train from scratch (as the original does). |
| Predicting raw/complex content (no phase shortcut) | **No** | Magnitude-only + input noisy phase is a deliberate FACETpy adaptation: halves the regression target on a small dataset and reuses the spectrogram convention. Complex/2-channel head is a flagged follow-up only. |

## Documented deviations summary

1. **Structural center-epoch mask** instead of MAE 75% random masking — the
   artifact location is known, so this is the correct supervised inductive bias.
2. **Magnitude-only prediction with input noisy phase** — halves the regression
   target on the tiny dataset; complex head deferred as a flagged follow-up.
3. **From-scratch training** — ImageNet ViT transfer is dimensionally and
   data-volume-wise inappropriate here.
4. **Optimizer recipe is config-level** — MAE base lr / linear scaling / betas /
   warmup are documented in the YAML; grad clip is a FACETpy stability addition.
