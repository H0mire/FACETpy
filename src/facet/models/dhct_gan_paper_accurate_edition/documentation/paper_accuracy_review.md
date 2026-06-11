# DHCT-GAN Paper-Accuracy Review

Paper: **Cai, Y., Meng, Z., Huang, X. "DHCT-GAN: Improving EEG Signal Quality with
a Dual-Branch Hybrid CNN-Transformer Network." MDPI _Sensors_ 2025, 25(1):231.**

This review documents every discrepancy between the original
`facet.models.dhct_gan` implementation and the paper, what the paper-accurate
edition (`facet.models.dhct_gan_paper_accurate_edition`) does about each, and an
EEG-fMRI applicability assessment of each paper technique.

## Discrepancy table

| # | Aspect | Paper specifies | Original impl | Severity | This edition |
| --- | --- | --- | --- | --- | --- |
| 1 | Reconstruction loss | MSE on every generator output branch (Eq. 10); all curves MSE | `nn.L1Loss` for artifact + consistency | high | **Fixed** → `nn.MSELoss` everywhere |
| 2 | Adversarial / D objective | LSGAN (Eqs. 12-13), real-valued score, no sigmoid | vanilla BCE-with-logits | high | **Fixed** → LSGAN least-squares for both D and G |
| 3 | Feature / perceptual loss | `lambda1·L_feat` = MSE of intermediate D conv features (Eq. 11) | omitted entirely | high | **Fixed** → discriminator returns feature maps; `lambda_feat·MSE(phi(Y), phi(G(X)))` |
| 4 | Gating fusion | 2 gating networks (2 FC + tanh) → masks `Ymask1`/`Ymask2`; `Ypre = Ymask1·Y1 + Ymask2·(Xraw-Y2)` (Eqs. 4-5); masks not summing to 1 | 1 conv gate (sigmoid), `g·clean+(1-g)·(x-art)` | medium | **Partial / architecture-only** → two independent conv gating heads (tanh) exist in `_compute_outputs`, but under the single-optimizer wrapper the loss never reads them (see Limitation below); conv-not-FC documented |
| 5 | Number of discriminators | 3 (D1 clean, D2 noise, D3 fused) (Eqs. 6-9, 13) | 1 PatchGAN on artifact head | medium | **Fixed** → 3 discriminators with private optimizers in the loss module |
| 6 | Per-branch generator loss | `Loss1 + Loss2 + Loss3` each MSE + feat + adv (Eqs. 6-9) | single combined objective | medium | **Fixed** → explicit Loss1/Loss2/Loss3 sum |
| 7 | Encoder depth/widths | 5 CNN-LGTB stages, 64→1024, 1024-sample inputs | 4 stages, 16→128, 512 samples | low | **Deviation kept** (depth/width kwarg-configurable) — see below |
| 8 | LGTB inner depth + local-attention scheme | inner residual stack x5; local SA = fixed 8-block split→concat; global SA; parallel CNN path | depth 1; fixed `window_size`; LayerNorm; sequential CNN+transformer | medium | **Fixed** → configurable `n_lgtb` (default 2), `n_local_blocks` (default 8) fixed-block split, **parallel CNN path** added; LayerNorm-vs-BN documented |
| 9 | Preprocessing stem + pool | 2 convs to 32-dim + AvgPool | 2 convs to `base_channels`, no pool | low | **Optional** `stem_pool` AvgPool added; stem width tracks `base_channels` |
| 10 | Generator optimizer betas | Gen Adam (0.5, 0.9); Disc Adam (0.9, 0.999); lr 1e-4 (or 1e-3→1e-4) | wrapper hard-codes AdamW (0.9, 0.999) on gen; D Adam (0.9, 0.999) | medium | **Deviation documented**: D betas matched (0.9/0.999); gen 0.5/0.9 not settable via the standard CLI wrapper |
| 11 | Input segment length | 1024 (2 s @ 512 Hz) | 512 | low | **Deviation kept**: 512 = Niazy NPZ window; model length-agnostic; `epoch_samples` configurable |
| 12 | Exported / deployed head | fused clean `Ypre` (output_type CLEAN) | artifact head Y2 (output_type ARTIFACT) | low | **Deviation kept**: artifact head exported for the FACETpy subtraction contract |

## EEG-fMRI applicability assessment

Kept for EEG-fMRI gradient-artifact removal:

- **Dual-branch generator (clean Y1 / artifact Y2).** Core ablation-validated
  contribution. The artifact branch is exactly what the ARTIFACT-subtraction
  pipeline supervises; cheap at small widths. Kept; artifact branch exported.
- **Two independent gating networks (Eqs. 4-5).** Adaptively trusts clean vs.
  raw-minus-noise per sample — sensible when an AAS-derived artifact estimate is
  imperfect over a TR. Implemented as two trace-friendly conv heads.
- **Local + Global Transformer Block (fixed-block local attention).** Global
  attention captures TR-periodic gradient structure; local captures fine slice
  detail. Kept at reduced `n_lgtb` (~2), fixed `n_local_blocks` (8), small head
  dims so a single forward+backward stays in the few-second CPU budget.
- **Three discriminators D1/D2/D3.** Headline GAN-stability contribution. Hosted
  inside the loss module with private optimizers (preserves the single-CLI-optimizer
  trick). Negligible CPU cost at smoke dims; adversarial weight kept small.
- **LSGAN adversarial / discriminator loss (Eqs. 12-13).** Exactly the paper spec,
  strictly more faithful than the original BCE, and more stable for regression-like
  1-D signal GANs. Free to adopt.
- **MSE reconstruction (Eq. 10).** Paper loss; penalizes large-amplitude artifact
  peaks quadratically — appropriate for gradient-artifact regression. Adopted.
- **Discriminator feature-matching `L_feat` (Eq. 11).** Cheap (reuses the
  discriminator forward), stabilizes training. Adopted with configurable
  `lambda_feat` (documented default 1.0).
- **Generator Adam betas (0.5, 0.9).** Faithful and harmless if reachable, but the
  standard `facet-train` wrapper hard-codes AdamW (0.9, 0.999) over generator
  params, so this cannot be set via the CLI. Discriminator betas already matched.

Dropped / adapted for EEG-fMRI (documented deviations):

- **5 stages / 64–1024 widths / 1024-sample inputs / batch 40 / 1000 epochs.**
  Oversized for ~25k single-channel 512-sample Niazy windows and narrow-band
  gradient artifacts; 512-sample windows cannot survive 5× downsampling. Kept the
  4-stage / 16–128 reduction but made depth/width/batch/epochs configurable.
- **Synthetic SNR mixing + physiological EMG/EOG/ECG artifacts.** Dataset-construction
  detail specific to physiological artifacts; not applicable. FACETpy targets fMRI
  gradient artifacts from the Niazy AAS-derived NPZ. The existing per-channel reader
  is kept; the artifact-domain difference is documented.
- **Returning fused clean `Ypre` (output_type CLEAN).** FACETpy subtracts a
  predicted artifact (output_type ARTIFACT), more robust for high-amplitude periodic
  gradient artifacts than regressing clean EEG directly. The artifact head is
  exported; documented as an intentional pipeline-driven deviation.

## Known limitation: gating heads + clean decoder are not trained in this edition

The `facet-train` wrapper calls `loss = loss_fn(model(x), target)` where `model(x)`
returns **only the artifact head Y2**. `DHCTGanLossPA` therefore never receives the
generator instance and cannot read the generator's clean head (Y1), gating heads
(`Ymask1`/`Ymask2`) or the gated fused output (Ypre). Instead it reconstructs the
clean/fused estimate analytically as `noisy − artifact` (the ARTIFACT-subtraction
identity) and supervises that.

Concrete consequences, verified by inspecting which parameters receive gradient:

- `clean_decoder`, `clean_head`, `gate1`, and `gate2` receive **zero gradient**
  during training (~16% of generator parameters). They are present in the module
  graph and in `_compute_outputs`, but are inert — trained to nothing and unused at
  inference (the exported head is the artifact decoder only).
- Because the loss sets `clean_pred = fused_pred = noisy − artifact`, the three
  discriminators reduce to **two distinct tasks**: D2 supervises the artifact head,
  while D1 (clean) and D3 (fused) are fed identical real/fake pairs and are therefore
  redundant duplicates. The paper's Y1-vs-Ypre distinction is not realized.
- `Loss1` and `Loss3` are computed on identical tensors, so the per-branch
  decomposition is two effective branches, not three.

This is the same structural limitation as the original `dhct_gan` (whose single
gate + clean head are likewise dead code under the wrapper), so it is **not a
regression**. The effective, gradient-carrying improvements over the original are
real: MSE reconstruction (Eq. 10), LSGAN (Eqs. 12-13), discriminator
feature-matching (Eq. 11), additional discriminator capacity, the parallel CNN
path, fixed-block local attention, and the configurable LGTB inner depth.

Fully realizing Eqs. 4-9 (a genuinely supervised dual-branch generator with two
live gating heads and three distinct discriminators) requires the loss to call the
generator's `_compute_outputs` directly — e.g. a custom wrapper that passes the
generator (or its full output dict) into the loss, rather than the artifact head
alone. That is out of scope here because it would change the `facet-train`
single-optimizer contract.

## Compatibility notes

- The exported `forward()` returns only the artifact head, so `torch.jit.trace`
  works and `DeepLearningCorrection` subtracts the predicted artifact unchanged.
- The three discriminators live in `DHCTGanLossPA`, run alternating LSGAN steps on
  detached predictions only when grad is enabled, and are skipped under
  `eval`/`no_grad`. The single `facet-train` optimizer continues to train only the
  generator.
- All shape arithmetic in the generator uses symbolic shapes (no Python branches on
  tensor-valued dimensions), so the model is trace-safe at any sequence length.
