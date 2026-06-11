# Nested-GAN — paper-accurate edition

A more **Restormer-faithful** re-implementation of the Nested-GAN *generator*,
living beside the original `facet.models.nested_gan` (the original is untouched).

## Primary paper unavailable — read this first

> **The primary Nested-GAN paper is paywalled.** *Biomed. Phys. Eng. Express*
> 2025, DOI [10.1088/2057-1976/ae1a8c](https://doi.org/10.1088/2057-1976/ae1a8c),
> PMID 41183389 (IOP, not on arXiv). Only the PubMed abstract is public; it
> reports metrics (MSE 0.098, Pearson r 0.892, RRMSE 0.065, 71.6 % temporal /
> 76.9 % spectral artifact reduction) but discloses **no** architecture depth/
> width, STFT parameters, optimizer schedule, segment length, channel layout,
> or the exact GAN/discriminator structure.

Because of this, the headline "nested GAN" structure (two GANs, four
discriminators, gradient-balance scheme) **cannot be verified**. This edition
therefore does **not** blindly re-add discriminators. Instead it makes the part
that *is* documented — the **Restormer** generator backbone (Zamir et al., CVPR
2022, *"Restormer: Efficient Transformer for High-Resolution Image
Restoration"*) — genuinely faithful, and keeps the deterministic
multi-resolution STFT loss as a documented surrogate for the paper's
multi-resolution discriminators. See `documentation/paper_accuracy_review.md`
for the full discrepancy table and rationale.

## What changed vs. the original `nested_gan`

| Area | Original | This edition |
| --- | --- | --- |
| Inner spectral branch topology | **Flat** stack of 4 identical Restormer blocks at a single resolution | Genuine **hierarchical** Restormer encoder-decoder: per-level channel doubling, per-level heads/depth, pixel-unshuffle/shuffle down/up, skip-concat + 1×1 reduce, optional refinement stage |
| Global residual (Restormer `out = in + R`) | Absent in the inner branch | **Added**: inner predicts `center_spec + R` in the 2-channel real/imag space |
| GDFN expansion γ | 2.0 | **2.66** (Restormer's value); overridable |
| LayerNorm | With learnable bias only | Optional **bias-free** variant (`inner_bias_free_norm`) matching Restormer's denoising config |
| Inner-branch context | Center epoch only | Optional center ±N neighbour epochs as extra STFT channels (`inner_neighbor_epochs`, default 0) — an EEG-fMRI improvement, **not** a paper claim |

## What deliberately stayed the same (documented, not bugs)

- **Generator-only recipe + MR-STFT loss surrogate.** The facet-train CLI takes
  a single `(pred, target) -> scalar` loss and one optimizer. A true alternating
  4-discriminator nested GAN needs a custom training-loop wrapper that is out of
  scope, not CPU-cheap, and — crucially — **unverifiable** against the paywalled
  paper. The multi-resolution STFT magnitude loss (HiFi-GAN lineage) is kept as
  the documented spectral-fidelity surrogate.
- **MDTA channel attention with a learnable per-head temperature.** This
  deliberately replaces Vaswani et al. (2017)'s fixed `1/√d_k` scaling and is
  faithful to Restormer — it is intentional, *not* a bug.
- **Outer time-domain U-Net refiner** over the multi-epoch context with
  center-slot residual injection. This is a FACETpy-appropriate adaptation (not
  a Restormer feature) that fixes trigger-boundary phase discontinuities — the
  model's main EEG-fMRI value-add.

## Contracts (unchanged)

- `build_model / build_loss / build_dataset` honour the facet-train factory
  contract (injected kwargs `input_shape`, `epoch_samples`, `context_epochs`,
  `n_channels`, `sfreq`, `target_type`, … swallowed via `**_`; explicit YAML
  `model.kwargs` override).
- `processor.py` exposes `NestedGANPaperAccurateAdapter` (subclasses
  `EpochContextArtifactAdapter`) and `@register_processor`
  `NestedGANPaperAccurateCorrection` with the **globally unique** registry name
  `nested_gan_paper_accurate_correction`. The TorchScript forward signature
  `(B, C_epochs, 1, T) -> (B, 1, T)` matches the original, so inference is a
  drop-in replacement.

## Files

- `training.py` — paper-faithful architecture + factories.
- `processor.py` — inference adapter + registered correction (unique name).
- `training_niazy_proof_fit_smoke.yaml` — illustrative CPU smoke config (tiny
  dims, `device: cpu`, `max_epochs: 1`); **not executed** by the pytest smoke.
- `documentation/paper_accuracy_review.md` — discrepancy table + EEG-fMRI
  applicability + the "paper paywalled / GAN structure unverified" banner.

## References

- M. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang. *Restormer:
  Efficient Transformer for High-Resolution Image Restoration.* CVPR 2022.
- A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017 (baseline that
  MDTA's learnable-temperature channel attention departs from).
- Nested-GAN primary paper: *Biomed. Phys. Eng. Express* 2025, DOI
  10.1088/2057-1976/ae1a8c, PMID 41183389 (paywalled; abstract only).
