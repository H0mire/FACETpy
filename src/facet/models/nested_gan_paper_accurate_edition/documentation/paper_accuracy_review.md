# Nested-GAN — paper-accuracy review

> ## ⚠ Primary paper paywalled — GAN/nesting structure UNVERIFIED
>
> The primary Nested-GAN paper (*Biomed. Phys. Eng. Express* 2025, DOI
> **10.1088/2057-1976/ae1a8c**, PMID **41183389**, IOP, not on arXiv) is
> **paywalled**. Only the PubMed abstract is public. It reports summary metrics
> (MSE 0.098, Pearson r 0.892, RRMSE 0.065, 71.6 % temporal / 76.9 % spectral
> artifact reduction) but discloses **none** of: architecture depth/width, STFT
> parameters, optimizer schedule, segment length, channel layout, or the exact
> GAN/discriminator structure.
>
> Consequently the headline **"nested GAN"** design — two GANs (inner STFT
> generator + 2 metric discriminators; outer waveform generator + 2
> multi-resolution discriminators) with a four-discriminator gradient-balance
> scheme — **cannot be verified**. This edition does **not** blindly re-add it.
> The paper-accuracy work targets the *documented* generator backbone,
> **Restormer** (Zamir et al., CVPR 2022), and keeps a deterministic
> multi-resolution STFT loss as the documented surrogate for the paper's
> multi-resolution discriminators.

## Discrepancy table (paper / documented backbone vs. implementations)

| # | Aspect | Paper / Restormer specifies | Original `nested_gan` | This edition | Severity addressed |
| - | --- | --- | --- | --- | --- |
| 1 | GAN wrapper / nesting / 4 discriminators (the model's defining feature) | Two nested GANs; inner generator + 2 metric discriminators on STFT; outer generator + 2 multi-resolution discriminators on the waveform; gradient-balance across the four | Generator-only; no discriminators; MR-STFT magnitude loss as surrogate | **Kept generator-only + MR-STFT surrogate** (unverifiable; incompatible with the single-loss/one-optimizer CLI; not CPU-cheap). Surfaced prominently as the top caveat. | high — documented deviation |
| 2 | Inner-branch topology | Restormer 4-level hierarchical encoder-decoder, pixel-unshuffle/shuffle, blocks [4,6,6,8], channels [48,96,192,384], heads [1,2,4,8], skip-concat + 1×1-halve, refinement stage | **Flat** stack of 4 identical blocks at one resolution | **Hierarchical** `HierarchicalSpectrogramRestormer`: per-level channel doubling, per-level heads/depth, pixel-unshuffle/shuffle down/up, skip-concat + 1×1 reduce, optional refinement stage (downscaled for CPU) | high — **fixed** |
| 3 | Global residual learning (`out = in + R`) | Final conv produces residual R; output = degraded_input + R | Inner branch projects directly to output, no residual | **Added**: inner predicts `center_spec + R` in the real/imag space | medium — **fixed** |
| 4 | GDFN expansion ratio γ | 2.66 | 2.0 | **2.66** default (overridable; smoke uses 2.0) | low — **fixed** |
| 5 | Attention scaling vs. vanilla Transformer | MDTA: learnable per-head temperature on channel cross-covariance, Q/K L2-normalized — a deliberate departure from Vaswani 2017's fixed 1/√d_k | Correctly implements learnable temperature + channel attention | **Preserved exactly** — the 1/√d_k departure is intentional and faithful, **not a bug** | low — no change (documented) |
| 6 | Inner branch sees center epoch only | Not disclosed; project lesson (DHCT-GAN) is that full context must inform the prediction | Inner sees center only; only outer sees context | Optional `inner_neighbor_epochs` feeds center ±N neighbours as extra STFT channels (default 0). Documented as an EEG-fMRI improvement, **not** a paper claim | medium — optional fix |
| 7 | Optimizer / LR schedule / progressive learning | Restormer: AdamW (0.9, 0.999), wd 1e-4, L1, lr 3e-4 cosine→1e-6 over 300K iters, progressive patch sizes. Primary schedule undisclosed | Full YAML: AdamW lr 5e-4, wd 1e-4, no cosine, no progressive | Smoke YAML stays minimal (lr 1e-3, 1 epoch). lr 3e-4 + cosine recommended for full runs if the CLI exposes a scheduler. **Progressive patch-size learning is inapplicable** (fixed 512-sample trigger-locked epochs) — documented, not implemented | low — documented |
| 8 | Smoke YAML device | Harness rule: `device: cpu` | Smoke YAML used `device: cuda` | Smoke YAML uses **`device: cpu`**, `max_epochs: 1`, tiny dims (illustrative; not executed by the pytest smoke) | medium — **fixed** |
| 9 | Bias-free design for denoising | Restormer uses bias-free convs + bias-free LayerNorm in its denoising config | Convs already bias-free; LayerNorm had a learnable bias | Optional **bias-free LayerNorm** (`inner_bias_free_norm`); default keeps bias for backward compatibility | low — optional fix |

## EEG-fMRI applicability assessment

| Paper method | Keep for EEG-fMRI? | Rationale |
| --- | :-: | --- |
| Restormer 4-level hierarchical encoder-decoder (pixel-unshuffle/shuffle, multi-scale blocks/channels/heads) | ✅ | The multi-scale hierarchy is Restormer's core and makes sense on the ~33×33 STFT image of a 512-sample epoch: gradient-artifact harmonics live at multiple time-frequency scales. A small 2–3-level version is CPU-cheap. **Kept**, downscaled for the single-EEG-channel setting. |
| MDTA channel cross-covariance attention with learnable temperature | ✅ | Linear-complexity channel attention suits small spectrogram images on CPU. The learnable-temperature departure from 1/√d_k is faithful to Restormer and helps. **Kept**. |
| GDFN gated feed-forward (γ = 2.66) | ✅ | Cheap, standard Restormer block; gating improves fidelity. **Kept**, γ fixed to 2.66. |
| Global residual learning (`out = in + R`) | ✅ | Residual prediction suits artifact removal where the artifact is a large structured perturbation. **Added** to the inner branch. |
| Full nested GAN with 2 metric + 2 multi-resolution discriminators + gradient balance | ❌ | Unverifiable (paywalled); needs an alternating-GAN loop incompatible with the single-loss/one-optimizer CLI; not CPU-cheap; metric discriminators predict a perceptual (PESQ-like) scalar with no EEG analogue. **Replaced** by the MR-STFT magnitude loss (documented surrogate). |
| Metric-discriminator perceptual scoring (CMGAN/PESQ lineage) | ❌ | PESQ/speech perceptual metrics have no EEG-fMRI equivalent; FACETpy's RMS/SNR metrics are not differentiable perceptual scalars. **Replaced** by the deterministic MR-STFT loss. |
| Multi-resolution STFT discriminators (HiFi-GAN lineage) | ✅ (as loss) | Their spectral-fidelity objective is well-approximated by a deterministic multi-resolution STFT magnitude loss — differentiable, single-loss-CLI compatible, CPU-cheap. **Kept** as the loss surrogate. |
| Progressive learning over growing image-patch sizes | ❌ | EEG epochs are fixed-length trigger-locked windows (512 samples); there is no 2D patch-cropping notion. **Documented as inapplicable**, not implemented. |
| Outer time-domain U-Net refiner over multi-epoch context with center-slot residual injection | ✅ | Not a Restormer feature but a sound EEG-fMRI adaptation: injects neighbour-epoch context to fix trigger-boundary phase discontinuities (the DHCT-GAN lesson). **Kept** — the model's main FACETpy value-add. |
| Sinusoidal positional encoding (Vaswani 2017) | ❌ | Restormer/MDTA use convolutions for implicit positional info and channel-axis attention; explicit positional encodings are neither used by the backbone nor needed. Correctly **absent**. |

## Documented deviations (summary)

1. **No discriminators / no nested-GAN loop.** Generator-only + MR-STFT
   surrogate, because the GAN structure is unverifiable against the paywalled
   paper and is incompatible with the single-loss/one-optimizer facet-train CLI
   and CPU-cheap smoke requirement. *This is the single most important caveat.*
2. **Metric (PESQ-like) discriminators replaced** by the deterministic MR-STFT
   magnitude loss — no EEG-fMRI perceptual analogue exists.
3. **Progressive patch-size learning not implemented** — inapplicable to
   fixed-length trigger-locked EEG epochs.
4. **`inner_neighbor_epochs` context extension and bias-free LayerNorm** are
   optional and default off / off-bias, presented as EEG-fMRI / Restormer-config
   improvements rather than verified paper claims.
5. **The learnable-temperature MDTA** intentionally departs from Vaswani 2017's
   fixed `1/√d_k`; preserved exactly per Restormer (not a bug).

## References

- M. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang. *Restormer:
  Efficient Transformer for High-Resolution Image Restoration.* CVPR 2022.
- A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
- Nested-GAN primary paper: *Biomed. Phys. Eng. Express* 2025, DOI
  10.1088/2057-1976/ae1a8c, PMID 41183389 (paywalled; abstract only).
