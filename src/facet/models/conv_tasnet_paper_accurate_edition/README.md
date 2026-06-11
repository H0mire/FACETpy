# Conv-TasNet (paper-accurate edition)

A more faithful re-implementation of

> Y. Luo and N. Mesgarani, **"Conv-TasNet: Surpassing Ideal Time-Frequency
> Magnitude Masking for Speech Separation,"** *IEEE/ACM Transactions on Audio,
> Speech, and Language Processing*, vol. 27, no. 8, pp. 1256-1266, 2019.
> [arXiv:1809.07454v3]

than the original `facet.models.conv_tasnet` package, while staying compatible
with the FACETpy training/inference contracts and CPU-cheap for
single-/few-channel EEG-fMRI gradient-artifact removal.

This package only **adds** new files; the original `conv_tasnet/` package is
untouched and both editions coexist (the processor here registers under the
unique name `conv_tasnet_paper_accurate_correction`).

## What is Conv-TasNet (briefly)

A fully-convolutional, time-domain, single-channel source separator that avoids
the STFT entirely (Fig. 1A/B of the paper):

1. **Encoder** — one 1-D conv with `N` basis filters of length `L` and stride
   `L/2` (50% overlap) maps each mixture segment to a representation `w`.
2. **Separation** — global layer norm (gLN) over channel+time, a 1x1 bottleneck
   `N -> B`, then a TCN of `R` repeats of `X` dilated conv blocks (dilation
   `2^x`). Each block (Fig. 1C) is `1x1-conv -> PReLU -> Norm -> depthwise
   D-conv -> PReLU -> Norm -> {residual 1x1 (H->B), skip 1x1 (H->Sc)}`. Skips
   from all blocks are summed, then `PReLU -> 1x1-conv (Sc -> C*N) -> mask
   activation` produces `C` masks in `[0, 1]`.
3. **Mask + Decoder** — each masked representation is inverted by a **single
   shared** 1-D transposed conv (`N -> 1`, length `L`, stride `L/2`) and
   overlap-added.

Paper headline **non-causal "gLN"** config (Table II/IV, ~5.1M params):
`N=512, L=16, B=128, H=512, Sc=128, P=3, X=8, R=3`, **linear encoder + Sigmoid
mask** (Table III best row), SI-SNR + uPIT loss.

## What changed vs. the original edition (and why)

| Change | Original | This edition | Rationale |
| --- | --- | --- | --- |
| **Encoder nonlinearity** | ReLU, hard-coded | `encoder_activation="linear"` by **default** (ReLU still selectable) | The paper's BEST published result (Table III), used for all headline experiments, pairs a **linear** overcomplete encoder with a Sigmoid mask. The original's ReLU encoder is a lower-scoring row. This is the single most impactful faithfulness fix and is free on CPU. |
| **Skip-connection width `Sc`** | Conflated with `B` (skip 1x1-conv emits `B`; mask reads `B`) | Explicit `skip_channels` (Sc); skip 1x1-conv `H->Sc`; mask 1x1-conv `Sc->C*N` (default `Sc=128`) | Fig. 1C / Table I define `Sc` as an independently-tuned hyperparameter separate from `B`. Making it explicit turns the model into a true superset of the paper and avoids silently breaking if a user sets `B != 128` expecting `Sc=128`. |
| **Source-additivity loss** | none | optional `consistency_mse` loss | FACETpy-appropriate enhancement: the dataset obeys `noisy = clean + artifact` *exactly*, so a penalty that the predicted sources sum to the mixture is well-founded (the paper merely *relaxes* the unit-summation mask constraint; we can do strictly better). |

### Deliberate, documented EEG-fMRI deviations (NOT copied from the paper)

* **Loss = MSE on ordered sources, no uPIT.** The two sources (clean EEG,
  gradient artifact) are known and **ordered**, so permutation-invariant
  training is unnecessary. SI-SNR's *scale invariance* would discard amplitude
  information that is meaningful for our deterministic AAS-derived data where
  the artifact must be subtracted at true scale. SI-SNR (`si_sdr_neg`) is kept
  **only as an ablation**; no PIT is implemented.
* **Input-length-matched dims.** Defaults `N=256, B=128, H=256, Sc=128, P=3,
  X=8, R=2` are sized for the ~512-sample (~100 ms) EEG epochs, not the paper's
  4-second/8 kHz speech utterances. With `X=8` the dilations reach `2^7=128`
  frames; the receptive field already covers the entire `T'≈63`-frame latent, so
  `R=2` vs `R=3` changes *capacity*, not *coverage*. This is a deliberate match,
  not a budget cut. The paper's `512/512/3` config is the documented reference.
* **gLN only (no causal cLN).** Offline correction is non-causal, and the paper
  shows cLN costs ~2.5 dB SI-SNRi, so gLN is the correct choice.
* **Raw-input demeaning.** The dataset/adapter demean the raw segment to remove
  DC offset (an EEG preprocessing step not in the paper, which only normalises
  the encoder latent via gLN). Train-time and inference-time demeaning are
  identical so they never diverge.
* **Depthwise-separable conv folding.** The paired pointwise (1x1) mixing of the
  S-conv is folded into the residual/skip convs, matching mainstream reference
  repos (asteroid, kaituoxu/Conv-TasNet); see `documentation/paper_accuracy_review.md`.

## Faithful pieces preserved from the original

* gLN over channel+time with per-channel learned gamma/beta — matches eq. 9-11.
* TCN block ordering `conv -> PReLU -> Norm` — matches Fig. 1C.
* A **single shared** transposed-conv decoder reused for every source — matches
  eq. 3/5.
* Encoder/decoder bias absent; encoder stride `L/2` (50% overlap).

## Usage

```yaml
model:
  factory: facet.models.conv_tasnet_paper_accurate_edition.training:build_model
  kwargs:
    encoder_activation: linear   # paper best config
    skip_channels: 128           # explicit Sc
  loss_factory: facet.models.conv_tasnet_paper_accurate_edition.training:build_loss
  loss_kwargs:
    name: mse                    # or consistency_mse for the additivity penalty
```

See `training_niazy_proof_fit_smoke.yaml` for a tiny illustrative CPU config and
`tests/models/conv_tasnet_paper_accurate_edition/test_training_smoke.py` for the
executable smoke check.
