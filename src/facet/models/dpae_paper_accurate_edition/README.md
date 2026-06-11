# DPAE -- Paper-Accurate Edition

`dpae_paper_accurate_edition` is a more faithful re-implementation of the
Dual-Pathway Autoencoder from

> H. Xiong, Y. Ma, and Y. Li, "A general dual-pathway network for EEG
> denoising," *Frontiers in Neuroscience*, vol. 17, art. 1258024, 2023.
> doi:10.3389/fnins.2023.1258024.

It exists alongside the original `facet.models.dpae` package (which is left
untouched). We implement the paper's **1D-CNN** instantiation, the variant the
paper reports as most robust on real multichannel EEG when applied
channel-by-channel (Sec. 4) -- which is exactly FACETpy's per-channel,
per-epoch correction setting.

## What changed vs. the original `dpae`

| # | Aspect | Original `dpae` | This edition (paper-faithful) | Paper ref |
|---|--------|-----------------|-------------------------------|-----------|
| 1 | **Fusion module** | concat -> BN -> single 1x1 conv -> SeLU | **Symmetric fusion**: concat -> BN -> Fusion Encoder (1x1 convs C->C/2->C/4->C/8) -> Fusion Decoder (mirror back to C) | Fig. 2, Table 1 |
| 2 | **Residual skip** | learned scalar on the **raw input signal** (`out = decoder(x) + s*x`) | identity **Resnet skip wrapping the fusion module** on feature maps (`fusion_out = decode(encode(z)) + z`) | Fig. 2 "Resnet Connection" (Drozdzal 2016) |
| 3 | **Target / loss** | predicts the **artifact**, MSE on artifact | reconstructs the **clean EEG**, MSE on clean (default); `target_type='artifact'` still supported | Sec. 2.3 |
| 4 | **CNN pathways** | k=3 with dilation (1,2,4,8) + MaxPool; k=15/11/7 + MaxPool | **k=3 stride 1** (fine) and **k=5 stride 4** (coarse); downsampling via stride | Sec. 2.2 |
| 5 | **Shrinkage ratios** | flat widths (F, F, 2F, 2F) | asymmetric **0.75** (expand-then-contract) and **0.45** (contract) channel-width ratios | Sec. 2.2, Table 1 |
| 6 | **Normalization** | demean only (train + inference) | per-segment **subtract std, divide by max-abs** (train + inference), rescaled back on output | Sec. 3.1 |
| 7 | **Length constraint** | hard `input_size % 4 == 0` | length-safe: decoder upsamples by the realised pathway factor; arbitrary lengths handled | -- |
| 8 | **Dataset contract** | subset exposed no attributes; `epoch_samples` missing | full contract incl. `epoch_samples`, propagated onto `train_val_split` subsets | FACETpy contract |

Faithful in both: **SeLU activations throughout**, **MSE loss**, **512-sample
2 s @ 256 Hz** reference input, **BatchNorm** on the joint representation
(Santurkar 2018), **lightweight / channel-count-independent** design.

## EEG-fMRI-appropriate deviations (documented, not blindly copied)

- **Training data.** The paper trains on EEGdenoiseNet clean EEG mixed with
  scaled EOG/EMG via its eq. 3 (SNR in [-7, 2] dB). We do not have EEGdenoiseNet
  here and our target artifact is the fMRI **gradient** artifact, not EOG/EMG.
  We therefore train on the FACETpy Niazy proof-fit NPZ bundle
  (`noisy = clean + artifact`). The **architecture is unchanged**; only the noise
  source differs. The multiscale dual-pathway inductive bias transfers well: the
  gradient pulse has both sharp EPI-readout edges (fine scale, Pathway1) and a
  slow volume/slice envelope (coarse scale, Pathway2).
- **Variants.** The paper's MLP and 1D-RNN instantiations are out of scope; the
  1D-CNN is the one suited to single-channel time-domain per-epoch correction.
- **Optimizer / schedule.** The paper uses plain Adam, batch 128, 200 epochs,
  lr 1e-3. The FACETpy harness uses AdamW + grad-clip + early stopping, which is
  an acceptable practical adaptation. Paper recipe is the documented reference.

See `documentation/paper_accuracy_review.md` for the full discrepancy table and
the EEG-fMRI applicability assessment.

## Architecture

```
input (B,1,L)
  ├─ Pathway1: Conv1d k=3 stride 1 stack, widths via ratio 0.75 (expand once, then contract)  ┐
  └─ Pathway2: Conv1d k=5 stride 4 stack, widths via ratio 0.45 (contract)                     ├─ concat (channel axis)
                                                                                               ┘
  -> BatchNorm -> Fusion Encoder (compress channels) -> Fusion Decoder (expand) (+Resnet skip)
  -> BatchNorm -> Decoder (ConvTranspose1d upsample to L) -> Conv1d 1x1 -> (B,1,L)
```

Both pathways downsample by the **same total factor** (`pathway2_stride`, default
4) so their bottleneck feature maps share a length and can be concatenated.

### Parameter budget

The paper reports the 1D-CNN at **~2.0M params / 3.9M FLOPs** (one-tenth to
one-twentieth the FLOPs of competing CNNs). The realised count here scales with
`base_filters`:

| `base_filters` | params |
|---|---|
| 32 (smoke-ish) | ~22 K |
| 96 | ~191 K |
| 256 | ~1.35 M |
| ~320 | **~2.0 M** (paper budget) |
| 384 | ~3.0 M |

The lightweight claim holds: the 1x1-conv fusion bottleneck and strided pathways
keep FLOPs low. Reaching the paper's exact 2M budget needs `base_filters≈320`
(the paper's MLP table uses much wider per-layer neuron counts than a CNN needs).
Pick `base_filters` per your compute budget; the structure -- not an ad-hoc width
-- realises the dual-ratio + symmetric-fusion mechanism.

## Training

```bash
uv run facet-train src/facet/models/dpae_paper_accurate_edition/training_niazy_proof_fit_smoke.yaml
```

The factory functions live in `training.py`:

- `build_model(input_shape=..., base_filters=32, shrink_ratio_low=0.45,
  shrink_ratio_high=0.75, pathway_layers=4, pathway2_stride=4, fusion_depth=3)`
- `build_loss(name='mse')` -- MSE (paper); `l1` / `smooth_l1` available.
- `build_dataset(path=..., target_type='clean', normalize=True, max_examples=...)`
  -- loads `noisy_center` / `clean_center` / `artifact_center` from the NPZ and
  yields per-channel `(1, samples)` items with std/max-abs normalisation.

## Inference

`processor.py` registers `dpae_paper_accurate_correction` (globally-unique name;
the original is `dpae_correction`). The adapter resamples each native
trigger-to-trigger epoch to `epoch_samples`, applies the same std/max-abs
normalisation as training, runs the TorchScript model per channel, rescales the
output back to native amplitude, and returns either `clean_data` (default) or
`artifact_data`. `DeepLearningCorrection` applies it (for the clean target it
derives `artifact = original - clean` and subtracts).

```python
from facet.models.dpae_paper_accurate_edition import DPAEPaperAccurateCorrection

step = DPAEPaperAccurateCorrection(
    checkpoint_path="exports/dpae_paper_accurate.ts",
    epoch_samples=512,
    target_type="clean",
)
```

## Evaluation

The paper evaluates with RRMSE (temporal and spectral) and Pearson correlation
over SNR sweeps. These metrics transfer to gradient-artifact denoising and are
part of FACETpy's evaluation standard (`src/facet/models/evaluation_standard.md`).
They are not exercised by the cheap smoke test.
