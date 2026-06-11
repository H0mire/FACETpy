# DPAE Paper-Accuracy Review

**Paper.** H. Xiong, Y. Ma, and Y. Li, "A general dual-pathway network for EEG
denoising," *Frontiers in Neuroscience*, vol. 17, art. 1258024, 2023.
doi:10.3389/fnins.2023.1258024.

DPAE is a lightweight supervised denoising autoencoder. Its defining
contribution (Fig. 2) is **two parallel encoder pathways at different scales**
feeding a **symmetric fusion module** (Fusion Encoder -> BatchNorm -> Fusion
Decoder) wrapped by a **residual ("Resnet") skip**, followed by a decoder that
reconstructs the **clean** signal. The template is instantiated as MLP, 1D-CNN
and 1D-RNN; we implement the **1D-CNN** (most robust on real multichannel EEG,
applied channel-by-channel). Target = clean EEG, loss = MSE, all activations
SeLU, input = single-channel 512-sample (2 s @ 256 Hz) segment, per-segment
inference normalisation = subtract std then divide by max-abs (Sec. 3.1).

This edition (`dpae_paper_accurate_edition`) is a more faithful re-implementation
than the original `facet.models.dpae`. The original is left untouched.

## Discrepancy table

| # | Aspect | Paper specifies | Original `dpae` | Severity | Fix in this edition |
|---|--------|-----------------|-----------------|----------|---------------------|
| 1 | Fusion module topology | Symmetric Fusion Encoder (compress joint rep, e.g. MLP 429->193->86->39) + Fusion Decoder (expand 39->86->193); BN on the joint-representation layer. Compress-then-reconstruct "common feature coding" (Fig. 2, Table 1). | concat -> BN -> single 1x1 conv -> SeLU. No fusion-encoder compression, no fusion-decoder reconstruction. | **high** | `_SymmetricFusion`: concat -> BN -> 1x1-conv encoder (C->C/2->C/4->C/8) -> mirror 1x1-conv decoder (->C). |
| 2 | Residual (Resnet) placement | Skip wraps the **fusion module**: input and output of the fusion module summed (identity of joint rep, Drozdzal 2016) on **feature maps**. | learned scalar on the **raw input signal**: `out = decoder(x) + s*x`. Pulls toward identity, conflicts with a clean target. | **high** | `fusion_out = fusion_decode(fusion_encode(z)) + z`. Scalar input-residual removed. |
| 3 | Prediction target / loss | Decoder reconstructs **clean** EEG; MSE(decoder, clean). | predicts the **artifact** (`artifact_center`, output_type ARTIFACT); MSE on artifact. | medium | Default `target_type='clean'`: dataset target = `clean_center`, adapter `output_type=CLEAN` returning `clean_data`. `'artifact'` switch retained; equivalent under additivity. |
| 4 | 1D-CNN pathway kernels/strides | Two conv pathways differing by **kernel** (1x3 vs 1x5) and **stride** (1x1 vs 1x4); stride realises the two scales. | dilation (1,2,4,8) + MaxPool on Path1; k=15/11/7 + MaxPool on Path2. Dilation, k=15/11/7 and strided convs are invented. | **high** | Pathway1 = Conv1d k=3 stride 1; Pathway2 = Conv1d k=5 stride 4. Downsampling via stride. |
| 5 | Shrinkage ratios | Low-dim pathway ratio **0.45** (512->230->103->46), high-dim pathway ratio **0.75** (512->682->511->383); high pathway expands once then contracts. | flat widths (F, F, 2F, 2F), no ratio logic, no expand-then-contract. | medium | `shrink_ratio_low=0.45` / `shrink_ratio_high=0.75` drive conv channel widths; high pathway expands once via `ceil(F/ratio)` then contracts. |
| 6 | Decoder | Post-fusion, post-residual reconstruction head; BN -> Dense -> Output 512 (MLP); CNN: symmetric mirror back to 512/1ch. | 2x ConvTranspose1d (fixed 4x) + Conv1d k3 + Conv1d k1; forced `%4` input. | low | Decoder upsamples by the **realised** pathway factor (length-safe), pre-decoder BN, 1x1 conv to 1 channel. No `%4` constraint. |
| 7 | Inference normalisation | Two-step (Sec. 3.1): subtract std, divide by max-abs per segment; rescale output back. | demean input only; optional prediction-mean removal. No std/max-abs, no inverse rescale. | medium | `normalize_segment` (subtract std, divide by max-abs) in `build_dataset` AND the processor; output rescaled back by the stored `(std, scale)`. |
| 8 | Depth / param budget | 16 hidden layers; 1D-CNN ~2.0M params, 3.9M FLOPs; "lightweight" is a central claim. | ~10 conv stages, widths reconstructed to ~2M but not via the paper's mechanism. | low | Layer count + width follow the dual-ratio + symmetric-fusion mechanism; realised param count tabulated in README (~2M at `base_filters≈320`). |
| 9 | Activations / optimizer / schedule | SeLU; Adam; batch 128; 200 epochs; lr 1e-3; MSE. | SeLU (faithful); MSE default (faithful); AdamW + clip + early stopping (close). | low | SeLU + MSE kept. Paper Adam/200-epoch recipe documented as reference; FACETpy AdamW + clip + early stopping noted as practical adaptation. |
| 10 | Dataset contract attributes | (FACETpy contract) `build_dataset` must expose `target_type`, `epoch_samples`, `trigger_aligned`, `sfreq`, `n_channels`, `chunk_size`, `input_shape`, `target_shape`, `n_chunks`, and propagate to split subsets. | `ChannelWiseArtifactDataset` missing `epoch_samples`; `_SubsetDataset` exposed none. | medium | `epoch_samples` added; all attributes propagated onto `_SubsetDataset`. Smoke test asserts the full contract on both full and split datasets. |

## EEG-fMRI applicability assessment

| Paper method | Kept? | Rationale |
|--------------|-------|-----------|
| Dual parallel pathways (fine vs coarse) feeding a fusion module | **Keep** | Gradient pulse has sharp EPI-readout edges (fine) and a slow volume/slice envelope (coarse); a multiscale dual pathway is the right inductive bias and is channel-agnostic on a single 1D segment. |
| Symmetric fusion module (Fusion Encoder -> BN -> Fusion Decoder) | **Keep (faithfully)** | The paper's core contribution; cheap (1x1 convs on the bottleneck); improves convergence per the ablation (Fig. 8); negligible CPU cost at tiny dims. |
| Residual skip around the fusion module | **Keep** | Identity mapping of the joint representation aids gradient flow; cheap; placed on feature maps, not the raw input. |
| Clean-EEG reconstruction target with MSE | **Keep (default)** | Equivalent to MSE-on-artifact under linear additivity `noisy = clean + artifact`; `target_type` exposed so FACETpy's subtract path still works. |
| k=3/k=5 kernels, stride 1/4 | **Keep** | Small, cheap, the literal paper spec; 4x strided downsampling is fine on 512-sample CPU segments. Replaces the invented dilation + k=15/11/7. |
| 0.45 / 0.75 shrinkage ratios | **Keep** | Cheap channel-width multipliers giving the two pathways genuinely different capacity (the paper's point). |
| Per-segment normalisation (subtract std, divide max-abs, rescale back) | **Keep** | EEG-fMRI gradient amplitudes vary across epochs/channels; amplitude-standardising before the net (and rescaling the output) improves generalisation. Applied identically in train and inference. |
| EEGdenoiseNet EOG/EMG noise model, eq. 3 SNR mixing | **Drop / adapt** | That is physiological-artifact training data, not gradient artifacts, and we do not have EEGdenoiseNet here. Train on the Niazy proof-fit NPZ instead; architecture unchanged. |
| RRMSE-temporal/spectral, Pearson-CC over SNR sweeps | **Keep (eval)** | Transfer to gradient-artifact denoising; part of FACETpy's evaluation standard. Not run by the smoke test. |
| 200-epoch / batch-128 / Adam recipe | **Keep (reference)** | Sensible reference; FACETpy AdamW + early stopping is an acceptable practical adaptation. Smoke uses `max_epochs 1`. |
| MLP and 1D-RNN DPAE variants | **Drop** | Out of scope; the 1D-CNN matches single-channel time-domain per-epoch correction. RNN would be slower on CPU. |

## Equivalence note (clean vs. artifact target)

Under the dataset's exact additivity `noisy = clean + artifact`, MSE on the
clean target equals MSE on the artifact target up to the (constant) std shift
used during normalisation. The clean target is the paper's formulation; the
artifact target is FACETpy's subtract-the-artifact convenience. The processor
returns `clean_data` for the clean target and lets `DeepLearningCorrection`
derive `artifact = original - clean`; for the artifact target it returns
`artifact_data` directly. Both paths leave the uncorrected samples untouched.
