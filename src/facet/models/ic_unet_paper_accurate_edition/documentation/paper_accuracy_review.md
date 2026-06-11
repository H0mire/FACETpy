# IC-U-Net paper-accuracy review

**Paper.** Chuang, Chang, Huang & Jung (2022), *IC-U-Net: A U-Net-based
denoising autoencoder using mixtures of independent components for automatic EEG
artifact removal*, NeuroImage 263:119586; arXiv:2111.10026; repo
`roseDwayane/AIEEG`.

**Scope.** This edition makes the FACETpy IC-U-Net model more faithful to the
source paper for single-/few-channel EEG-fMRI **gradient**-artifact removal,
without breaking the facet-train / inference contracts and while staying cheap
on CPU. Paper techniques that do not transfer to this domain are adapted and
documented rather than copied.

## 1. Discrepancy table (original edition → paper-accurate edition)

| Aspect | Paper specifies | Original `ic_unet` | Severity | Fix in this edition |
|--------|-----------------|--------------------|----------|---------------------|
| ICA's role | ICA + ICLabel synthesise training pairs (mixB clean / mixBnB noisy); runtime network is sensor-level (Sec 2.3, Conclusion). | Frozen `W` / `W_pinv` buffers sandwich the U-Net in IC space. | **high** | Default model is a pure sensor-level U-Net. `use_frozen_ica=True` keeps the sandwich as a documented non-paper extension (OFF by default). |
| Frequency loss term | `L_freq` = MSE of z-scored **PSD** `\|rfft\|²` over **1-50 Hz** bins (Eq. 4). | Raw `\|rfft\|` magnitude, full spectrum, no PSD/band/z-score. | **high** | `L_freq` = MSE of per-channel z-scored `\|rfft\|²` restricted to 1-50 Hz; `sfreq` resolves the band, hi clamped below Nyquist, short-signal fallback to full positive band. |
| Decoder upsampling | 1-D transposed convolution (deconvolution) with CBR (Sec 2.1). | `nn.Upsample(mode='linear')` (parameter-free) + DoubleConv. | medium | `nn.ConvTranspose1d(stride=2)` learned deconvolution + concatenative skip + CBR double-conv, with pad/crop length guard. |
| CBR activation | Conv → BatchNorm → **ReLU** (Sec 2.1). | `LeakyReLU(0.1)`. | medium | Default `activation='relu'`; `'leaky_relu'` opt-in. |
| Ensemble weighting | `(1/Σαᵢ)·Σαᵢ Lᵢ`, best `α=[1,1,1,1]` (Eq. 2, Fig 4A). | `[1,1,1,0.5]` summed, no `1/Σαᵢ`. | medium | Default `α=[1,1,1,1]`, divide by `Σαᵢ`; weights still configurable. |
| Input/target normalisation | Per-time-series **z-score** before training (Sec 3.1/4.1). | Per-channel **demean** only. | medium | Default per-channel z-score in `build_dataset`; per-example scale recoverable via `scale_for`. `'demean'`/`'none'` opt-in. |
| Output/target semantics | DDAE reconstructs the **clean** signal `Y ≈ X` (Eqs 1-3). | Predicts the **artifact** (`artifact_center`). | low | Default `target_type='clean'` (clean head; FACET subtracts clean from noisy). `'artifact'` kept compatible. Adapter wires `clean_data` vs `artifact_data` accordingly. |
| Depth / filter ladder | Doubling encoder / halving decoder; exact Fig 2A kernels not in prose. | Hardcoded 4-level `64→512`, kernels `7,7,5,3` / `3`. | low | Keep doubling/halving ladder; expose `depth`/`base_channels`/`kernel_size`. Default 4-level 64-base. Note that exact Fig 2A kernels were taken from the reference repo. |
| Smoke device / pytest | N/A (FACETpy infra: CPU smoke in seconds). | Smoke YAML used `device: cuda`; no pytest smoke. | low | New smoke YAML uses `device: cpu` tiny dims; new pytest CPU smoke asserts loss decreases and forward shape. |

## 2. EEG-fMRI applicability assessment

### Kept (paper-faithful, sensible for EEG-fMRI gradient removal)

- **Sensor-level 1-D U-Net** (CBR encoder + transposed-conv decoder +
  concatenative skips). Signal-agnostic, works for `n_channels ≥ 1`, directly
  learns the gradient-artifact morphology from paired noisy/clean data. This is
  the heart of the model.
- **Four-term ensemble loss** (amp + velocity + acceleration + frequency).
  Gradient artifacts are broadband and high-frequency; the velocity/acceleration
  /frequency terms counter the frequency-principle low-frequency bias and help
  fit sharp GA transients. Cheap on CPU. Fixed to the paper's normalised,
  equal-weighted, 1-50-Hz z-scored-PSD form (band clamped ≤ Nyquist for safety).
- **Per-time-series z-score normalisation.** Standard, cheap, the paper's stated
  preprocessing; stabilises a ReLU/BN network on unbounded EEG amplitudes;
  per-example scale tracked for inference denormalisation.
- **Clean-reconstruction target** (`target_type='clean'`). The paper-faithful
  objective; the corrector subtracts the reconstructed clean signal from the
  noisy one. Default, with the FACETpy artifact head kept as a compatible option.
- **Multichannel (joint) processing.** The U-Net naturally handles `n_channels`
  in/out; joint processing exploits cross-channel GA structure (e.g. the 30-ch
  Niazy montage) and degrades cleanly to single-channel.

### Dropped from the faithful default (documented deviations)

- **In-graph frozen ICA `W`/`W_pinv`.** *Not in the paper* — the paper's network
  is sensor-level. For single-/few-channel EEG-fMRI a small (e.g. 30×30) ICA
  matrix is ill-posed and FastICA on GA-dominated data is unstable (the original
  code falls back to identity). Kept only as `use_frozen_ica=True` extension.
- **ICA + ICLabel mixB/mixBnB data synthesis (Eqs 5-6).** ICLabel classifies
  eye/muscle/heart/line/channel-noise ICs — *none* of which is the fMRI gradient
  artifact — and needs many channels plus a resting-state IC library we lack.
  FACETpy already supplies physically paired noisy/clean (AAS-referenced)
  examples in the NPZ, which is the appropriate domain substitute. This is the
  principled deviation: we keep the *idea* (train on paired clean/noisy data)
  while replacing the *mechanism* (ICA backprojection) with FACET's physical
  pairing.
- **Published hyperparameters** (lr=0.01, batch=128, 150 epochs, 256 Hz,
  1024-sample segments). Tuned for 150k resting-state segments at 256 Hz; the
  Niazy proof-fit bundle is small, higher-rate, and uses a 7×512 context. The
  paper values are documented as references; FACETpy-appropriate values (lower
  lr, smaller batch) drive the actual proof-fit run, and the smoke run shrinks
  everything for CPU.

## 3. Summary

The high-severity corrections (drop in-graph ICA → sensor-level network; fix the
frequency loss to a z-scored 1-50 Hz PSD) plus the medium fixes (ReLU CBR,
transposed-conv decoder, normalised equal-weight ensemble, z-score
normalisation) and the low-severity clean-target default make this edition
materially more faithful to Chuang et al. 2022 than the original. Every change
remains configurable so the original behaviour is reproducible, and every
domain-driven deviation (ICA/ICLabel synthesis, hyperparameters) is documented
with rationale rather than silently copied or discarded.
