# DenoiseMamba — paper-accurate edition

A from-scratch rebuild of the `denoise_mamba` model that follows the source
paper far more closely than the original package, while staying compatible with
FACETpy's facet-train factory contract, the deep-learning inference adapter
contract, and CPU-only execution on a laptop.

## Source paper

> Chen, Li, Zheng, Shi. **"DenoiseMamba: An Innovative Approach for EEG Artifact
> Removal Leveraging Mamba and CNN."** *IEEE Journal of Biomedical and Health
> Informatics*, vol. 29, no. 9, Sept. 2025, pp. 6551–6562. (IEEE Xplore
> 11012652, PMID 40408214.)

The original `src/facet/models/denoise_mamba/` package was, by its own
`research_notes.md` admission, reconstructed **without paper access**. It is a
flat residual stack of Mamba‑1 selective‑scan blocks — not the paper's
architecture. This edition rebuilds the model to the paper's specification.

## What changed vs the original

| Area | Original `denoise_mamba` | This edition (paper‑accurate) |
|------|--------------------------|-------------------------------|
| Topology | Flat stack: `Conv1d(1→64,k7)` → N identical ConvSSD blocks at constant width → `LayerNorm` → `1×1 Conv` | **U‑shaped** encoder/decoder: SignalEmbedding → 3×[ConvSSD + downsample] (64→128→256… , L→L/2→L/4) → bottleneck → 3×[upsample + skip‑concat + ConvSSD] → projection head (Fig. 1) |
| ConvSSD block | Single pre‑norm residual wrapping one Mamba‑1 block | **Channel split** into a local conv branch (`Conv3‑BN‑PReLU‑Conv3‑BN‑PReLU‑Dropout`) and a global SSD branch, fused with learnable `r1, r2`, refined by a depthwise‑separable conv (Fig. 2, Eqs. 6–10) |
| SSD branch | One selective‑scan path + SiLU gate | `y = Linear(RMSNorm(y_a1 + y_a2 + SiLU(Linear(x))))` with two parallel `SSD(DWConv(Linear(x)))` axes + a no‑SSD SiLU skip (Eqs. 6–9) |
| State‑space op | **Mamba‑1** (S6): input‑dependent matrix `A`, sequential per‑timestep Python loop | **Mamba‑2 / SSD** (Eq. 5): scalar‑per‑head `A`, chunked semiseparable‑matrix scan |
| Feature fusion | None | Two learnable scalars `r1, r2` scaling the conv/SSD halves before concat (Eq. 10) |
| Signal embedding | Single `Conv1d(1→64, k7)` | `Conv3 → PReLU → BatchNorm → Conv3` (Fig. 1 inset) |
| Projection head | `LayerNorm → 1×1 Conv` | `GAP → Linear → PReLU → LayerNorm → Linear` summary fused with a length‑preserving conv (Fig. 1 inset) |
| Activations | SiLU everywhere | **PReLU** in conv/embedding/head, **SiLU** only on the SSD skip (Eq. 8) |
| Normalisation | LayerNorm only; demean‑only data | **RMSNorm** (SSD), **BatchNorm** (conv), **LayerNorm** (head); per‑epoch **z‑score** standardisation (Fig. 6) |
| Prediction target | Artifact (`artifact_center`), subtracted | **Clean** signal by default (paper target, MSE loss); artifact target still supported for FACETpy parity |

## Deliberate, documented deviations

These keep the model sensible for single‑/few‑channel EEG‑fMRI gradient‑artifact
removal and CPU‑cheap, rather than blindly copying paper tricks that need data
or hardware we do not have. Full rationale in
[`documentation/paper_accuracy_review.md`](documentation/paper_accuracy_review.md).

1. **Spatial scan → bidirectional temporal scan.** The paper's spatiotemporal
   scan (Fig. 3) scans a 2D multi‑channel EEG map along both a spatial (channel)
   and a temporal axis. FACETpy corrects gradient artifacts *channel‑wise* (one
   channel per forward), so there is no spatial axis to scan. The two SSD axes
   are realised as **forward‑time and reverse‑time** scans (bidirectional
   temporal SSD), preserving the dual‑path structure while remaining meaningful
   for a single channel.
2. **Portable PyTorch SSD, not the CUDA kernel.** The Mamba‑2 SSD is a pure
   PyTorch chunked scalar‑`A` scan, not the `mamba-ssm` CUDA kernel, so it runs
   on CPU / Apple MPS. At the short epoch lengths used here (L ≤ 512) this is
   cheap.
3. **Length‑preserving projection head.** The paper's GAP collapses time into a
   global summary; we add the pooled summary back as a per‑channel bias to a
   length‑preserving conv path so the output stays `(B, 1, L)` for FACETpy's
   sample‑aligned subtraction.
4. **Smoke‑scaled hyper‑parameters.** Paper defaults (base 64, 3 stages, d_state
   16, dropout 0.2, batch 128, 50 epochs, AdamW + ReduceLROnPlateau) are the
   factory defaults; the smoke YAML/test shrink every dimension for a
   few‑second CPU run.

## Factory contract (facet-train)

`training.py` exposes:

- `build_model(**kwargs) → nn.Module` — accepts all facet‑train‑injected kwargs
  (`n_channels`, `chunk_size`, `sfreq`, `target_type`, `input_shape`,
  `target_shape`, `epoch_samples`, …) via `**_`; `epoch_samples` is resolved from
  `input_shape` when absent. Paper defaults: `base_channels=64, n_stages=3,
  d_state=16, dropout=0.2`.
- `build_loss(name='mse', **kwargs) → nn.Module` — MSE (paper), plus L1/SmoothL1.
- `build_dataset(path=None, context_path=None, max_examples=None,
  target_type='clean', normalize='zscore', **_)` — loads `noisy_center`,
  `clean_center`, `artifact_center` from the Niazy proof‑fit NPZ and serves
  channel‑wise `(1, L)` items. Exposes `__len__`, `__getitem__`,
  `train_val_split`, and the required attributes (`n_channels`, `chunk_size`,
  `input_shape`, `target_shape`, `n_chunks`, `target_type`, `trigger_aligned`,
  `sfreq`, `epoch_samples`).

## Inference (processor.py)

`PaperAccurateDenoiseMambaAdapter` (CLEAN output by default; ARTIFACT supported)
plus `@register_processor DenoiseMambaPaperAccurateCorrection` with the
**globally unique** name `denoise_mamba_paper_accurate_correction` (the original
keeps `denoise_mamba_correction`). For a CLEAN model the adapter restores the
per‑chunk input DC level so the cleaned signal stays on‑scale before
`DeepLearningCorrection` writes it back.

## Files

- `training.py` — paper‑faithful architecture + factories.
- `processor.py` — adapter + unique‑named correction processor.
- `training_niazy_proof_fit_smoke.yaml` — illustrative CPU smoke config (NOT run
  by the test).
- `documentation/paper_accuracy_review.md` — discrepancy table + EEG‑fMRI
  applicability assessment.
- `tests/models/denoise_mamba_paper_accurate_edition/test_training_smoke.py` —
  CPU smoke test (forward shape, loss‑decrease, dataset shapes/attrs/split,
  registry uniqueness).
