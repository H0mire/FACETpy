# DHCT-GAN v2 — Paper-Accuracy Review

**Source paper:** Cai, Y., Meng, Z. & Huang, J. *DHCT-GAN: Improving EEG Signal
Quality with a Dual-Branch Hybrid CNN-Transformer Network.* MDPI **Sensors**
25(1), 231 (2025). https://doi.org/10.3390/s25010231

This document records (1) the discrepancies between the *original* FACETpy
`dhct_gan_v2` and the paper, (2) which were fixed in this paper-accurate
edition, and (3) an EEG-fMRI applicability assessment for each paper technique.

## 1. Discrepancy table

| Aspect | Paper specifies | Original impl | Severity | Status in this edition |
|--------|-----------------|---------------|----------|------------------------|
| Reconstruction loss | `L_mse = mean((Y − G(X))²)` MSE for every branch (Eq. 10) | `nn.L1Loss()` for both recon + consistency | medium | **Fixed**: MSE is the default (`recon="mse"`); L1 kept as documented option |
| Adversarial / discriminator loss | LSGAN: `L_adv = mean((D(G(X))−1)²)`, `L_D = 0.5·mean(D(G(X))²) + 0.5·mean((D(Y)−1)²)` (Eq. 12-13) | vanilla `binary_cross_entropy_with_logits` for both | high | **Fixed**: LSGAN least-squares; discriminators output raw scores |
| Feature-matching loss | `L_feat = mean((φ(Y) − φ(G(X)))²)` on intermediate disc features, weight λ1 (Eq. 11) | absent | high | **Fixed**: each discriminator taps a mid-block feature; MSE feature-matching weighted by `lambda_feat` |
| Number of discriminators | three (D1 clean / D2 noise / D3 fused), shared architecture (Algorithm 1) | one PatchGAN on the artifact head | medium | **Fixed**: `D_clean`, `D_noise`, `D_fused`, one shared private Adam |
| Discriminator architecture | M=8 strided convs (k=3,s=2,p=1) + BN + activation, channels 64,64,128,128,256,256,512,512, + FC scalar | 4-layer PatchGAN (k=4,s=2), 16→32→64→128, no FC | medium | **Fixed**: configurable strided-conv blocks with paper progression (scaled for CPU), global pool + FC scalar, feature tap |
| Gating | two independent tanh FC gating nets → `mask1`, `mask2`; `Y_pre = mask1⊙Y1 + mask2⊙(X−Y2)` (Eq. 4-5) | single Conv-Tanh-Conv-Sigmoid gate `g`, complementary `g`/`1−g` | medium | **Fixed**: two independent tanh gating networks (conv-for-FC), non-complementary masks |
| LSA block partition + LGTB repeat | LSA splits sequence into 8 fixed equal blocks; LGTB (LSA→FF→GSA→FF) repeated x5 per stage | sliding `window_size`; ONE LGTB/stage; single trailing FF | medium | **Fixed**: configurable equal-block split (default 8), FF after both LSA and GSA, `lgtb_depth` repeats (default 2; paper 5 documented; smoke 1) |
| Generator branching | two fully-duplicated parallel branch generators (separate encoders) | shared encoder + dual decoders | low | **Deviation (kept)**: shared encoder halves params/CPU cost; documented |
| Output exposed | fused clean signal `Y_pre` is the denoised output | `forward` returns artifact head only | low | **Deviation (kept)**: returns artifact `noisy_center − Y_pre` for subtractive correction, but derived from the full fused+gating path |
| Generator optimizer betas | gen Adam (0.5, 0.9); disc Adam (0.9, 0.999); lr 1e-3 / 1e-4 | gen optimizer owned by facet-train (AdamW defaults); disc (0.9, 0.999) ✓ | low | **Documented**: recommended `betas: [0.5, 0.9]` in the smoke YAML; disc betas already match |
| Segment length / depth / widths | 1024-sample, 5 stages, 64..1024 channels | 512-sample, depth 4, 16..128 | low | **Deviation (kept)**: reduced for the tiny proof-fit NPZ + CPU; all kwargs, full-scale possible on GPU |
| Positional encoding / masking / diffusion / graph | none used | none | low | **Matches**: no change required |
| Input conditioning | single 2 s segment | 7-epoch trigger-aligned context (v1→v2 diff) | n/a | **Deviation (kept)**: gradient artifact is TR-periodic; cross-epoch context is essential and beneficial |
| Data domain | EMG/EOG/ECG at −7..2 dB SNR | Niazy fMRI gradient-artifact proof-fit NPZ | n/a | **Deviation (kept)**: different artifact class; FACETpy target domain |

## 2. EEG-fMRI applicability assessment

| Paper method | Keep for EEG-fMRI? | Rationale |
|--------------|--------------------|-----------|
| LSGAN adversarial + discriminator loss (Eq. 12-13) | **Yes** | Cheap; more stable than vanilla BCE on continuous 1D regression targets; directly addresses the documented v1/v2 GAN-instability hypothesis. |
| Feature-matching loss `L_feat` (Eq. 11) | **Yes** | One of the three core innovations; one extra MSE on a discriminator intermediate; stabilizes training and preserves waveform detail; fully applicable to single-channel gradient artifacts. |
| Three discriminators (clean/noise/fused) | **Yes (scaled)** | Multi-discriminator stability is exactly what v2 needs. All three inside the loss with one private optimizer to keep the single-optimizer facet-train contract. Dims shrunk for CPU smoke. |
| Two independent tanh gating networks (Eq. 4-5) | **Yes** | Cheap; independent (non-complementary) masks give the network freedom to reconcile clean/noise branches. Implemented as 1D convs (variable-length epochs); conv-for-FC documented. |
| MSE reconstruction (Eq. 10) | **Yes** | Faithful default; cheap. L1 retained only as a documented spike-robust alternative. |
| LSA 8-block split + GSA + per-attention FF, LGTB x5 | **Yes (configurable)** | Local/global attention applicable; 8-block split is the faithful local design. Block-count and `lgtb_depth` configurable; x5 is the dominant CPU cost, so default reduced (2) with the paper value documented; smoke uses 1. |
| Two fully-duplicated parallel branch generators | **No** | Doubling the encoder ~doubles params/CPU for little benefit on a tiny single-channel proof-fit dataset. Shared-encoder + dual-decoder adaptation kept; full duplication is a GPU-only option. |
| Generator emits fused clean `Y_pre` | **No** | FACETpy correction is subtractive (subtract predicted gradient artifact), more robust for gradient artifacts. Return artifact `noisy_center − Y_pre` derived from the full fused+gating machinery. |
| 1024-sample / 5-stage / 64-1024-channel full-scale arch | **No** | Proof-fit epochs are 512-sample resampled trigger-to-trigger; full-scale dims are CPU-prohibitive and unnecessary. Reduced dims kept; exposed as kwargs for optional GPU runs. |
| Physiological-artifact data (EMG/EOG/ECG, −7..2 dB) | **No** | Different artifact class; FACETpy targets fMRI gradient artifacts. Retain the FACETpy NPZ context dataset; data-domain difference documented. |
| 7-epoch trigger-aligned context input | **Yes** (not from the paper) | The key FACETpy adaptation: gradient artifacts are strongly TR-periodic, so cross-epoch context is essential. Retain the v2 stem-mixing of the 7-epoch context. |

## 3. Documented deviations (summary)

1. 7-epoch trigger-aligned context input (FACETpy conditioning; paper is single-segment).
2. Shared encoder + dual decoders instead of two duplicated branch generators (CPU/parameter economy).
3. Reduced architecture dims (512-sample, depth 4, 16..128 channels) for the tiny proof-fit NPZ and CPU; all exposed as kwargs.
4. Subtractive artifact output (`noisy_center − fused_clean`) instead of emitting the clean signal `Y_pre`.
5. Conv-for-FC gating networks (variable-length resampled epochs).
6. fMRI gradient-artifact data domain instead of physiological artifacts.
7. Clean-branch discriminator/recon terms use the fused clean estimate as a proxy, since the facet-train `(pred, target)` loss contract exports the artifact (and hence the fused clean) rather than the explicit `Y1` head; this keeps the `(B,3,T)` target packing intact while still realising LSGAN + feature-matching across all three discriminators.
8. Generator Adam betas (0.5, 0.9) recommended via the smoke YAML; if facet-train cannot override the optimizer betas, its AdamW default is a documented deviation (the discriminator betas (0.9, 0.999) are set internally and match the paper).
