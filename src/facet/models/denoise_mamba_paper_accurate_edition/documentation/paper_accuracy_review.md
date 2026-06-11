# DenoiseMamba — paper accuracy review

Reference: Chen, Li, Zheng, Shi, "DenoiseMamba: An Innovative Approach for EEG
Artifact Removal Leveraging Mamba and CNN", *IEEE J. Biomed. Health Inform.*,
vol. 29, no. 9, Sept. 2025, pp. 6551–6562 (IEEE Xplore 11012652, PMID
40408214).

This document records, per discrepancy, what the paper specifies, what the
**original** `denoise_mamba` package did, and what this **paper‑accurate**
edition does. It closes with an EEG‑fMRI applicability assessment for each paper
method.

## Discrepancy table

| # | Aspect | Paper specifies | Original impl | This edition | Severity (orig.) |
|---|--------|-----------------|---------------|--------------|------------------|
| 1 | Overall topology | U‑shaped: SignalEmbedding → 3×[ConvSSD+DownSample] (64×L,128×L/2,256×L/4,512×L/8) → 3×[ConvSSD+UpSample] with skip‑concat → projection head (Fig. 1) | Flat residual stack at constant width/length; no down/up, no skips, no pyramid | **Fixed.** Parametric U‑Net: SignalEmbedding → `n_stages` encoder [ConvSSD + stride‑2 conv, doubling channels] → bottleneck → `n_stages` decoder [transpose‑conv upsample + skip‑concat + ConvSSD] → projection head | High |
| 2 | ConvSSD block structure | Channel split into halves; upper local branch `Conv3‑BN‑PReLU‑Conv3‑BN‑PReLU‑Dropout`; lower SSD branch; concat then DW‑separable conv (Fig. 2) | Single pre‑norm residual wrapping one Mamba‑1 block; no split, no conv branch, no DW‑separable fusion | **Fixed.** `ConvSSDBlock` channel‑splits, runs the conv branch + SSD branch, scales by `r1, r2`, concatenates, refines with a depthwise + pointwise conv | High |
| 3 | SSD branch composition | `y_s = SSD_spatial(DWConv(Linear(x)))`, `y_t = SSD_temporal(DWConv(Linear(x)))`, `y_skip = SiLU(Linear(x))`, `y = Linear(RMSNorm(y_s+y_t+y_skip))` (Eqs. 6–9) | Single SSM path + SiLU gate; one direction, no RMSNorm, no additive skip‑before‑norm | **Fixed.** `SSDBranch` runs two parallel `SSD(DWConv(Linear(x)))` axes + a SiLU(Linear) skip, summed, RMSNorm’d, projected. The two axes are forward/reverse time (see deviation D1) | High |
| 4 | State‑space mechanism | Mamba‑2 / SSD (Eq. 5: `y = SSD(A,B,C)x = Mx`, semiseparable matrix) | Mamba‑1 (S6) sequential selective scan with input‑dependent matrix `A` and a Python time loop | **Fixed.** `SSDLayer` uses scalar‑per‑head `A` (Mamba‑2 head structure) and a **chunked semiseparable** scan (intra‑chunk dense decay matrix + inter‑chunk state recurrence). Portable PyTorch, not the CUDA kernel (deviation D2) | Medium |
| 5 | Feature fusion (Eq. 10) | `Output = Concat(input_Conv·r1, input_SSD·r2)` with two learnable scalars | None | **Fixed.** `nn.Parameter` `r1, r2` (init 1.0) scale the conv/SSD halves before concat | Medium |
| 6 | Signal embedding | `Conv3 → PReLU → BatchNorm → Conv3` → 64 ch (Fig. 1 inset) | Single `Conv1d(1→64, k7)`; no PReLU/BN/second conv; unjustified k7 | **Fixed.** `SignalEmbedding` = `Conv3 → PReLU → BatchNorm1d → Conv3` (kernel 3 per paper) | Medium |
| 7 | Projection head | `GAP → Linear → PReLU → LayerNorm → Linear` + length restore (Fig. 1 inset) | `LayerNorm → 1×1 Conv` | **Fixed.** `ProjectionHead` = GAP→Linear→PReLU→LayerNorm→Linear summary, injected as a per‑channel bias to a length‑preserving 1×1 conv (deviation D3) | Low |
| 8 | Activation function | PReLU in conv/embedding/head; SiLU only on SSD skip (Eq. 8) | SiLU everywhere; no PReLU | **Fixed.** PReLU in conv/embedding/head paths; SiLU only on the SSD skip | Low |
| 9 | Prediction target | Clean (denoised) EEG; MSE to clean ground truth | Predicts artifact (`artifact_center`), subtracts it | **Fixed (dual).** Default `target_type='clean'` (output_type CLEAN, `DeepLearningPrediction(clean_data=…)`); `target_type='artifact'` supported for FACETpy parity | Medium |
| 10 | Normalisation regime | RMSNorm (SSD), BatchNorm (conv), LayerNorm (head); z‑score standardised data (Fig. 6) | LayerNorm only; demean‑only data | **Fixed.** All three norms present in their paper locations; dataset default `normalize='zscore'` (per‑epoch), with `demean`/`none` fallbacks | Medium |
| 11 | Spatiotemporal (2D) scan | Spatial‑first + temporal‑first scans over the multi‑channel map (Fig. 3) | Single temporal scan; no spatial axis | **Adapted (D1).** Forward‑time + reverse‑time bidirectional temporal SSD; documented single‑channel substitution | Low |
| 12 | Optimizer/schedule/dropout | AdamW lr 1e‑3 + weight decay, ReduceLROnPlateau(0.5, patience 3), dropout 0.2, batch 128, 50 epochs | AdamW lr 1e‑3, wd 1e‑4, grad‑clip 1.0, dropout 0.1, batch 64, 60 epochs; no scheduler reflected | **Documented.** Full‑run YAML uses dropout 0.2, AdamW + weight decay; ReduceLROnPlateau(0.5, patience 3) documented as the intended scheduler (configure if facet‑train supports it). Batch/epochs scaled for smoke | Low |

## Documented deviations (with rationale)

- **D1 — Spatial scan → bidirectional temporal scan.** The paper's spatial scan
  needs many simultaneous channels formed into a 2D map. FACETpy's gradient‑
  artifact correction is **channel‑wise** (a single channel per forward), so a
  spatial axis does not exist for us. We keep the *dual‑path* structure by
  scanning the single channel **forward in time** (axis 1) and **reverse in
  time** (axis 2). This gives the SSD bidirectional temporal context — useful
  for the symmetric slice‑readout structure of gradient artifacts — without
  inventing a channel dimension that the pipeline does not provide. The code
  comments in `SSDBranch.forward` mark this substitution explicitly.

- **D2 — Portable PyTorch SSD, not the CUDA `mamba-ssm` kernel.** The faithful
  mechanism is the Mamba‑2 SSD (scalar‑`A`, semiseparable matrix). We implement
  it as a chunked PyTorch scan: an **intra‑chunk** dense lower‑triangular decay
  matrix (the diagonal blocks of the semiseparable form) plus an **inter‑chunk**
  state recurrence carrying the running state across chunk boundaries. This is
  mathematically the SSD form, is matmul‑friendly, and runs on CPU / Apple MPS.
  No CUDA dependency exists in the deployment (laptop‑only), so the portable
  implementation is the correct engineering choice. At L ≤ 512 it is cheap.

- **D3 — Length‑preserving projection head.** The paper's GAP collapses the time
  axis into a global summary; a verbatim head would not return a per‑sample
  signal. FACETpy subtracts/replaces the signal **per sample**, so the output
  must stay `(B, 1, L)`. We compute the GAP→Linear→PReLU→LayerNorm→Linear
  summary and add it back as a per‑channel additive bias to a length‑preserving
  conv path, preserving both the paper's head computation and a sample‑aligned
  output.

- **D4 — Smoke scaling.** Paper hyper‑parameters (base 64, 3 stages, d_state 16,
  dropout 0.2, batch 128, 50 epochs) are the factory/full‑run defaults. The
  smoke YAML and pytest smoke shrink every dimension (base 8, 1–2 stages,
  d_state 4, chunk 64, batch 2–4, a few steps, SSD chunk 16) so forward+backward
  is milliseconds on CPU. This is a test‑speed concession, not an architectural
  deviation.

## EEG‑fMRI applicability assessment

| Paper method | Keep for EEG‑fMRI? | Rationale |
|--------------|--------------------|-----------|
| U‑shaped encoder/decoder, 3 down/up stages, skip‑concat | **Yes** | Multi‑scale temporal context fits MR gradient artifacts: fine scale for steep slice‑readout edges, coarse scale for slow TR/volume periodicity. Works on one 512‑sample channel and stays cheap at small width. Depth/width are parametric so smoke uses tiny stages. |
| ConvSSD channel split + dual conv/SSD branch + DW‑separable fusion | **Yes** | Core contribution. Conv branch captures local gradient‑pulse morphology; SSD branch captures long‑range periodicity. Channel split halves per‑branch width → CPU‑cheap. Kept faithfully. |
| Mamba‑2 SSD (semiseparable matrix) layer | **Yes (portable form)** | The SSD formulation is the paper's distinguishing mechanism vs the original Mamba‑1 scan. A pure‑PyTorch chunked scalar‑`A` SSD is portable and cheap at L ≤ 256–512 on CPU. Kept as an SSD layer, not the CUDA kernel (D2). |
| Spatial SSD scan over the channel axis (Fig. 3) | **No (adapted)** | True spatial scanning needs many simultaneous channels formed into a 2D map; FACETpy processes gradient artifacts channel‑wise. Adapted to forward/reverse bidirectional temporal SSD (D1). |
| Learnable `r1/r2` feature fusion (Eq. 10) | **Yes** | Two scalars are nearly free and let the model balance local‑conv vs global‑SSD evidence — useful when gradient artifacts dominate amplitude. Kept. |
| Signal Embedding + Linear Projection head | **Yes** | Standard, cheap stem/head. Kept faithfully; the GAP head is paired with a length‑preserving path so output stays `(B, 1, L)` (D3). |
| Predict clean (denoised) EEG with MSE loss | **Yes (dual support)** | Matches the paper and is valid for EEG‑fMRI. FACETpy's correction subtracts an artifact, so we support both via `target_type`: default `clean` for faithfulness, `artifact` for pipeline parity. |
| Z‑score standardisation of inputs | **Yes** | Variance normalisation stabilises training across channels with very different gradient‑artifact amplitudes. Kept per‑epoch z‑score; demean‑only retained as a fallback toggle for baseline comparison. |
| AdamW + ReduceLROnPlateau, dropout 0.2, batch 128, 50 epochs | **Yes (scaled)** | Sensible recipe. AdamW/scheduler/dropout kept. Batch 128 / 50 epochs are fine on the Niazy bundle; the smoke run shrinks to 1 epoch / batch 2–4. |
