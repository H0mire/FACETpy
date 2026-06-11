# Paper-Accuracy Review — Demucs (Time-Domain)

Source paper: Défossez, Usunier, Bottou, Bach (2019),
*Music Source Separation in the Waveform Domain*,
[arXiv:1911.13254](https://arxiv.org/abs/1911.13254).

This document records the discrepancies between the **original**
`facet.models.demucs` edition and the source paper, what this **paper-accurate
edition** changed, and a per-method assessment of whether each paper technique
makes sense for single-/few-channel EEG-fMRI gradient-artifact removal.

## 1. Discrepancy table

| Aspect | Paper specifies | Original impl | Severity | Status in this edition |
|--------|-----------------|---------------|----------|------------------------|
| **Arbitrary-length handling** (valid_length pad + center-trim) | Reference pads input to a *valid length* (multiple of `stride^depth`) and center-trims skips / output so any length works; output length == input length. | `forward` summed skips with no length alignment; worked only when `total_samples` divisible by `stride^depth`. **Verified crash**: `depth=2`, length 700 → `RuntimeError` (172 vs 175). | **High** | **Fixed.** `valid_length(length)` rounds up to the next multiple of `stride^depth`, `F.pad`s the input, center-crops each encoder skip to the decoder feature length, and center-crops the final output back to the original length. Verified for divisible *and* non-divisible lengths in the smoke test. |
| **2x resampling trick** (Sec 4.1) | Upsample input ×2 (sinc/Kaiser), run net, downsample ×2; inside the end-to-end loss. Ablation: 6.03 → 6.28 SDR. | Not implemented. | **Medium** | **Added.** Constructor flag `resample` (default 2; 1 disables). Dependency-free sinc/Kaiser FIR via `F.conv1d`, TorchScript-traceable. `valid_length` is recomputed on the upsampled signal. Disabled in the smoke config to stay fast. |
| **Depth L** (6 in paper, 4 here) | `L=6`, 64→128→256→512→1024→2048 for ~11 s of 44.1 kHz audio. | `depth=4` hard-coded, justified in comments. | **Low** (deliberate adaptation) | **Made principled.** `auto_depth` caps depth at the max keeping the (optionally upsampled) bottleneck ≥ 1 sample. `initial_channels=64` (paper best) kept. Documented as a necessary EEG adaptation. |
| **Init rescaling zeroes biases** | Sec 4.3 rescales weights only (`α=std(w)/a`, `w'=w/√α`, `a=0.1`); silent on biases. | Additionally `bias.zero_()` on every conv. | **Low** | **Fixed.** Weight-rescale formula kept; `bias.zero_()` removed (PyTorch default bias init retained). |
| **Test-time shift trick** (Sec 4.4) | Average predictions over `S=10` random circular time-shifts (inverse-shifted); +0.3 SDR. | Single forward pass per channel/center epoch. | **Medium** | **Added.** Processor `n_shifts` param (default 1). `>1` rolls the concatenated context, runs the model, inverse-rolls and averages before slicing the center epoch. Evenly-spaced (reproducible) offsets. |
| **Reconstruction loss** (L1, summed over sources) | L1 over waveform samples, summed across S sources (Eq. 2); L1 default over MSE. | `L1Loss()` default (correct); S=1. | **Low** | **Unchanged (already faithful).** L1 default; `mse`/`smooth_l1`/`huber` ablation options kept. S=1 documented as the single-artifact EEG adaptation. |
| **Training augmentations** (pitch/tempo, remix, gain, channel-swap) | Heavy augmentation incl. ±1/±2 semitone pitch, tempo [0.88,1.12], source remixing, channel swap, gain [0.25,1.25]. | None (demean only). | **Low** | **Mostly omitted (documented).** Music/stereo/multi-source augmentations do not transfer. Added an opt-in FACETpy substitute: small random gain + small circular time-shift (off by default, off for the proof fit). |
| **Decoder conv kernel size** (K=3 vs K=1) | Decoder first conv `K=3` + GLU; K=1 worse (6.11 vs 6.28). | `Conv1d(K=3, padding=1)` + GLU. | **Low** | **Unchanged (verified match).** K=3 kept. |

## 2. EEG-fMRI applicability assessment

| Paper method | Keep for EEG-fMRI? | Rationale |
|--------------|--------------------|-----------|
| Waveform-domain U-Net, strided `Conv1d(K=8,S=4)` + GLU | **Yes** | Directly applicable to 1D EEG; the gradient artifact is a periodic broadband waveform with multi-scale structure. Cheap on CPU at reduced width/depth. |
| Bidirectional 2-layer LSTM bottleneck + Linear | **Yes** | The single most important component per the paper's ablation (−0.88 SDR without it). The artifact has long-range periodic temporal dependencies (slice/volume repetition) across the multi-epoch context — exactly what the BiLSTM models. |
| Summed U-Net skips + valid-length pad/center-trim | **Yes** | Skip sums preserve high-frequency artifact detail through downsampling. The pad/center-trim machinery is **essential** so the net is length-agnostic for variable `epoch_samples` (fixes the verified crash). |
| Init weight rescaling (`a=0.1`), no batch norm | **Yes** | Faithful, cheap, stabilises training without BN. Kept; bias-zeroing dropped to match the paper. |
| GLU after 1×1 (encoder) / K=3 (decoder) convs | **Yes** | Ablation shows GLU beats ReLU there. Already correct; kept. |
| 2× input upsample / output downsample (resampling trick) | **Yes** | Applies to any waveform, small real gain, and buys one extra usable U-Net level on the short EEG context. Dependency-free FIR, opt-in, off in smoke. |
| Test-time shift trick (average over S shifts) | **Yes** | The model's only inference-time trick; transfers directly to EEG via `np.roll` + inverse. Added as optional `n_shifts`; small values keep CPU cost low. |
| L1 reconstruction loss (vs L2) | **Yes** | Paper default, robust to outliers. Kept as default with mse/smooth_l1 options. |
| Pitch/tempo shift, source remixing, channel swap, gain aug | **No** (mostly) | Pitch/tempo and 4-instrument remixing are music/multi-source specific; channel swap assumes stereo. None map to single-channel single-source gradient artifact. Omitted and documented; only a cheap gain + small time-shift substitute is offered (off by default). |
| `L=6`, `C_L=2048`, 11 s 44.1 kHz stereo, ~1 GB + DiffQ quant. | **No** | Sized for ~485 k-sample stereo audio; the EEG context is 3584 samples single-channel. Use auto-capped depth (~4), `C_1=64` (full) / 8 (smoke), single channel/source. Quantization irrelevant at this scale. Documented necessary downscaling. |
| Stereo I/O (`C_0=2`, S×2 channels out) | **No** | EEG is processed channel-wise (`in_channels=1`) with a single artifact source (S=1) to decouple the checkpoint from EEG channel count. Documented adaptation; consistent with the sibling `cascaded_*_dae` models. |

## 3. Net effect

The paper-accurate edition fixes the one **high-severity correctness bug**
(length-agnostic forward), adds the two **medium-severity** paper techniques
that transfer cleanly to EEG (the 2× resampling trick and the test-time shift
trick), and corrects the **low-severity** init-bias deviation — while keeping
every documented EEG adaptation (channel-wise single-source I/O, reduced depth,
no music augmentation) that the paper's tricks cannot sensibly cover. All
additions are CPU-cheap and the smoke test runs in a couple of seconds.
