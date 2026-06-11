# Conv-TasNet paper-accuracy review

Source paper: Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing Ideal
Time-Frequency Magnitude Masking for Speech Separation," IEEE/ACM TASLP 27(8):
1256-1266, 2019 (arXiv:1809.07454v3).

Compared against the original `facet.models.conv_tasnet` implementation. This
document records the full discrepancy table, the fixes adopted in this edition,
and the EEG-fMRI applicability assessment for each paper technique.

## Discrepancy table

| # | Aspect | Paper specifies | Original impl | Severity | Action in this edition |
| --- | --- | --- | --- | --- | --- |
| 1 | **Skip-connection width `Sc`** | Fig. 1C / Table I: a distinct skip width `Sc` (e.g. 128) for the skip 1x1-conv `H->Sc`, separate from residual width `B`; mask 1x1-conv maps `Sc->C*N`. | Both residual and skip emit `B`; mask maps `B->C*N`. No `Sc` param; conflated with `B=128`. Coincides with the paper value but is not configurable and breaks silently if `B != 128`. | low | **Fixed.** Added explicit `skip_channels` (Sc); skip 1x1-conv `H->Sc`, mask 1x1-conv `Sc->C*N`. Default `Sc=128`. |
| 2 | **Encoder nonlinearity / best config** | Original TasNet uses ReLU, but Conv-TasNet's BEST published config (Table III, used for all headline results) uses a **LINEAR** encoder + **Sigmoid** mask; encoder bias absent. | ReLU applied unconditionally; not configurable. Corresponds to the lower-scoring "ReLU encoder + Sigmoid" row. | medium | **Fixed.** `encoder_activation` configurable; **defaults to `linear`** with sigmoid mask. ReLU kept as an option. |
| 3 | **Depthwise-separable conv (D-conv + 1x1 pointwise)** | Sec. II-D, eq. 6-7: S-conv = depthwise D-conv immediately followed by a 1x1 pointwise conv before PReLU+Norm. | Only the grouped depthwise conv; channel mixing happens at the residual/skip 1x1-convs at block end. | low | **Documented equivalence (no change).** Mainstream reference repos (asteroid, kaituoxu/Conv-TasNet) fold the pointwise into the residual/skip convs exactly like this; it is paper-faithful enough and CPU-cheap. |
| 4 | **Loss (SI-SNR + uPIT)** | Maximise SI-SNR (eq. 15) with utterance-level PIT to resolve speaker-order ambiguity; scale invariance is the point for speech. | Default plain MSE; `si_sdr_neg` available but not permutation-invariant and not default. No PIT. | low | **Justified deviation (kept MSE default).** Sources are known and ORDERED so PIT is unnecessary; SI-SNR's scale invariance discards meaningful amplitude. MSE/weighted-MSE is correct; SI-SNR offered as ablation only. |
| 5 | **Hyperparameter defaults (N, H, R)** | Best non-causal config: `N=512, L=16, B=128, H=512, Sc=128, P=3, X=8, R=3` (5.1M params). | `N=256, H=256, R=2` (Sc=B=128), sized for ~512-sample EEG epochs. | low | **Kept input-matched defaults; documented paper reference.** All params exposed (incl. new `Sc`). With `X=8` (dilations to `2^7=128`) the receptive field already exceeds `T'≈63` frames, so `R=2` vs `R=3` changes capacity not coverage. |
| 6 | **Decoder count (shared vs per-source)** | Single shared decoder basis `V` (eq. 3/5) applied to each masked rep. | Single shared `ConvTranspose1d` reused for every source. | low | **Correctly faithful (no change).** |
| 7 | **TCN block ordering (`conv -> PReLU -> Norm`)** | Fig. 1C: PReLU and Norm AFTER each of the first 1x1-conv and the D-conv. | `expand -> PReLU -> gLN`, `D-conv -> PReLU -> gLN`. | low | **Correctly faithful (no change).** Preserved. |
| 8 | **Encoder input demeaning** | gLN applied to encoder output `w`; SI-SNR target/estimate zero-meaned. Raw waveform input NOT demeaned. | Dataset/adapter demean the raw segment and targets; model also pre-norms latent (gLN, correct). | low | **Justified EEG deviation (kept).** Raw-input demeaning removes DC offset; applied identically train/inference so consistent. Documented as not-in-paper. |

## EEG-fMRI applicability assessment

| Paper method | Keep for EEG-fMRI? | Rationale |
| --- | --- | --- |
| 1-D learned conv encoder/decoder (`L=16`, stride `L/2`), no STFT | **Yes** | A short learned filterbank on one EEG channel is cheap on CPU and ideal for broadband, phase-coherent gradient artifacts. `L=16`/stride 8 on 512 samples gives `T'≈63` frames. |
| Dilated TCN with exponential dilation `2^x` + skip-sum | **Yes** | Core, cheap. With `X=8` the ~255-frame receptive field already covers the whole 63-frame latent, so `R` can stay small (`R=2`) for CPU speed. |
| Global layer norm (gLN) over channel+time | **Yes** | Correct for non-causal offline use, faithfully implemented, cheap. cLN is unnecessary and costs ~2.5 dB. |
| Sigmoid mask on a LINEAR (non-ReLU) overcomplete encoder | **Yes — adopted as default** | The paper's BEST config; applies cleanly and is free on CPU. The single most impactful faithfulness fix. |
| Distinct skip width `Sc` separate from `B` | **Yes — added** | Trivial to add, makes the model a true superset of the paper, gives the mask predictor a properly-sized skip space. Default `Sc=128`. |
| SI-SNR objective + utterance-level PIT | **No** | Sources are KNOWN and ORDERED — no permutation problem. SI-SNR's scale invariance discards amplitude that IS meaningful for a deterministic AAS dataset where the artifact is subtracted at true scale. MSE/weighted-MSE on ordered sources is the correct deviation; SI-SNR offered only as ablation. |
| Unit-summation mask constraint relaxation (`sum m_i = 1`) | **Yes — and extended** | Paper already relaxes it. For EEG-fMRI we go further with an OPTIONAL source-additivity/consistency penalty (`clean + artifact = noisy`, exactly true here) — a FACETpy-appropriate enhancement beyond the paper. |
| 4-s / 8 kHz mixtures, 100-epoch training, `N=512/H=512/R=3` | **No (scaled)** | Our epochs are ~512 samples (~100 ms). The large dims are sized for long speech; we keep input-length-matched dims (smaller H/R) and document it as a deliberate match, not a budget cut. Full-scale 100-epoch training is for the user's GPU run, not the CPU smoke. |
| Depthwise-separable conv (D-conv + 1x1 pointwise) per block | **Yes** | Keep the parameter-efficient separable structure. The fold-pointwise-into-residual/skip form matches mainstream repos and is CPU-cheap; documented as an accepted S-conv equivalent. |

## Summary of net faithfulness gain

The two structural fixes that move this edition strictly closer to the paper are
(1) the **linear-encoder + sigmoid-mask default** (the headline best config) and
(2) the **explicit `Sc` skip width** (Fig. 1C / Table I). Everything the original
already got right — gLN over channel+time, `conv -> PReLU -> Norm` block ordering,
the single shared decoder, biasless encoder/decoder with 50% stride — is preserved
verbatim. The remaining differences from the paper (MSE instead of SI-SNR+uPIT,
input-matched dims, raw-input demeaning, folded S-conv) are deliberate,
EEG-fMRI-justified deviations rather than oversights, and each is documented above
with its rationale.
