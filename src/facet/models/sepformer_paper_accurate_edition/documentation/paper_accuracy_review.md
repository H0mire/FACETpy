# SepFormer paper-accuracy review

**Paper:** C. Subakan, M. Ravanelli, S. Cornell, M. Bronzi, J. Zhong,
*Attention is All You Need in Speech Separation*, ICASSP 2021
(arXiv:2010.13154). Transformer fundamentals from A. Vaswani et al.,
*Attention Is All You Need*, NeurIPS 2017.

This document compares the **paper-accurate edition**
(`facet.models.sepformer_paper_accurate_edition`) against the **original**
FACETpy SepFormer (`facet.models.sepformer`) and assesses each paper technique
for EEG-fMRI gradient-artifact removal.

## Discrepancy table

| # | Aspect | Paper specifies | Original FACETpy | This edition | Severity (orig.) |
|---|--------|-----------------|------------------|--------------|------------------|
| 1 | Masking-net bracketing stages | Fig. 2: `LayerNorm+Linear` before chunking; `PReLU+Linear` after the SepFormer block; `FeedForward+ReLU` mask generator after OverlapAdd | Only `LayerNorm+Linear` pre-chunk and a single `Conv1d+ReLU` mask head; `PReLU+Linear` and the `FFW+ReLU` stage missing/collapsed | **Fixed.** All three stages reproduced in exact Fig. 2 order (`pre_chunk_norm/linear`, `post_block_prelu/linear` before overlap-add, `mask_ffn` after) | medium |
| 2 | Whole-stack residual (Eq. 6) | `f(z) = g^K(z + e) + z`: PE added once, K layers, then residual from the input across the whole stack | No outer residual; only per-layer pre-norm residuals | **Fixed.** `_SBTransformerStack` saves the pre-PE input and adds it back (`whole_stack_residual=True`) | medium |
| 3 | Intra/Inter depth K | `K=8` for both IntraT and InterT; ablation: 8 > 3, IntraT depth matters most | `intra=4, inter=4` | **Improved + documented.** Default `intra=8, inter=4` (honours the ablation); fully configurable; smoke `1/1` | low |
| 4 | Encoder/decoder filters, d_model, d_ffn, heads | 256 filters, `d_model=256`, `d_ffn=1024`, 8 heads (~26M params) | 128 / 128 / 256 / 4 heads | **Documented deviation.** Compact `128 / 128 / 512 / 8` default for the small proof-fit set + CPU; full config exposed as an option | low |
| 5 | Multiplicative source mask | `Ns` soft ReLU masks, `m_k * h`, decode per source | Single ReLU mask (`Ns=1`), `mask * h`, one decode | **Kept (faithful single-source reduction).** `Ns=1` ReLU mask; documented as the artifact-removal specialization | low |
| 6 | Loss: SI-SNR + PIT + 30 dB clip | Scale-invariant SI-SNR, utterance-level PIT, clipped at 30 dB | Default plain MSE; SI-SNR / SI-SNR+MSE without the 30 dB clip; no PIT | **Fixed.** Default = negative SI-SNR with `clamp(si_snr, max=30)`; PIT documented as N/A (single target); MSE/SI-SNR+MSE still selectable | medium |
| 7 | Feature-axis chunking, `C`, 50% overlap | Chunk the encoded FEATURE sequence into `C=250` chunks, 50% overlap | Feature-axis chunking, `chunk_size=64`, 50% hop (correct mechanism) | **Kept + corrected framing.** Default `chunk_size` tied to feature length (`~len//8`, clamped `[16,64]`, even); `C=250` documented as speech-scale; "epoch=chunk" framing corrected | low |
| 8 | Positional encoding placement | Sinusoidal PE added once at the input of each Intra/Inter transformer | PE added once per stack (faithful) | **Kept.** PE feeds the Eq. 6 residual using the pre-PE input | low |
| 9 | Attention scaling / Vaswani math | Scaled dot-product, `1/sqrt(d_head)`, softmax, head split/concat | Hand-written, faithful (trace-safe) | **Kept.** Functionally identical to `nn.MultiheadAttention` | low |
| 10 | FFN inner activation | ReLU (paper default); SpeechBrain also offers GELU | ReLU | **Kept.** ReLU default; GELU exposed via `ffn_activation` | low |
| 11 | Optimizer / LR schedule / grad-clip 5 | Adam `lr=15e-5` + warmup, halve after epoch 65, grad-clip 5, batch 1, 200 epochs, AMP, Dynamic Mixing | AdamW `lr=5e-4`, grad-clip 1.0, batch 64, 50 epochs, no warmup/halving/DM | **Documented deviation.** FACET-train AdamW + early-stopping kept; grad-clip set to paper `5.0` in YAML; warmup/anneal/DM documented as omitted | low |
| 12 | `skip_around_intra` residual | SpeechBrain default `True` | `True` | **Kept.** `True` | low |

## EEG-fMRI applicability assessment

| Paper method | Keep for EEG-fMRI? | Rationale |
|--------------|--------------------|-----------|
| Conv1d encoder + ConvTranspose1d decoder (learned STFT-like front-end, kernel 16 / stride 8, ReLU) | **Yes** | Works directly on a single-channel EEG waveform, CPU-cheap. The kernel/stride 50% overlap roughly matches the few-ms gradient-artifact morphology after upsampling. Encoder filters kept reduced (128) for the small dataset. |
| Dual-path chunking (intra + inter attention) on the FEATURE sequence, 50% overlap | **Yes** | The core reason to use SepFormer here. Intra-chunk captures local artifact morphology, inter-chunk captures slower drift. Must chunk the feature axis (not the epoch axis); the new edition fixes that framing. With ~hundreds of feature frames and `C ~ 16-64`, both attention matrices stay tiny / CPU-cheap. |
| Pre-norm layers + whole-stack residual (Eq. 6) + `PReLU+Linear` / `FFW+ReLU` bracketing | **Yes** | Generic sequence-model components, domain-independent. Adding the missing outer residual and bracketing stages improves faithfulness and gradient flow at negligible parameter cost. Depth kept modest (8/4 full, 1/1 smoke). |
| Sinusoidal positional encoding | **Yes** | Essentially free on CPU; the paper's ablation shows PE helps (+~0.7 dB). The artifact is trigger-locked and periodic, so intra-chunk ordering is meaningful. |
| Multiplicative soft ReLU mask × encoder features, decoded per source | **Yes (`Ns=1`)** | Artifact removal is the degenerate single-source case of separation: one mask, multiply with the encoded mixture, decode the artifact. The mechanism transfers cleanly. |
| SI-SNR + utterance-level PIT + 30 dB clip | **Partial** | Keep SI-SNR with the 30 dB clip as the faithful default, but **drop PIT** (one target, no permutation ambiguity). SI-SNR is scale-invariant, which is risky for artifact subtraction where absolute amplitude matters, so MSE / SI-SNR+MSE are offered to anchor amplitude. |
| Dynamic Mixing (remix + speed perturbation 95-105%) | **No** | Speech-specific. One artifact morphology per recording, not a pool of speakers; speed perturbation would corrupt the trigger-locked timing that defines the gradient artifact. (A FACET substitute — small amplitude jitter / additive EEG-background noise — is out of scope for the proof-fit smoke.) |
| Warmup + halve-after-65 schedule, 200 epochs, batch 1, AMP, grad-clip 5 | **No (except grad-clip 5)** | Tuned to a 30 h speech corpus on V100s; not transferable to an ~hundreds-of-example proof-fit set on CPU. Keep FACET-train's AdamW + early stopping; adopt only grad-clip `5.0`. |
| Full-scale capacity (`d_model=256`, `d_ffn=1024`, 8 heads, `K=8/8`, ~26M params) | **No (compact default)** | Full scale would overfit the tiny dataset immediately and is too heavy for CPU smoke. Compact default (`d_model=128`, `d_ffn=512`, 4-8 heads); full config exposed as an option. The reduction is dataset-size-driven, not EEG-domain-driven, and is documented. |

## Summary of documented deviations

1. **Capacity reduction** (`d_model=128`, `d_ffn=512`, `K=8/4`) — dataset size + CPU; full config exposed.
2. **`Ns=1` single mask + no PIT** — artifact removal is single-source separation.
3. **Dynamic Mixing excluded** — speech-specific; would corrupt trigger-locked timing.
4. **Optimizer schedule** — FACET-train AdamW + early stopping retained; only grad-clip `5.0` adopted; warmup/anneal/200-epoch/AMP omitted.
5. **`C=250` not used** — speech-scale; `chunk_size` tied to the EEG feature length.
6. **SI-SNR amplitude caveat** — scale-invariant loss kept as the faithful default, with MSE / SI-SNR+MSE offered because subtraction depends on absolute amplitude.
