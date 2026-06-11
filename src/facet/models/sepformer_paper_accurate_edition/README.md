# SepFormer (paper-accurate edition)

A more **paper-faithful** re-implementation of the FACETpy SepFormer artifact
predictor. It does **not** modify the original `facet.models.sepformer`
package; both editions coexist and can be imported in the same process.

**Source paper:** C. Subakan, M. Ravanelli, S. Cornell, M. Bronzi, J. Zhong,
"Attention is All You Need in Speech Separation," ICASSP 2021
(arXiv:2010.13154). Canonical transformer math (sinusoidal positional
encoding, scaled dot-product attention, the original post-norm sublayer that
SepFormer converts to pre-norm) follows A. Vaswani et al., "Attention Is All
You Need," NeurIPS 2017.

The task remains: predict the **centre-epoch gradient artifact** from an
odd-length channel-wise epoch context of shape `(B, context_epochs, 1, S)`,
returning `(B, 1, S)`. This preserves the FACET-train factory contract and the
`EpochContextArtifactAdapter` inference contract.

## What changed vs the original (and why)

All of these increase fidelity to Fig. 2 / Sec. 2-3 of the paper:

1. **Masking-network bracketing stages (Fig. 2).** The paper brackets the
   dual-path stack with three distinct learned stages; the original collapsed
   two of them. The new edition reproduces the exact ordering:
   - `LayerNorm + Linear` **before** chunking (`pre_chunk_norm` /
     `pre_chunk_linear`);
   - `PReLU + Linear` **after** the dual-path blocks, applied on the chunked
     feature axis **before** overlap-add (`post_block_prelu` /
     `post_block_linear`);
   - a 2-layer `FeedForward + ReLU` mask generator **after** overlap-add
     (`mask_ffn`), then the mask activation.

2. **Whole-stack residual `f(z) = g^K(z + e) + z` (Eq. 6).** Each Intra/Inter
   transformer now adds the *pre-PE* input back across the entire K-layer
   stack. The original had only the per-layer pre-norm residuals; this outer
   residual materially improves gradient flow and matches the paper.

3. **Feature-space chunking, corrected framing.** Chunking happens in the
   encoded **feature** sequence (50% overlap), never in epoch space. The
   default `chunk_size` is tied to the feature length
   (`~feature_length // 8`, clamped to `[16, 64]`, forced even) so the model
   gets a small but meaningful number of overlapping chunks. The paper's
   `C = 250` is speech-scale and inappropriate for the ~hundreds-of-frames EEG
   feature sequence; it is exposed as an explicit override. The original
   research notes' "7 epochs = 7 chunks" framing was misleading and is
   corrected here.

4. **Paper-faithful default loss.** `build_loss` now defaults to the negative
   **scale-invariant SI-SNR with 30 dB clipping** (`clamp(si_snr, max=30)`
   before negation), exactly as in the paper. `mse`, `l1`, `smooth_l1` and
   `si_snr_mse` remain selectable. **PIT is inapplicable** (a single target —
   the centre-epoch artifact — has no source-permutation ambiguity) and is
   documented as such rather than implemented.

5. **Depth follows the ablation.** Defaults are `intra_layers=8`,
   `inter_layers=4` (the paper's Table 2 shows IntraT depth matters more than
   InterT depth). Both are configurable; the smoke uses `1/1`.

6. **Configurable Vaswani details retained / exposed.** Hand-written
   multi-head attention (functionally identical to `nn.MultiheadAttention`,
   kept for trace stability), sinusoidal PE added once per transformer,
   `skip_around_intra=True`, ReLU FFN inner activation (GELU exposed as a
   documented switch).

## Documented deviations (paper technique not transferred verbatim)

These are deliberate and justified by the EEG-fMRI domain and/or CPU cost; see
`documentation/paper_accuracy_review.md` for the full rationale.

- **Compact capacity.** `d_model=128`, `d_ffn=512`, 4-8 heads, `K=8/4` rather
  than the paper's `d_model=256`, `d_ffn=1024`, 8 heads, `K=8/8` (~26M
  params). The full config is exposed as an option; the compact default avoids
  overfitting the small proof-fit set and keeps CPU runs feasible.
- **`Ns = 1` multiplicative ReLU mask.** Artifact removal is the degenerate
  single-source case of separation. The masking mechanism is faithful; PIT is
  dropped because it is unnecessary with one source.
- **Dynamic Mixing dropped.** On-the-fly speaker remixing + speed perturbation
  is speech-specific; there is one artifact morphology per recording and speed
  perturbation would corrupt the trigger-locked timing that defines the
  gradient artifact.
- **Optimizer schedule.** The paper's Adam `lr=15e-5` + warmup + halve-after-
  epoch-65 + 200-epoch / batch-1 / AMP regime is tuned to the 30 h WSJ0 corpus
  on V100s. FACET-train's AdamW + early-stopping regime is retained; only the
  grad-clip is set to the paper's `5.0` in the YAML.

## Factory entry points

- `build_model(**kwargs)` — accepts all facet-train injected kwargs via `**_`;
  resolves `context_epochs` / `epoch_samples` from `input_shape`.
- `build_loss(name="si_snr", si_snr_max=30.0, ...)` — paper-faithful default.
- `build_dataset(path=..., target_type="artifact", ...)` — reads the Niazy
  proof-fit bundle (`noisy_center`, `clean_center`, `artifact_center`,
  `sfreq`) and builds sliding odd-length context windows channel-wise. Honors
  `target_type` ("artifact" or "clean").

## Inference

`processor.py` exposes `SepFormerPaperAccurateAdapter` and the registered
`SepFormerPaperAccurateCorrection` (registry name
**`sepformer_paper_accurate_correction`** — distinct from the original
`sepformer_correction`). Both subclass the shared
`EpochContextArtifactAdapter` / `DeepLearningCorrection` and return
`DeepLearningPrediction(artifact_data=...)`.

## Smoke

```bash
uv run pytest tests/models/sepformer_paper_accurate_edition/ -q
```

`training_niazy_proof_fit_smoke.yaml` is illustrative only and is not executed
by the test.
