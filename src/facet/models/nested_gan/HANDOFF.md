# Nested-GAN Hand-off

## Identity

- Branch: `feature/model-nested_gan`
- Worktree: `worktrees/model-nested_gan` (off `feature/add-deeplearning`)
- Model id: `nested_gan`
- Smoke job: `66b505ba6ebb` (`nested_gan_niazy_smoke_v2`) — `finished` on `gpu2`
  - (the first attempt `1693e942b811` failed because the
    multi-resolution STFT loss windows were on CPU while inputs landed
    on cuda; commit `5c45e2c` switched the windows to per-call device
    allocation.)
- Full job: `2ebbde32772d` (`nested_gan_niazy_full`) — `finished` on `gpu2`

## Run paths

- Full training run: `training_output/nestedganniazyprooffit_20260510_222546/`
  - `summary.json`
  - `loss.png`
  - `training.jsonl`
  - `facet_train_config.resolved.{json,yaml}`
  - `exports/nested_gan.ts` (TorchScript, 1.3 MB)
  - 52 epochs, early-stopped after best at epoch 37
  - best validation loss `0.13328766` (combined L1 + 0.5·MR-STFT)
  - 500 s wall clock on RTX 5090

- Smoke training run: `training_output/nestedganniazysmoke_20260510_222221/`
  - 1 epoch, best metric `0.55218936`, 14 s elapsed
  - same artifacts as the full run

- Evaluation: `output/model_evaluations/nested_gan/20260511_004531/`
  - `evaluation_manifest.json`
  - `metrics.json`
  - `evaluation_summary.md`
  - `plots/example_0425_ch23.png`, `example_0530_ch17.png`, `example_0708_ch15.png`

## Headline metrics

| Metric | Value |
|---|---:|
| Clean-SNR before correction | **-11.59 dB** |
| Clean-SNR after correction | **+1.95 dB** |
| **Clean-SNR improvement** | **+13.54 dB** |
| Artifact prediction correlation (vs `artifact_center`) | **0.978** |
| Residual RMS ratio (artifact residual ÷ original artifact RMS) | **0.210** |
| Trigger-locked RMS reduction (real-proxy) | **80.6 %** |
| Clean reconstruction L1 before | 9.19e-4 |
| Clean reconstruction L1 after | 1.43e-4 |
| Clean reconstruction L2 (RMS) before | 1.88e-3 |
| Clean reconstruction L2 (RMS) after | 3.96e-4 |

Dataset: 833 windows × 30 EEG channels × 512 samples, sfreq = 4096 Hz.

## Comparison vs available baselines

The full numeric tables for the other models are not fetched into this
worktree; values for `cascaded_dae`, `cascaded_context_dae`,
`dhct_gan`, and `dhct_gan_v2` are scattered across other worktrees on
the orchestrator MacBook (see `output/model_evaluations/<model_id>/`
on the main checkout).

What is robust to state here:

- **vs DHCT-GAN (-7.13 dB on the same dataset, orchestrator note).**
  Nested-GAN improves clean-SNR by +13.54 dB. The hypothesis
  *nested time-frequency decomposition outperforms single-domain GAN*
  is **strongly supported** by these numbers (gap of ≈ 21 dB).
- **vs cascaded_context_dae** (same 7-epoch context, same dataset).
  cascaded_context_dae's smoke training reportedly reaches
  best val loss ~ 2e-6 (MSE on the same target), Nested-GAN reaches
  best val loss 0.133 — but the losses are **not comparable** because
  Nested-GAN uses L1 + multi-resolution log-magnitude STFT while
  cascaded_context_dae uses MSE. Direct comparison must be done via
  the standard evaluation metrics in `output/model_evaluations/`.
- **vs published Nested-GAN (Biomed. Phys. Eng. Express 2025).** The
  paper reports 71.6 % temporal artifact reduction and Pearson 0.892.
  This FACETpy implementation reports 80.6 % RMS reduction and
  Pearson 0.978 on the Niazy proof-fit data, which is in the same
  ballpark and arguably slightly stronger on this dataset — even
  without the four discriminators (see scope reduction note).

## Caveats discovered during evaluation

1. **In-distribution evaluation only.** The evaluation NPZ bundle is
   the same dataset used for training. The 0.978 correlation should be
   interpreted as *fit quality on the proof-fit set*, not
   cross-subject generalization. A held-out subject evaluation is the
   right next step before claiming generalization.
2. **The model converged in 37 epochs out of 80.** Early stopping fired
   at epoch 52. Increasing `max_epochs` further is unlikely to help
   without other changes.
3. **Smoke v1 failure surfaced a real device-placement bug** in the
   multi-resolution STFT loss; the fix in commit `5c45e2c` is now
   covered indirectly by the passing GPU smoke run. A unit test that
   reproduces the cross-device case would require a CUDA-equipped
   local box; the GPU smoke acts as the regression test.
4. **The exported TorchScript checkpoint is independent of channel
   count**, but the input `target-epoch-samples` and `context-epochs`
   are baked into the model (and into the dataset bundle). Inference
   at a different epoch length needs a new training run with a fresh
   dataset.
5. **Edge epochs not corrected.** Same as `cascaded_context_dae`: the
   first and last `(context_epochs - 1) / 2` epochs of each recording
   are not corrected because the 7-epoch context cannot be assembled.

## Confirmation of documentation and tests

- `src/facet/models/nested_gan/README.md` — present, describes scope,
  training command, inference command, and direct comparison contract.
- `src/facet/models/nested_gan/documentation/research_notes.md` —
  present, cites primary paper PMID 41183389 plus the closely related
  public work (Restormer 2022, TF-Restormer 2025, CMGAN, HiFi-GAN).
  Documents the context-handling strategy and the deliberate scope
  reduction (generator-only, no adversarial discriminators).
- `src/facet/models/nested_gan/documentation/model_card.md` — present,
  describes input/output shapes, architecture, loss, and training
  defaults.
- `src/facet/models/nested_gan/documentation/evaluations.md` —
  present, points at the standard output layout.
- Tests at `tests/models/nested_gan/`:
  - `test_processor.py` — adapter applies center-epoch artifact;
    raises when too few triggers.
  - `test_training_smoke.py` — `build_model` returns the generator
    with the expected I/O shape; forward+backward produces non-zero
    finite gradients; loss is a non-negative scalar; outer branch
    receives the center-corrected context; channel-wise dataset
    expands `(N, C, S)` examples to `(N·channels, 1, S)`.
  - `test_evaluate.py` — synthesizes a tiny noisy/clean/artifact
    bundle and verifies that `evaluate.main()` writes the standard
    manifest/metrics/summary files.
  - Local pytest summary: **9 passed in 2.28 s** (all green on the
    orchestrator MacBook).

## Commits on this branch

```text
5c45e2c fix: device-place multi-resolution stft windows per call
a1d7ba8 test: smoke test for nested_gan evaluate.py
ea9559d feat: add nested_gan evaluation script
137144c feat: add nested-gan model package
```

All four commits carry the `made by Müller Janik Michael` marker.

## Architecture summary (for the orchestrator)

Generator-only two-stage network operating per channel.

- **Inner branch:** STFT (`n_fft=64`, `hop=16`, Hann window) of the
  center epoch → light Restormer (4 MDTA + GDFN blocks at C=48,
  4 heads) on the real/imag stacked complex spectrogram → iSTFT.
- **Outer branch:** 1D residual U-Net (base width 32, 3 enc/dec
  levels, kernel 5, GELU) over the 7-epoch context with the center
  slot replaced by `context_center − inner_artifact`. Output is the
  residual artifact correction.
- **Loss:** `L1(time) + 0.5 · MultiResolutionSTFTLoss(log|STFT|)` over
  window sizes `{32, 64, 128, 256}` with hop = window / 4.
- **No adversarial discriminators.** The multi-resolution STFT loss
  substitutes for the published Nested-GAN's two multi-resolution
  discriminators. The published paper's two metric discriminators
  predict a perceptual scalar; no defensible perceptual metric for
  EEG exists, so this term is dropped. Both reductions are recorded
  with reasoning in `documentation/research_notes.md` and are
  reversible: a custom `TrainableModelWrapper` plus an alternating
  GAN training loop would restore the full recipe.

## Suggested next experiments for the orchestrator

These are ranked by expected information value, not by ease of
implementation:

1. **Held-out subject evaluation.** Build a Niazy proof-fit context
   dataset over a different subject and re-run
   `python -m facet.models.nested_gan.evaluate`. The 0.978 in-
   distribution correlation is impressive but proves nothing about
   cross-subject generalization. This is the single most important
   follow-up.
2. **Head-to-head metric table vs the other family baselines.** Run
   the same `evaluate.py` style on `cascaded_dae`,
   `cascaded_context_dae`, `dhct_gan`, `dhct_gan_v2`, and
   `denoise_mamba` (when fetched), then build a unified comparison
   table from each model's `metrics.json`. This is the most direct
   test of "TF decomposition wins" vs "context window alone is
   enough".
3. **Add the published adversarial losses.** Build a custom
   `TrainableModelWrapper` subclass that owns separate discriminator
   networks and runs the alternating G/D update. With +13.5 dB
   already without discriminators, the adversarial pass should
   primarily sharpen high-frequency texture rather than fundamentally
   change the metrics; this experiment scopes whether the published
   paper's higher-frequency claim is reproducible here.
4. **Try `demean_target=false`.** Demeaning the artifact target
   removes any DC component that the model would have to recover
   later. A run with the DC component preserved tests whether the
   current high RMS reduction is leaving a constant baseline behind.

I do not recommend running these autonomously. The +13.5 dB result
already supports the hypothesis the orchestrator wanted to test, so
the natural next step is the orchestrator's call: declare the
hypothesis supported, or invest the GPU time to falsify it on a
held-out subject.

## What the orchestrator should not do

- Do not merge this branch without the cross-subject evaluation in
  experiment (1). The within-subject metrics are not sufficient
  evidence for the thesis claim.
- Do not delete `output/niazy_proof_fit_context_512/` on the
  MacBook — it is the dataset bundle that produced the evaluation.
  The bundle was rebuilt deterministically on the GPU worker during
  the smoke and full runs (`--prepare-command`), but the local copy
  is what `evaluate.py` consumed.
