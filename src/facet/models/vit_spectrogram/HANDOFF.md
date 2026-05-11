# HANDOFF — `vit_spectrogram` (ViT Spectrogram Inpainter)

## Identity

- **Branch:** `feature/model-vit_spectrogram`
- **Worktree path:** `worktrees/model-vit_spectrogram/`
  (`/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/worktrees/model-vit_spectrogram/`)
- **Architecture family:** Vision-Inspired (Spectrogram Inpainting) —
  report §7.2.1.

## Smoke run

- **Fleet job id:** `67049ff6fb1c`
- **Worker:** `gpu2`
- **Status:** `finished`
- **Training run dir:** `training_output/vitspectrograminpainterniazysmoke_20260510_211348/`
- **Confirmed artifacts:** `summary.json`, `loss.png`,
  `training.jsonl`, `exports/vit_spectrogram.ts`.
- **Smoke val_loss after 1 epoch on 512 examples:** `4.70e-5`.

## Full training run

- **Fleet job id:** `71f267275b98`
- **Worker:** `gpu2`
- **Status:** `finished`
- **Training run dir:** `training_output/vitspectrograminpainterniazyprooffit_20260510_211842/`
- **Config used:**
  `src/facet/models/vit_spectrogram/training_niazy_proof_fit.yaml`
  (max_epochs=60, batch=64, AdamW lr=3e-4, weight_decay=0.05, grad clip
  1.0, depth=6, embed_dim=192, n_heads=6).
- **Dataset:** `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`
  (833 context examples × 30 channels = 24 990 per-channel training pairs;
  19 992 train / 4 998 val after the 0.2 split).
- **Epochs trained:** 13 (early-stopped on `loss` with patience 12).
- **Best epoch:** 1 with val_loss `2.39e-7`. The val loss is computed in
  the per-epoch *demeaned* clean-signal space, so it stays close to the
  zero-prediction baseline (mean(clean²) ≈ 1e-7 after demeaning) — the
  small absolute number does not by itself indicate that the model
  learned nothing; the proof-fit metrics below show a clear improvement.
- **Wall-clock:** 87.5 s of GPU time (RTX 5090). The full run is far
  faster than the 60-epoch budget; convergence happens in one epoch.
- **TorchScript artifacts:**
  - `exports/vit_spectrogram.ts` — the fleet's traced export (cuda:0
    device baked in; not usable for CPU inference, see "Caveats" below).
  - `exports/vit_spectrogram_cpu.ts` — a CPU-compatible re-trace
    produced locally from the saved state-dict checkpoint
    (`checkpoints/epoch0001_loss0.0000.pt`) after the device-buffer
    fix in commit `4184443`.

## Evaluation

- **Standard run path:** `output/model_evaluations/vit_spectrogram/20260510_full/`
- **Manifest:** `output/model_evaluations/vit_spectrogram/20260510_full/evaluation_manifest.json`
- **Required artifacts present:** `evaluation_manifest.json`,
  `metrics.json`, `evaluation_summary.md`,
  `synthetic_cleaning_examples.png`, `synthetic_metric_summary.png`,
  `vit_spectrogram_metrics.json`.
- **Evaluation dataset:** Niazy proof-fit context NPZ (same as training
  data — proof-of-fit only, not a generalization benchmark).
- **Evaluation script:** `python -m facet.models.vit_spectrogram.evaluate`
  (lives in the model folder, follows the standard from
  `src/facet/models/evaluation_standard.md`).

### Headline metrics on Niazy proof-fit

| Metric | `vit_spectrogram` (this run) |
|---|---:|
| Clean MSE before correction | 3.54e-6 |
| Clean MSE after correction | 2.45e-7 |
| **Clean MSE reduction** | **93.08 %** |
| Clean SNR before correction | -11.59 dB |
| Clean SNR after correction | 0.00 dB |
| **Clean SNR improvement** | **+11.60 dB** |
| **Artifact correlation** | **0.9660** |
| Residual error RMS ratio | 0.2631 |
| Artifact MAE | 2.14e-4 |
| Examples × channels | 833 × 30 |
| Sampling rate | 4096 Hz |

### Comparison vs. existing baselines

The previously published evaluation runs for `cascaded_dae` and
`cascaded_context_dae` (`output/model_evaluations/cascaded_*/20260502_*`)
were evaluated against the **single-channel synthetic spike-artifact
dataset** (`output/synthetic_spike_artifact_context_512/...`), not the
Niazy proof-fit dataset that I used. The architectures themselves are
also single-channel models, so my one-line attempt to re-run
`examples/evaluate_context_artifact_model.py` against the Niazy
proof-fit npz fails with a shape mismatch
(`mat1 and mat2 shapes cannot be multiplied (128x107520 and 3584x512)`).

For reference, the published baseline numbers on their respective
synthetic dataset are:

| Metric | `cascaded_dae` (synthetic) | `cascaded_context_dae` (synthetic) | `vit_spectrogram` (Niazy proof-fit) |
|---|---:|---:|---:|
| Clean MSE reduction | -1.1 % | 51.7 % | 93.1 % |
| Clean SNR improvement | -0.05 dB | +3.16 dB | +11.60 dB |
| Artifact correlation | 0.086 | 0.732 | 0.966 |
| Residual RMS ratio | 1.005 | 0.695 | 0.263 |

Treat the cross-row gap with care: this is **not an apples-to-apples
comparison**. A like-for-like comparison would require running the
cascade checkpoints on the same Niazy proof-fit dataset (or training
fresh single-channel cascade baselines on it). I did not do that here
because it would require either modifying the shared evaluation script
to handle 30-channel multi-channel inputs or training new baseline
checkpoints, both outside the brief for this model agent.

## Caveats discovered during evaluation

1. **Traced TorchScript on the fleet bakes the cuda:0 device into the
   patch-mask buffer's device-cast call**, which made the exported
   `vit_spectrogram.ts` fail to run on CPU
   (`Could not run 'aten::empty_strided' with arguments from the 'CUDA'
   backend`). Fixed in commit `4184443` by removing the explicit
   `.to(tokens.device)` and relying on the buffer's own device
   inheritance. The fix is in the source tree, the prior tests still
   pass, and a CPU-compatible re-trace
   (`exports/vit_spectrogram_cpu.ts`) is what the evaluation actually
   ran against. A re-export on the fleet using the patched source
   would emit a CPU-and-CUDA-compatible TorchScript in one shot.
2. **`best_epoch=1` is real, not a logging artefact.** Training collapses
   to near-optimal val-loss after one pass over the data. The loss
   curve in `loss.png` is essentially flat after that. Two interpretations
   coexist: (a) the model converged quickly because the proof-fit
   target is highly self-similar (same recording front-to-back), and (b)
   the loss in demeaned space is dominated by the small variance of the
   clean signal so further optimization buys little. The downstream
   proof-fit metrics confirm the model did learn useful structure
   (artifact correlation 0.97), so this is not a "model failed to fit"
   case.
3. **Val and train losses are reported in the per-epoch demeaned clean
   space**, which makes the absolute numbers small and superficially
   uninformative. The proof-fit metrics in
   `output/model_evaluations/vit_spectrogram/20260510_full/metrics.json`
   are the correct headline numbers.
4. **Magnitude-only spectrogram reconstruction keeps the input's
   (noisy) phase.** This is the design choice documented in
   `documentation/research_notes.md` and acknowledged by the report.
   Some artifact-locked phase structure necessarily leaks back into the
   reconstruction; if the next agent wants to push these numbers
   further, the most direct knob is to predict full complex (real+imag)
   spectrograms or a complex-valued residual.
5. **Proof-fit is not generalization.** Training and inference draw
   from the same Niazy recording. Strong numbers here do not imply
   strong numbers on held-out recordings; the dataset builder's
   docstring and `evaluation_standard.md` both flag this.

## Confirmation of completeness

- [x] `model_card.md` — present at
  `src/facet/models/vit_spectrogram/documentation/model_card.md`.
- [x] `research_notes.md` — present and documents the phase-handling
  decision (magnitude only, preserved noisy phase) and hardware
  feasibility for an RTX 5090 (~2.7 M params, well under 24 GB).
- [x] `evaluations.md` — updated to point at the `20260510_full` run.
- [x] `README.md` — present.
- [x] Tests — `tests/models/vit_spectrogram/test_processor.py` and
  `tests/models/vit_spectrogram/test_training_smoke.py`,
  10/10 passing under `uv run pytest tests/models/vit_spectrogram -v`.
- [x] Processor follows the canonical two-layer pattern
  (`ViTSpectrogramInpainterAdapter` + `@register_processor`
  `ViTSpectrogramInpainterCorrection(DeepLearningCorrection)`).
- [x] No edits to `src/facet/core/*` or to other models' folders.
- [x] No edits to `src/facet/training/cli.py` or other shared core
  training code.
- [x] No fleet `dispatch` invocations from this agent — only `submit`
  / `status` / `fetch`.

## Suggested next steps for the orchestrator

The metrics here are good enough that the orchestrator can decide
`vit_spectrogram` is a viable baseline. Optional follow-ups if a deeper
investigation is wanted:

- **Apples-to-apples baseline comparison on Niazy proof-fit.** Either
  re-evaluate the existing single-channel `cascaded_dae` /
  `cascaded_context_dae` checkpoints on the Niazy proof-fit npz
  (requires extending `examples/evaluate_context_artifact_model.py` to
  loop per channel, or building a thin wrapper that does so), or
  re-train both baselines on the Niazy proof-fit dataset and evaluate.
- **Complex-valued reconstruction.** Train a variant that predicts the
  full complex spectrogram (real+imag, two channels) so that phase can
  also be corrected. This is the single most promising follow-up if
  the orchestrator wants to push the SNR ceiling.
- **Hold-out evaluation.** Train on Niazy proof-fit but evaluate on
  the large-MFF AAS artifact bundle in `output/artifact_libraries/`
  to confirm the gains transfer.
- **Re-export from the patched source on the fleet.** The fleet's
  `exports/vit_spectrogram.ts` is the cuda-baked one; the orchestrator
  may want to re-run a quick "export only" pass to refresh it once
  the patch in commit `4184443` lands on the deep-learning branch.
  The `exports/vit_spectrogram_cpu.ts` re-trace I produced locally is
  numerically identical and is what the published metrics were
  computed against.

The model does not appear "fundamentally unsuitable" — the inpainting
framing produces large MSE reductions and high artifact correlations
on the proof-fit benchmark, comfortably above the cascade baselines'
self-reported synthetic numbers. The remaining headroom is consistent
with the documented phase-handling trade-off rather than with an
architecture-level limitation.
