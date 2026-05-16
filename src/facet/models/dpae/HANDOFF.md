# DPAE Hand-Off

## Identity

- Model id: `dpae`
- Model name: `Dual-Pathway Autoencoder` (Xiong et al. 2023, 1D-CNN variant)
- Architecture family: Discriminative (CNN / Autoencoder)
- Branch: `feature/model-dpae`
- Worktree: `worktrees/model-dpae` (i.e.
  `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/worktrees/model-dpae`)
- Commits on the branch (newest first):
  - `571eeaa` feat: add DPAE evaluation script for niazy proof-fit dataset
  - `ae2c32b` feat: add DPAE (Dual-Pathway Autoencoder) model

## Smoke run

- Job id: `29d774016542` (`dpae_niazy_smoke`)
- Worker: `gpu1`
- Status in `.facet_gpu_fleet/queue.json`: `finished` (exit=0)
- Run dir on the orchestrator MacBook:
  `training_output/dualpathwayautoencoderniazysmoke_20260510_192605/`
- Verified artifacts:
  - `summary.json` present, best metric `5.88e-4` after 1 epoch on 256 examples
  - `loss.png` present
  - `exports/dpae.ts` present
  - `training.jsonl` present

## Full run

- Job id: `b6f78ee64cf1` (`dpae_niazy_full`)
- Worker: `gpu1`
- Status in `.facet_gpu_fleet/queue.json`: `finished` (exit=0)
- Run dir on the orchestrator MacBook:
  `training_output/dualpathwayautoencoderniazyprooffit_20260510_192929/`
- Headline numbers from `summary.json`:
  - 24,990 channel-wise examples (19,992 train / 4,998 val)
  - Total epochs run: 19 (early-stopped at patience=15 on validation loss)
  - Best epoch: 16
  - Best validation MSE: `3.45e-7`
  - Wall clock: `48.0 s` on RTX 5090
- Exported checkpoint:
  `training_output/dualpathwayautoencoderniazyprooffit_20260510_192929/exports/dpae.ts`

## Evaluation

- Run id: `20260510_193431`
- Path:
  `output/model_evaluations/dpae/20260510_193431/`
- `evaluation_manifest.json`: present
- `metrics.json` (Niazy proof-fit, 24,990 examples × 512 samples):
  - `clean_mse_before` = `3.54e-6`
  - `clean_mse_after`  = `6.33e-7`
  - `clean_mse_reduction_pct` = `82.1 %`
  - `clean_snr_improvement_db` = `+7.48 dB`
  - `artifact_corr` = `0.913`
  - `artifact_snr_db` = `7.48 dB`
  - `residual_error_rms_ratio` = `0.423` (corrected residual is 42 % of the
    noisy residual RMS)
- Plot: `plots/niazy_proof_fit_examples.png` (4 noisy/clean/corrected and
  true/predicted artifact overlays)
- `evaluation_summary.md`: present

## Comparison vs `cascaded_dae` and `cascaded_context_dae`

The two existing baselines have evaluation runs under
`output/model_evaluations/cascaded_dae/` and
`output/model_evaluations/cascaded_context_dae/`, but those were run against
**different datasets** than the new Niazy proof-fit bundle that DPAE was trained
on. Their metric keys are therefore not row-equivalent. Closest cross-references:

| Metric (closest concept)           | DPAE (proof-fit)  | cascaded_context_dae (synthetic) | cascaded_dae (synthetic) |
|---|---:|---:|---:|
| `clean_snr_improvement_db`         | **+7.48 dB**      | +3.16 dB                          | -0.05 dB                  |
| `artifact_corr`                    | **0.913**         | 0.732                             | 0.086                     |

Caveat: those baseline numbers come from the synthetic spike artifact context
dataset, while DPAE's numbers come from the Niazy AAS proof-fit dataset, which
is a much narrower (and easier) target. The fair comparison would be to evaluate
all three models on the Niazy proof-fit set or all three on the synthetic set.
The orchestrator may want to extend `examples/model_evaluation/evaluate_context_artifact_model.py`
to register `dpae` alongside the others, or to re-run the existing synthetic
evaluation against the DPAE checkpoint for an apples-to-apples comparison.

## Caveats

- Clean reference in the proof-fit dataset is the AAS-corrected Niazy surrogate,
  not an independent clean source. The 82 % MSE reduction is therefore a
  proof-of-fit number for the AAS template's morphology — not an
  out-of-distribution generalisation claim.
- The model parameter count came out at ~270 k, considerably below the
  paper's ~2 M for the 1D-CNN variant. The paper does not tabulate the CNN
  layer-by-layer, so the channel widths were chosen by structural intuition
  rather than to hit the published budget. Convergence to `3.45e-7` validation
  MSE in 19 epochs suggests the smaller capacity is sufficient for this
  dataset, but a wider configuration is straightforward via
  `model.kwargs.base_filters` / `latent_filters` in
  `training_niazy_proof_fit.yaml` if a future run needs more capacity.
- Inference uses the trigger-locked epoch resampling pattern of
  `cascaded_context_dae`, so DPAE inference still requires trigger metadata
  even though the training data does not. A no-trigger fixed-chunk fallback
  could be added if the orchestrator wants DPAE to work on raw EEG without
  trigger detection.
- Only trained and evaluated on the Niazy proof-fit recording; no
  cross-subject and no cross-dataset evaluation. The architecture should
  generalise (it is channel-wise and channel-count-agnostic), but that is
  unmeasured.

## Documentation completeness

- `src/facet/models/dpae/README.md`: present
- `src/facet/models/dpae/documentation/model_card.md`: present
- `src/facet/models/dpae/documentation/research_notes.md`: present
- `src/facet/models/dpae/documentation/evaluations.md`: present (auto-extended
  by `ModelEvaluationWriter` with the run-id index)
- `tests/models/dpae/test_processor.py`,
  `tests/models/dpae/test_training_smoke.py`: present, **8 tests passing**
  (`uv run pytest tests/models/dpae/ -v --no-cov`)

## Next steps the orchestrator could consider

- Register `dpae` in `examples/model_evaluation/evaluate_context_artifact_model.py` so it
  participates in the synthetic-dataset comparison alongside the two existing
  baselines (this requires editing a non-`dpae` file, which is outside this
  agent's allowed scope).
- Decide whether to keep the smaller (~270 k) parameter footprint or rerun
  full training with `base_filters=64, latent_filters=192` (~2 M params)
  to match the published budget.
- Merge `feature/model-dpae` into `feature/add-deeplearning` once the
  orchestrator is satisfied with the comparison; nothing in this branch
  modifies `src/facet/core/`, `src/facet/training/`, or any other model.
