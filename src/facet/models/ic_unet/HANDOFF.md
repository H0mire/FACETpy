# IC-U-Net Hand-off

## Branch & worktree

- Branch: `feature/model-ic_unet` (off `feature/add-deeplearning`).
- Worktree: `worktrees/model-ic_unet/`.
- Last commit on this branch: `086e73f feat: add IC-U-Net multichannel artifact correction model`.

## Status

**Metrics are good — orchestrator can merge.** IC-U-Net clears the DPAE
+7.48 dB target by a large margin and slightly edges out the closest peer
(`cascaded_context_dae`).

## Runs

| Phase | Job id | Worker | Result |
|---|---|---|---|
| Smoke (1 epoch, 64 examples, MSE) | `438ad63798b1` | gpu2 | `training_output/icunetniazysmoke_20260510_221545/` — finished, all artifacts present |
| Full (≤120 epochs, ensemble loss; early-stopped at 16 epochs) | `78364e361ab8` | gpu2 | `training_output/icunetniazyprooffit_20260510_223556/` — finished, 17.6 s wall-clock on RTX 5090 |

Final checkpoint (TorchScript, 11.9 MB):

```
training_output/icunetniazyprooffit_20260510_223556/exports/ic_unet.ts
```

## Evaluation

- Script: `examples/model_evaluation/evaluate_ic_unet.py`
- Manifest:
  `output/model_evaluations/ic_unet/20260510_full/evaluation_manifest.json`
- Metrics: `output/model_evaluations/ic_unet/20260510_full/metrics.json`
- Summary: `output/model_evaluations/ic_unet/20260510_full/evaluation_summary.md`
- Plot: `output/model_evaluations/ic_unet/20260510_full/plots/synthetic_cleaning_examples.png`

### Headline metrics (Niazy proof-fit, 833 examples × 30 channels × 512 samples)

| Metric | IC-U-Net | DPAE | cascaded_context_dae |
|---|---:|---:|---:|
| `clean_snr_db_after` | **+0.17** | −4.12 | +0.00 |
| `clean_snr_improvement_db` | **+11.77** | +7.48 | +11.60 |
| `clean_mse_reduction_pct` | **93.3 %** | 82.1 % | 93.1 % |
| `artifact_corr` | **0.967** | 0.913 | 0.966 |
| `residual_error_rms_ratio` | **0.258** | 0.423 | 0.263 |
| `trigger_locked_rms_reduction_pct` | 97.0 % | – | – |

Comparison data:
`output/model_evaluations/dpae/20260510_193431/metrics.json`
and `output/model_evaluations/cascaded_context_dae/20260510_vit_compare/metrics.json`.

## Caveats

- **Proof-fit dataset, not generalisation.** The Niazy proof-fit dataset
  trains and evaluates on the same recording; reported numbers describe how
  well IC-U-Net captures the AAS estimate, not how well it generalises to
  unseen subjects or scanners. This caveat applies equally to all peers.
- **AAS-derived clean target.** The "clean" reference is AAS-corrected; the
  absolute clean-SNR floor inherits AAS bias.
- **Early-stopping fired at epoch 16.** Patience 15 was set for the full run;
  on this small dataset the loss plateaus very quickly, so the schedule
  could safely be shortened in a future iteration.
- **Sigmoid replaced by `LeakyReLU(0.1)`.** Documented in
  `documentation/research_notes.md`; the reference repo's sigmoid bounds the
  output and would not fit unbounded EEG amplitudes. This is the only
  deliberate architectural divergence from `roseDwayane/AIEEG`.
- **FastICA is fit once, then frozen.** Refitting per-recording at inference
  time is not supported by the current adapter — the `W` matrix is baked
  into the TorchScript checkpoint.
- **Multichannel checkpoint.** Tied to the 30-channel Niazy montage. Other
  montages or channel counts require retraining.

## What's complete

- ✅ `processor.py` (adapter + `DeepLearningCorrection` subclass with
  registry decoration).
- ✅ `training.py` (`IcUnet1D`, `IcUnetWithIca`, `IcUnetEnsembleLoss`,
  `NiazyContextIcDataset`, `build_model`, `build_loss`, `build_dataset`).
- ✅ `training_niazy_proof_fit.yaml` (full) and
  `training_niazy_proof_fit_smoke.yaml` (smoke), both `device: cuda`.
- ✅ `README.md`, `documentation/model_card.md`,
  `documentation/research_notes.md`, `documentation/evaluations.md`.
- ✅ `examples/model_evaluation/evaluate_ic_unet.py`.
- ✅ `tests/models/ic_unet/test_processor.py` and
  `test_training_smoke.py` (9 tests, all passing locally:
  `uv run pytest tests/models/ic_unet -v`).
