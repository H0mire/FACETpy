# ST-GNN Hand-off

## Identity

- Model id: `st_gnn`
- Branch: `feature/model-st_gnn`
- Worktree: `worktrees/model-st_gnn` (off `feature/add-deeplearning`)
- Architecture family: Graph (GNN), report Section 7.3
- Source paper: Yu, Yin, Zhu 2018 (arXiv:1709.04875), with EEG-GCNN
  (arXiv:2011.12107) used as the inspiration for the fixed scalp-graph
  adjacency.
- Documentation: `src/facet/models/st_gnn/documentation/{model_card,research_notes,evaluations}.md`
- Tests: `tests/models/st_gnn/`

## Runs

### Smoke

- Job id: `79d92b6846ca`
- Worker: `gpu2`
- Config: `src/facet/models/st_gnn/training_niazy_proof_fit_smoke.yaml`
- Status: `finished`
- Run dir: `training_output/spatiotemporalgnnniazysmoke_20260510_211134/`
- TorchScript export: `exports/st_gnn.ts` ✓
- `summary.json`, `loss.png`, `training.jsonl` all present.
- 1 epoch, 64 examples, val MSE `5.41e-04`, elapsed 1.3 s.

### Full

- Job id: `f7536755f724`
- Worker: `gpu2`
- Config: `src/facet/models/st_gnn/training_niazy_proof_fit.yaml`
- Status: `finished`
- Run dir: `training_output/spatiotemporalgnnniazyprooffit_20260510_211512/`
- TorchScript export: `exports/st_gnn.ts` ✓
- 13 epochs (early-stopped from `max_epochs=60`, `patience=10`,
  `min_delta=1e-6`), val MSE `2.70e-07`, elapsed 84 s.

## Evaluation

- Manifest: `output/model_evaluations/st_gnn/niazy_proof_fit_20260510_211512/evaluation_manifest.json`
- Metrics: `output/model_evaluations/st_gnn/niazy_proof_fit_20260510_211512/metrics.json`
- Summary: `output/model_evaluations/st_gnn/niazy_proof_fit_20260510_211512/evaluation_summary.md`
- Plots: `validation_examples.png`, `per_channel_rmse.png`,
  `loss_log.png` (under `plots/`).

Validation-set headline metrics (166 examples, seed=42, val_ratio=0.2):

| Metric                              | Value     |
| ----------------------------------- | --------- |
| `clean_reconstruction_mse_before`   | 3.56e-06  |
| `clean_reconstruction_mse_after`    | 2.82e-07  |
| `clean_snr_db_before`               | -10.94 dB |
| `clean_snr_db_after`                |  +0.06 dB |
| `artifact_prediction_mse`           | 2.82e-07  |
| `artifact_prediction_correlation`   | 0.960     |
| `residual_rms_ratio`                | 0.277     |
| `rms_noisy`                         | 1.91e-03  |
| `rms_corrected`                     | 1.27e-04  |

## Comparison Vs Baselines (Niazy proof-fit, val MSE on artifact target)

Pulled from `training_output/<run>/summary.json` for runs that targeted
the same Niazy proof-fit context dataset:

| Model               | Best epoch | Best val MSE | Total epochs | Elapsed (s) |
| ------------------- | ---------: | -----------: | -----------: | ----------: |
| `dpae`              |        16  |    3.45e-07  |          19  |       48    |
| `conv_tasnet`       |        12  |    1.98e-08  |          13  |      158    |
| **`st_gnn` (this)** |        13  |    2.70e-07  |          13  |       84    |
| `d4pm`              |         4  |    1.07e-01  |          14  |      105    |

(Values for the cascaded baselines on this exact dataset are not
present as full runs in `training_output/`; the
`cascaded_context_dae` smoke run after one epoch was 7.84e-07.
The cascaded_context_dae README cites ~2e-6 best val on its own
context dataset.)

ST-GNN is roughly on par with the dpae autoencoder family on this
dataset, an order of magnitude behind conv_tasnet, and substantially
better than diffusion (`d4pm`) which optimises a different surrogate
loss. The artifact correlation of 0.960 confirms the predicted
waveform tracks the AAS target in shape, not just magnitude.

## Caveats Discovered

1. **Early stopping fired at min_delta**: `min_delta=1e-6` is *larger*
   than the absolute val-MSE values being compared (~3e-7), so any
   epoch that does not improve by 1e-6 in absolute terms triggers the
   patience counter. The model very likely has additional headroom if
   `min_delta` is tightened (e.g. `1e-9`) or set as a relative
   threshold. The exporter still saved the best-epoch checkpoint, so
   metrics here are conservative.
2. **Channel order is baked into the checkpoint**: the Chebyshev
   Laplacian buffer is computed once at `build_model` time from
   `NIAZY_PROOF_FIT_CHANNELS`. Inference on a recording with a
   different montage / order requires rebuilding and retraining. The
   adapter raises `ProcessorValidationError` when channels don't match.
3. **`torch_geometric` is a stated dependency but unused at runtime**:
   the model implements a self-contained Chebyshev recursion on a
   dense `(30, 30)` rescaled Laplacian so `torch.jit.trace` produces a
   clean TorchScript artefact. `torch_geometric` is added to
   `pyproject.toml` as instructed by the brief; it is available for
   future hybrids but is not on the live forward path.
4. **Niazy uses old 10-20 names** (`T3/T4/T5/T6`) which the adjacency
   builder maps to MNE's `standard_1005` `T7/T8/P7/P8`. Datasets with
   the modern names work without further configuration.
5. **`filterwarnings = error` in pytest.ini interacts with TracerWarnings**
   from `torch.jit.trace`. The trace tests use a per-test
   `pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")` to
   keep the project-wide policy intact while letting tracing succeed.

## Confirmation Of Required Artefacts

- `model_card.md` ✓ (`src/facet/models/st_gnn/documentation/model_card.md`)
- `research_notes.md` ✓ (`src/facet/models/st_gnn/documentation/research_notes.md`)
- `evaluations.md` ✓ (`src/facet/models/st_gnn/documentation/evaluations.md`)
- `README.md` ✓ (`src/facet/models/st_gnn/README.md`)
- `tests` ✓ (`tests/models/st_gnn/`, 13 tests passing locally)

## Suggested Follow-ups (For The Orchestrator)

These are **not** autonomous experiments; the orchestrator decides
whether to spawn them.

1. Re-run the full training with a relative early-stopping threshold
   (`min_delta=0` plus a relative-improvement guard, or simply
   `min_delta=1e-9`) and `max_epochs=60`. Expectation: 30+ epochs of
   training, additional ~3-5x reduction in val MSE before convergence.
2. Sweep `knn_k ∈ {3, 4, 6, 8}` and `k_order ∈ {1, 2, 3, 4}` to
   confirm the spatial graph contributes beyond per-channel temporal
   modelling. `k_order=1` collapses to per-channel temporal conv and
   is the cleanest ablation control.
3. Compare against `cascaded_context_dae` and `cascaded_dae` on the
   exact same Niazy proof-fit context dataset, not on their original
   datasets, so the comparison column in `evaluations.md` is on equal
   footing.
4. Inspect the per-electrode RMSE bar chart
   (`output/model_evaluations/st_gnn/niazy_proof_fit_20260510_211512/plots/per_channel_rmse.png`)
   for spatial outliers. Channels with persistently higher RMSE than
   their neighbours would suggest the graph constraint is over- or
   under-smoothing locally.

## Ready For Merge?

Tests pass locally, smoke and full runs are green, evaluation artefacts
exist, documentation is complete. From the model agent's side: **yes**.
The orchestrator owns the merge decision and should compare the
metrics column above against what is acceptable for the thesis.
