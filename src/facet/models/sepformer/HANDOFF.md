# SepFormer Handoff Note

## Identity

- Model id: `sepformer`
- Model family: Audio-Inspired Source Separation
- Branch: `feature/model-sepformer`
- Worktree: `worktrees/model-sepformer/`
- Source paper: Subakan et al. 2021, "Attention is All You Need in Speech Separation", arXiv:2010.13154

## Status: green

The model trained to convergence on the Niazy proof-fit context dataset
in 11 epochs and clearly out-performs the FC autoencoder baselines on
every supervised metric. All deliverables are present.

## Smoke Run

- Run id: `sepformerniazysmoke_20260510_225537` (fleet job `2eb1675c36bc`, worker `gpu2`)
- Status: `finished`, exit code 0
- Best val loss (MSE): `2.69e-06` (1 epoch)
- Elapsed: 1.09 s on the smoke config
- Results path: `training_output/sepformerniazysmoke_20260510_225537/`
- Artifacts verified: `summary.json`, `loss.png`, `exports/sepformer.ts`,
  `training.jsonl`.

## Full Run

- Run id: `sepformerniazyprooffit_20260510_230104` (fleet job `6ec407384bd2`, worker `gpu2`)
- Status: `finished`, exit code 0
- Total epochs: 11 (early stopping triggered; best at epoch 11)
- Best val loss (MSE): `4.32e-08`
- Elapsed: 465 s (~7.75 min)
- 19 992 train + 4 998 val channel-wise pairs of shape `(7, 1, 512)` → `(1, 512)`
- Results path: `training_output/sepformerniazyprooffit_20260510_230104/`
- Checkpoint: `training_output/sepformerniazyprooffit_20260510_230104/exports/sepformer.ts`
- Parameter count: ≈ 2.22 M

## Evaluation

- Standardized run id: `20260511_012740`
- Evaluation manifest: `output/model_evaluations/sepformer/20260511_012740/evaluation_manifest.json`
- Metrics: `output/model_evaluations/sepformer/20260511_012740/metrics.json`
- Summary: `output/model_evaluations/sepformer/20260511_012740/evaluation_summary.md`
- Plots: `output/model_evaluations/sepformer/20260511_012740/plots/sepformer_examples.png` and `sepformer_metric_summary.png`
- Script: `examples/model_evaluation/evaluate_sepformer_niazy_proof_fit.py`
- Evaluation dataset: deterministic val split of the Niazy proof-fit
  context dataset (val_ratio = 0.2, seed = 42), identical to the
  facet-train split.

### Key metrics on the val split (4998 channel-wise pairs)

| Metric | SepFormer (this) | demucs | conv_tasnet | cascaded_context_dae | cascaded_dae |
|---|---:|---:|---:|---:|---:|
| `clean_snr_improvement_db` | **19.05** | 31.28 | 22.03 | 3.16 | −0.05 |
| `artifact_corr` | **0.9938** | 0.9996 | 0.997 | 0.732 | 0.086 |
| `residual_error_rms_ratio` | **0.112** | 0.027 | 0.079 | 0.695 | 1.005 |
| `clean_mse_after` | **4.30e-08** | 2.58e-09 | — | 1.47e-06 | — |
| `artifact_snr_db` | **19.05** | — | — | 3.16 | — |

Baseline numbers for `demucs`, `conv_tasnet`, `cascaded_dae`, and
`cascaded_context_dae` are taken verbatim from
`output/model_evaluations/<model>/<run>/metrics.json`. They are written
into the SepFormer manifest under the `baseline_reference.*` flat-metric
keys for direct cross-model comparison.

## Comparison vs cascaded baselines (the catalog requirement)

- vs **`cascaded_context_dae`**: SepFormer reduces clean MSE on the val
  split by ~ 34× (`1.47e-06 → 4.30e-08`), improves SNR by another
  ~ 15.9 dB on top of cascaded_context_dae's 3.2 dB, and lifts the
  artifact correlation from 0.73 to 0.99. This is consistent with the
  research-notes hypothesis: a model with explicit intra-/inter-chunk
  attention captures the trigger-locked artifact morphology much more
  faithfully than a 3-layer fully-connected encoder over the same
  context.
- vs **`cascaded_dae`** (single-epoch, no context): cascaded_dae fails
  to learn this task at all on the proof-fit dataset
  (`clean_snr_improvement_db = -0.05`, correlation 0.086). SepFormer's
  use of the seven-epoch context is decisive.

## Comparison vs the rest of the audio family

- `conv_tasnet` (22.0 dB SNR↑, corr 0.997, ratio 0.079) and `demucs`
  (31.3 dB SNR↑, corr 0.9996, ratio 0.027) currently lead the
  audio-source-separation family by a clear margin on this dataset.
- SepFormer is a strong third in the same family: it dominates the
  cascaded baselines, validates the family's inductive bias for this
  problem, and shows that *attention*-based separation (no recurrent
  or U-Net structure) is sufficient on its own.

## Caveats

1. **Proof-fit only.** The dataset uses the same Niazy recording for
   training and evaluation, with an AAS-derived clean surrogate. The
   reported numbers answer "does SepFormer fit?" — not generalization
   to a held-out recording.
2. **Compact, not full SepFormer.** The implementation uses 2 dual-path
   blocks × 4 intra/inter layers, `d_model = 128`, `d_ffn = 256`,
   4 attention heads, encoder filters = 128 (kernel 16, stride 8). The
   original SepFormer (8 + 8 layers per block, `d_model = 256`,
   `d_ffn = 1024`) was scaled down to fit the 833-example dataset and
   the 24 GB RTX 5090 envelope — see `documentation/research_notes.md`
   for the parameter-count derivation (≈ 2.22 M params measured;
   the original is ≈ 26 M). Going larger is a documented next
   experiment if generalization to other recordings is later required.
3. **`nn.MultiheadAttention` replaced.** PyTorch 2.11's
   `torch.jit.trace(check_trace=True)` (the path the FACETpy training
   CLI uses for export) does not handle the module mangling of
   `nn.MultiheadAttention` reliably, so the model uses a hand-written
   multi-head self-attention with the same math but a stable trace
   graph. `nn.GroupNorm(1, C)` was similarly swapped for a small
   `_ChannelLayerNorm` because its private `_verify_batch_size` helper
   trips on traced shapes. Functionally equivalent; just trace-safe.
4. **MSE loss, not SI-SNR.** The first full run trains with plain MSE
   to stay aligned with the cascaded/demucs/conv_tasnet baseline
   contracts. `build_loss` also supports `si_snr` and `si_snr_mse` for
   follow-up experiments.

## Documentation Completeness

- [x] `src/facet/models/sepformer/README.md`
- [x] `src/facet/models/sepformer/documentation/model_card.md`
- [x] `src/facet/models/sepformer/documentation/research_notes.md`
- [x] `src/facet/models/sepformer/documentation/evaluations.md`
- [x] `src/facet/models/sepformer/processor.py` (adapter +
      `DeepLearningCorrection` subclass, follows the preferred
      integration path from `src/facet/models/README.md`)
- [x] `src/facet/models/sepformer/training.py` (model + `build_model`,
      `build_loss`, `build_dataset`, plus a self-contained
      `ChannelWiseContextArtifactDataset`)
- [x] `src/facet/models/sepformer/training_niazy_proof_fit.yaml`
- [x] `src/facet/models/sepformer/training_niazy_proof_fit_smoke.yaml`
- [x] `tests/models/sepformer/test_processor.py` (factory, forward,
      backward, dataset, TorchScript round-trip, processor pipeline)
- [x] `tests/models/sepformer/test_training_smoke.py` (loss factory,
      TorchScript export round-trip)
- [x] `examples/model_evaluation/evaluate_sepformer_niazy_proof_fit.py`

All 9 tests pass:

```text
tests/models/sepformer/test_processor.py::test_sepformer_factory_returns_module PASSED
tests/models/sepformer/test_processor.py::test_sepformer_forward_output_shape PASSED
tests/models/sepformer/test_processor.py::test_sepformer_one_batch_backward_updates_gradients PASSED
tests/models/sepformer/test_processor.py::test_sepformer_torchscript_roundtrip_preserves_shape PASSED
tests/models/sepformer/test_processor.py::test_channel_wise_dataset_expands_channels PASSED
tests/models/sepformer/test_processor.py::test_sepformer_correction_applies_center_epochs PASSED
tests/models/sepformer/test_training_smoke.py::test_sepformer_si_snr_loss_negative_or_zero PASSED
tests/models/sepformer/test_training_smoke.py::test_sepformer_si_snr_mse_loss_finite_on_random_pair PASSED
tests/models/sepformer/test_training_smoke.py::test_sepformer_torchscript_export_smoke PASSED
```

## Files Outside the Model Folder

- `examples/model_evaluation/evaluate_sepformer_niazy_proof_fit.py` — model-specific
  evaluation script that uses `ModelEvaluationWriter`. No FACETpy core
  edits were required.

## Suggested Next Experiments (Optional)

If the orchestrator wants to push SepFormer further on this dataset:

1. **Scale up.** Increase `n_blocks` to 4 and `intra_layers/inter_layers`
   to 8, push `d_model` to 192 or 256. Memory budget is fine; the
   bottleneck is the 833-example dataset, not VRAM.
2. **SI-SNR + MSE loss.** Switch `loss.kind` to `si_snr_mse`. This
   matches the original SepFormer recipe and might reduce the residual
   RMS ratio.
3. **Skip-around-intra ablation.** Toggle the `skip_around_intra` flag
   and re-measure. The current value is on (matches the paper).
4. **Multichannel variant.** SepFormer's inter-chunk attention is a
   natural fit for cross-channel correlation. A 30-channel multichannel
   variant could be tried by collapsing the channel dim into the
   "speaker" dim. Would require a model-specific dataset builder.

The orchestrator decides whether to spawn a follow-up agent — this
agent stops here.
