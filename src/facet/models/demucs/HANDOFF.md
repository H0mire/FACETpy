# Demucs Hand-Off

## Identity

- Branch: `feature/model-demucs`
- Worktree: `worktrees/model-demucs/`
- Base commit: `32190a0` (`feature/add-deeplearning`).
- Model id: `demucs`
- Model name: `Demucs`
- Architecture family: Audio-Inspired Source Separation
- Report section: 7.1.2 (Demucs: Deep Music Separation)
- Author marker: `made by Müller Janik Michael`

## What Was Done

1. **Phase 1 mandatory reading**: CLAUDE.md, AGENTS.md,
   `docs/PROCESSOR_GUIDELINES.md`, `src/facet/models/README.md`,
   `docs/source/development/training_cli_architecture.md`,
   `dl_eeg_gradient_artifacts.pdf` §7.1.2, `architecture_catalog.md`,
   `deep_learning_parallel_runpod_workflow.md`, `evaluation_standard.md`,
   `cascaded_dae/processor.py`, full `cascaded_context_dae/`.
2. **Phase 2 research notes**: `src/facet/models/demucs/documentation/research_notes.md`
   covers primary papers (arXiv 1911.13254, with v4 noted but not adopted),
   architecture knobs derived from the published reference implementation,
   the 7-epoch context mapping, the L=4 depth derivation for the 3584-sample
   input, hardware feasibility, and open questions.
3. **Phase 3 reuse**: training routes through `facet-train` →
   `PyTorchModelWrapper`; dataset is a custom `FlatContextArtifactDataset`
   that loads the existing Niazy proof-fit `.npz` directly (the standard
   `NPZContextArtifactDataset` requires a 3D target while we need the full
   4D `artifact_context`); evaluation uses `facet.evaluation.ModelEvaluationWriter`.
   No new dataset builder, no edits to `src/facet/core/*` or
   `src/facet/training/*`.
4. **Phase 4 worktree** created off `feature/add-deeplearning`.
5. **Phase 5 implementation**:
   - `src/facet/models/demucs/training.py` — `Demucs` (Conv1d/GLU encoder,
     2-layer BiLSTM bottleneck with linear projection, ConvTranspose1d
     decoder, U-Net skip-sums, weight-rescale-at-init), `FlatContextArtifactDataset`,
     `build_model`/`build_loss`/`build_dataset` factories. Depth-collapse
     check rejects configs that would shrink the bottleneck below 1 sample.
   - `src/facet/models/demucs/processor.py` — `DemucsAdapter`
     (subclasses `DeepLearningModelAdapter`, declares
     `DeepLearningModelSpec(architecture=AUDIO_SOURCE_SEPARATION, execution_granularity=CHANNEL, supports_multichannel=False, uses_triggers=True)`,
     flattens 7 trigger-defined epochs per channel into a 3584-sample
     waveform, slices the center 512 samples from the prediction, resamples
     back to the native trigger-to-trigger length) and `DemucsCorrection`
     (`@register_processor` as `demucs_correction`, subclasses
     `DeepLearningCorrection`, refuses `parallel=True`).
   - YAMLs: `training_niazy_proof_fit.yaml` (full: depth=4, C₁=64,
     batch=32, 60 epochs, L1, lr=3e-4) and `training_niazy_proof_fit_smoke.yaml`
     (max_examples=512, 1 epoch, C₁=32, batch=16). Both `device: cuda`.
   - Documentation: `README.md`, `documentation/model_card.md`,
     `documentation/research_notes.md`, `documentation/evaluations.md`.
   - Tests: `tests/models/demucs/test_processor.py` (factory shape,
     depth-collapse guard, one-batch backward, loss variants, dataset
     shape/demean/split, factory round-trip, `demucs_correction` registry)
     and `test_training_smoke.py` (build→one step→trace→reload).
6. **Phase 6 smoke**: submitted as `c8013b0e0071` on gpu2, `finished` (exit 0)
   in ~1.3 s of training (1 epoch, 512 examples, C₁=32). All required
   artifacts present: `summary.json`, `loss.png`, `exports/demucs.ts`.
7. **Phase 7 full training**: submitted as `a097d80ea0e1` on gpu2,
   `finished` (exit 0) in 406 s of training (52 epochs, early-stopped at
   patience 12, best epoch 49, best val L1 = 2.78e-05).
8. **Phase 8 evaluation**: `examples/evaluate_demucs.py` slices the
   center-epoch artifact from the 3584-sample prediction and compares
   against `clean_center` / `artifact_center` on the held-out 20 % val split
   (same RNG seed as training). Standard manifest/metrics/summary/plots
   written via `ModelEvaluationWriter`. Run id `20260511_005636`. A second
   CPU-traced TorchScript (`exports/demucs_cpu.ts`) was created locally
   from `checkpoints/epoch0049_loss0.0000.pt` because the CUDA-traced
   export baked CUDA tensor metadata into the LSTM graph and could not be
   loaded on CPU even with `map_location="cpu"` — see Caveats below.
9. **Phase 9 tests**: `uv run pytest tests/models/demucs -v` passes 11/11.

## Run Identifiers and Paths

| Item | Path |
| --- | --- |
| Smoke job id | `c8013b0e0071` |
| Smoke run dir | `training_output/demucsniazysmoke_20260510_224439/` |
| Smoke export | `training_output/demucsniazysmoke_20260510_224439/exports/demucs.ts` |
| Full job id | `a097d80ea0e1` |
| Full run dir | `training_output/demucsniazyprooffit_20260510_224653/` |
| Full export (CUDA) | `training_output/demucsniazyprooffit_20260510_224653/exports/demucs.ts` |
| Full export (CPU re-trace) | `training_output/demucsniazyprooffit_20260510_224653/exports/demucs_cpu.ts` |
| Best checkpoint | `training_output/demucsniazyprooffit_20260510_224653/checkpoints/epoch0049_loss0.0000.pt` |
| Eval manifest | `output/model_evaluations/demucs/20260511_005636/evaluation_manifest.json` |
| Eval metrics | `output/model_evaluations/demucs/20260511_005636/metrics.json` |
| Eval summary | `output/model_evaluations/demucs/20260511_005636/evaluation_summary.md` |
| Eval plots | `output/model_evaluations/demucs/20260511_005636/plots/` |

## Headline Metrics (Niazy proof-fit val split, 4998 pairs)

| Metric | Value |
| --- | ---: |
| Clean SNR before | -11.61 dB |
| Clean SNR after | +19.67 dB |
| **Clean SNR improvement** | **+31.28 dB** |
| **Clean MSE reduction** | **99.93 %** |
| **Artifact correlation** | **0.9996** |
| Artifact SNR | +31.28 dB |
| Residual error RMS ratio | 0.027 |
| Best epoch | 49 / 52 |
| Best val L1 | 2.78e-05 |
| Wall clock | 406 s on RTX 5090 |
| Parameter count | ≈ 16.6 M |

## Comparison vs. Other Audio-Family And Cascaded Baselines

Apples-to-oranges caveat: the cascaded baselines were evaluated on the
synthetic-spike dataset, not Niazy proof-fit. Conv-TasNet and Demucs share
exactly the same Niazy proof-fit val split and seed, so the Demucs vs
Conv-TasNet comparison is the most direct.

| Model | Eval dataset | Clean SNR Δ | Artifact corr | Residual RMS ratio |
| --- | --- | ---: | ---: | ---: |
| `cascaded_dae` 20260502_115914 | synthetic_spike | -0.05 dB | 0.09 | 1.005 |
| `cascaded_context_dae` 20260502_115926 | synthetic_spike | +3.16 dB | 0.73 | 0.695 |
| `conv_tasnet` 20260510_224113 | niazy_proof_fit | +22.03 dB | 0.997 | 0.079 |
| **`demucs` 20260511_005636** | **niazy_proof_fit** | **+31.28 dB** | **0.9996** | **0.027** |

The U-Net + BiLSTM inductive bias beat the dilated-TCN sibling by ~9 dB SNR
improvement, ~3× lower residual RMS, and one extra digit of artifact
correlation on the same dataset.

## Caveats Discovered During Evaluation

- **CUDA-only TorchScript export.** `facet-train` traces the model with
  `torch.jit.trace(model.eval(), example_input_on_cuda)`. The LSTM's
  internal hidden-state allocation is captured as a CUDA `aten::empty`
  call, so `torch.jit.load(map_location="cpu")` fails at first forward
  pass on a CPU host. Workaround: re-trace from the `.pt` checkpoint on
  CPU. I added the re-traced `exports/demucs_cpu.ts` alongside the
  original. A clean fix would be either (a) change the CLI export path
  to `torch.jit.script` for LSTM-bearing models, or (b) add a small
  "retrace_on_cpu" post-step in `_export_pytorch_torchscript`. Out of
  scope here — flagged for the orchestrator.
- **Validation is in-distribution.** The val split shares the same
  artifact library (`niazy_aas_2x_direct`) used during training. The
  metric tells us the model learnt the mapping on this distribution; it
  does not show cross-subject, cross-scanner, or cross-artifact-library
  generalisation.
- **Center-only scoring.** Demucs predicts 7 epochs (3584 samples) at a
  time but `DemucsCorrection` only uses the center epoch. Boundary
  epochs are scored implicitly through the L1 training loss but are not
  surfaced in the supervised metrics, matching `conv_tasnet`'s
  evaluation contract.
- **No real-EEG trigger-locked metrics.** Same gap as Conv-TasNet — a
  unified evaluator that runs real Niazy EDF metrics across all models
  is still missing. Adding it is a worthwhile next step but was out of
  scope for this agent.
- **`pytest` `filterwarnings = error`.** `torch.jit.trace` emits a
  `TracerWarning` because the LSTM internally compares a Python int to
  `input.size(-1)`. The test suite turns warnings into errors, so the
  smoke test wraps the trace call in `warnings.catch_warnings()` +
  `warnings.simplefilter("ignore")`. The production CLI does not hit
  this because it runs outside pytest.

## Completeness Confirmation

- `model_card.md` ✅
- `research_notes.md` ✅
- `evaluations.md` updated with run `20260511_005636` ✅
- `README.md` ✅
- `tests/models/demucs/` 11/11 passing ✅
- Smoke run finished, full run finished, evaluation written via
  `ModelEvaluationWriter` ✅
- Pipeline integration via `DeepLearningCorrection` (no
  `facet.core` changes; no edits outside
  `src/facet/models/demucs/`, `tests/models/demucs/`, and the new
  `examples/evaluate_demucs.py`).

The branch is ready for orchestrator review. Not pushed, not merged.
