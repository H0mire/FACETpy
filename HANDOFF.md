# Conv-TasNet Hand-Off

## Identity

- Branch: `feature/model-conv_tasnet`
- Worktree: `worktrees/model-conv_tasnet/`
- Latest commit: `dd3fd88` (on top of upstream `32190a0`).
- Model id: `conv_tasnet`
- Model name: `Conv-TasNet`
- Architecture family: Audio-Inspired Source Separation
- Report section: 7.1.1 (Conv-TasNet: Time-Domain Audio Separation Network)

## What Was Done

1. **Phase 1 mandatory reading** complete (CLAUDE.md, AGENTS.md,
   `docs/PROCESSOR_GUIDELINES.md`, `src/facet/models/README.md`,
   `docs/source/development/training_cli_architecture.md`,
   `dl_eeg_gradient_artifacts.pdf` §7.1.1, `architecture_catalog.md`,
   `deep_learning_parallel_runpod_workflow.md`,
   `evaluation_standard.md`, `cascaded_dae/processor.py`, full
   `cascaded_context_dae/`).
2. **Phase 2 research notes**: `src/facet/models/conv_tasnet/documentation/research_notes.md`.
3. **Phase 3 reuse**: training routes through `facet-train` →
   `PyTorchModelWrapper`; dataset is a `ChannelWiseSourceSeparationDataset`
   over the existing Niazy proof-fit `.npz` bundle (no new dataset
   builder needed); evaluation uses
   `facet.evaluation.ModelEvaluationWriter`.
4. **Phase 4 worktree** created off `feature/add-deeplearning`.
5. **Phase 5 implementation**:
   - `src/facet/models/conv_tasnet/training.py` — `ConvTasNetSeparator`
     (encoder + dilated TCN + decoder, gLN, skip-sum, sigmoid masks),
     `ChannelWiseSourceSeparationDataset`, `_NegSISDR` and
     `_WeightedSourceMSE`, `build_model` / `build_loss` / `build_dataset`.
   - `src/facet/models/conv_tasnet/processor.py` — `ConvTasNetAdapter`
     (subclasses `DeepLearningModelAdapter`, declares
     `DeepLearningModelSpec(supports_chunking=True, chunk_size_samples=512,
     execution_granularity=CHANNEL, supports_multichannel=False)`,
     extracts source 1 (artifact) into `DeepLearningPrediction`) and
     `ConvTasNetCorrection` (`@register_processor`, subclasses
     `DeepLearningCorrection`, refuses parallel mode).
   - `src/facet/models/conv_tasnet/training_niazy_proof_fit.yaml` and
     `training_niazy_proof_fit_smoke.yaml` (both `device: cuda`).
   - Documentation: `README.md`, `documentation/model_card.md`,
     `documentation/research_notes.md`,
     `documentation/evaluations.md`.
   - Tests: `tests/models/conv_tasnet/test_processor.py` (factory,
     forward shape, one-batch backward, loss variants, dataset shape,
     correction-subtracts-artifact end-to-end) and
     `test_training_smoke.py` (5-step loss-decreases sanity).
6. **Phase 6 smoke** submitted as `5f0e7fd306e4` on gpu2,
   `finished` (val_loss 5.48e-7 after 1 epoch). All required
   artefacts present: `summary.json`, `loss.png`, `exports/conv_tasnet.ts`.
7. **Phase 7 full training** submitted as `4901c5314d85` on gpu2,
   `finished`. Required hot-patch fix (`32190a0 fix: resolve training
   config relative to submit worktree`) was rebased into this branch
   before resubmit.
8. **Phase 8 evaluation**: `examples/evaluate_conv_tasnet.py` runs the
   trained TorchScript on the held-out 20 % val split of the Niazy
   proof-fit `.npz` and writes manifest/metrics/summary/plots via
   `ModelEvaluationWriter`. Run id `20260510_224113`.
9. **Phase 9 tests**: `uv run pytest tests/models/conv_tasnet -v`
   passes 7/7.

## Run Identifiers and Paths

| Item | Path |
| --- | --- |
| Smoke job id | `5f0e7fd306e4` |
| Smoke run dir | `training_output/convtasnetniazysmoke_20260510_195511/` |
| Smoke export | `training_output/convtasnetniazysmoke_20260510_195511/exports/conv_tasnet.ts` |
| Full job id | `4901c5314d85` |
| Full run dir | `training_output/convtasnetniazyprooffit_20260510_202818/` |
| Full export | `training_output/convtasnetniazyprooffit_20260510_202818/exports/conv_tasnet.ts` |
| Eval manifest | `output/model_evaluations/conv_tasnet/20260510_224113/evaluation_manifest.json` |
| Eval metrics | `output/model_evaluations/conv_tasnet/20260510_224113/metrics.json` |
| Eval summary | `output/model_evaluations/conv_tasnet/20260510_224113/evaluation_summary.md` |
| Eval plots | `output/model_evaluations/conv_tasnet/20260510_224113/plots/` |

## Headline Metrics (Niazy proof-fit val split)

| Metric | Value |
| --- | ---: |
| Clean SNR before | -11.61 dB |
| Clean SNR after | 10.42 dB |
| **Clean SNR improvement** | **+22.03 dB** |
| **Clean MSE reduction** | **99.37 %** |
| **Artifact correlation** | **0.997** |
| Artifact SNR | 22.03 dB |
| Residual error RMS ratio | 0.079 |
| Predicted source-sum MSE | 5.52e-09 |
| Best epoch | 12 / 13 |
| Best val MSE | 1.98e-08 |
| Wall clock | 158 s on RTX 5090 |
| Parameter count | ≈ 1.72 M |

## Comparison vs. Cascaded Baselines

Apples-to-oranges note: the only published baselines available are on
the synthetic-spike dataset, not the Niazy proof-fit set Conv-TasNet
trains on. Numbers below are for orientation only.

| Model | Eval dataset | Clean SNR Δ | Artifact corr | Residual RMS ratio |
| --- | --- | ---: | ---: | ---: |
| `cascaded_dae` 20260502_115914 | synthetic_spike | -0.05 dB | 0.09 | 1.005 |
| `cascaded_context_dae` 20260502_115926 | synthetic_spike | +3.16 dB | 0.73 | 0.695 |
| **`conv_tasnet` 20260510_224113** | **niazy_proof_fit** | **+22.03 dB** | **0.997** | **0.079** |

Even allowing for the dataset difference, this is a clearly working
model. The per-source consistency `clean + artifact ≈ noisy` falls out
of training without an explicit consistency penalty (sum MSE 5.5e-9,
five orders of magnitude below the supervised loss).

## Caveats Discovered During Evaluation

- Validation is **in-distribution**: the val split shares the same
  artifact library (`niazy_aas_2x_direct`) used during training. The
  metric tells us the model learnt the mapping, not that it
  generalises across subjects, scanners, or to MFF data.
- The evaluator does **not** run real Niazy EDF trigger-locked metrics
  (the existing `examples/evaluate_context_artifact_model.py` is
  hard-wired to the cascaded model families). Adding that path —
  ideally as a unified evaluator that includes Conv-TasNet alongside
  the cascaded models — is a worthwhile next step but was out of scope
  for this agent.
- Source-additivity (`clean + artifact = noisy`) is not enforced
  during training. `DeepLearningCorrection` only consumes the
  artifact source, so any residual that the network splits between the
  two sources stays out of the corrected signal.

## Completeness Confirmation

- `model_card.md` ✅
- `research_notes.md` ✅
- `evaluations.md` updated with run `20260510_224113` ✅
- `README.md` ✅
- `tests/models/conv_tasnet/` 7/7 passing ✅
- Smoke run finished, full run finished, evaluation written via
  `ModelEvaluationWriter` ✅

Pipeline integration via `DeepLearningCorrection` (no
`facet.core` changes; no edits outside
`src/facet/models/conv_tasnet/`, `tests/models/conv_tasnet/`, and
the new `examples/evaluate_conv_tasnet.py`).

The branch is ready for orchestrator review. Not pushed, not merged.
