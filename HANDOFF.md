# DenoiseMamba — Agent Hand-off

`denoise_mamba` (Section 6.2 — ConvSSD state-space) is trained, evaluated, and
ready for the orchestrator to merge or compare. Metrics are clearly above the
DPAE baseline on the same Niazy proof-fit dataset.

## Identity

- **Branch:** `feature/model-denoise_mamba`
- **Worktree:** `worktrees/model-denoise_mamba` (off `feature/add-deeplearning`)
- **Last commits on branch:**
  - `ca18c3d` — eval script (ModelEvaluationWriter)
  - `22bda53` — model implementation + tests
- **Model ID / name:** `denoise_mamba` / `DenoiseMamba`
- **Architecture family:** Sequence Modeling (State Space / Mamba)
- **Source paper:** IEEE Xplore 11012652 — *DenoiseMamba: An Innovative Approach
  for EEG Artifact Removal Leveraging Mamba and CNN*
- **Background:** Mamba/SSD (arXiv 2405.21060), HiPPO (NeurIPS 2020).

## Smoke run

- Fleet job id: `493f86a3fdb7`
- Session: `denoise_mamba_niazy_smoke`
- Worker: `gpu1` (RTX 5090)
- Status: `finished` (exit 0)
- Run dir on worktree:
  `worktrees/model-denoise_mamba/training_output/denoisemambaniazysmoke_20260510_193254/`
- Wall-clock: ~5 s of actual training (1 epoch, tiny config). Smoke produced
  `training.jsonl`, `loss.png`, `summary.json`, and
  `exports/denoise_mamba.ts`.

## Full run

- Fleet job id: `ef18b567acc3`
- Session: `denoise_mamba_niazy_full`
- Worker: `gpu1`
- Status: `finished` (exit 0)
- Run dir on worktree:
  `worktrees/model-denoise_mamba/training_output/denoisemambaniazyprooffit_20260510_193847/`
- Wall-clock: **5 h 02 min** (18 155 s). Early stopping triggered at
  **epoch 11** (`patience=10` on training-loss `min_delta=1e-6`).
- Best epoch / metric: **11 / 2.30e-7** (validation MSE).
- Exported checkpoint:
  `training_output/denoisemambaniazyprooffit_20260510_193847/exports/denoise_mamba.ts`
  (CUDA-traced — see *Caveats* below).
- A CPU-portable re-export was saved as
  `exports/denoise_mamba_cpu.ts` for local evaluation.

### Training trajectory

| Epoch | train_loss | val_loss |
| ---: | ---: | ---: |
| 1  | 4.56e-03 | 2.87e-07 |
| 2  | 5.44e-06 | 2.74e-07 |
| 3  | 3.24e-06 | 2.52e-07 |
| 4  | 2.01e-06 | 2.41e-07 |
| 5  | 1.30e-06 | 2.35e-07 |
| 6  | 8.92e-07 | 2.35e-07 |
| 7  | 7.25e-07 | 5.70e-06 |
| 8  | 1.84e-06 | 2.52e-07 |
| 9  | 4.40e-06 | 2.48e-07 |
| 10 | 5.32e-06 | 2.33e-07 |
| 11 | 5.69e-06 | **2.30e-07** |

The model essentially converged after epoch 1; the remaining epochs trimmed
val loss by ~20 % at the cost of train-loss oscillation.

## Evaluation

- Script: `src/facet/models/denoise_mamba/evaluate.py` (mirrors the DPAE and
  Cascaded-Context-DAE metric schema).
- Evaluation run id: `niazy_proof_fit_full`
- Manifest:
  `worktrees/model-denoise_mamba/output/model_evaluations/denoise_mamba/niazy_proof_fit_full/evaluation_manifest.json`
- Metrics: same dir, `metrics.json` (flat + nested under `niazy_proof_fit`).
- Summary: same dir, `evaluation_summary.md`.
- Plot: `plots/niazy_proof_fit_examples.png` (4 example traces).
- Evaluated 24 990 channel-wise examples (833 trigger-defined epochs × 30 EEG
  channels) of 512 samples each, on Mac CPU.

### Key metrics on the Niazy proof-fit dataset

| Metric | Value |
| --- | ---: |
| Clean SNR before correction | -11.59 dB |
| Clean SNR after correction | +0.21 dB |
| **Clean SNR improvement** | **+11.80 dB** |
| Clean MSE before correction | 3.54e-06 |
| Clean MSE after correction | 2.34e-07 |
| **Clean MSE reduction** | **93.39 %** |
| Predicted-vs-true artifact correlation | **0.966** |
| Residual error RMS ratio | **0.257** |
| Artifact SNR | 11.80 dB |
| Predicted artifact edge / center ratio | 0.60 |
| Inference (CPU, 24 990 segments) | 328 s |

### Comparison vs other models in the same `output/model_evaluations/` tree

| Model | Dataset group | SNR gain (dB) | Artifact corr | Residual RMS ratio | MSE reduction (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| **denoise_mamba** | `niazy_proof_fit` | **11.80** | **0.966** | **0.257** | **93.39** |
| dpae | `niazy_proof_fit` | 7.48 | 0.913 | 0.423 | 82.12 |
| vit_spectrogram | `synthetic` | 11.60 | 0.966 | 0.263 | 93.08 |
| cascaded_context_dae | `synthetic` | 3.16 | 0.732 | 0.695 | 51.68 |
| cascaded_dae | `synthetic` | -0.05 | 0.086 | 1.005 | -1.07 |
| demucs | `synthetic_niazy_proof_fit_val_split` | 31.28 | 0.9996 | 0.027 | 99.93 |

**Caveat on cross-model comparison:** only `dpae` and `denoise_mamba` were
evaluated on exactly the same `niazy_proof_fit` group, so the apples-to-apples
result is *DenoiseMamba beats DPAE by +4.32 dB SNR, +0.05 artifact correlation,
−0.17 residual RMS ratio*. `cascaded_dae` / `cascaded_context_dae` numbers are
from an older synthetic spike dataset and `demucs` from a 1-channel val split
— do **not** read those as worse/better than DenoiseMamba without rerunning
them through the same evaluation script first.

## Tests

`uv run pytest tests/models/denoise_mamba -v` — **10 passed** (last run 2026-05-11 03:30 CEST).
Covered:

- Factory returns the expected `nn.Module` with `(B, 1, samples)` IO.
- Forward / one-batch backward pass on CPU.
- Multi-channel input is rejected at the model boundary.
- `ChannelWiseSingleEpochArtifactDataset` channel expansion.
- `build_dataset` consumes the real NPZ bundle shape.
- TorchScript-loaded `DenoiseMambaCorrection` applies chunk-wise artifact
  subtraction end-to-end through a `ProcessingContext`.

## Documentation completeness

- `src/facet/models/denoise_mamba/README.md` — usage + architectural decision.
- `src/facet/models/denoise_mamba/documentation/research_notes.md` — sources,
  ConvSSD reasoning, hyperparameter justification, hardware feasibility.
- `src/facet/models/denoise_mamba/documentation/model_card.md` — formal model
  card.
- `src/facet/models/denoise_mamba/documentation/evaluations.md` — index of
  evaluation runs (the writer also auto-indexes new runs).

## Caveats discovered during this run

1. **Pure-PyTorch selective scan is the wall-clock bottleneck.** The
   sequential scan over 512 time-steps × 4 blocks gives ~28 min/epoch on one
   RTX 5090 with `d_model=64`, `d_state=16`, batch 64, on 19 992 train
   chunks. 60 epochs would have cost ~28 h; early stopping kept it to 5 h.
2. **CUDA-traced TorchScript is not CPU-portable.** The cuda-trained
   `denoise_mamba.ts` contains a baked `device=cuda:0` inside
   `torch.zeros(...)` of the SelectiveSSM state, so loading it on CPU fails
   with `aten::empty.memory_format ... only available for [CPU, MPS, ...]`.
   For Mac/CPU evaluation, re-export the state_dict on CPU and re-trace.
   Helper command used:
   ```python
   model = build_model(epoch_samples=512, d_model=64, d_state=16, expand=2,
                       d_conv=4, n_blocks=4, dropout=0.1, input_kernel_size=7)
   state = torch.load("checkpoints/epoch0011_loss0.0000.pt",
                      map_location="cpu", weights_only=True)
   model.load_state_dict(state["model_state_dict"])
   model.eval()
   torch.jit.trace(model, torch.randn(1, 1, 512)).save("denoise_mamba_cpu.ts")
   ```
   This is fine for evaluation, but production inference should keep the
   cuda-traced artifact on GPU workers. Long term: replace the explicit
   `torch.zeros(..., device=x.device, ...)` in `SelectiveSSM` with
   `x.new_zeros(...)` and trace on CPU, OR move to `torch.jit.script` once
   the `assert` / shape-check lines are scriptable.
3. **Train-loss / val-loss divergence after epoch 6.** Train loss climbs
   from 7.3e-7 (epoch 7) to 5.7e-6 (epoch 11) while val loss keeps
   decreasing very slowly. The selective-scan path is mildly unstable at
   this tiny loss scale. Not a blocker for the proof-fit result; worth
   watching if anyone scales to harder datasets.
4. **`tools/gpu_fleet/fleet.py` was hot-patched into this worktree at
   2026-05-10 22:20** by what looks like an orchestrator action (added a
   `_resolve_main_worktree` helper that resolves `REPO_ROOT` to the main
   worktree so all linked worktrees share `.facet_gpu_fleet/queue.json`).
   That change is **not** committed on this branch. If you want it on
   `feature/add-deeplearning`, pick it up from whichever agent owns the
   author commit.

## Files written / modified by this agent

```text
src/facet/models/denoise_mamba/__init__.py
src/facet/models/denoise_mamba/README.md
src/facet/models/denoise_mamba/processor.py
src/facet/models/denoise_mamba/training.py
src/facet/models/denoise_mamba/training_niazy_proof_fit.yaml
src/facet/models/denoise_mamba/training_niazy_proof_fit_smoke.yaml
src/facet/models/denoise_mamba/evaluate.py
src/facet/models/denoise_mamba/documentation/research_notes.md
src/facet/models/denoise_mamba/documentation/model_card.md
src/facet/models/denoise_mamba/documentation/evaluations.md
tests/models/denoise_mamba/__init__.py
tests/models/denoise_mamba/test_processor.py
tests/models/denoise_mamba/test_training_smoke.py
HANDOFF.md
```

Generated artifacts (ignored by git, kept on the worktree only):

```text
training_output/denoisemambaniazysmoke_20260510_193254/
training_output/denoisemambaniazyprooffit_20260510_193847/
  └── exports/denoise_mamba.ts        (cuda-traced)
  └── exports/denoise_mamba_cpu.ts    (cpu re-trace for local eval)
remote_logs/denoise_mamba_niazy_smoke.{log,sh,runner.sh,exitcode}
remote_logs/denoise_mamba_niazy_full.{log,sh,runner.sh,exitcode}
output/model_evaluations/denoise_mamba/niazy_proof_fit_full/
  └── evaluation_manifest.json
  └── metrics.json
  └── evaluation_summary.md
  └── plots/niazy_proof_fit_examples.png
```

I did not push, merge, or modify other agents' worktrees, did not run the
fleet dispatcher, and did not iterate hyperparameters after the slow full
run. Hand-off is ready.
