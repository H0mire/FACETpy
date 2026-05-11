# Demucs Evaluations

Standardized evaluation runs for `demucs` should be stored under:

```text
output/model_evaluations/demucs/<run_id>/
```

Each run must include:

- `evaluation_manifest.json`
- `metrics.json`
- `evaluation_summary.md`

The first evaluation should compare:

- Synthetic supervised correction on the Niazy proof-fit context dataset
  (held-out validation split).
- Trigger-locked real-data proxy metrics on the Niazy recording.
- Visual plots for noisy, predicted artifact, corrected, and reference clean
  (AAS-corrected) waveforms.
- Direct comparison vs `cascaded_dae`, `cascaded_context_dae`, and
  `conv_tasnet` (the audio-family sibling).

## Recorded Runs

### `20260511_005636` — Niazy proof-fit, 60-epoch full training

- Checkpoint: `training_output/demucsniazyprooffit_20260510_224653/exports/demucs.ts`
  (the CUDA-traced export saved by `facet-train`; the evaluator used a
  re-traced CPU copy under `exports/demucs_cpu.ts` because the LSTM trace
  baked in a CUDA backend hint that prevented map-loading the CUDA `.ts` on
  CPU).
- Validation split: 4998 (example, channel) pairs (20 %, seed 42).
- Headline metrics on the center epoch (matches how `DemucsCorrection`
  applies the correction):

| Metric | Value |
| --- | ---: |
| Clean SNR before | -11.61 dB |
| Clean SNR after | +19.67 dB |
| **Clean SNR improvement** | **+31.28 dB** |
| **Clean MSE reduction** | **99.93 %** |
| **Artifact correlation** | **0.9996** |
| Artifact SNR | +31.28 dB |
| Residual error RMS ratio | 0.027 |

- Artifacts: `output/model_evaluations/demucs/20260511_005636/`
  (`evaluation_manifest.json`, `metrics.json`, `evaluation_summary.md`,
  `plots/demucs_examples.png`, `plots/demucs_metric_summary.png`).
