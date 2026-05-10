# ViT Spectrogram Inpainter — Evaluations Index

This file lists evaluation runs for `vit_spectrogram`. Generated artifacts
live under `output/model_evaluations/vit_spectrogram/<run_id>/` as defined
by `src/facet/models/evaluation_standard.md`.

## Runs

| Run id | Dataset | Notes | Manifest |
|---|---|---|---|
| _pending — first full Niazy proof-fit evaluation will be appended after the orchestrator fetches the full-run results._ | — | — | — |

## Methodology

- Uses `facet.evaluation.ModelEvaluationWriter` so every run emits
  `evaluation_manifest.json`, `metrics.json`, and `evaluation_summary.md`
  side-by-side under `output/model_evaluations/vit_spectrogram/<run_id>/`.
- Required metric groups for the Niazy proof-fit (synthetic supervised)
  dataset are taken directly from
  `src/facet/models/evaluation_standard.md`:
  - clean reconstruction error before/after correction
  - clean SNR before/after correction
  - artifact prediction error
  - artifact correlation
  - residual RMS ratio
- Baselines for comparison: `cascaded_dae` (no context) and
  `cascaded_context_dae` (same 7-epoch context, time-domain
  autoencoder).

## Caveats Carried Forward From Research Notes

- Magnitude-only reconstruction with preserved noisy phase is lossy at
  GA-dominated bins (see `research_notes.md`, "Phase-handling decision").
  If `vit_spectrogram` underperforms `cascaded_context_dae` on residual
  metrics specifically — but performs well on clean-signal reconstruction
  in artifact-free regions — that pattern is consistent with the phase
  bottleneck rather than with a fundamentally unsuitable architecture.
- The Niazy proof-fit dataset uses AAS-corrected EEG as the "clean"
  surrogate target. Improvements over AAS are bounded by AAS's own
  residual artifact level. Generalization to held-out recordings is **not
  validated** by this evaluation; that is an explicit limitation of the
  proof-fit benchmark and is documented in the report and in the dataset
  builder's docstring.
