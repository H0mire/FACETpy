# IC-U-Net Evaluations

This file links specific evaluation runs to their generated artifacts.
Generated outputs live in `output/model_evaluations/ic_unet/<run_id>/` and
follow `src/facet/models/evaluation_standard.md`.

## Runs

_Populated after each evaluation run completes._

## Comparison Baselines

- `cascaded_context_dae` — closest peer; same 7-epoch context but channel-wise.
- `cascaded_dae` — single-channel windowed DAE baseline.
- DPAE family reference (+7.48 dB clean-SNR improvement) — from
  `docs/research/dl_eeg_gradient_artifacts.pdf` Section 3.2.
