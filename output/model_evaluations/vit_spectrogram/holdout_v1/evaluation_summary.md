# Evaluation Run: ViT Spectrogram Inpainter

## Identity

- model id: `vit_spectrogram`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.960525 |
| `unified_holdout.artifact_mae` | 2.344912e-04 |
| `unified_holdout.artifact_mse` | 2.859251e-07 |
| `unified_holdout.artifact_snr_db` | 10.947212 |
| `unified_holdout.clean_mae_after` | 2.344912e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 2.859251e-07 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 91.959578 |
| `unified_holdout.clean_snr_db_after` | 0.002265 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 10.947212 |
| `unified_holdout.inference_seconds` | 8.582679 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.283556 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

vit_spectrogram on the unified holdout split: SNR improvement = +10.95 dB (before=-10.94 dB → after=+0.00 dB). Artifact correlation with ground truth = +0.9605. Residual RMS ratio (after / before) = 0.284.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/vit_spectrogram/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "vit_spectrogram",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/vitspectrograminpainterniazyprooffit_20260510_211842/exports/vit_spectrogram.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Vision (MAE)",
  "notes": "(1,7,1,512) in \u2192 (1,1,512) CLEAN out (not artifact). artifact = noisy_demeaned - pred_clean."
}
```

## Raw Metrics

```json
{
  "unified_holdout": {
    "n_examples": 166,
    "n_channels": 30,
    "samples_per_epoch": 512,
    "sfreq_hz": 4096.0,
    "clean_mse_before": 3.556095862222719e-06,
    "clean_mse_after": 2.8592512535396963e-07,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.0002344912354601547,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 0.002265466609969735,
    "artifact_mse": 2.8592512535396963e-07,
    "artifact_mae": 0.0002344912354601547,
    "artifact_corr": 0.9605251892044411,
    "artifact_snr_db": 10.947212219238281,
    "residual_error_rms_ratio": 0.2835563787833825,
    "clean_mse_reduction_pct": 91.95957768204671,
    "clean_snr_improvement_db": 10.94721175567247,
    "inference_seconds": 8.582679,
    "device": "cpu"
  }
}
```
