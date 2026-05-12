# Evaluation Run: SepFormer

## Identity

- model id: `sepformer`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.993291 |
| `unified_holdout.artifact_mae` | 1.048927e-04 |
| `unified_holdout.artifact_mse` | 4.787764e-08 |
| `unified_holdout.artifact_snr_db` | 18.708406 |
| `unified_holdout.clean_mae_after` | 1.048927e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 4.787764e-08 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 98.653646 |
| `unified_holdout.clean_snr_db_after` | 7.763461 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 18.708407 |
| `unified_holdout.inference_seconds` | 100.856996 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.116032 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

sepformer on the unified holdout split: SNR improvement = +18.71 dB (before=-10.94 dB → after=+7.76 dB). Artifact correlation with ground truth = +0.9933. Residual RMS ratio (after / before) = 0.116.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/sepformer/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "sepformer",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/sepformerniazyprooffit_20260510_230104/exports/sepformer.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Audio (Transformer)",
  "notes": "(1,7,1,512) in \u2192 (1,1,512) artifact out. Per-epoch demean."
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
    "clean_mse_after": 4.787764140701256e-08,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.00010489272972336039,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 7.763461112976074,
    "artifact_mse": 4.787764140701256e-08,
    "artifact_mae": 0.00010489272972336039,
    "artifact_corr": 0.9932905120930949,
    "artifact_snr_db": 18.708406448364258,
    "residual_error_rms_ratio": 0.1160324938002232,
    "clean_mse_reduction_pct": 98.65364592907552,
    "clean_snr_improvement_db": 18.708407402038574,
    "inference_seconds": 100.856996,
    "device": "cpu"
  }
}
```
