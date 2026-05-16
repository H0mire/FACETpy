# Evaluation Run: Cascaded Context DAE

## Identity

- model id: `cascaded_context_dae`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.993602 |
| `unified_holdout.artifact_mae` | 1.624090e-04 |
| `unified_holdout.artifact_mse` | 4.559828e-08 |
| `unified_holdout.artifact_snr_db` | 18.920250 |
| `unified_holdout.clean_mae_after` | 1.624090e-04 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 4.559828e-08 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | 98.717743 |
| `unified_holdout.clean_snr_db_after` | 7.975304 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | 18.920250 |
| `unified_holdout.inference_seconds` | 0.108307 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 0.113237 |
| `unified_holdout.rms_baseline_noisy_ratio` | 7.254694 |
| `unified_holdout.rms_recovery_distance` | 0.569076 |
| `unified_holdout.rms_recovery_ratio` | 1.514302 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

cascaded_context_dae on the unified holdout split: SNR improvement = +18.92 dB (before=-10.94 dB → after=+7.98 dB). Artifact correlation with ground truth = +0.9936. Residual RMS ratio (after / before) = 0.113.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/cascaded_context_dae/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "cascaded_context_dae",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/cascadedcontextdenoisingautoencoderniazyprooffit_20260516_152603/exports/cascaded_context_dae.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "Autoencoder (context MLP)",
  "notes": "(1,7,1,512) in \u2192 (1,1,512) artifact out. Per-epoch demean. Glob picks newest matching training_output run."
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
    "clean_mse_after": 4.55982807068267e-08,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.0001624089782126248,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": 7.975304126739502,
    "artifact_mse": 4.559827715411302e-08,
    "artifact_mae": 0.0001624089782126248,
    "artifact_corr": 0.9936016833360306,
    "artifact_snr_db": 18.920249938964844,
    "residual_error_rms_ratio": 0.11323677582097232,
    "rms_recovery_ratio": 1.5143024921417236,
    "rms_recovery_distance": 0.5690756440162659,
    "rms_baseline_noisy_ratio": 7.25469446182251,
    "clean_mse_reduction_pct": 98.71774320846555,
    "clean_snr_improvement_db": 18.920250415802002,
    "inference_seconds": 0.108307,
    "device": "cpu"
  }
}
```
