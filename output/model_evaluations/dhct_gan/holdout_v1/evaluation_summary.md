# Evaluation Run: DHCT-GAN (v1, deprecated)

## Identity

- model id: `dhct_gan`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.157284 |
| `unified_holdout.artifact_mae` | 0.001277 |
| `unified_holdout.artifact_mse` | 1.830413e-05 |
| `unified_holdout.artifact_snr_db` | -7.115756 |
| `unified_holdout.clean_mae_after` | 0.001277 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 1.830413e-05 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | -414.725393 |
| `unified_holdout.clean_snr_db_after` | -18.060701 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | -7.115755 |
| `unified_holdout.inference_seconds` | 27.101549 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 2.268756 |
| `unified_holdout.rms_baseline_noisy_ratio` | 7.254694 |
| `unified_holdout.rms_recovery_distance` | 20.993982 |
| `unified_holdout.rms_recovery_ratio` | 21.993982 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

dhct_gan on the unified holdout split: SNR improvement = -7.12 dB (before=-10.94 dB → after=-18.06 dB). Artifact correlation with ground truth = +0.1573. Residual RMS ratio (after / before) = 2.269.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/dhct_gan/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "dhct_gan",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/dhctganniazyprooffit_20260510_213159/exports/dhct_gan.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "GAN (single-epoch input, failed)",
  "notes": "(1,1,512) in \u2192 (1,1,512) artifact out. Single epoch only, no context. Kept for completeness."
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
    "clean_mse_after": 1.8304128388990648e-05,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.0012767703738063574,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": -18.060701370239258,
    "artifact_mse": 1.8304128388990648e-05,
    "artifact_mae": 0.0012767703738063574,
    "artifact_corr": 0.15728382457439363,
    "artifact_snr_db": -7.115756034851074,
    "residual_error_rms_ratio": 2.268755995514912,
    "rms_recovery_ratio": 21.993982315063477,
    "rms_recovery_distance": 20.993982315063477,
    "rms_baseline_noisy_ratio": 7.25469446182251,
    "clean_mse_reduction_pct": -414.72539262621837,
    "clean_snr_improvement_db": -7.115755081176758,
    "inference_seconds": 27.101549,
    "device": "cpu"
  }
}
```
