# Evaluation Run: DHCT-GAN v2

## Identity

- model id: `dhct_gan_v2`
- run id: `holdout_v1`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `unified_holdout.artifact_corr` | 0.564440 |
| `unified_holdout.artifact_mae` | 0.001853 |
| `unified_holdout.artifact_mse` | 4.659726e-06 |
| `unified_holdout.artifact_snr_db` | -1.173869 |
| `unified_holdout.clean_mae_after` | 0.001853 |
| `unified_holdout.clean_mae_before` | 9.214178e-04 |
| `unified_holdout.clean_mse_after` | 4.659726e-06 |
| `unified_holdout.clean_mse_before` | 3.556096e-06 |
| `unified_holdout.clean_mse_reduction_pct` | -31.034884 |
| `unified_holdout.clean_snr_db_after` | -12.118815 |
| `unified_holdout.clean_snr_db_before` | -10.944946 |
| `unified_holdout.clean_snr_improvement_db` | -1.173869 |
| `unified_holdout.inference_seconds` | 31.965136 |
| `unified_holdout.n_channels` | 30 |
| `unified_holdout.n_examples` | 166 |
| `unified_holdout.residual_error_rms_ratio` | 1.144705 |
| `unified_holdout.rms_baseline_noisy_ratio` | 7.254694 |
| `unified_holdout.rms_recovery_distance` | 4.841935 |
| `unified_holdout.rms_recovery_ratio` | 5.841934 |
| `unified_holdout.samples_per_epoch` | 512 |
| `unified_holdout.sfreq_hz` | 4096.000000 |

## Interpretation

dhct_gan_v2 on the unified holdout split: SNR improvement = -1.17 dB (before=-10.94 dB → after=-12.12 dB). Artifact correlation with ground truth = +0.5644. Residual RMS ratio (after / before) = 1.145.

## Artifacts

- `holdout_examples`: `/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/model_evaluations/dhct_gan_v2/holdout_v1/plots/holdout_examples.png`

## Limitations

- Unified holdout split: window-level seed=42 val_ratio=0.2 → 166 windows (4980 channel-windows).
- Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.
- Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.

## Configuration

```json
{
  "model_id": "dhct_gan_v2",
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/dhctganv2niazyprooffit_20260510_220534/exports/dhct_gan_v2.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "holdout_split_hash": "sha256:ddaa64a504e062fd",
  "holdout_seed": 42,
  "holdout_val_ratio": 0.2,
  "holdout_n_windows": 166,
  "family": "GAN (hybrid CNN+Transformer, ctx fix)",
  "notes": "(1,7,512) in \u2014 note flat 7-epoch packing, not (1,7,1,512). Per-epoch demean."
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
    "clean_mse_after": 4.659726073441561e-06,
    "clean_mae_before": 0.0009214178426191211,
    "clean_mae_after": 0.0018529613735154271,
    "clean_snr_db_before": -10.9449462890625,
    "clean_snr_db_after": -12.118815422058105,
    "artifact_mse": 4.659726073441561e-06,
    "artifact_mae": 0.0018529613735154271,
    "artifact_corr": 0.5644395476653311,
    "artifact_snr_db": -1.1738691329956055,
    "residual_error_rms_ratio": 1.1447047136947361,
    "rms_recovery_ratio": 5.8419342041015625,
    "rms_recovery_distance": 4.841934680938721,
    "rms_baseline_noisy_ratio": 7.25469446182251,
    "clean_mse_reduction_pct": -31.03488359082116,
    "clean_snr_improvement_db": -1.1738691329956055,
    "inference_seconds": 31.965136,
    "device": "cpu"
  }
}
```
