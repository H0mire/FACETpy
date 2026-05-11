# Evaluation Run: D4PM

## Identity

- model id: `d4pm`
- run id: `20260510_d4pm_full_e4`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `runtime.inference_seconds_per_segment` | 0.153311 |
| `runtime.inference_seconds_total` | 19.623851 |
| `runtime.num_steps_train` | 200 |
| `runtime.sample_steps` | 30 |
| `shape.n_channels` | 4 |
| `shape.n_examples` | 32 |
| `shape.n_samples` | 512 |
| `shape.sfreq_hz` | 4096.000000 |
| `synthetic_supervised.artifact_corr` | 0.725136 |
| `synthetic_supervised.artifact_pred_rms_error` | 7.118406e-04 |
| `synthetic_supervised.artifact_rms` | 0.001049 |
| `synthetic_supervised.clean_rms_after` | 7.119074e-04 |
| `synthetic_supervised.clean_rms_before` | 0.001049 |
| `synthetic_supervised.clean_snr_db_after` | -13.148385 |
| `synthetic_supervised.clean_snr_db_before` | -16.355381 |
| `synthetic_supervised.residual_rms_ratio` | 0.698610 |

## Interpretation

Single-branch conditional DDPM evaluated on the supervised Niazy proof-fit pairs. Lower residual_rms_ratio is better. clean_snr_db_after - clean_snr_db_before measures correction gain.

## Artifacts

- `example_correction_plot`: `output/model_evaluations/d4pm/20260510_d4pm_full_e4/plots/example_correction.png`

## Limitations

- Single-branch reduction of D4PM; dual-branch joint posterior not implemented.
- Iterative sampling cost grows linearly with sample_steps.
- Niazy proof-fit pairs are synthetic-supervised; real-recording proxy metrics not reported here.

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/d4pmartifactdiffusionniazyprooffit_20260510_201242/checkpoints/epoch0004_loss0.1069.pt",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "epoch_samples": 512,
  "num_steps": 200,
  "sample_steps": 30,
  "data_consistency_weight": 0.5,
  "feats": 64,
  "d_model": 128,
  "d_ff": 512,
  "n_heads": 2,
  "n_layers": 2,
  "embed_dim": 128,
  "device": "cpu",
  "max_channels": 4,
  "max_examples": 32
}
```

## Raw Metrics

```json
{
  "synthetic_supervised": {
    "clean_rms_before": 0.0010492284782230854,
    "clean_rms_after": 0.0007119073998183012,
    "clean_snr_db_before": -16.35538101196289,
    "clean_snr_db_after": -13.148385047912598,
    "artifact_rms": 0.0010491814464330673,
    "artifact_pred_rms_error": 0.0007118406356312335,
    "artifact_corr": 0.7251355051994324,
    "residual_rms_ratio": 0.6986104249954224
  },
  "runtime": {
    "inference_seconds_total": 19.62385082244873,
    "inference_seconds_per_segment": 0.1533113345503807,
    "sample_steps": 30,
    "num_steps_train": 200,
    "device": "cpu"
  },
  "shape": {
    "n_examples": 32,
    "n_channels": 4,
    "n_samples": 512,
    "sfreq_hz": 4096.0
  }
}
```
