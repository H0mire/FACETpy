# Evaluation Run: Nested-GAN

## Identity

- model id: `nested_gan`
- run id: `20260511_004531`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `real_proxy.predicted_artifact_rms` | 0.001848 |
| `real_proxy.rms_reduction_pct` | 80.601731 |
| `real_proxy.trigger_locked_rms_after` | 3.705312e-04 |
| `real_proxy.trigger_locked_rms_before` | 0.001910 |
| `synthetic.artifact_correlation` | 0.977631 |
| `synthetic.artifact_prediction_l1` | 1.433118e-04 |
| `synthetic.artifact_prediction_rms` | 3.957818e-04 |
| `synthetic.clean_reconstruction_l1_after` | 1.433118e-04 |
| `synthetic.clean_reconstruction_l1_before` | 9.189959e-04 |
| `synthetic.clean_reconstruction_l2_after` | 3.957818e-04 |
| `synthetic.clean_reconstruction_l2_before` | 0.001881 |
| `synthetic.clean_snr_db_after` | 1.946879 |
| `synthetic.clean_snr_db_before` | -11.593128 |
| `synthetic.clean_snr_improvement_db` | 13.540007 |
| `synthetic.epoch_samples` | 512 |
| `synthetic.n_channels` | 30 |
| `synthetic.n_examples` | 833 |
| `synthetic.residual_rms_ratio` | 0.210378 |

## Interpretation

Clean-SNR improvement: 13.54 dB. Artifact correlation: 0.978. Residual RMS ratio: 0.210.

## Artifacts

- `example_0425_ch23.png`: `plots/example_0425_ch23.png`
- `example_0530_ch17.png`: `plots/example_0530_ch17.png`
- `example_0708_ch15.png`: `plots/example_0708_ch15.png`

## Limitations

- Evaluation uses the same NPZ bundle that trained the model; no held-out subject.
- Edge epochs at the start/end of each recording are not corrected.

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/nestedganniazyprooffit_20260510_222546/exports/nested_gan.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "device": "cpu",
  "batch_size": 64,
  "demean_input": true,
  "remove_prediction_mean": true,
  "ch_names": [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
    "AF4",
    "AF3",
    "FC2",
    "FC1",
    "CP1",
    "CP2",
    "PO3",
    "PO4",
    "FC6",
    "FC5",
    "CP5"
  ],
  "sfreq": 4096.0
}
```

## Raw Metrics

```json
{
  "synthetic": {
    "n_examples": 833,
    "n_channels": 30,
    "epoch_samples": 512,
    "clean_reconstruction_l1_before": 0.0009189959382638335,
    "clean_reconstruction_l1_after": 0.00014331183047033846,
    "clean_reconstruction_l2_before": 0.0018812920895137598,
    "clean_reconstruction_l2_after": 0.00039578184596752585,
    "clean_snr_db_before": -11.593128071318722,
    "clean_snr_db_after": 1.946879120755847,
    "artifact_prediction_l1": 0.00014331183047033846,
    "artifact_prediction_rms": 0.00039578184596725745,
    "artifact_correlation": 0.9776313969988457,
    "residual_rms_ratio": 0.21037766978612063,
    "clean_snr_improvement_db": 13.540007192074569
  },
  "real_proxy": {
    "trigger_locked_rms_before": 0.001910124991888904,
    "trigger_locked_rms_after": 0.0003705311840507346,
    "predicted_artifact_rms": 0.001847858500231697,
    "rms_reduction_pct": 80.6017310058689
  }
}
```
