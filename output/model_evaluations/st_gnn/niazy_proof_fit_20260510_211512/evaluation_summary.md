# Evaluation Run: Spatiotemporal Graph Neural Network

## Identity

- model id: `st_gnn`
- run id: `niazy_proof_fit_20260510_211512`
- schema: `facetpy.model_evaluation.v1`

## Metric Summary

| Metric | Value |
| --- | ---: |
| `per_channel.artifact_rmse_AF3` | 2.273125e-04 |
| `per_channel.artifact_rmse_AF4` | 1.533639e-04 |
| `per_channel.artifact_rmse_C3` | 3.304672e-04 |
| `per_channel.artifact_rmse_C4` | 4.575627e-04 |
| `per_channel.artifact_rmse_CP1` | 3.879861e-04 |
| `per_channel.artifact_rmse_CP2` | 5.886260e-04 |
| `per_channel.artifact_rmse_CP5` | 5.762668e-04 |
| `per_channel.artifact_rmse_Cz` | 4.334658e-04 |
| `per_channel.artifact_rmse_F3` | 2.279353e-04 |
| `per_channel.artifact_rmse_F4` | 4.033467e-04 |
| `per_channel.artifact_rmse_F7` | 1.885649e-04 |
| `per_channel.artifact_rmse_F8` | 3.973043e-04 |
| `per_channel.artifact_rmse_FC1` | 1.909804e-04 |
| `per_channel.artifact_rmse_FC2` | 5.302907e-04 |
| `per_channel.artifact_rmse_FC5` | 2.502839e-04 |
| `per_channel.artifact_rmse_FC6` | 4.594355e-04 |
| `per_channel.artifact_rmse_Fp1` | 3.917952e-04 |
| `per_channel.artifact_rmse_Fp2` | 3.265463e-04 |
| `per_channel.artifact_rmse_Fz` | 2.956298e-04 |
| `per_channel.artifact_rmse_O1` | 8.092501e-04 |
| `per_channel.artifact_rmse_O2` | 0.001078 |
| `per_channel.artifact_rmse_P3` | 6.844032e-04 |
| `per_channel.artifact_rmse_P4` | 8.730928e-04 |
| `per_channel.artifact_rmse_PO3` | 6.438371e-04 |
| `per_channel.artifact_rmse_PO4` | 8.520593e-04 |
| `per_channel.artifact_rmse_Pz` | 6.793730e-04 |
| `per_channel.artifact_rmse_T3` | 3.912307e-04 |
| `per_channel.artifact_rmse_T4` | 7.233492e-04 |
| `per_channel.artifact_rmse_T5` | 4.868780e-04 |
| `per_channel.artifact_rmse_T6` | 3.805635e-04 |
| `synthetic.artifact_prediction_correlation` | 0.959548 |
| `synthetic.artifact_prediction_mse` | 2.821578e-07 |
| `synthetic.clean_reconstruction_mse_after` | 2.821748e-07 |
| `synthetic.clean_reconstruction_mse_before` | 3.556096e-06 |
| `synthetic.clean_snr_db_after` | 0.059607 |
| `synthetic.clean_snr_db_before` | -10.944946 |
| `synthetic.residual_rms_ratio` | 0.277410 |
| `synthetic.rms_corrected` | 1.273809e-04 |
| `synthetic.rms_noisy` | 0.001915 |
| `synthetic.rms_residual` | 5.312013e-04 |

## Interpretation

Validation set: 166 examples (seed=42, val_ratio=0.2). Chebyshev order K=3, hidden=16, two ST-Conv blocks. Compare flat_metrics.synthetic.* against cascaded_context_dae and cascaded_dae on the same Niazy proof-fit split.

## Artifacts

- `loss_log`: `plots/loss_log.png`
- `per_channel_rmse`: `plots/per_channel_rmse.png`
- `validation_examples`: `plots/validation_examples.png`

## Limitations

- Niazy proof-fit dataset uses AAS-derived clean and artifact targets; absolute metrics overstate generalisation to independent recordings.
- Validation split is in-recording; the model has been exposed to the same artifact morphology family at training time.
- Per-electrode adjacency assumes the Niazy 30-channel layout in the exact stored order.

## Configuration

```json
{
  "checkpoint": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/training_output/spatiotemporalgnnniazyprooffit_20260510_211512/exports/st_gnn.ts",
  "dataset": "/Users/janikmueller/Documents/Projects/FACETpy/git-repos/facetpy/output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz",
  "split": {
    "val_ratio": 0.2,
    "seed": 42,
    "n_train": 667,
    "n_val": 166
  },
  "device": "cpu",
  "architecture": {
    "context_epochs": 7,
    "n_channels": 30,
    "samples": 512,
    "hidden_channels": 16,
    "k_order": 3,
    "time_kernel": 3,
    "knn_k": 4
  }
}
```

## Raw Metrics

```json
{
  "synthetic": {
    "clean_reconstruction_mse_before": 3.556095862222719e-06,
    "clean_reconstruction_mse_after": 2.821747671077901e-07,
    "clean_snr_db_before": -10.944945993106439,
    "clean_snr_db_after": 0.059606852154090034,
    "artifact_prediction_mse": 2.821578277689696e-07,
    "artifact_prediction_correlation": 0.9595478773117065,
    "residual_rms_ratio": 0.2774100377918546,
    "rms_noisy": 0.0019148595165461302,
    "rms_corrected": 0.00012738093209918588,
    "rms_residual": 0.0005312012508511543
  },
  "per_channel": {
    "artifact_rmse_Fp1": 0.0003917951544281095,
    "artifact_rmse_Fp2": 0.0003265462873969227,
    "artifact_rmse_F7": 0.00018856489623431116,
    "artifact_rmse_F3": 0.00022793530661147088,
    "artifact_rmse_Fz": 0.00029562978306785226,
    "artifact_rmse_F4": 0.0004033466975670308,
    "artifact_rmse_F8": 0.00039730427670292556,
    "artifact_rmse_T3": 0.00039123071474023163,
    "artifact_rmse_C3": 0.00033046715543605387,
    "artifact_rmse_Cz": 0.0004334657860454172,
    "artifact_rmse_C4": 0.0004575627390295267,
    "artifact_rmse_T4": 0.0007233492215164006,
    "artifact_rmse_T5": 0.00048687795060686767,
    "artifact_rmse_P3": 0.0006844031740911305,
    "artifact_rmse_Pz": 0.0006793730426579714,
    "artifact_rmse_P4": 0.000873092794790864,
    "artifact_rmse_T6": 0.0003805634914897382,
    "artifact_rmse_O1": 0.0008092501084320247,
    "artifact_rmse_O2": 0.0010775693226605654,
    "artifact_rmse_AF4": 0.0001533638860564679,
    "artifact_rmse_AF3": 0.00022731252829544246,
    "artifact_rmse_FC2": 0.000530290650203824,
    "artifact_rmse_FC1": 0.00019098036864306778,
    "artifact_rmse_CP1": 0.0003879861324094236,
    "artifact_rmse_CP2": 0.0005886259605176747,
    "artifact_rmse_PO3": 0.0006438370910473168,
    "artifact_rmse_PO4": 0.0008520593401044607,
    "artifact_rmse_FC6": 0.00045943548320792615,
    "artifact_rmse_FC5": 0.00025028386153280735,
    "artifact_rmse_CP5": 0.00057626684429124
  }
}
```
