# Evaluation Index — Niazy Proof-Fit Comparison

This file is the entry point into the consolidated evaluation outputs for
every deep-learning model trained against the Niazy proof-fit context
dataset under the parallel-agent workflow.

Branch of record: `feature/proof_fit_consolidated`
Dataset: `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`
  (833 examples × 7 context epochs × 30 channels × 512 samples, sfreq 4096 Hz)

For the narrative analysis, ranking, caveats, and thesis recommendations,
see [`docs/research/thesis_results_report.md`](../../docs/research/thesis_results_report.md).

## Layout Convention

Every model writes evaluation artifacts to:

```
output/model_evaluations/<model_id>/<run_id>/
├── evaluation_manifest.json    # identity, schema version, file paths
├── metrics.json                # nested + flat_metrics
├── evaluation_summary.md       # human-readable per-run summary
├── training_summary.json       # copy of training_output/<run>/summary.json
├── training.jsonl              # per-epoch training metrics log
└── plots/
    ├── training_loss.png       # training/validation loss curve
    └── <model>_*.png           # evaluation result plots (examples, summary, etc.)
```

This layout is defined in
[`src/facet/models/evaluation_standard.md`](../../src/facet/models/evaluation_standard.md)
and produced by `facet.evaluation.ModelEvaluationWriter` plus a
post-evaluation copy step from `training_output/<run>/` (which is
gitignored).

## Cross-Model Ranking (Niazy Proof-Fit Validation)

Normalized from each model's `flat_metrics`. SNR improvement = `clean_snr_db_after − clean_snr_db_before` where the model reported them as separate values.

| Rank | Model | Family | SNR↑ dB | art.corr | res.RMS | Run | n examples |
|---:|---|---|---:|---:|---:|---|---:|
| 1 | demucs | Audio (U-Net+LSTM) | +31.28 | 0.9996 | 0.027 | 20260511_005636 | 4998 |
| 2 | conv_tasnet | Audio (TCN) | +22.03 | 0.9969 | 0.079 | 20260510_224113 | 4998 |
| 3 | sepformer | Audio (Transformer) | +19.05 | 0.9938 | 0.112 | 20260511_012740 | 4998 |
| 4 | nested_gan | GAN (TF+Time) | +13.54 | 0.9776 | 0.210 | 20260511_004531 | 833 |
| 5 | denoise_mamba | SSM | +11.80 | 0.9664 | 0.257 | niazy_proof_fit_full | 24990 |
| 6 | ic_unet | Discriminative + ICA | +11.77 | 0.9667 | 0.258 | 20260510_full | 833 |
| 7 | vit_spectrogram | Vision (MAE) | +11.60 | 0.9660 | 0.263 | 20260510_full | 833 |
| 8 | st_gnn | Graph (GNN) | +11.00 | 0.9595 | 0.277 | niazy_proof_fit_20260510_211512 | – |
| 9 | dpae | Discriminative | +7.48 | 0.9132 | 0.423 | 20260510_193431 | 24990 |
| 10 | d4pm | Diffusion | +3.21\* | 0.7251 | 0.699 | 20260510_d4pm_full_e4 | 32 (4 ch) |
| 11 | dhct_gan_v2 | GAN (hybrid CNN+Transformer, ctx fix) | +1.69 | 0.5673 | 0.824 | 20260511_proof_fit | – |
| — | dhct_gan | GAN (single-epoch input, failed) | −7.13 | 0.1577 | 2.272 | 20260510_233500_proof_fit | – |

\* D4PM evaluated on only 32 examples × 4 channels because of diffusion
inference cost; absolute number is not directly comparable to the full-set
evaluations. Re-evaluation on the full validation split is recommended.

## Per-Model Links

| Model | Code | Hand-off | Eval folder |
|---|---|---|---|
| dpae | [src](../../src/facet/models/dpae/) | [HANDOFF.md](../../src/facet/models/dpae/HANDOFF.md) | [dpae/20260510_193431/](dpae/20260510_193431/) |
| ic_unet | [src](../../src/facet/models/ic_unet/) | [HANDOFF.md](../../src/facet/models/ic_unet/HANDOFF.md) | [ic_unet/20260510_full/](ic_unet/20260510_full/) |
| denoise_mamba | [src](../../src/facet/models/denoise_mamba/) | [HANDOFF.md](../../src/facet/models/denoise_mamba/HANDOFF.md) | [denoise_mamba/niazy_proof_fit_full/](denoise_mamba/niazy_proof_fit_full/) |
| vit_spectrogram | [src](../../src/facet/models/vit_spectrogram/) | [HANDOFF.md](../../src/facet/models/vit_spectrogram/HANDOFF.md) | [vit_spectrogram/20260510_full/](vit_spectrogram/20260510_full/) |
| st_gnn | [src](../../src/facet/models/st_gnn/) | [HANDOFF.md](../../src/facet/models/st_gnn/HANDOFF.md) | [st_gnn/niazy_proof_fit_20260510_211512/](st_gnn/niazy_proof_fit_20260510_211512/) |
| conv_tasnet | [src](../../src/facet/models/conv_tasnet/) | [HANDOFF.md](../../src/facet/models/conv_tasnet/HANDOFF.md) | [conv_tasnet/20260510_224113/](conv_tasnet/20260510_224113/) |
| demucs | [src](../../src/facet/models/demucs/) | [HANDOFF.md](../../src/facet/models/demucs/HANDOFF.md) | [demucs/20260511_005636/](demucs/20260511_005636/) |
| sepformer | [src](../../src/facet/models/sepformer/) | [HANDOFF.md](../../src/facet/models/sepformer/HANDOFF.md) | [sepformer/20260511_012740/](sepformer/20260511_012740/) |
| nested_gan | [src](../../src/facet/models/nested_gan/) | [HANDOFF.md](../../src/facet/models/nested_gan/HANDOFF.md) | [nested_gan/20260511_004531/](nested_gan/20260511_004531/) |
| d4pm | [src](../../src/facet/models/d4pm/) | [HANDOFF.md](../../src/facet/models/d4pm/HANDOFF.md) | [d4pm/20260510_d4pm_full_e4/](d4pm/20260510_d4pm_full_e4/) |
| dhct_gan | [src](../../src/facet/models/dhct_gan/) | [HANDOFF.md](../../src/facet/models/dhct_gan/HANDOFF.md) | [dhct_gan/20260510_233500_proof_fit/](dhct_gan/20260510_233500_proof_fit/) |
| dhct_gan_v2 | [src](../../src/facet/models/dhct_gan_v2/) | [HANDOFF.md](../../src/facet/models/dhct_gan_v2/HANDOFF.md) | [dhct_gan_v2/20260511_proof_fit/](dhct_gan_v2/20260511_proof_fit/) |

## Pre-Existing Baselines (Different Dataset)

These two were evaluated on the synthetic spike artifact dataset before the
proof-fit workflow; absolute metrics are not directly comparable to the table
above, but are kept here for completeness:

- [`cascaded_dae/`](cascaded_dae/) — original channel-wise DAE baseline
- [`cascaded_context_dae/`](cascaded_context_dae/) — 7-epoch context DAE baseline

## Reproducing Any Result

```bash
git checkout feature/proof_fit_consolidated
git worktree add ../proof_fit_repro feature/proof_fit_consolidated
cd ../proof_fit_repro
uv sync
uv run python tools/gpu_fleet/fleet.py submit \
  --name <model_id>_niazy_full \
  --worktree . \
  --training-config src/facet/models/<model_id>/training_niazy_proof_fit.yaml \
  --prepare-command "uv run python examples/build_niazy_proof_fit_context_dataset.py --artifact-bundle output/artifact_libraries/niazy_aas_2x_direct/niazy_aas_direct_artifact.npz --target-epoch-samples 512 --context-epochs 7 --output-dir output/niazy_proof_fit_context_512"
```

With the central dispatcher running on the orchestrator
(`uv run python tools/gpu_fleet/fleet.py dispatch --loop --interval 60`).

## Outstanding Gaps

The thesis report flags four caveats that should be addressed before
publishing absolute numbers:

1. **Test-split size confound** — some models evaluated on 833 windows,
   others on 4998 channel-wise pairs, D4PM on 32 only. Recommended:
   single common holdout re-evaluation across all 12 models.
2. **AAS-fidelity ceiling** — "clean" target is AAS-corrected, not true
   ground truth. Metrics measure fidelity to AAS, not physical denoising
   beyond AAS.
3. **Input contract audit** — DHCT-GAN v1 demonstrated 9 dB swing from
   single-center-epoch vs full 7-epoch context. Other plateau-tier models
   should be audited for the same issue.
4. **TorchScript device baking** — some exports may be CUDA-locked; see
   commit `4184443` in `worktrees/model-vit_spectrogram` for the fix pattern.

See [`docs/research/thesis_results_report.md`](../../docs/research/thesis_results_report.md#5-critical-caveats)
for the full discussion.
