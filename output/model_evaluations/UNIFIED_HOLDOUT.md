# Unified Holdout Re-Evaluation

Implements the cross-model re-evaluation on a unified test split requested in
[docs/research/thesis_results_report.md §5](../../docs/research/thesis_results_report.md#5-critical-caveats)
and [docs/research/run_2_plan.md §5.1](../../docs/research/run_2_plan.md). Every
model is evaluated on the **same 166 held-out windows** (4980 channel-windows*),
producing directly comparable absolute numbers — the test-split-size confound
from Run 1 is eliminated.

\* 166 windows × 30 channels = **4980** channel-windows. Note: in Run 1, d4pm was
evaluated on only 32 windows × 4 channels (= 128 ch-w) and ic_unet/vit_spectrogram
/nested_gan on all 833 windows; here every model sees the same 166 windows.

Dataset: `output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz`
Holdout: seed=42, val_ratio=0.2 at the window level.
Split hash: `sha256:ddaa64a504e062fd`.
Holdout indices: `output/niazy_proof_fit_context_512/holdout_v1_indices.json`.
Driver: [`tools/eval_unified_holdout.py`](../../tools/eval_unified_holdout.py).

## Cross-Model Ranking (Unified Holdout)

| Rank | Model | Family | SNR↑ dB (holdout) | SNR before | SNR after | art.corr | res.RMS ratio | Δ vs Run 1 | t [s] |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | demucs | Audio (U-Net+LSTM) | +31.30 | -10.94 | +20.35 | +0.9996 | 0.027 | +0.02 | 147.7 |
| 2 | conv_tasnet | Audio (TCN) | +22.74 | -10.94 | +11.79 | +0.9973 | 0.073 | +0.71 | 923.0 |
| 3 | cascaded_context_dae | Autoencoder (context MLP) | +18.92 | -10.94 | +7.98 | +0.9936 | 0.113 | +0.08 | 0.1 |
| 4 | sepformer | Audio (Transformer) | +18.71 | -10.94 | +7.76 | +0.9933 | 0.116 | -0.34 | 135.6 |
| 5 | cascaded_dae | Autoencoder (cascaded MLP) | +18.06 | -10.94 | +7.11 | +0.9923 | 0.125 | +0.27 | 0.5 |
| 6 | nested_gan | GAN (TF+Time) | +11.71 | -10.94 | +0.77 | +0.9746 | 0.260 | -1.83 | 167.9 |
| 7 | denoise_mamba | SSM | +11.20 | -10.94 | +0.26 | +0.9614 | 0.275 | -0.60 | 101.1 |
| 8 | ic_unet | Discriminative + ICA | +11.11 | -10.94 | +0.16 | +0.9613 | 0.278 | -0.66 | 1.7 |
| 9 | st_gnn | Graph (GNN) | +11.00 | -10.94 | +0.06 | +0.9595 | 0.282 | +0.00 | 5.3 |
| 10 | vit_spectrogram | Vision (MAE) | +10.95 | -10.94 | +0.00 | +0.9605 | 0.284 | -0.65 | 12.4 |
| 11 | dpae | Discriminative | +7.28 | -10.94 | -3.66 | +0.9092 | 0.432 | -0.20 | 29.8 |
| 12 | d4pm | Diffusion | +4.81 | -10.94 | -6.14 | +0.9265 | 0.575 | +1.60 | 1332.8 |
| 13 | dhct_gan_v2 | GAN (hybrid CNN+Transformer, ctx fix) | -1.17 | -10.94 | -12.12 | +0.5644 | 1.145 | -2.86 | 32.0 |
| 14 | dhct_gan | GAN (single-epoch input, failed) | -7.12 | -10.94 | -18.06 | +0.1573 | 2.269 | +0.01 | 27.1 |

**Reading the table:**
- `SNR↑ dB` is the primary thesis metric. Higher is better.
- `art.corr` = Pearson correlation between predicted and true artifact (all channels × samples). Closer to +1 is better.
- `res.RMS ratio` = RMS(corrected − clean) / RMS(noisy − clean). Lower is better.
- `Δ vs Run 1` shows how the unified-holdout number differs from the original Run 1 ranking. Most values are within 1 dB — confirming the original ranking was qualitatively correct despite the split confound, except d4pm (now properly evaluated on 4980 ch-w instead of 128) which jumped from +3.21 to +4.81.

## Methodology Notes

- **Holdout split** is at the **window level** (n=833 → 166), not channel-window level. This means all 30 channels of each holdout window go through every model, and per-channel models see them as 4980 channel-windows. The split is deterministic given `seed=42`.
- **Metric formulas** are reused verbatim from `examples/evaluate_conv_tasnet.py` so the absolute numbers stay backwards-compatible with Run 1 per-model evaluations.
- **Inference paths**:
  - 10 models: TorchScript export (`.ts` file) from `training_output/<run>/exports/`.
  - `denoise_mamba`: Source module + `last.pt` checkpoint, because the TorchScript bakes `device='cuda:0'` into the SSM scan (`run_2_plan §3.5` device-baking anti-pattern).
  - `d4pm`: Source module + `last.pt` checkpoint, because the `d4pm.ts` export is a zero-stub (the DDPM reverse loop wasn't traced).
- **vit_spectrogram** is the only model whose underlying TS predicts the *clean* center epoch, not the artifact. The artifact is recovered in the driver via `artifact = noisy_demeaned − pred_clean` — matching the per-model adapter.

## Per-model artifacts

- [`demucs/holdout_v1/`](../demucs/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`conv_tasnet/holdout_v1/`](../conv_tasnet/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`cascaded_context_dae/holdout_v1/`](../cascaded_context_dae/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`sepformer/holdout_v1/`](../sepformer/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`cascaded_dae/holdout_v1/`](../cascaded_dae/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`nested_gan/holdout_v1/`](../nested_gan/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`denoise_mamba/holdout_v1/`](../denoise_mamba/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`ic_unet/holdout_v1/`](../ic_unet/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`st_gnn/holdout_v1/`](../st_gnn/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`vit_spectrogram/holdout_v1/`](../vit_spectrogram/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`dpae/holdout_v1/`](../dpae/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`d4pm/holdout_v1/`](../d4pm/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`dhct_gan_v2/holdout_v1/`](../dhct_gan_v2/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
- [`dhct_gan/holdout_v1/`](../dhct_gan/holdout_v1/) — `metrics.json`, `evaluation_summary.md`, `plots/holdout_examples.png`
