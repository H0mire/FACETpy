"""Standardized SepFormer evaluation on the Niazy proof-fit val split.

Reproduces the train/val split deterministically (seed=42, val_ratio=0.2)
from the channel-wise dataset wrapper, runs the trained TorchScript
checkpoint on the held-out channel examples, and emits the standard
``ModelEvaluationWriter`` payload under
``output/model_evaluations/sepformer/<run_id>/``.

Schema mirrors the existing ``demucs`` and ``conv_tasnet`` evaluation
runs so flat-metric keys can be compared directly.

Example::

    uv run python examples/model_evaluation/evaluate_sepformer_niazy_proof_fit.py \\
        --checkpoint training_output/sepformerniazyprooffit_20260510_230104/exports/sepformer.ts \\
        --dataset output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from facet.evaluation import ModelEvaluationWriter
from facet.models.sepformer.training import (
    ChannelWiseContextArtifactDataset,
)
from facet.training.dataset import NPZContextArtifactDataset


DEFAULT_CHECKPOINT = Path(
    "training_output/sepformerniazyprooffit_20260510_230104/exports/sepformer.ts"
)
DEFAULT_DATASET = Path(
    "output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
)
MODEL_ID = "sepformer"
MODEL_NAME = "SepFormer"
MODEL_DESCRIPTION = (
    "Compact channel-wise SepFormer (Subakan et al. 2021) for fMRI gradient artifact "
    "removal. Dual-path Transformer with intra-chunk and inter-chunk self-attention "
    "applied to a 1D-conv encoded representation of the 7-epoch context; mask + "
    "transposed-conv decoder reconstructs the centre-epoch artifact."
)

BASELINES = {
    "cascaded_dae__synthetic_spike_20260502_115914": {
        "clean_snr_improvement_db": -0.0463,
        "artifact_corr": 0.0862,
        "residual_error_rms_ratio": 1.0053,
    },
    "cascaded_context_dae__synthetic_spike_20260502_115926": {
        "clean_snr_improvement_db": 3.1585,
        "artifact_corr": 0.7318,
        "residual_error_rms_ratio": 0.6951,
    },
    "conv_tasnet__niazy_proof_fit_20260510_224113": {
        "clean_snr_improvement_db": 22.03,
        "artifact_corr": 0.997,
        "residual_error_rms_ratio": 0.079,
    },
    "demucs__niazy_proof_fit_20260511_005636": {
        "clean_snr_improvement_db": 31.2793,
        "artifact_corr": 0.9996,
        "residual_error_rms_ratio": 0.0273,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-epochs", type=int, default=7)
    parser.add_argument("--keep-input-mean", action="store_true")
    parser.add_argument("--keep-prediction-mean", action="store_true")
    parser.add_argument("--n-plot-examples", type=int, default=4)
    return parser.parse_args()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _snr_db(reference: np.ndarray, error: np.ndarray) -> float:
    return float(
        10.0 * np.log10(
            (np.mean(np.square(reference)) + 1e-20) / (np.mean(np.square(error)) + 1e-20)
        )
    )


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _gather_val_examples(
    dataset: ChannelWiseContextArtifactDataset, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize the val split into (noisy_context, artifact_center) arrays.

    Uses the same numpy permutation rule as the training factory so the
    val split here is identical to the one used during training.
    """
    n_total = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total).tolist()
    n_val = max(1, int(n_total * val_ratio))
    val_idx_set = set(indices[:n_val])
    val_indices = [i for i in range(n_total) if i in val_idx_set]

    first_noisy, first_target = dataset[val_indices[0]]
    noisy = np.empty((len(val_indices),) + first_noisy.shape, dtype=np.float32)
    target = np.empty((len(val_indices),) + first_target.shape, dtype=np.float32)
    for i, base_idx in enumerate(val_indices):
        nb, tb = dataset[base_idx]
        noisy[i] = nb.astype(np.float32, copy=False)
        target[i] = tb.astype(np.float32, copy=False)
    return noisy, target


def _predict(checkpoint: Path, noisy: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    import torch

    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, noisy.shape[0], batch_size):
            batch_np = noisy[start : start + batch_size]
            batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            pred = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(pred)
    return np.concatenate(predictions, axis=0)


def _compute_metrics(
    noisy_center: np.ndarray,
    artifact_center: np.ndarray,
    pred_artifact: np.ndarray,
    *,
    input_mean_removed: bool,
    prediction_mean_removed: bool,
    samples_per_epoch: int,
) -> dict[str, Any]:
    # noisy_center here is the centre channel-wise noisy epoch from the
    # training-aligned dataset (already de-meaned if input_mean_removed).
    # The clean target reconstructed from the artifact target.
    clean_center = noisy_center - artifact_center
    corrected = noisy_center - pred_artifact

    clean_mse_before = _mse(noisy_center, clean_center)
    clean_mse_after = _mse(corrected, clean_center)
    clean_mae_before = _mae(noisy_center, clean_center)
    clean_mae_after = _mae(corrected, clean_center)
    clean_snr_before = _snr_db(clean_center, noisy_center - clean_center)
    clean_snr_after = _snr_db(clean_center, corrected - clean_center)
    artifact_mse = _mse(pred_artifact, artifact_center)
    artifact_mae = _mae(pred_artifact, artifact_center)
    artifact_corr = _corrcoef(pred_artifact, artifact_center)
    artifact_snr_db = _snr_db(artifact_center, pred_artifact - artifact_center)
    residual_rms_ratio = _rms(corrected - clean_center) / max(
        _rms(noisy_center - clean_center), 1e-20
    )
    return {
        "n_examples": int(noisy_center.shape[0]),
        "samples_per_epoch": int(samples_per_epoch),
        "input_mean_removed": bool(input_mean_removed),
        "prediction_mean_removed": bool(prediction_mean_removed),
        "clean_mse_before": clean_mse_before,
        "clean_mse_after": clean_mse_after,
        "clean_mae_before": clean_mae_before,
        "clean_mae_after": clean_mae_after,
        "clean_snr_db_before": clean_snr_before,
        "clean_snr_db_after": clean_snr_after,
        "artifact_mse": artifact_mse,
        "artifact_mae": artifact_mae,
        "artifact_corr": artifact_corr,
        "artifact_snr_db": artifact_snr_db,
        "residual_error_rms_ratio": residual_rms_ratio,
        "clean_mse_reduction_pct": 100.0 * (1.0 - clean_mse_after / max(clean_mse_before, 1e-30)),
        "clean_snr_improvement_db": clean_snr_after - clean_snr_before,
    }


def _plot_examples(
    output_path: Path,
    noisy_center: np.ndarray,
    artifact_center: np.ndarray,
    pred_artifact: np.ndarray,
    n_examples: int,
    sfreq_hz: float,
) -> None:
    n = min(n_examples, noisy_center.shape[0])
    rng = np.random.default_rng(0)
    indices = rng.choice(noisy_center.shape[0], size=n, replace=False)
    samples = noisy_center.shape[-1]
    t = np.arange(samples) / sfreq_hz * 1000.0

    fig, axes = plt.subplots(n, 2, figsize=(10, 2.0 * n), constrained_layout=True)
    if n == 1:
        axes = axes[np.newaxis, :]
    clean = noisy_center - artifact_center
    corrected = noisy_center - pred_artifact
    for row, idx in enumerate(indices):
        ax_a, ax_b = axes[row]
        ax_a.plot(t, artifact_center[idx, 0], color="C3", label="target artifact")
        ax_a.plot(t, pred_artifact[idx, 0], color="C0", label="predicted artifact", alpha=0.7)
        ax_a.set_ylabel(f"ex {idx}\namplitude")
        if row == 0:
            ax_a.legend(loc="upper right", fontsize=8)
            ax_a.set_title("Artifact: target vs prediction")
        ax_b.plot(t, clean[idx, 0], color="C2", label="target clean")
        ax_b.plot(t, corrected[idx, 0], color="C0", label="corrected", alpha=0.7)
        if row == 0:
            ax_b.legend(loc="upper right", fontsize=8)
            ax_b.set_title("Clean: target vs corrected")
        ax_b.set_ylabel("")
    axes[-1, 0].set_xlabel("time (ms)")
    axes[-1, 1].set_xlabel("time (ms)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _plot_metric_summary(
    output_path: Path,
    sepformer_metrics: dict[str, Any],
    baselines: dict[str, dict[str, float]],
) -> None:
    keys = ["clean_snr_improvement_db", "artifact_corr", "residual_error_rms_ratio"]
    labels = ["SNR↑Δ (dB)", "artifact corr", "residual RMS ratio"]
    models = ["sepformer (this work)"] + list(baselines.keys())
    sep_values = [float(sepformer_metrics[k]) for k in keys]
    matrix = [sep_values]
    for name in baselines:
        matrix.append([float(baselines[name].get(k, np.nan)) for k in keys])

    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.0), constrained_layout=True)
    y_pos = np.arange(len(models))
    for i, (ax, key, label) in enumerate(zip(axes, keys, labels)):
        values = [row[i] for row in matrix]
        colors = ["C0"] + ["#888888"] * len(baselines)
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        if i == 0:
            ax.set_yticklabels(models, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(label)
        ax.axvline(0, color="black", linewidth=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    base = NPZContextArtifactDataset(
        path=args.dataset,
        input_key="noisy_context",
        target_key="artifact_center",
        demean_input=False,
        demean_target=False,
    )
    dataset = ChannelWiseContextArtifactDataset(
        base,
        context_epochs=args.context_epochs,
        demean_input=not args.keep_input_mean,
        demean_target=not args.keep_prediction_mean,
    )
    noisy, artifact = _gather_val_examples(dataset, val_ratio=args.val_ratio, seed=args.seed)

    pred_artifact = _predict(args.checkpoint, noisy, args.batch_size, args.device)
    if not args.keep_prediction_mean:
        pred_artifact = pred_artifact - pred_artifact.mean(axis=-1, keepdims=True)

    noisy_center = noisy[:, args.context_epochs // 2, :, :]
    metrics_val = _compute_metrics(
        noisy_center=noisy_center,
        artifact_center=artifact,
        pred_artifact=pred_artifact,
        input_mean_removed=not args.keep_input_mean,
        prediction_mean_removed=not args.keep_prediction_mean,
        samples_per_epoch=dataset.epoch_samples,
    )
    dataset_info = {
        "n_pairs_evaluated": int(noisy.shape[0]),
        "n_channels_per_example": int(dataset.n_channels),
        "samples_per_epoch": int(dataset.epoch_samples),
        "context_epochs": int(args.context_epochs),
        "sfreq_hz": float(dataset.sfreq),
    }

    writer = ModelEvaluationWriter(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        model_description=MODEL_DESCRIPTION,
        run_id=run_id,
    )

    run = writer.run
    plots_dir = run.run_dir / "plots"
    examples_plot = plots_dir / "sepformer_examples.png"
    summary_plot = plots_dir / "sepformer_metric_summary.png"
    _plot_examples(
        examples_plot,
        noisy_center=noisy_center,
        artifact_center=artifact,
        pred_artifact=pred_artifact,
        n_examples=args.n_plot_examples,
        sfreq_hz=float(dataset.sfreq) if not np.isnan(dataset.sfreq) else 4096.0,
    )
    _plot_metric_summary(summary_plot, metrics_val, BASELINES)

    config = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "device": args.device,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "context_epochs": args.context_epochs,
        "keep_input_mean": bool(args.keep_input_mean),
        "keep_prediction_mean": bool(args.keep_prediction_mean),
    }
    interpretation = (
        "SepFormer was trained channel-wise on the Niazy proof-fit context dataset. "
        "The reported metrics are computed on the deterministic val-split slice that "
        "facet-train held out at training time (val_ratio=0.2, seed=42), so they "
        "answer the proof-of-fit question 'can SepFormer learn the AAS-estimated "
        "artifact morphology' rather than a generalization claim."
    )
    limitations = [
        "Clean target is an AAS surrogate of the same Niazy recording used to train the model.",
        "Not a generalization benchmark — no held-out subject or independent recording is used.",
        "Baseline reference metrics are copied from prior evaluation runs and not re-computed on this exact dataset for cascaded_dae / cascaded_context_dae.",
    ]
    metrics_payload = {
        "synthetic_niazy_proof_fit_val_split": metrics_val,
        "dataset": dataset_info,
        "baseline_reference": BASELINES,
    }
    writer.write(
        metrics=metrics_payload,
        config=config,
        artifacts={
            "examples_plot": str(examples_plot),
            "summary_plot": str(summary_plot),
        },
        interpretation=interpretation,
        limitations=limitations,
    )
    print(json.dumps({"run_id": run_id, "run_dir": str(run.run_dir)}, indent=2))


if __name__ == "__main__":
    main()
