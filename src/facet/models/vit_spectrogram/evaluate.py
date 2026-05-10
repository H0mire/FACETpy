"""Evaluate a trained ViT Spectrogram Inpainter on the Niazy proof-fit dataset.

The model is trained to output the *clean* center epoch; this script computes
the artifact estimate as ``noisy_center - predicted_clean`` and reports both
supervised metrics (against the AAS-corrected surrogate target) and a small
set of residual-error diagnostics so the run can be compared against
``cascaded_dae`` and ``cascaded_context_dae``.

Outputs follow ``src/facet/models/evaluation_standard.md`` via
:class:`facet.evaluation.ModelEvaluationWriter`.

Usage:
    uv run python -m facet.models.vit_spectrogram.evaluate \\
        --checkpoint training_output/<run>/exports/vit_spectrogram.ts \\
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

MODEL_ID = "vit_spectrogram"
MODEL_NAME = "ViT Spectrogram Inpainter"
MODEL_DESCRIPTION = (
    "Vision-Transformer spectrogram inpainter that reconstructs the clean "
    "center epoch from a 7-epoch context via magnitude STFT and "
    "phase-preserving iSTFT."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="TorchScript checkpoint.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"),
        help="Niazy proof-fit context dataset NPZ.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/model_evaluations"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--demean-input",
        action="store_true",
        default=True,
        help="Subtract per-epoch mean before model inference (matches training config).",
    )
    parser.add_argument(
        "--keep-input-mean",
        dest="demean_input",
        action="store_false",
        help="Disable per-epoch input demeaning.",
    )
    return parser.parse_args()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def _snr_db(reference: np.ndarray, error: np.ndarray) -> float:
    return float(10.0 * np.log10((np.mean(np.square(reference)) + 1e-20) / (np.mean(np.square(error)) + 1e-20)))


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _predict_clean(checkpoint: Path, noisy_context: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """Run TorchScript model on ``(N, 7, n_channels, samples)`` per-channel context.

    The TorchScript model takes input shape ``(B, 7, 1, samples)`` and
    returns ``(B, 1, samples)``. We expand the per-channel axis to batch
    dimension so the model sees one channel at a time.
    """
    import torch

    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()
    n_examples, context_epochs, n_channels, samples = noisy_context.shape
    flat = noisy_context.transpose(0, 2, 1, 3).reshape(n_examples * n_channels, context_epochs, 1, samples)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            batch = torch.as_tensor(flat[start : start + batch_size], dtype=torch.float32, device=device)
            out = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(out)
    stacked = np.concatenate(predictions, axis=0)
    return stacked.reshape(n_examples, n_channels, 1, samples).squeeze(2)


def evaluate(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    with np.load(args.dataset, allow_pickle=True) as data:
        noisy_context = data["noisy_context"].astype(np.float32, copy=False)
        noisy_center = data["noisy_center"].astype(np.float32, copy=False)
        clean_center = data["clean_center"].astype(np.float32, copy=False)
        artifact_center = data["artifact_center"].astype(np.float32, copy=False)
        sfreq = float(data["sfreq"][0])

    if args.demean_input:
        noisy_context = noisy_context - noisy_context.mean(axis=-1, keepdims=True)

    pred_clean = _predict_clean(args.checkpoint, noisy_context, args.batch_size, args.device)
    if args.demean_input:
        pred_clean = pred_clean - pred_clean.mean(axis=-1, keepdims=True)

    # Restore the noisy center's original mean before computing the artifact
    # so we don't lose the DC offset between noisy and predicted clean.
    noisy_center_demeaned = noisy_center - noisy_center.mean(axis=-1, keepdims=True) if args.demean_input else noisy_center
    pred_artifact = noisy_center_demeaned - pred_clean
    corrected = noisy_center - pred_artifact

    before_error = noisy_center - clean_center
    after_error = corrected - clean_center

    metrics = {
        "n_examples": int(noisy_center.shape[0]),
        "n_channels": int(noisy_center.shape[1]),
        "sfreq_hz": sfreq,
        "clean_mse_before": _mse(noisy_center, clean_center),
        "clean_mse_after": _mse(corrected, clean_center),
        "clean_mae_before": _mae(noisy_center, clean_center),
        "clean_mae_after": _mae(corrected, clean_center),
        "clean_snr_db_before": _snr_db(clean_center, before_error),
        "clean_snr_db_after": _snr_db(clean_center, after_error),
        "artifact_mse": _mse(pred_artifact, artifact_center),
        "artifact_mae": _mae(pred_artifact, artifact_center),
        "artifact_corr": _corrcoef(pred_artifact, artifact_center),
        "artifact_snr_db": _snr_db(artifact_center, pred_artifact - artifact_center),
        "residual_error_rms_ratio": _rms(after_error) / (_rms(before_error) + 1e-20),
        "input_mean_removed": bool(args.demean_input),
    }
    metrics["clean_mse_reduction_pct"] = 100.0 * (
        1.0 - metrics["clean_mse_after"] / (metrics["clean_mse_before"] + 1e-20)
    )
    metrics["clean_snr_improvement_db"] = metrics["clean_snr_db_after"] - metrics["clean_snr_db_before"]

    _plot_examples(
        output_dir / "synthetic_cleaning_examples.png",
        noisy_center,
        clean_center,
        corrected,
        artifact_center,
        pred_artifact,
        sfreq,
        seed=args.seed,
    )
    _plot_metric_summary(output_dir / "synthetic_metric_summary.png", metrics)
    return metrics


def _plot_examples(
    path: Path,
    noisy: np.ndarray,
    clean: np.ndarray,
    corrected: np.ndarray,
    true_artifact: np.ndarray,
    pred_artifact: np.ndarray,
    sfreq: float,
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = min(4, noisy.shape[0])
    indices = rng.choice(noisy.shape[0], size=n, replace=False)
    time_ms = np.arange(noisy.shape[-1]) / sfreq * 1000.0

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), squeeze=False)
    for row, idx in enumerate(indices):
        axes[row, 0].plot(time_ms, noisy[idx, 0] * 1e6, label="noisy", color="#9a3412", linewidth=1.0)
        axes[row, 0].plot(time_ms, clean[idx, 0] * 1e6, label="clean target", color="#0f766e", linewidth=1.1)
        axes[row, 0].plot(time_ms, corrected[idx, 0] * 1e6, label="corrected", color="#1d4ed8", linewidth=1.0)
        axes[row, 0].set_title(f"Proof-fit sample {int(idx)}: EEG before/after")
        axes[row, 0].set_ylabel("uV")
        axes[row, 0].legend(loc="upper right", fontsize=8)

        axes[row, 1].plot(time_ms, true_artifact[idx, 0] * 1e6, label="artifact target", color="#374151")
        axes[row, 1].plot(time_ms, pred_artifact[idx, 0] * 1e6, label="predicted artifact", color="#dc2626")
        axes[row, 1].set_title("Artifact target vs prediction")
        axes[row, 1].legend(loc="upper right", fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("ms")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_metric_summary(path: Path, metrics: dict[str, Any]) -> None:
    labels = ["MSE before", "MSE after", "MAE before", "MAE after"]
    values = [
        metrics["clean_mse_before"],
        metrics["clean_mse_after"],
        metrics["clean_mae_before"],
        metrics["clean_mae_after"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#9a3412", "#1d4ed8", "#9a3412", "#1d4ed8"])
    ax.set_yscale("log")
    ax.set_title("ViT spectrogram: supervised error before vs after correction")
    ax.set_ylabel("log-scaled error")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / MODEL_ID / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate(args, output_dir)
    metrics_path = output_dir / "vit_spectrogram_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    writer = ModelEvaluationWriter(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        model_description=MODEL_DESCRIPTION,
        run_id=args.run_id,
    )
    standard_run = writer.write(
        metrics={"synthetic": metrics},
        config={
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "device": args.device,
            "batch_size": args.batch_size,
            "input_mean_removed": bool(args.demean_input),
        },
        artifacts={
            "metrics_json": metrics_path,
            "synthetic_cleaning_examples": output_dir / "synthetic_cleaning_examples.png",
            "synthetic_metric_summary": output_dir / "synthetic_metric_summary.png",
        },
        interpretation=(
            "Niazy proof-fit metrics use the AAS-corrected EEG as a surrogate clean target. "
            "Clean-signal improvements are bounded by the AAS surrogate's residual; "
            "metrics should be compared head-to-head with cascaded_context_dae on the same dataset."
        ),
        limitations=[
            "Proof-fit only: training and inference draw from the same Niazy recording.",
            "Magnitude-only spectrogram reconstruction; phase is preserved from the noisy input "
            "and therefore retains some artifact-locked phase structure (see research_notes.md).",
        ],
    )

    print(f"Saved evaluation under: {output_dir}")
    print(f"Standard manifest: {standard_run.manifest_path}")
    print(f"Clean MSE reduction: {metrics['clean_mse_reduction_pct']:.2f}%")
    print(f"Clean SNR improvement: {metrics['clean_snr_improvement_db']:.3f} dB")
    print(f"Artifact correlation: {metrics['artifact_corr']:.4f}")


if __name__ == "__main__":
    main()
