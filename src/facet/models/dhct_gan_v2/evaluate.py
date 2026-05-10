"""Standardized supervised evaluation for the DHCT-GAN v2 model.

Reads the trained TorchScript checkpoint, predicts the center-epoch artifact
for each per-channel window of the Niazy proof-fit dataset using the full
7-epoch noisy context as input, and writes the canonical evaluation triple
(``evaluation_manifest.json``, ``metrics.json``, ``evaluation_summary.md``)
plus a small plot. Run after `facet-train fit` and a `fleet.py fetch` so the
checkpoint exists locally.

Usage:

    uv run python -m facet.models.dhct_gan_v2.evaluate \
        --checkpoint training_output/<run>/exports/dhct_gan_v2.ts \
        --dataset ./output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from facet.evaluation import ModelEvaluationWriter

MODEL_ID = "dhct_gan_v2"
MODEL_NAME = "DHCT-GAN v2"
MODEL_DESCRIPTION = (
    "Multi-epoch context dual-branch hybrid CNN-Transformer GAN "
    "(v2 of Cai et al., MDPI Sensors 25/1/231, 2025; 7-epoch context input)."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Niazy proof-fit NPZ bundle.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-epochs", type=int, default=7)
    parser.add_argument("--demean-input", action="store_true", default=True)
    parser.add_argument("--remove-prediction-mean", action="store_true", default=True)
    parser.add_argument(
        "--keep-input-mean",
        dest="demean_input",
        action="store_false",
        help="Disable demeaning of the per-window noisy context.",
    )
    parser.add_argument(
        "--keep-prediction-mean",
        dest="remove_prediction_mean",
        action="store_false",
        help="Disable demeaning of the predicted artifact.",
    )
    return parser.parse_args()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def _snr_db(reference: np.ndarray, error: np.ndarray) -> float:
    return float(
        10.0
        * np.log10(
            (np.mean(np.square(reference)) + 1e-20)
            / (np.mean(np.square(error)) + 1e-20)
        )
    )


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _flatten_windows(
    noisy_context: np.ndarray,
    noisy_center: np.ndarray,
    clean_center: np.ndarray,
    artifact_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # noisy_context shape: (E, K, C, T) where K = context_epochs.
    # Other arrays shape: (E, C, T).
    n_examples, n_context, n_channels, n_samples = noisy_context.shape
    # Per-channel flattening: re-order to (E*C, K, T) and (E*C, 1, T).
    # Stack channel axis to the leading example axis.
    context_flat = np.transpose(noisy_context, (0, 2, 1, 3)).reshape(
        n_examples * n_channels, n_context, n_samples
    ).astype(np.float32, copy=False)
    center_shape = (n_examples * n_channels, 1, n_samples)
    return (
        context_flat,
        noisy_center.reshape(center_shape).astype(np.float32, copy=False),
        clean_center.reshape(center_shape).astype(np.float32, copy=False),
        artifact_center.reshape(center_shape).astype(np.float32, copy=False),
    )


def _predict_artifact(
    checkpoint: Path, context: np.ndarray, batch_size: int, device: str
) -> np.ndarray:
    import torch

    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, context.shape[0], batch_size):
            batch = torch.as_tensor(
                context[start : start + batch_size], dtype=torch.float32, device=device
            )
            pred = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(pred)
    return np.concatenate(predictions, axis=0)


def _plot_examples(
    path: Path,
    noisy: np.ndarray,
    clean: np.ndarray,
    corrected: np.ndarray,
    artifact: np.ndarray,
    predicted: np.ndarray,
    sfreq: float,
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = min(4, noisy.shape[0])
    idxs = rng.choice(noisy.shape[0], size=n, replace=False)
    time_ms = np.arange(noisy.shape[-1]) / sfreq * 1000.0

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), squeeze=False)
    for row, idx in enumerate(idxs):
        axes[row, 0].plot(time_ms, noisy[idx, 0] * 1e6, label="noisy", color="#9a3412", linewidth=0.9)
        axes[row, 0].plot(time_ms, clean[idx, 0] * 1e6, label="clean target", color="#0f766e", linewidth=1.0)
        axes[row, 0].plot(time_ms, corrected[idx, 0] * 1e6, label="corrected", color="#1d4ed8", linewidth=0.9)
        axes[row, 0].set_title(f"window {int(idx)}: EEG before / target / corrected")
        axes[row, 0].set_ylabel("uV")
        axes[row, 0].legend(loc="upper right", fontsize=8)

        axes[row, 1].plot(time_ms, artifact[idx, 0] * 1e6, label="artifact target", color="#374151")
        axes[row, 1].plot(time_ms, predicted[idx, 0] * 1e6, label="predicted artifact", color="#dc2626")
        axes[row, 1].set_title("artifact target vs prediction")
        axes[row, 1].legend(loc="upper right", fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("ms")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    with np.load(args.dataset, allow_pickle=True) as bundle:
        noisy_context = bundle["noisy_context"].astype(np.float32, copy=False)
        noisy_center = bundle["noisy_center"].astype(np.float32, copy=False)
        clean_center = bundle["clean_center"].astype(np.float32, copy=False)
        artifact_center = bundle["artifact_center"].astype(np.float32, copy=False)
        sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

    expected_context = int(args.context_epochs)
    if noisy_context.shape[1] != expected_context:
        raise ValueError(
            f"Dataset has {noisy_context.shape[1]} context epochs but evaluation expects {expected_context}"
        )

    context_flat, noisy_flat, clean_flat, artifact_flat = _flatten_windows(
        noisy_context, noisy_center, clean_center, artifact_center
    )

    model_input = context_flat.copy()
    if args.demean_input:
        model_input = model_input - model_input.mean(axis=-1, keepdims=True)
    pred_artifact = _predict_artifact(args.checkpoint, model_input, args.batch_size, args.device)
    if args.remove_prediction_mean:
        pred_artifact = pred_artifact - pred_artifact.mean(axis=-1, keepdims=True)
    corrected = noisy_flat - pred_artifact

    before_error = noisy_flat - clean_flat
    after_error = corrected - clean_flat

    metrics = {
        "synthetic": {
            "n_windows": int(noisy_flat.shape[0]),
            "sfreq_hz": sfreq,
            "context_epochs": expected_context,
            "clean_mse_before": _mse(noisy_flat, clean_flat),
            "clean_mse_after": _mse(corrected, clean_flat),
            "clean_mae_before": _mae(noisy_flat, clean_flat),
            "clean_mae_after": _mae(corrected, clean_flat),
            "clean_snr_db_before": _snr_db(clean_flat, before_error),
            "clean_snr_db_after": _snr_db(clean_flat, after_error),
            "clean_snr_improvement_db": (
                _snr_db(clean_flat, after_error) - _snr_db(clean_flat, before_error)
            ),
            "clean_mse_reduction_pct": 100.0
            * (1.0 - _mse(corrected, clean_flat) / (_mse(noisy_flat, clean_flat) + 1e-20)),
            "artifact_mse": _mse(pred_artifact, artifact_flat),
            "artifact_mae": _mae(pred_artifact, artifact_flat),
            "artifact_corr": _corrcoef(pred_artifact, artifact_flat),
            "artifact_snr_db": _snr_db(artifact_flat, pred_artifact - artifact_flat),
            "residual_error_rms_ratio": _rms(after_error) / (_rms(before_error) + 1e-20),
        }
    }

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = ModelEvaluationWriter(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        model_description=MODEL_DESCRIPTION,
        run_id=run_id,
    )
    plots_dir = writer.run.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "supervised_examples.png"
    _plot_examples(
        plot_path,
        noisy_flat,
        clean_flat,
        corrected,
        artifact_flat,
        pred_artifact,
        sfreq,
        seed=args.seed,
    )

    config = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "device": args.device,
        "batch_size": args.batch_size,
        "context_epochs": expected_context,
        "demean_input": bool(args.demean_input),
        "remove_prediction_mean": bool(args.remove_prediction_mean),
    }
    artifacts = {"supervised_examples": str(plot_path.relative_to(writer.run.run_dir))}
    interpretation = (
        "DHCT-GAN v2 supervised proof-fit metrics on the Niazy bundle, with "
        "7-epoch context input. Clean SNR improvement and artifact correlation "
        "should both move in the right direction relative to the no-correction "
        "baseline and relative to v1's single-epoch predictions."
    )
    limitations = [
        "Targets are AAS-derived. The clean reference is the AAS-corrected "
        "Niazy surrogate, so generalization claims beyond the AAS estimate are not warranted.",
        "Per-window demeaning is applied before inference; absolute baseline drift is not corrected.",
        "Real-data trigger-locked proxies and the FACET framework metric battery are not yet included.",
    ]
    run = writer.write(
        metrics=metrics,
        config=config,
        artifacts=artifacts,
        interpretation=interpretation,
        limitations=limitations,
    )
    print(f"wrote evaluation: {run.manifest_path}")
    print(f"metrics: {run.metrics_path}")
    print(f"summary: {run.summary_path}")
    for key, value in metrics["synthetic"].items():
        print(f"  {key}: {value}")


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
