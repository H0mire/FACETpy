"""Evaluate a trained Nested-GAN checkpoint on the Niazy proof-fit context dataset.

This script runs the model in inference mode on the same NPZ bundle that was
used for training, computes the metric groups required by
``src/facet/models/evaluation_standard.md``, and writes manifest, metrics,
summary, and plots via :class:`facet.evaluation.ModelEvaluationWriter`.

Usage::

    uv run python -m facet.models.nested_gan.evaluate \
        --checkpoint training_output/<run>/exports/nested_gan.ts \
        --dataset output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from facet.evaluation import ModelEvaluationWriter


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _load_bundle(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as bundle:
        return {
            "noisy_context": bundle["noisy_context"].astype(np.float32),
            "noisy_center": bundle["noisy_center"].astype(np.float32),
            "artifact_center": bundle["artifact_center"].astype(np.float32),
            "clean_center": bundle["clean_center"].astype(np.float32),
            "sfreq": float(bundle["sfreq"][0]) if "sfreq" in bundle.files else float("nan"),
            "ch_names": (
                [str(name) for name in bundle["ch_names"]] if "ch_names" in bundle.files else []
            ),
        }


def _predict_artifact(
    model: Any,
    torch: Any,
    noisy_context: np.ndarray,
    device: str,
    *,
    batch_size: int = 64,
    demean_input: bool = True,
    remove_prediction_mean: bool = True,
) -> np.ndarray:
    n_examples, ctx, n_channels, n_samples = noisy_context.shape
    flat = noisy_context.transpose(0, 2, 1, 3).reshape(n_examples * n_channels, ctx, 1, n_samples)
    if demean_input:
        flat = flat - flat.mean(axis=-1, keepdims=True)

    predictions = np.empty((flat.shape[0], 1, n_samples), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            stop = min(start + batch_size, flat.shape[0])
            chunk = torch.as_tensor(flat[start:stop], dtype=torch.float32, device=device)
            out = model(chunk).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions[start:stop] = out

    artifact = predictions.reshape(n_examples, n_channels, n_samples)
    if remove_prediction_mean:
        artifact = artifact - artifact.mean(axis=-1, keepdims=True)
    return artifact


def _flat_rms(values: np.ndarray) -> float:
    flat = values.reshape(-1).astype(np.float64)
    if flat.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(flat * flat)))


def _snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    s = signal.reshape(-1).astype(np.float64)
    n = noise.reshape(-1).astype(np.float64)
    if n.size == 0:
        return float("inf")
    noise_power = float(np.mean(n * n))
    signal_power = float(np.mean(s * s))
    if noise_power <= 0.0:
        return float("inf")
    if signal_power <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = x.reshape(-1).astype(np.float64)
    b = y.reshape(-1).astype(np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(a * b) / denom)


def _compute_metrics(
    noisy_center: np.ndarray,
    artifact_center: np.ndarray,
    clean_center: np.ndarray,
    predicted_artifact: np.ndarray,
) -> dict[str, dict[str, float]]:
    corrected = noisy_center - predicted_artifact
    artifact_residual = artifact_center - predicted_artifact

    metrics: dict[str, dict[str, float]] = {
        "synthetic": {
            "n_examples": int(noisy_center.shape[0]),
            "n_channels": int(noisy_center.shape[1]),
            "epoch_samples": int(noisy_center.shape[2]),
            "clean_reconstruction_l1_before": float(
                np.mean(np.abs(noisy_center - clean_center))
            ),
            "clean_reconstruction_l1_after": float(
                np.mean(np.abs(corrected - clean_center))
            ),
            "clean_reconstruction_l2_before": float(_flat_rms(noisy_center - clean_center)),
            "clean_reconstruction_l2_after": float(_flat_rms(corrected - clean_center)),
            "clean_snr_db_before": _snr_db(clean_center, noisy_center - clean_center),
            "clean_snr_db_after": _snr_db(clean_center, corrected - clean_center),
            "artifact_prediction_l1": float(np.mean(np.abs(artifact_residual))),
            "artifact_prediction_rms": float(_flat_rms(artifact_residual)),
            "artifact_correlation": _pearson_corr(predicted_artifact, artifact_center),
            "residual_rms_ratio": (
                float(_flat_rms(artifact_residual) / max(_flat_rms(artifact_center), 1e-12))
            ),
        },
        "real_proxy": {
            "trigger_locked_rms_before": float(_flat_rms(noisy_center)),
            "trigger_locked_rms_after": float(_flat_rms(corrected)),
            "predicted_artifact_rms": float(_flat_rms(predicted_artifact)),
            "rms_reduction_pct": (
                100.0
                * (1.0 - _flat_rms(corrected) / max(_flat_rms(noisy_center), 1e-12))
            ),
        },
    }
    metrics["synthetic"]["clean_snr_improvement_db"] = (
        metrics["synthetic"]["clean_snr_db_after"]
        - metrics["synthetic"]["clean_snr_db_before"]
    )
    return metrics


def _plot_examples(
    *,
    noisy_center: np.ndarray,
    artifact_center: np.ndarray,
    clean_center: np.ndarray,
    predicted_artifact: np.ndarray,
    out_dir: Path,
    sfreq: float,
    n_examples: int = 3,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    rng = np.random.default_rng(0)
    total = noisy_center.shape[0] * noisy_center.shape[1]
    if total == 0:
        return paths
    flat_indices = rng.choice(total, size=min(n_examples, total), replace=False)

    samples = noisy_center.shape[2]
    t = np.arange(samples) / sfreq if np.isfinite(sfreq) and sfreq > 0 else np.arange(samples)

    for idx in flat_indices:
        example_idx, channel_idx = divmod(int(idx), noisy_center.shape[1])
        noisy = noisy_center[example_idx, channel_idx]
        artifact = artifact_center[example_idx, channel_idx]
        clean = clean_center[example_idx, channel_idx]
        pred = predicted_artifact[example_idx, channel_idx]
        corrected = noisy - pred

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(t, noisy, label="noisy", color="tab:gray")
        axes[0].plot(t, clean, label="clean", color="tab:green", alpha=0.8)
        axes[0].set_title(f"example {example_idx}, channel {channel_idx}")
        axes[0].legend(loc="upper right")
        axes[1].plot(t, artifact, label="artifact (target)", color="tab:red")
        axes[1].plot(t, pred, label="predicted artifact", color="tab:blue", alpha=0.7)
        axes[1].legend(loc="upper right")
        axes[2].plot(t, clean, label="clean", color="tab:green")
        axes[2].plot(t, corrected, label="corrected", color="tab:blue", alpha=0.8)
        axes[2].legend(loc="upper right")
        axes[2].set_xlabel("time (s)" if np.isfinite(sfreq) and sfreq > 0 else "sample")

        plot_path = out_dir / f"example_{example_idx:04d}_ch{channel_idx:02d}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        paths.append(plot_path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Nested-GAN on the Niazy proof-fit dataset.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--demean-input", action="store_true", default=True)
    parser.add_argument("--no-demean-input", dest="demean_input", action="store_false")
    parser.add_argument(
        "--remove-prediction-mean", action="store_true", default=True
    )
    parser.add_argument(
        "--no-remove-prediction-mean", dest="remove_prediction_mean", action="store_false"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id; defaults to a timestamp.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("output/model_evaluations"),
        type=Path,
        help="Root under which output/model_evaluations/nested_gan/<run_id>/ is created.",
    )
    parser.add_argument(
        "--docs-root",
        default=Path("src/facet/models"),
        type=Path,
        help="Root under which the model documentation index is updated.",
    )
    args = parser.parse_args(argv)

    import torch

    device = _resolve_device(args.device)
    model = torch.jit.load(str(args.checkpoint.expanduser()), map_location=device)
    model.eval()

    bundle = _load_bundle(args.dataset.expanduser())
    predicted_artifact = _predict_artifact(
        model,
        torch,
        bundle["noisy_context"],
        device=device,
        batch_size=int(args.batch_size),
        demean_input=bool(args.demean_input),
        remove_prediction_mean=bool(args.remove_prediction_mean),
    )

    metrics = _compute_metrics(
        bundle["noisy_center"],
        bundle["artifact_center"],
        bundle["clean_center"],
        predicted_artifact,
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = ModelEvaluationWriter(
        model_id="nested_gan",
        model_name="Nested-GAN",
        model_description=(
            "Inner spectrogram Restormer cascaded into an outer time-domain refiner "
            "(generator-only training with multi-resolution STFT loss)."
        ),
        output_root=Path(args.output_root),
        docs_root=Path(args.docs_root),
        run_id=run_id,
    )

    plots_dir = writer.run.run_dir / "plots"
    plot_paths = _plot_examples(
        noisy_center=bundle["noisy_center"],
        artifact_center=bundle["artifact_center"],
        clean_center=bundle["clean_center"],
        predicted_artifact=predicted_artifact,
        out_dir=plots_dir,
        sfreq=bundle["sfreq"],
    )

    config = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "device": device,
        "batch_size": int(args.batch_size),
        "demean_input": bool(args.demean_input),
        "remove_prediction_mean": bool(args.remove_prediction_mean),
        "ch_names": bundle["ch_names"],
        "sfreq": bundle["sfreq"],
    }

    run = writer.write(
        metrics=metrics,
        config=config,
        artifacts={
            plot_path.name: str(plot_path.relative_to(writer.run.run_dir))
            for plot_path in plot_paths
        },
        interpretation=(
            f"Clean-SNR improvement: "
            f"{metrics['synthetic']['clean_snr_improvement_db']:.2f} dB. "
            f"Artifact correlation: {metrics['synthetic']['artifact_correlation']:.3f}. "
            f"Residual RMS ratio: {metrics['synthetic']['residual_rms_ratio']:.3f}."
        ),
        limitations=[
            "Evaluation uses the same NPZ bundle that trained the model; no held-out subject.",
            "Edge epochs at the start/end of each recording are not corrected.",
        ],
    )
    print(json.dumps({"run_dir": str(run.run_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
