"""Evaluate a trained DPAE checkpoint on the Niazy proof-fit context dataset.

Single-purpose evaluation that mirrors the metric set used by
``cascaded_context_dae`` so DPAE can be compared row-for-row in
``output/model_evaluations/<model>/<run>/metrics.json``.

Example:
    uv run python -m facet.models.dpae.evaluate \
        --checkpoint training_output/dualpathwayautoencoderniazyprooffit_<run>/exports/dpae.ts \
        --dataset output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from facet.evaluation import ModelEvaluationWriter

DEFAULT_DATASET = Path("./output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz")
DEFAULT_OUTPUT_ROOT = Path("./output/model_evaluations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="TorchScript .ts checkpoint.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Niazy proof-fit dataset NPZ.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Evaluation root.")
    parser.add_argument("--run-id", default=None, help="Stable run id; defaults to a UTC timestamp.")
    parser.add_argument("--device", default="cpu", help="PyTorch device for inference.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Limit number of (example, channel) pairs evaluated; 0 means all.",
    )
    parser.add_argument(
        "--demean-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract per-segment mean before forward pass.",
    )
    parser.add_argument(
        "--remove-prediction-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract per-segment mean from the predicted artifact.",
    )
    return parser.parse_args()


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as bundle:
        keys = ("noisy_center", "clean_center", "artifact_center", "sfreq")
        return {key: np.asarray(bundle[key]) for key in keys}


def _flatten_channels(arr: np.ndarray) -> np.ndarray:
    """``(examples, channels, samples)`` -> ``(examples * channels, 1, samples)``."""
    n_examples, n_channels, n_samples = arr.shape
    return arr.reshape(n_examples * n_channels, 1, n_samples).astype(np.float32, copy=False)


def _predict_artifact(
    model: torch.jit.ScriptModule,
    noisy: np.ndarray,
    *,
    device: str,
    batch_size: int,
    demean_input: bool,
    remove_prediction_mean: bool,
) -> np.ndarray:
    predictions = np.empty_like(noisy)
    for start in range(0, noisy.shape[0], batch_size):
        stop = min(start + batch_size, noisy.shape[0])
        batch = noisy[start:stop]
        if demean_input:
            batch = batch - batch.mean(axis=-1, keepdims=True)
        tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(tensor)
        out_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
        if remove_prediction_mean:
            out_np = out_np - out_np.mean(axis=-1, keepdims=True)
        predictions[start:stop] = out_np
    return predictions


def _safe_mean_squared(x: np.ndarray) -> float:
    return float(np.mean(np.square(x.astype(np.float64))))


def _compute_metrics(
    *,
    noisy: np.ndarray,
    clean: np.ndarray,
    artifact: np.ndarray,
    predicted_artifact: np.ndarray,
) -> dict[str, float | int | bool]:
    corrected = noisy - predicted_artifact

    clean_mse_before = _safe_mean_squared(noisy - clean)
    clean_mse_after = _safe_mean_squared(corrected - clean)
    clean_mae_before = float(np.mean(np.abs((noisy - clean).astype(np.float64))))
    clean_mae_after = float(np.mean(np.abs((corrected - clean).astype(np.float64))))

    clean_power = _safe_mean_squared(clean)
    eps = 1e-12

    def _snr_db(signal_power: float, noise_power: float) -> float:
        if noise_power <= eps:
            return float("inf")
        return float(10.0 * np.log10((signal_power + eps) / (noise_power + eps)))

    clean_snr_db_before = _snr_db(clean_power, clean_mse_before)
    clean_snr_db_after = _snr_db(clean_power, clean_mse_after)

    artifact_residual = predicted_artifact - artifact
    artifact_mse = _safe_mean_squared(artifact_residual)
    artifact_mae = float(np.mean(np.abs(artifact_residual.astype(np.float64))))

    flat_pred = predicted_artifact.reshape(-1).astype(np.float64)
    flat_true = artifact.reshape(-1).astype(np.float64)
    if flat_pred.std() < eps or flat_true.std() < eps:
        artifact_corr = 0.0
    else:
        artifact_corr = float(np.corrcoef(flat_pred, flat_true)[0, 1])

    artifact_power = _safe_mean_squared(artifact)
    artifact_snr_db = _snr_db(artifact_power, artifact_mse)

    rms_before = float(np.sqrt(_safe_mean_squared(noisy - clean)) + eps)
    rms_after = float(np.sqrt(_safe_mean_squared(corrected - clean)) + eps)
    residual_error_rms_ratio = rms_after / rms_before

    edge_samples = max(1, predicted_artifact.shape[-1] // 8)
    edge = np.concatenate(
        [predicted_artifact[..., :edge_samples], predicted_artifact[..., -edge_samples:]],
        axis=-1,
    )
    center_start = (predicted_artifact.shape[-1] - 2 * edge_samples) // 2
    center = predicted_artifact[..., center_start : center_start + 2 * edge_samples]
    edge_abs_mean = float(np.mean(np.abs(edge.astype(np.float64))))
    center_abs_mean = float(np.mean(np.abs(center.astype(np.float64))))

    clean_mse_reduction_pct = 0.0
    if clean_mse_before > eps:
        clean_mse_reduction_pct = float(100.0 * (1.0 - clean_mse_after / clean_mse_before))

    return {
        "n_examples": int(noisy.shape[0]),
        "n_samples": int(noisy.shape[-1]),
        "clean_mse_before": clean_mse_before,
        "clean_mse_after": clean_mse_after,
        "clean_mae_before": clean_mae_before,
        "clean_mae_after": clean_mae_after,
        "clean_snr_db_before": clean_snr_db_before,
        "clean_snr_db_after": clean_snr_db_after,
        "artifact_mse": artifact_mse,
        "artifact_mae": artifact_mae,
        "artifact_corr": artifact_corr,
        "artifact_snr_db": artifact_snr_db,
        "residual_error_rms_ratio": residual_error_rms_ratio,
        "predicted_artifact_edge_abs_mean": edge_abs_mean,
        "predicted_artifact_center_abs_mean": center_abs_mean,
        "predicted_artifact_edge_to_center_abs_ratio": (
            float(edge_abs_mean / center_abs_mean) if center_abs_mean > eps else 0.0
        ),
        "clean_mse_reduction_pct": clean_mse_reduction_pct,
        "clean_snr_improvement_db": clean_snr_db_after - clean_snr_db_before,
    }


def _plot_examples(
    *,
    noisy: np.ndarray,
    clean: np.ndarray,
    artifact: np.ndarray,
    predicted_artifact: np.ndarray,
    sfreq: float,
    out_path: Path,
    n_examples: int = 4,
) -> None:
    n_examples = min(n_examples, noisy.shape[0])
    indices = np.linspace(0, noisy.shape[0] - 1, n_examples, dtype=int)
    samples = noisy.shape[-1]
    time_axis = np.arange(samples) / max(sfreq, 1.0)

    fig, axes = plt.subplots(n_examples, 2, figsize=(11, 2.4 * n_examples), sharex=True)
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        axes[row, 0].plot(time_axis, noisy[idx, 0], label="noisy", color="#666", lw=0.8)
        axes[row, 0].plot(time_axis, clean[idx, 0], label="clean (ref)", color="#1f77b4", lw=0.8)
        axes[row, 0].plot(
            time_axis,
            (noisy[idx, 0] - predicted_artifact[idx, 0]),
            label="corrected",
            color="#d62728",
            lw=0.9,
        )
        axes[row, 0].set_ylabel(f"ex {idx}")
        if row == 0:
            axes[row, 0].set_title("Noisy / clean / corrected")
            axes[row, 0].legend(loc="upper right", fontsize=7)

        axes[row, 1].plot(time_axis, artifact[idx, 0], label="true artifact", color="#1f77b4", lw=0.8)
        axes[row, 1].plot(
            time_axis,
            predicted_artifact[idx, 0],
            label="predicted artifact",
            color="#d62728",
            lw=0.8,
        )
        if row == 0:
            axes[row, 1].set_title("Artifact: true vs predicted")
            axes[row, 1].legend(loc="upper right", fontsize=7)

    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle("DPAE proof-fit cleaning examples (Niazy)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bundle = _load_dataset(args.dataset)
    sfreq = float(np.atleast_1d(bundle["sfreq"])[0])

    noisy = _flatten_channels(bundle["noisy_center"])
    clean = _flatten_channels(bundle["clean_center"])
    artifact = _flatten_channels(bundle["artifact_center"])

    if args.max_examples and noisy.shape[0] > args.max_examples:
        noisy = noisy[: args.max_examples]
        clean = clean[: args.max_examples]
        artifact = artifact[: args.max_examples]

    model = torch.jit.load(str(args.checkpoint), map_location=args.device)
    model.eval()

    started = time.time()
    predicted_artifact = _predict_artifact(
        model,
        noisy,
        device=args.device,
        batch_size=args.batch_size,
        demean_input=args.demean_input,
        remove_prediction_mean=args.remove_prediction_mean,
    )
    elapsed = time.time() - started

    metrics = _compute_metrics(
        noisy=noisy,
        clean=clean,
        artifact=artifact,
        predicted_artifact=predicted_artifact,
    )
    metrics["sfreq_hz"] = sfreq
    metrics["inference_seconds"] = elapsed
    metrics["device"] = args.device

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    writer = ModelEvaluationWriter(
        model_id="dpae",
        model_name="DPAE",
        model_description="Dual-Pathway Autoencoder (Xiong et al. 2023, 1D-CNN variant).",
        output_root=args.output_root,
        run_id=run_id,
    )
    plot_path = writer.run.run_dir / "plots" / "niazy_proof_fit_examples.png"
    _plot_examples(
        noisy=noisy,
        clean=clean,
        artifact=artifact,
        predicted_artifact=predicted_artifact,
        sfreq=sfreq,
        out_path=plot_path,
    )

    config = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "device": args.device,
        "batch_size": args.batch_size,
        "demean_input": args.demean_input,
        "remove_prediction_mean": args.remove_prediction_mean,
        "max_examples": args.max_examples or None,
    }
    interpretation = (
        f"DPAE evaluated on the Niazy proof-fit context dataset "
        f"({metrics['n_examples']} channel-wise examples of {metrics['n_samples']} samples). "
        f"Clean MSE drops by {metrics['clean_mse_reduction_pct']:.2f}% and clean-signal SNR "
        f"improves by {metrics['clean_snr_improvement_db']:.2f} dB. "
        "The clean reference is the AAS-corrected Niazy surrogate, so this run measures proof-fit "
        "of artifact morphology, not generalisation to an independent recording."
    )
    limitations = [
        "Clean ground truth is the AAS-corrected Niazy surrogate, not an independent clean source.",
        "Single-recording proof-fit; no cross-subject generalisation tested.",
    ]

    writer.write(
        metrics={"niazy_proof_fit": metrics},
        config=config,
        artifacts={"niazy_proof_fit_examples": str(plot_path)},
        interpretation=interpretation,
        limitations=limitations,
    )
    print(f"Wrote evaluation run to {writer.run.run_dir}")


if __name__ == "__main__":
    main()
