"""Evaluate the Conv-TasNet model on the Niazy proof-fit context dataset.

Loads the noisy/clean/artifact center arrays produced by
``examples/build_niazy_proof_fit_context_dataset.py``, runs the trained
TorchScript checkpoint on each (channel, example) mixture, and writes the
standard ``ModelEvaluationWriter`` outputs under
``output/model_evaluations/conv_tasnet/<run_id>/``.

Conv-TasNet returns two ordered sources per mixture: source 0 = clean EEG
estimate, source 1 = gradient artifact estimate. Supervised metrics are
computed against ``clean_center`` and ``artifact_center``. The corrected
signal used for SNR/RMS metrics is ``noisy_center - predicted_artifact``,
matching how :class:`ConvTasNetCorrection` modifies the pipeline raw.

Example::

    uv run python examples/evaluate_conv_tasnet.py \
        --checkpoint training_output/<run>/exports/conv_tasnet.ts \
        --dataset output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from facet.evaluation import ModelEvaluationWriter

DEFAULT_DATASET = Path(
    "./output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
)
DEFAULT_OUTPUT_ROOT = Path("./output/model_evaluations")
MODEL_ID = "conv_tasnet"
MODEL_NAME = "Conv-TasNet"
MODEL_DESCRIPTION = (
    "Channel-wise time-domain source separator (Luo & Mesgarani 2019) adapted "
    "to gradient-artifact removal: source 0 is clean EEG, source 1 is the "
    "gradient artifact."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="TorchScript checkpoint.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Niazy proof-fit NPZ bundle.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output run directory (default: timestamped).")
    parser.add_argument("--run-id", default=None, help="Stable evaluation run id (default: timestamp).")
    parser.add_argument("--device", default="cpu", help="PyTorch device.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Held-out validation fraction (matches training_niazy_proof_fit.yaml).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Train/val split seed (matches training).")
    parser.add_argument(
        "--keep-input-mean",
        action="store_true",
        help="Do not remove the per-mixture mean before inference (defaults to remove).",
    )
    parser.add_argument(
        "--keep-prediction-mean",
        action="store_true",
        help="Do not remove the per-source predicted mean (defaults to remove).",
    )
    parser.add_argument("--plot-seed", type=int, default=7, help="Plot example sampling seed.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Cap (channel, epoch) pairs evaluated. 0 means use the full validation split.",
    )
    return parser.parse_args()


def _resolve_output(args: argparse.Namespace) -> tuple[Path, str]:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / MODEL_ID / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_id


def _val_indices(n: int, val_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    n_val = max(1, int(n * val_ratio))
    return np.asarray(sorted(indices[:n_val]), dtype=np.int64)


def _load_validation_split(args: argparse.Namespace) -> dict[str, np.ndarray]:
    with np.load(args.dataset, allow_pickle=True) as bundle:
        noisy = bundle["noisy_center"].astype(np.float32, copy=False)
        clean = bundle["clean_center"].astype(np.float32, copy=False)
        artifact = bundle["artifact_center"].astype(np.float32, copy=False)
        sfreq = float(bundle["sfreq"][0])

    n_examples, n_channels, n_samples = noisy.shape
    total_pairs = n_examples * n_channels
    val_pairs = _val_indices(total_pairs, args.val_ratio, args.seed)

    if args.max_examples > 0:
        val_pairs = val_pairs[: args.max_examples]

    example_idx = val_pairs // n_channels
    channel_idx = val_pairs % n_channels

    return {
        "noisy": noisy[example_idx, channel_idx, :],
        "clean": clean[example_idx, channel_idx, :],
        "artifact": artifact[example_idx, channel_idx, :],
        "sfreq": sfreq,
        "n_pairs": int(len(val_pairs)),
        "n_channels": int(n_channels),
        "n_samples": int(n_samples),
    }


def _run_inference(
    checkpoint: Path,
    mixtures: np.ndarray,
    *,
    batch_size: int,
    device: str,
    keep_input_mean: bool,
) -> np.ndarray:
    import torch

    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()

    inputs = mixtures.astype(np.float32, copy=True)
    if not keep_input_mean:
        inputs = inputs - inputs.mean(axis=-1, keepdims=True)
    inputs = inputs[:, np.newaxis, :]

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = torch.as_tensor(inputs[start : start + batch_size], dtype=torch.float32, device=device)
            output = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(output)
    return np.concatenate(predictions, axis=0)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def _snr_db(reference: np.ndarray, error: np.ndarray) -> float:
    return float(
        10.0
        * np.log10((np.mean(np.square(reference)) + 1e-20) / (np.mean(np.square(error)) + 1e-20))
    )


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _compute_metrics(
    *,
    noisy: np.ndarray,
    clean: np.ndarray,
    artifact: np.ndarray,
    pred_clean: np.ndarray,
    pred_artifact: np.ndarray,
    keep_input_mean: bool,
    keep_prediction_mean: bool,
) -> dict[str, float | int | bool]:
    corrected = noisy - pred_artifact
    before_error = noisy - clean
    after_error = corrected - clean

    metrics: dict[str, float | int | bool] = {
        "n_examples": int(noisy.shape[0]),
        "samples_per_epoch": int(noisy.shape[1]),
        "input_mean_removed": not keep_input_mean,
        "prediction_mean_removed": not keep_prediction_mean,
        "clean_mse_before": _mse(noisy, clean),
        "clean_mse_after": _mse(corrected, clean),
        "clean_mae_before": _mae(noisy, clean),
        "clean_mae_after": _mae(corrected, clean),
        "clean_snr_db_before": _snr_db(clean, before_error),
        "clean_snr_db_after": _snr_db(clean, after_error),
        "artifact_mse": _mse(pred_artifact, artifact),
        "artifact_mae": _mae(pred_artifact, artifact),
        "artifact_corr": _corrcoef(pred_artifact, artifact),
        "artifact_snr_db": _snr_db(artifact, pred_artifact - artifact),
        "predicted_clean_mse": _mse(pred_clean, clean),
        "predicted_clean_corr": _corrcoef(pred_clean, clean),
        "predicted_source_sum_mse": _mse(pred_clean + pred_artifact, noisy),
        "residual_error_rms_ratio": _rms(after_error) / (_rms(before_error) + 1e-20),
    }
    metrics["clean_mse_reduction_pct"] = 100.0 * (
        1.0 - metrics["clean_mse_after"] / (metrics["clean_mse_before"] + 1e-20)
    )
    metrics["clean_snr_improvement_db"] = metrics["clean_snr_db_after"] - metrics["clean_snr_db_before"]
    return metrics


def _plot_examples(
    path: Path,
    noisy: np.ndarray,
    clean: np.ndarray,
    artifact: np.ndarray,
    pred_clean: np.ndarray,
    pred_artifact: np.ndarray,
    sfreq: float,
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = min(4, noisy.shape[0])
    indices = rng.choice(noisy.shape[0], size=n, replace=False)
    time_ms = np.arange(noisy.shape[-1]) / sfreq * 1000.0
    corrected = noisy - pred_artifact

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), squeeze=False)
    for row, idx in enumerate(indices):
        axes[row, 0].plot(time_ms, noisy[idx] * 1e6, label="noisy mixture", color="#9a3412", linewidth=1.0)
        axes[row, 0].plot(time_ms, clean[idx] * 1e6, label="clean target", color="#0f766e", linewidth=1.1)
        axes[row, 0].plot(time_ms, corrected[idx] * 1e6, label="corrected", color="#1d4ed8", linewidth=1.0)
        axes[row, 0].plot(time_ms, pred_clean[idx] * 1e6, label="predicted clean", color="#7c3aed", alpha=0.7, linewidth=0.9)
        axes[row, 0].set_title(f"Conv-TasNet sample {int(idx)}: clean vs corrected")
        axes[row, 0].set_ylabel("uV")
        axes[row, 0].legend(loc="upper right", fontsize=8)

        axes[row, 1].plot(time_ms, artifact[idx] * 1e6, label="artifact target", color="#374151")
        axes[row, 1].plot(time_ms, pred_artifact[idx] * 1e6, label="predicted artifact", color="#dc2626")
        axes[row, 1].set_title("Artifact target vs prediction")
        axes[row, 1].legend(loc="upper right", fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel("ms")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_summary(path: Path, metrics: dict[str, float | int | bool]) -> None:
    labels = ["clean MSE before", "clean MSE after", "artifact MSE", "predicted clean MSE"]
    values = [
        float(metrics["clean_mse_before"]),
        float(metrics["clean_mse_after"]),
        float(metrics["artifact_mse"]),
        float(metrics["predicted_clean_mse"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#9a3412", "#1d4ed8", "#dc2626", "#7c3aed"])
    ax.set_yscale("log")
    ax.set_title("Conv-TasNet supervised errors (log scale)")
    ax.set_ylabel("log-scaled error")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _baseline_summary() -> dict[str, dict[str, float]]:
    """Return the most recent published baseline metrics for cross-model context.

    These come from the cascaded_dae and cascaded_context_dae evaluation runs
    on the synthetic_spike dataset (the only supervised eval available at the
    time of writing). They are reported alongside the Conv-TasNet
    Niazy-proof-fit metrics for orientation only — they are not directly
    comparable because the source datasets differ.
    """
    return {
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
    }


def main() -> None:
    args = parse_args()
    output_dir, run_id = _resolve_output(args)

    val = _load_validation_split(args)
    sources = _run_inference(
        args.checkpoint,
        val["noisy"],
        batch_size=args.batch_size,
        device=args.device,
        keep_input_mean=args.keep_input_mean,
    )
    if sources.ndim != 3 or sources.shape[1] < 2:
        raise SystemExit(f"Unexpected model output shape {sources.shape}; expected (N, >=2, samples)")
    pred_clean = sources[:, 0, :]
    pred_artifact = sources[:, 1, :]
    if not args.keep_prediction_mean:
        pred_clean = pred_clean - pred_clean.mean(axis=-1, keepdims=True)
        pred_artifact = pred_artifact - pred_artifact.mean(axis=-1, keepdims=True)

    supervised = _compute_metrics(
        noisy=val["noisy"],
        clean=val["clean"],
        artifact=val["artifact"],
        pred_clean=pred_clean,
        pred_artifact=pred_artifact,
        keep_input_mean=args.keep_input_mean,
        keep_prediction_mean=args.keep_prediction_mean,
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_examples(
        plots_dir / "conv_tasnet_examples.png",
        val["noisy"],
        val["clean"],
        val["artifact"],
        pred_clean,
        pred_artifact,
        val["sfreq"],
        seed=args.plot_seed,
    )
    _plot_summary(plots_dir / "conv_tasnet_metric_summary.png", supervised)

    metrics = {
        "synthetic_niazy_proof_fit_val_split": supervised,
        "dataset": {
            "n_pairs_evaluated": val["n_pairs"],
            "n_channels_per_example": val["n_channels"],
            "samples_per_epoch": val["n_samples"],
            "sfreq_hz": val["sfreq"],
        },
        "baseline_reference": _baseline_summary(),
    }
    config = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "device": args.device,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "keep_input_mean": bool(args.keep_input_mean),
        "keep_prediction_mean": bool(args.keep_prediction_mean),
    }
    artifacts = {
        "examples_plot": str(plots_dir / "conv_tasnet_examples.png"),
        "summary_plot": str(plots_dir / "conv_tasnet_metric_summary.png"),
    }
    interpretation = (
        "Conv-TasNet was trained from scratch on the Niazy proof-fit context "
        "dataset (Niazy/AAS 2x direct artifact library, 512-sample center "
        "epochs). Metrics reported here are on the held-out 20 % validation "
        "split using the same RNG seed as training, so each test pair is one "
        "(example, channel) mixture with known clean and artifact targets. "
        "Supervised clean-MSE reduction and artifact correlation are the "
        "primary quality signals; predicted_source_sum_mse is reported as a "
        "consistency check (Conv-TasNet does not enforce clean + artifact = "
        "noisy during training). Comparison to cascaded_dae and "
        "cascaded_context_dae is recorded for orientation only — those runs "
        "were evaluated on a different synthetic-spike dataset, so the "
        "absolute numbers are not directly comparable."
    )
    limitations = [
        "Validation is on the same Niazy/AAS-derived artifact library used for training; it does not test cross-subject or cross-scanner generalisation.",
        "Real Niazy EDF trigger-locked metrics are not included in this run; they require the EDF input and a longer pipeline. Add later when a comparable run is available across all baseline models.",
        "Source consistency (clean + artifact = noisy) is reported but not enforced during training. The DeepLearningCorrection pipeline only consumes the artifact source.",
    ]

    writer = ModelEvaluationWriter(
        model_id=MODEL_ID,
        model_name=MODEL_NAME,
        model_description=MODEL_DESCRIPTION,
        run_id=run_id,
    )
    run = writer.write(
        metrics=metrics,
        config=config,
        artifacts=artifacts,
        interpretation=interpretation,
        limitations=limitations,
    )

    print(f"manifest: {run.manifest_path}")
    print(f"metrics:  {run.metrics_path}")
    print(f"summary:  {run.summary_path}")
    print(f"plots:    {plots_dir}")


if __name__ == "__main__":
    main()
