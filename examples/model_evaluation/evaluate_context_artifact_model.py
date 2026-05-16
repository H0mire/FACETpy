"""Compare context-model cleaning on synthetic data and the Niazy recording.

The synthetic dataset contains ground truth clean and artifact targets, so it
gets supervised metrics. The Niazy recording does not contain a true clean EEG;
it is therefore evaluated with trigger-locked proxy metrics before/after model
correction.

Example:
    uv run python examples/model_evaluation/evaluate_context_artifact_model.py \
        --checkpoint training_output/sevenepochcontextartifactnet_20260429_204945/exports/seven_epoch_context_artifact_net.ts
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
from facet.models.cascaded_dae import CascadedDenoisingAutoencoderCorrection
from facet.models.cascaded_context_dae import CascadedContextDenoisingAutoencoderCorrection
from facet.models.demo01 import EpochContextDeepLearningCorrection

from facet import (
    DownSample,
    DropChannels,
    HighPassFilter,
    TriggerAligner,
    TriggerDetector,
    UpSample,
    load,
)

DEFAULT_CHECKPOINT = Path(
    "./training_output/sevenepochcontextartifactnet_20260429_204945/exports/seven_epoch_context_artifact_net.ts"
)
DEFAULT_SYNTHETIC_DATASET = Path(
    "./output/synthetic_spike_artifact_context_from_generator/synthetic_spike_artifact_context_dataset.npz"
)
DEFAULT_NIAZY_INPUT = Path("./examples/datasets/NiazyFMRI.edf")
DEFAULT_OUTPUT_ROOT = Path("./output/model_evaluations")
DEFAULT_NON_EEG_CHANNELS = ["EKG", "EMG", "EOG", "ECG"]
MODEL_NAMES = {
    "cascaded_dae": "Cascaded DAE",
    "demo01": "Demo 01 Context CNN",
    "cascaded_context_dae": "Cascaded Context DAE",
}
MODEL_DESCRIPTIONS = {
    "cascaded_dae": "Channel-wise cascaded denoising autoencoder without trigger context.",
    "demo01": "Frozen seven-epoch context CNN proof-of-concept.",
    "cascaded_context_dae": "Seven-epoch channel-wise cascaded denoising autoencoder.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="TorchScript checkpoint.")
    parser.add_argument("--synthetic-dataset", type=Path, default=DEFAULT_SYNTHETIC_DATASET, help="Synthetic NPZ dataset.")
    parser.add_argument("--niazy-input", type=Path, default=DEFAULT_NIAZY_INPUT, help="Niazy EDF recording.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for metrics and plots.")
    parser.add_argument("--run-id", default=None, help="Stable evaluation run id. Defaults to a timestamp.")
    parser.add_argument(
        "--model-id",
        choices=sorted(MODEL_NAMES),
        default="demo01",
        help="Model package id used for standardized evaluation storage.",
    )
    parser.add_argument("--model-name", default=None, help="Human-readable model name for the evaluation report.")
    parser.add_argument("--trigger-regex", default=r"\b1\b", help="Niazy trigger regex.")
    parser.add_argument("--context-epochs", type=int, default=7, help="Odd number of context epochs.")
    parser.add_argument("--epoch-samples", type=int, default=292, help="Fixed model epoch length.")
    parser.add_argument("--batch-size", type=int, default=128, help="Synthetic model evaluation batch size.")
    parser.add_argument("--max-synthetic-examples", type=int, default=0, help="Limit synthetic examples; 0 means all.")
    parser.add_argument("--device", default="cpu", help="PyTorch device.")
    parser.add_argument(
        "--keep-input-mean",
        action="store_true",
        help="Do not remove the per-epoch input mean before model inference.",
    )
    parser.add_argument(
        "--keep-prediction-mean",
        action="store_true",
        help="Do not remove the per-epoch mean from predicted artifacts before subtraction.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Plot sampling seed.")
    return parser.parse_args()


def _resolve_output_args(args: argparse.Namespace) -> None:
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.model_id / args.run_id
    if args.model_name is None:
        args.model_name = MODEL_NAMES[args.model_id]


def _build_correction(args: argparse.Namespace):
    if args.model_id == "cascaded_dae":
        return CascadedDenoisingAutoencoderCorrection(
            checkpoint_path=args.checkpoint,
            chunk_size_samples=args.epoch_samples,
            chunk_overlap_samples=0,
            device=args.device,
            demean_input=not args.keep_input_mean,
            remove_prediction_mean=not args.keep_prediction_mean,
        )

    kwargs = {
        "checkpoint_path": args.checkpoint,
        "context_epochs": args.context_epochs,
        "epoch_samples": args.epoch_samples,
        "device": args.device,
        "demean_input": not args.keep_input_mean,
        "remove_prediction_mean": not args.keep_prediction_mean,
    }
    if args.model_id == "demo01":
        return EpochContextDeepLearningCorrection(**kwargs)
    if args.model_id == "cascaded_context_dae":
        return CascadedContextDenoisingAutoencoderCorrection(**kwargs)
    raise ValueError(f"Unsupported context model id: {args.model_id}")


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


def _predict_synthetic(checkpoint: Path, noisy_context: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    import torch

    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, noisy_context.shape[0], batch_size):
            batch_np = noisy_context[start : start + batch_size]
            batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            pred = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(pred)
    return np.concatenate(predictions, axis=0)


def _build_synthetic_model_input(args: argparse.Namespace, noisy_context: np.ndarray, noisy_center: np.ndarray) -> np.ndarray:
    if args.model_id == "cascaded_dae":
        model_input = noisy_center
    else:
        model_input = noisy_context
    if not args.keep_input_mean:
        model_input = model_input - model_input.mean(axis=-1, keepdims=True)
    return model_input


def evaluate_synthetic(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    with np.load(args.synthetic_dataset, allow_pickle=True) as data:
        noisy_context = data["noisy_context"].astype(np.float32, copy=False)
        noisy_center = data["noisy_center"].astype(np.float32, copy=False)
        clean_center = data["clean_center"].astype(np.float32, copy=False)
        artifact_center = data["artifact_center"].astype(np.float32, copy=False)
        sfreq = float(data["sfreq"][0])

    if args.max_synthetic_examples > 0:
        noisy_context = noisy_context[: args.max_synthetic_examples]
        noisy_center = noisy_center[: args.max_synthetic_examples]
        clean_center = clean_center[: args.max_synthetic_examples]
        artifact_center = artifact_center[: args.max_synthetic_examples]

    model_input = _build_synthetic_model_input(args, noisy_context, noisy_center)
    pred_artifact = _predict_synthetic(args.checkpoint, model_input, args.batch_size, args.device)
    if not args.keep_prediction_mean:
        pred_artifact = pred_artifact - pred_artifact.mean(axis=-1, keepdims=True)
    corrected = noisy_center - pred_artifact
    before_error = noisy_center - clean_center
    after_error = corrected - clean_center

    metrics = {
        "n_examples": int(noisy_center.shape[0]),
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
        "input_mean_removed": not args.keep_input_mean,
        "prediction_mean_removed": not args.keep_prediction_mean,
    }
    edge_width = min(8, pred_artifact.shape[-1] // 4)
    edge_abs = np.mean(np.abs(np.concatenate([pred_artifact[..., :edge_width], pred_artifact[..., -edge_width:]], axis=-1)))
    center_abs = np.mean(np.abs(pred_artifact[..., edge_width:-edge_width]))
    metrics["predicted_artifact_edge_abs_mean_uv"] = float(edge_abs * 1e6)
    metrics["predicted_artifact_center_abs_mean_uv"] = float(center_abs * 1e6)
    metrics["predicted_artifact_edge_to_center_abs_ratio"] = float(edge_abs / (center_abs + 1e-20))
    metrics["clean_mse_reduction_pct"] = 100.0 * (1.0 - metrics["clean_mse_after"] / (metrics["clean_mse_before"] + 1e-20))
    metrics["clean_snr_improvement_db"] = metrics["clean_snr_db_after"] - metrics["clean_snr_db_before"]

    _plot_synthetic_examples(
        output_dir / "synthetic_cleaning_examples.png",
        noisy_center,
        clean_center,
        corrected,
        artifact_center,
        pred_artifact,
        sfreq,
        seed=args.seed,
    )
    _plot_synthetic_summary(output_dir / "synthetic_metric_summary.png", metrics)
    return metrics


def _plot_synthetic_examples(
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
        axes[row, 0].set_title(f"Synthetic sample {int(idx)}: EEG before/after")
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


def _plot_synthetic_summary(path: Path, metrics: dict[str, Any]) -> None:
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
    ax.set_title("Synthetic supervised error before vs after correction")
    ax.set_ylabel("log-scaled error")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _epoch_boundaries(triggers: np.ndarray, offset_seconds: float, sfreq: float, n_times: int) -> tuple[np.ndarray, np.ndarray]:
    offset_samples = int(round(offset_seconds * sfreq))
    starts = triggers[:-1] + offset_samples
    stops = triggers[1:] + offset_samples
    valid = (starts >= 0) & (stops > starts) & (stops <= n_times)
    return starts[valid].astype(int), stops[valid].astype(int)


def _resample_epoch(epoch: np.ndarray, target_samples: int) -> np.ndarray:
    if epoch.shape[-1] == target_samples:
        return epoch.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, epoch.shape[-1], dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    return np.vstack([np.interp(x_new, x_old, channel) for channel in epoch]).astype(np.float32, copy=False)


def _trigger_locked_stack(data: np.ndarray, starts: np.ndarray, stops: np.ndarray, target_samples: int) -> np.ndarray:
    return np.stack([_resample_epoch(data[:, start:stop], target_samples) for start, stop in zip(starts, stops, strict=False)])


def evaluate_niazy(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    context = load(str(args.niazy_input), preload=True, artifact_to_trigger_offset=-0.005)
    context = (
        context
        | DropChannels(channels=DEFAULT_NON_EEG_CHANNELS)
        | TriggerDetector(regex=args.trigger_regex)
        | HighPassFilter(freq=1.0)
        | UpSample(factor=10)
        | TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False)
        | DownSample(factor=10)
    )

    raw_before = context.get_raw().copy()
    before = raw_before.get_data().astype(np.float32, copy=False)
    triggers = np.asarray(context.get_triggers(), dtype=int)
    starts, stops = _epoch_boundaries(
        triggers,
        context.metadata.artifact_to_trigger_offset,
        context.get_sfreq(),
        raw_before.n_times,
    )

    corrected_context = context | _build_correction(args)
    after = corrected_context.get_raw().get_data().astype(np.float32, copy=False)
    predicted = corrected_context.get_estimated_noise().astype(np.float32, copy=False)

    radius = args.context_epochs // 2
    eval_starts = starts[radius : len(starts) - radius]
    eval_stops = stops[radius : len(stops) - radius]
    before_epochs = _trigger_locked_stack(before, eval_starts, eval_stops, args.epoch_samples)
    after_epochs = _trigger_locked_stack(after, eval_starts, eval_stops, args.epoch_samples)
    pred_epochs = _trigger_locked_stack(predicted, eval_starts, eval_stops, args.epoch_samples)

    before_template = np.median(before_epochs, axis=0)
    after_template = np.median(after_epochs, axis=0)
    before_ptp = np.ptp(before_template, axis=-1)
    after_ptp = np.ptp(after_template, axis=-1)

    metrics = {
        "n_channels": int(before.shape[0]),
        "n_samples": int(before.shape[1]),
        "sfreq_hz": float(context.get_sfreq()),
        "n_triggers": int(len(triggers)),
        "corrected_center_epochs": int(len(eval_starts)),
        "epoch_length_min": int((eval_stops - eval_starts).min()),
        "epoch_length_median": float(np.median(eval_stops - eval_starts)),
        "epoch_length_max": int((eval_stops - eval_starts).max()),
        "trigger_locked_rms_before": _rms(before_epochs),
        "trigger_locked_rms_after": _rms(after_epochs),
        "template_rms_before": _rms(before_template),
        "template_rms_after": _rms(after_template),
        "template_peak_to_peak_median_before": float(np.median(before_ptp)),
        "template_peak_to_peak_median_after": float(np.median(after_ptp)),
        "predicted_artifact_rms": _rms(pred_epochs),
    }
    metrics["trigger_locked_rms_reduction_pct"] = 100.0 * (
        1.0 - metrics["trigger_locked_rms_after"] / (metrics["trigger_locked_rms_before"] + 1e-20)
    )
    metrics["template_rms_reduction_pct"] = 100.0 * (
        1.0 - metrics["template_rms_after"] / (metrics["template_rms_before"] + 1e-20)
    )
    metrics["template_peak_to_peak_reduction_pct"] = 100.0 * (
        1.0
        - metrics["template_peak_to_peak_median_after"] / (metrics["template_peak_to_peak_median_before"] + 1e-20)
    )

    _plot_niazy_templates(
        output_dir / "niazy_trigger_locked_templates.png",
        before_template,
        after_template,
        pred_epochs,
        context.get_sfreq(),
    )
    _plot_niazy_examples(
        output_dir / "niazy_cleaning_examples.png",
        before,
        after,
        predicted,
        eval_starts,
        eval_stops,
        context.get_sfreq(),
        seed=args.seed,
    )
    return metrics


def _plot_niazy_templates(
    path: Path,
    before_template: np.ndarray,
    after_template: np.ndarray,
    pred_epochs: np.ndarray,
    sfreq: float,
) -> None:
    pred_template = np.median(pred_epochs, axis=0)
    time_ms = np.arange(before_template.shape[-1]) / sfreq * 1000.0
    channels = [0, min(1, before_template.shape[0] - 1), min(2, before_template.shape[0] - 1)]

    fig, axes = plt.subplots(len(channels), 1, figsize=(11, 2.8 * len(channels)), squeeze=False)
    for row, ch_idx in enumerate(channels):
        ax = axes[row, 0]
        ax.plot(time_ms, before_template[ch_idx] * 1e6, label="before", color="#9a3412")
        ax.plot(time_ms, after_template[ch_idx] * 1e6, label="after", color="#1d4ed8")
        ax.plot(time_ms, pred_template[ch_idx] * 1e6, label="predicted artifact", color="#dc2626", alpha=0.85)
        ax.set_title(f"Niazy trigger-locked median template, channel {ch_idx}")
        ax.set_ylabel("uV")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("ms")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_niazy_examples(
    path: Path,
    before: np.ndarray,
    after: np.ndarray,
    predicted: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    sfreq: float,
    *,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n = min(4, len(starts))
    indices = rng.choice(len(starts), size=n, replace=False)
    channel = 0
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), squeeze=False)
    for row, epoch_idx in enumerate(indices):
        start = starts[epoch_idx]
        stop = stops[epoch_idx]
        time_ms = np.arange(stop - start) / sfreq * 1000.0
        ax = axes[row, 0]
        ax.plot(time_ms, before[channel, start:stop] * 1e6, label="before", color="#9a3412")
        ax.plot(time_ms, after[channel, start:stop] * 1e6, label="after", color="#1d4ed8")
        ax.plot(time_ms, predicted[channel, start:stop] * 1e6, label="predicted artifact", color="#dc2626", alpha=0.85)
        ax.set_title(f"Niazy epoch {int(epoch_idx)}, channel {channel}")
        ax.set_ylabel("uV")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("ms")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _resolve_output_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_metrics = evaluate_synthetic(args, args.output_dir)
    niazy_metrics = evaluate_niazy(args, args.output_dir)
    summary = {
        "schema": "legacy_context_artifact_model_metrics",
        "model_id": args.model_id,
        "model_name": args.model_name,
        "run_id": args.run_id,
        "checkpoint": str(args.checkpoint),
        "synthetic_dataset": str(args.synthetic_dataset),
        "niazy_input": str(args.niazy_input),
        "synthetic": synthetic_metrics,
        "niazy": niazy_metrics,
        "interpretation_note": (
            "Synthetic metrics are supervised against clean/artifact targets. "
            "Niazy metrics are unsupervised trigger-locked proxies because no true clean EEG is available."
        ),
    }

    metrics_path = args.output_dir / "context_artifact_model_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = args.output_dir / "context_artifact_model_evaluation_report.md"
    report_path.write_text(_format_report(summary), encoding="utf-8")
    writer = ModelEvaluationWriter(
        model_id=args.model_id,
        model_name=args.model_name,
        model_description=MODEL_DESCRIPTIONS[args.model_id],
        run_id=args.run_id,
    )
    standard_run = writer.write(
        metrics={
            "synthetic": synthetic_metrics,
            "niazy": niazy_metrics,
        },
        config={
            "checkpoint": str(args.checkpoint),
            "synthetic_dataset": str(args.synthetic_dataset),
            "niazy_input": str(args.niazy_input),
            "context_epochs": args.context_epochs,
            "epoch_samples": args.epoch_samples,
            "device": args.device,
            "input_mean_removed": not args.keep_input_mean,
            "prediction_mean_removed": not args.keep_prediction_mean,
            "trigger_regex": args.trigger_regex,
        },
        artifacts={
            "legacy_metrics": metrics_path,
            "legacy_report": report_path,
            "synthetic_cleaning_examples": args.output_dir / "synthetic_cleaning_examples.png",
            "synthetic_metric_summary": args.output_dir / "synthetic_metric_summary.png",
            "niazy_cleaning_examples": args.output_dir / "niazy_cleaning_examples.png",
            "niazy_trigger_locked_templates": args.output_dir / "niazy_trigger_locked_templates.png",
        },
        interpretation=summary["interpretation_note"],
        limitations=[
            "Synthetic metrics use generated clean/artifact targets and may not represent real scanner distributions.",
            "Niazy metrics are trigger-locked proxy metrics without clean EEG ground truth.",
        ],
    )
    print("Saved context artifact model evaluation:")
    print(f"  metrics               : {metrics_path}")
    print(f"  report                : {report_path}")
    print(f"  standard manifest     : {standard_run.manifest_path}")
    print(f"  standard summary      : {standard_run.summary_path}")
    print(f"  synthetic examples    : {args.output_dir / 'synthetic_cleaning_examples.png'}")
    print(f"  synthetic summary     : {args.output_dir / 'synthetic_metric_summary.png'}")
    print(f"  niazy examples        : {args.output_dir / 'niazy_cleaning_examples.png'}")
    print(f"  niazy trigger template: {args.output_dir / 'niazy_trigger_locked_templates.png'}")
    print(f"  synthetic SNR gain    : {synthetic_metrics['clean_snr_improvement_db']:.3f} dB")
    print(f"  niazy template RMS red: {niazy_metrics['template_rms_reduction_pct']:.2f} %")


def _format_report(summary: dict[str, Any]) -> str:
    synthetic = summary["synthetic"]
    niazy = summary["niazy"]
    return f"""# Context Artifact Model Evaluation

## Scope

This report compares the same TorchScript context-artifact model on two inputs:

- **Synthetic training-style dataset:** supervised evaluation with known clean center epoch and artifact target.
- **Niazy EEG-fMRI recording:** unsupervised/proxy evaluation because no true clean EEG reference is available.

Checkpoint: `{summary["checkpoint"]}`

## Synthetic Metrics

| Metric | Value |
| --- | ---: |
| examples | {synthetic["n_examples"]} |
| clean MSE before | {synthetic["clean_mse_before"]:.6e} |
| clean MSE after | {synthetic["clean_mse_after"]:.6e} |
| clean MSE reduction | {synthetic["clean_mse_reduction_pct"]:.2f} % |
| clean SNR before | {synthetic["clean_snr_db_before"]:.3f} dB |
| clean SNR after | {synthetic["clean_snr_db_after"]:.3f} dB |
| clean SNR improvement | {synthetic["clean_snr_improvement_db"]:.3f} dB |
| artifact MAE | {synthetic["artifact_mae"]:.6e} |
| artifact correlation | {synthetic["artifact_corr"]:.4f} |
| residual RMS ratio | {synthetic["residual_error_rms_ratio"]:.4f} |
| input mean removed | {synthetic["input_mean_removed"]} |
| prediction mean removed | {synthetic["prediction_mean_removed"]} |
| predicted artifact edge mean abs | {synthetic["predicted_artifact_edge_abs_mean_uv"]:.2f} uV |
| predicted artifact center mean abs | {synthetic["predicted_artifact_center_abs_mean_uv"]:.2f} uV |
| edge/center abs ratio | {synthetic["predicted_artifact_edge_to_center_abs_ratio"]:.2f} |

## Niazy Proxy Metrics

| Metric | Value |
| --- | ---: |
| channels | {niazy["n_channels"]} |
| triggers | {niazy["n_triggers"]} |
| corrected center epochs | {niazy["corrected_center_epochs"]} |
| native epoch length min/median/max | {niazy["epoch_length_min"]} / {niazy["epoch_length_median"]:.1f} / {niazy["epoch_length_max"]} samples |
| trigger-locked RMS before | {niazy["trigger_locked_rms_before"]:.6e} |
| trigger-locked RMS after | {niazy["trigger_locked_rms_after"]:.6e} |
| trigger-locked RMS reduction | {niazy["trigger_locked_rms_reduction_pct"]:.2f} % |
| median-template RMS before | {niazy["template_rms_before"]:.6e} |
| median-template RMS after | {niazy["template_rms_after"]:.6e} |
| median-template RMS reduction | {niazy["template_rms_reduction_pct"]:.2f} % |
| median-template peak-to-peak reduction | {niazy["template_peak_to_peak_reduction_pct"]:.2f} % |
| predicted artifact RMS | {niazy["predicted_artifact_rms"]:.6e} |

## Interpretation

The current checkpoint is a minimal demonstration model, not a validated artifact corrector. In this run it worsens
the supervised synthetic clean reconstruction and does not reduce the Niazy trigger-locked proxy metrics. This points
to insufficient training/model capacity and/or a mismatch between the synthetic target distribution and Niazy inference
distribution.

The model also shows a systematic boundary artifact: the predicted artifact amplitude is much larger at the first and
last samples than in the center of the epoch. Boundary-sensitive metrics and plots should therefore be interpreted
with care until the model is retrained with edge-safe augmentation and padding.

The Niazy metrics must not be interpreted as clean-ground-truth metrics. They only quantify whether trigger-locked
structure becomes smaller after correction.

## Plots

- `synthetic_cleaning_examples.png`
- `synthetic_metric_summary.png`
- `niazy_cleaning_examples.png`
- `niazy_trigger_locked_templates.png`
"""


if __name__ == "__main__":
    main()
