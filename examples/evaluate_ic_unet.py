"""IC-U-Net evaluation on the Niazy proof-fit context dataset.

Loads a TorchScript checkpoint exported by ``facet-train``, runs inference on
the same NPZ used for training, and writes the standardised
``ModelEvaluationWriter`` outputs.

Example:
    uv run python examples/evaluate_ic_unet.py \\
        --checkpoint training_output/icunetniazyprooffit_<run>/exports/ic_unet.ts
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


DEFAULT_DATASET = Path(
    "./output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
)
DEFAULT_MODEL_ID = "ic_unet"
DEFAULT_MODEL_NAME = "IC-U-Net"
DEFAULT_MODEL_DESCRIPTION = (
    "Multichannel 1-D U-Net with frozen FastICA preprocessing, adapted from "
    "Chuang et al. 2022."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="TorchScript checkpoint.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Niazy proof-fit NPZ path.")
    parser.add_argument("--run-id", default=None, help="Stable evaluation run id; defaults to a timestamp.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep-input-mean", action="store_true")
    parser.add_argument("--keep-prediction-mean", action="store_true")
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


def _flatten_context_to_multichannel(noisy_context: np.ndarray) -> np.ndarray:
    """Reshape ``(N, 7, channels, samples)`` to ``(N, channels, 7*samples)``."""
    n, ce, ch, s = noisy_context.shape
    return noisy_context.transpose(0, 2, 1, 3).reshape(n, ch, ce * s)


def _predict_artifacts(
    checkpoint: Path,
    multichannel_input: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, multichannel_input.shape[0], batch_size):
            batch_np = multichannel_input[start : start + batch_size]
            tensor = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            output = model(tensor).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(output)
    return np.concatenate(predictions, axis=0)


def evaluate(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    with np.load(args.dataset, allow_pickle=True) as data:
        noisy_context = data["noisy_context"].astype(np.float32, copy=False)
        noisy_center = data["noisy_center"].astype(np.float32, copy=False)
        clean_center = data["clean_center"].astype(np.float32, copy=False)
        artifact_center = data["artifact_center"].astype(np.float32, copy=False)
        sfreq = float(data["sfreq"][0])

    multichannel_input = _flatten_context_to_multichannel(noisy_context)
    if not args.keep_input_mean:
        multichannel_input = multichannel_input - multichannel_input.mean(axis=-1, keepdims=True)

    pred_artifact = _predict_artifacts(args.checkpoint, multichannel_input, args.batch_size, args.device)
    if not args.keep_prediction_mean:
        pred_artifact = pred_artifact - pred_artifact.mean(axis=-1, keepdims=True)

    corrected = noisy_center - pred_artifact
    before_error = noisy_center - clean_center
    after_error = corrected - clean_center

    metrics: dict[str, Any] = {
        "n_examples": int(noisy_center.shape[0]),
        "sfreq_hz": sfreq,
        "input_mean_removed": not args.keep_input_mean,
        "prediction_mean_removed": not args.keep_prediction_mean,
    }
    synthetic = {
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
    }
    synthetic["clean_snr_improvement_db"] = (
        synthetic["clean_snr_db_after"] - synthetic["clean_snr_db_before"]
    )
    synthetic["clean_mse_reduction_pct"] = 100.0 * (
        1.0 - synthetic["clean_mse_after"] / (synthetic["clean_mse_before"] + 1e-20)
    )
    metrics["synthetic"] = synthetic

    real_proxy = {
        "trigger_locked_rms_before_uv": _rms(noisy_center) * 1e6,
        "trigger_locked_rms_after_uv": _rms(corrected) * 1e6,
        "predicted_artifact_rms_uv": _rms(pred_artifact) * 1e6,
    }
    real_proxy["trigger_locked_rms_reduction_pct"] = 100.0 * (
        1.0 - real_proxy["trigger_locked_rms_after_uv"] / (real_proxy["trigger_locked_rms_before_uv"] + 1e-20)
    )
    metrics["real_proxy"] = real_proxy

    plot_path = run_dir / "plots" / "synthetic_cleaning_examples.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_examples(
        plot_path,
        noisy_center,
        clean_center,
        corrected,
        artifact_center,
        pred_artifact,
        sfreq,
        seed=args.seed,
    )
    return metrics, {"synthetic_examples": str(plot_path.relative_to(run_dir))}


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
        axes[row, 0].set_title(f"Niazy sample {int(idx)} channel 0")
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


def main() -> None:
    args = parse_args()
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    writer = ModelEvaluationWriter(
        model_id=DEFAULT_MODEL_ID,
        model_name=DEFAULT_MODEL_NAME,
        model_description=DEFAULT_MODEL_DESCRIPTION,
        run_id=args.run_id,
    )
    run = writer.run
    run.run_dir.mkdir(parents=True, exist_ok=True)

    metrics, artifacts = evaluate(args, run.run_dir)

    config = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "batch_size": int(args.batch_size),
        "device": args.device,
        "input_mean_removed": not args.keep_input_mean,
        "prediction_mean_removed": not args.keep_prediction_mean,
    }
    interpretation = (
        "IC-U-Net trained on the Niazy proof-fit context dataset. The clean "
        "target is the AAS-corrected Niazy signal; the artifact target is the "
        "AAS-estimated artifact from the same recording. SNR-after-vs-before "
        "and the artifact correlation indicate how closely the U-Net captures "
        "the AAS estimate."
    )
    limitations = [
        "The proof-fit dataset uses the same Niazy recording for training and "
        "evaluation; numbers reported here characterise fit quality, not "
        "generalisation.",
        "The clean target is itself an AAS estimate, so absolute clean-SNR "
        "values inherit AAS bias.",
    ]
    run = writer.write(
        metrics=metrics,
        config=config,
        artifacts=artifacts,
        interpretation=interpretation,
        limitations=limitations,
    )

    print(f"evaluation run: {run.run_dir}")
    print(json.dumps(metrics["synthetic"], indent=2))


if __name__ == "__main__":
    main()
