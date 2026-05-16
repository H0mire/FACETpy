"""Evaluate the cascaded (context) DAE on the Niazy proof-fit val split.

This is the Run-1-style per-model evaluator for the two cascaded autoencoders.
It mirrors the evaluation contract used by ``evaluate_conv_tasnet.py`` /
``evaluate_demucs.py`` so that the resulting metrics line up directly with
the other deep-learning baselines in
``output/model_evaluations/INDEX.md``.

The same per-channel pair val split is used for both variants:

* ``cascaded_dae``         : input ``(N, 1, 512)``  → output ``(N, 1, 512)`` artifact
* ``cascaded_context_dae`` : input ``(N, 7, 1, 512)`` → output ``(N, 1, 512)`` artifact

Val split (matches both training_niazy_proof_fit.yaml configs):

    n_pairs = n_examples * n_channels = 833 * 30 = 24990
    val_indices = sorted(np.random.default_rng(42).permutation(n_pairs)[:4998])

Example:

    uv run python examples/model_evaluation/evaluate_cascaded_daes_niazy_proof_fit.py \\
        --variant cascaded_dae \\
        --checkpoint training_output/cascadeddenoisingautoencoderniazyprooffit_<TS>/exports/cascaded_dae.ts

    uv run python examples/model_evaluation/evaluate_cascaded_daes_niazy_proof_fit.py \\
        --variant cascaded_context_dae \\
        --checkpoint training_output/cascadedcontextdenoisingautoencoderniazyprooffit_<TS>/exports/cascaded_context_dae.ts
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from facet.evaluation import ModelEvaluationWriter

DEFAULT_DATASET = Path(
    "./output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
)
DEFAULT_OUTPUT_ROOT = Path("./output/model_evaluations")

VARIANT_META = {
    "cascaded_dae": {
        "model_name": "Cascaded DAE",
        "description": (
            "Two-stage residual fully-connected denoising autoencoder. "
            "Per-channel single-epoch artifact predictor (Stage 1 estimates the "
            "artifact, Stage 2 predicts the residual after subtraction)."
        ),
    },
    "cascaded_context_dae": {
        "model_name": "Cascaded Context DAE",
        "description": (
            "Two-stage residual cascaded denoising autoencoder operating on a "
            "7-epoch trigger-aligned context per channel. Stage 2 sees the "
            "residual after the Stage-1 artifact estimate has been subtracted "
            "from the center epoch."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_META.keys()),
        required=True,
        help="Which cascaded DAE variant to evaluate.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="TorchScript checkpoint.")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="Niazy proof-fit NPZ bundle."
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output run directory.")
    parser.add_argument("--run-id", default=None, help="Stable evaluation run id.")
    parser.add_argument("--device", default="cpu", help="PyTorch device.")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Held-out validation fraction (matches training_niazy_proof_fit.yaml).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Train/val split seed.")
    parser.add_argument("--plot-seed", type=int, default=7, help="Plot example sampling seed.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Cap (channel, epoch) pairs evaluated. 0 = full val split.",
    )
    return parser.parse_args()


def _resolve_output(args: argparse.Namespace) -> tuple[Path, str]:
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.variant / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_id


def _val_indices(n: int, val_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    n_val = max(1, int(n * val_ratio))
    return np.asarray(sorted(indices[:n_val]), dtype=np.int64)


def _load_split(args: argparse.Namespace) -> dict[str, np.ndarray]:
    with np.load(args.dataset, allow_pickle=True) as bundle:
        noisy_center = bundle["noisy_center"].astype(np.float32, copy=False)
        clean_center = bundle["clean_center"].astype(np.float32, copy=False)
        artifact_center = bundle["artifact_center"].astype(np.float32, copy=False)
        noisy_context = (
            bundle["noisy_context"].astype(np.float32, copy=False)
            if args.variant == "cascaded_context_dae"
            else None
        )
        sfreq = float(bundle["sfreq"][0])

    n_examples, n_channels, n_samples = noisy_center.shape
    total_pairs = n_examples * n_channels
    val_pairs = _val_indices(total_pairs, args.val_ratio, args.seed)
    if args.max_examples > 0:
        val_pairs = val_pairs[: args.max_examples]
    example_idx = val_pairs // n_channels
    channel_idx = val_pairs % n_channels

    payload: dict[str, np.ndarray] = {
        "noisy": noisy_center[example_idx, channel_idx, :],
        "clean": clean_center[example_idx, channel_idx, :],
        "artifact": artifact_center[example_idx, channel_idx, :],
        "sfreq": sfreq,
        "n_pairs": int(len(val_pairs)),
        "n_channels": int(n_channels),
        "n_samples": int(n_samples),
    }
    if noisy_context is not None:
        # (n_examples, 7, n_channels, n_samples) -> (n_pairs, 7, n_samples)
        ctx_pairs = noisy_context[example_idx, :, channel_idx, :]
        payload["noisy_context"] = ctx_pairs.astype(np.float32, copy=False)
    return payload


def _run_inference(
    variant: str,
    checkpoint: Path,
    payload: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    model = torch.jit.load(str(checkpoint), map_location=device)
    model.eval()

    if variant == "cascaded_dae":
        # Per-segment demean, (N, 1, S) input
        x = payload["noisy"].astype(np.float32, copy=True)
        x = x - x.mean(axis=-1, keepdims=True)
        x = x[:, np.newaxis, :]
    else:
        # Per-epoch demean on the 7-epoch context, (N, 7, 1, S) input
        x = payload["noisy_context"].astype(np.float32, copy=True)  # (N, 7, S)
        x = x - x.mean(axis=-1, keepdims=True)
        x = x[:, :, np.newaxis, :]  # (N, 7, 1, S)

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(out)
    pred = np.concatenate(predictions, axis=0)
    # Both DAE variants output (N, 1, S); squeeze the channel dim.
    if pred.ndim == 3 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    elif pred.ndim != 2:
        raise RuntimeError(f"unexpected output shape: {pred.shape}")
    # Remove prediction DC (matches the adapter and other Run-1 evaluators)
    pred = pred - pred.mean(axis=-1, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Metrics — bit-identical formulas to evaluate_conv_tasnet._compute_metrics
# ---------------------------------------------------------------------------


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


def _compute_metrics(
    *,
    noisy: np.ndarray,
    clean: np.ndarray,
    artifact: np.ndarray,
    pred_artifact: np.ndarray,
) -> dict[str, float | int | bool]:
    corrected = noisy - pred_artifact
    before_error = noisy - clean
    after_error = corrected - clean
    pred_clean = noisy - pred_artifact  # DAEs predict the artifact only

    metrics: dict[str, float | int | bool] = {
        "n_examples": int(noisy.shape[0]),
        "samples_per_epoch": int(noisy.shape[1]),
        "input_mean_removed": True,
        "prediction_mean_removed": True,
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
    metrics["clean_snr_improvement_db"] = (
        metrics["clean_snr_db_after"] - metrics["clean_snr_db_before"]
    )
    return metrics


def _plot_examples(
    path: Path,
    noisy: np.ndarray,
    clean: np.ndarray,
    artifact: np.ndarray,
    pred_artifact: np.ndarray,
    sfreq: float,
    variant: str,
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
        axes[row, 0].plot(time_ms, noisy[idx] * 1e6, label="noisy", color="#9a3412", linewidth=1.0)
        axes[row, 0].plot(time_ms, clean[idx] * 1e6, label="clean target", color="#0f766e", linewidth=1.1)
        axes[row, 0].plot(time_ms, corrected[idx] * 1e6, label="corrected", color="#1d4ed8", linewidth=1.0)
        axes[row, 0].set_title(f"{variant} sample {int(idx)}: clean vs corrected")
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


def main() -> int:
    args = parse_args()
    output_dir, run_id = _resolve_output(args)
    meta = VARIANT_META[args.variant]

    payload = _load_split(args)
    print(
        f"Loaded val split: n_pairs={payload['n_pairs']}, sfreq={payload['sfreq']:.1f} Hz, "
        f"variant={args.variant}"
    )

    started = datetime.now(UTC)
    pred_artifact = _run_inference(
        args.variant,
        args.checkpoint,
        payload,
        batch_size=args.batch_size,
        device=args.device,
    )
    elapsed = (datetime.now(UTC) - started).total_seconds()

    metrics = _compute_metrics(
        noisy=payload["noisy"],
        clean=payload["clean"],
        artifact=payload["artifact"],
        pred_artifact=pred_artifact,
    )
    metrics["inference_seconds"] = float(elapsed)
    metrics["device"] = args.device

    writer = ModelEvaluationWriter(
        model_id=args.variant,
        model_name=meta["model_name"],
        model_description=meta["description"],
        output_root=DEFAULT_OUTPUT_ROOT,
        run_id=run_id,
    )
    plot_path = writer.run.run_dir / "plots" / f"{args.variant}_val_examples.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_examples(
        plot_path,
        payload["noisy"],
        payload["clean"],
        payload["artifact"],
        pred_artifact,
        payload["sfreq"],
        args.variant,
        seed=args.plot_seed,
    )

    config = {
        "model_id": args.variant,
        "model_name": meta["model_name"],
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "n_pairs": payload["n_pairs"],
        "n_channels": payload["n_channels"],
        "n_samples": payload["n_samples"],
    }
    writer.write(
        metrics={"synthetic_niazy_proof_fit_val_split": metrics},
        config=config,
        artifacts={"val_examples": str(plot_path)},
        interpretation=(
            f"{args.variant} on Niazy proof-fit val split "
            f"({payload['n_pairs']} pairs): "
            f"clean SNR Δ = {metrics['clean_snr_improvement_db']:+.2f} dB "
            f"(before {metrics['clean_snr_db_before']:+.2f} → "
            f"after {metrics['clean_snr_db_after']:+.2f}), "
            f"artifact correlation = {metrics['artifact_corr']:+.4f}, "
            f"residual RMS ratio = {metrics['residual_error_rms_ratio']:.3f}."
        ),
        limitations=[
            "Per-channel-pair val split (seed=42, val_ratio=0.2). Identical to "
            "the split used by evaluate_conv_tasnet.py / evaluate_demucs.py — "
            "directly comparable.",
            "Targets are AAS-corrected 'clean' and AAS-estimated 'artifact' "
            "(see dataset_metadata.json) — fidelity to AAS, not absolute ground truth.",
        ],
    )

    print(
        f"  SNR before={metrics['clean_snr_db_before']:+.2f} dB  "
        f"after={metrics['clean_snr_db_after']:+.2f} dB  "
        f"Δ={metrics['clean_snr_improvement_db']:+.2f} dB  "
        f"art_corr={metrics['artifact_corr']:+.4f}  "
        f"res_rms_ratio={metrics['residual_error_rms_ratio']:.3f}  "
        f"t={elapsed:.1f}s"
    )
    try:
        rel = writer.run.run_dir.relative_to(Path.cwd())
        print(f"Wrote evaluation to {rel}")
    except ValueError:
        print(f"Wrote evaluation to {writer.run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
