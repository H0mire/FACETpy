"""Evaluate a trained ST-GNN checkpoint on the Niazy proof-fit dataset.

This is a closed-beta evaluation helper that emits the standard
`facet.evaluation.ModelEvaluationWriter` artefacts under
``output/model_evaluations/st_gnn/<run_id>/`` for direct comparison
with the cascaded-DAE baselines.

Usage::

    uv run python examples/evaluate_st_gnn.py \
        --checkpoint training_output/spatiotemporalgnnniazyprooffit_<ts>/exports/st_gnn.ts \
        --dataset output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz \
        --run-id niazy_proof_fit_<ts>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from facet.evaluation import ModelEvaluationWriter
from facet.models.st_gnn.training import NIAZY_PROOF_FIT_CHANNELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run id under output/model_evaluations/st_gnn/.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def demean(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def run_inference(model: torch.jit.ScriptModule, noisy: np.ndarray, device: str) -> np.ndarray:
    """Predict centre-epoch artifacts for every example."""
    out = np.empty((noisy.shape[0], noisy.shape[2], noisy.shape[3]), dtype=np.float32)
    batch = 32
    with torch.no_grad():
        for start in range(0, noisy.shape[0], batch):
            stop = min(start + batch, noisy.shape[0])
            x = torch.from_numpy(noisy[start:stop]).float().to(device)
            y = model(x).cpu().numpy()
            out[start:stop] = y
    return out


def compute_metrics(
    noisy_center: np.ndarray,
    artifact_target: np.ndarray,
    clean_target: np.ndarray,
    artifact_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute the standard metric groups for synthetic supervised data."""
    corrected = noisy_center - artifact_pred

    def _mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))

    def _rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean(a ** 2)))

    def _flat_corr(a: np.ndarray, b: np.ndarray) -> float:
        a_f = a.reshape(-1)
        b_f = b.reshape(-1)
        a_f = a_f - a_f.mean()
        b_f = b_f - b_f.mean()
        denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a_f, b_f) / denom)

    def _snr_db(signal: np.ndarray, residual: np.ndarray) -> float:
        sig_pow = float(np.mean(signal ** 2))
        res_pow = float(np.mean(residual ** 2))
        if res_pow == 0.0:
            return float("inf")
        return 10.0 * float(np.log10(sig_pow / res_pow))

    rms_noisy = _rms(noisy_center)
    rms_corrected = _rms(corrected)
    rms_residual = _rms(corrected - clean_target)

    metrics = {
        "synthetic": {
            "clean_reconstruction_mse_before": _mse(noisy_center, clean_target),
            "clean_reconstruction_mse_after": _mse(corrected, clean_target),
            "clean_snr_db_before": _snr_db(clean_target, noisy_center - clean_target),
            "clean_snr_db_after": _snr_db(clean_target, corrected - clean_target),
            "artifact_prediction_mse": _mse(artifact_pred, artifact_target),
            "artifact_prediction_correlation": _flat_corr(artifact_pred, artifact_target),
            "residual_rms_ratio": rms_residual / rms_noisy if rms_noisy > 0 else float("nan"),
            "rms_noisy": rms_noisy,
            "rms_corrected": rms_corrected,
            "rms_residual": rms_residual,
        },
    }

    per_channel_artifact_rmse = np.sqrt(np.mean((artifact_pred - artifact_target) ** 2, axis=(0, 2)))
    metrics["per_channel"] = {
        f"artifact_rmse_{name}": float(per_channel_artifact_rmse[idx])
        for idx, name in enumerate(NIAZY_PROOF_FIT_CHANNELS)
    }
    return metrics


def plot_loss_curves(training_jsonl: Path, output: Path) -> None:
    if not training_jsonl.exists():
        return
    import json

    train_loss: list[float] = []
    val_loss: list[float] = []
    with training_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "train_loss" in row:
                train_loss.append(row["train_loss"])
            if "val_loss" in row:
                val_loss.append(row["val_loss"])

    if not train_loss:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_loss, label="train")
    if val_loss:
        ax.plot(val_loss, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("ST-GNN training/val loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def plot_examples(
    output: Path,
    noisy_center: np.ndarray,
    clean_target: np.ndarray,
    artifact_target: np.ndarray,
    artifact_pred: np.ndarray,
    examples: int = 3,
) -> None:
    fig, axes = plt.subplots(examples, 3, figsize=(11, 2.4 * examples), sharex=True)
    if examples == 1:
        axes = axes[np.newaxis, :]
    n_total = noisy_center.shape[0]
    sampled = np.linspace(0, n_total - 1, examples).astype(int)
    for row, idx in enumerate(sampled):
        # Pick the centre channel "Cz" if available.
        ch_idx = NIAZY_PROOF_FIT_CHANNELS.index("Cz")
        axes[row, 0].plot(noisy_center[idx, ch_idx], label="noisy", color="tab:gray")
        axes[row, 0].plot(clean_target[idx, ch_idx], label="clean target", color="tab:green", lw=0.8)
        axes[row, 0].set_title(f"example {idx} — Cz noisy vs clean")
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].plot(artifact_target[idx, ch_idx], label="target", color="tab:blue")
        axes[row, 1].plot(artifact_pred[idx, ch_idx], label="pred", color="tab:orange", lw=0.8)
        axes[row, 1].set_title(f"example {idx} — Cz artifact")
        axes[row, 1].legend(fontsize=7)
        corrected = noisy_center[idx, ch_idx] - artifact_pred[idx, ch_idx]
        axes[row, 2].plot(corrected, label="corrected", color="tab:red")
        axes[row, 2].plot(clean_target[idx, ch_idx], label="clean target", color="tab:green", lw=0.8)
        axes[row, 2].set_title(f"example {idx} — Cz corrected")
        axes[row, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def plot_per_channel_rmse(
    output: Path,
    artifact_target: np.ndarray,
    artifact_pred: np.ndarray,
) -> None:
    rmse = np.sqrt(np.mean((artifact_pred - artifact_target) ** 2, axis=(0, 2)))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(rmse.shape[0]), rmse)
    ax.set_xticks(np.arange(rmse.shape[0]))
    ax.set_xticklabels(NIAZY_PROOF_FIT_CHANNELS, rotation=70, fontsize=7)
    ax.set_ylabel("artifact RMSE")
    ax.set_title("ST-GNN per-channel artifact RMSE on validation set")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()

    bundle = np.load(dataset_path, allow_pickle=True)
    noisy_context = bundle["noisy_context"].astype(np.float32, copy=False)
    artifact_center = bundle["artifact_center"].astype(np.float32, copy=False)
    clean_center = bundle["clean_center"].astype(np.float32, copy=False)
    noisy_center = bundle["noisy_center"].astype(np.float32, copy=False)

    train_idx, val_idx = split_indices(noisy_context.shape[0], args.val_ratio, args.seed)
    val_noisy_context = demean(noisy_context[val_idx])
    val_noisy_center = noisy_center[val_idx]
    val_artifact_center = artifact_center[val_idx]
    val_clean_center = clean_center[val_idx]

    val_artifact_demeaned = demean(val_artifact_center)

    model = torch.jit.load(str(checkpoint), map_location=args.device)
    model.eval()
    pred_artifact = run_inference(model, val_noisy_context, args.device)

    metrics = compute_metrics(
        noisy_center=val_noisy_center,
        artifact_target=val_artifact_demeaned,
        clean_target=val_clean_center,
        artifact_pred=pred_artifact,
    )

    writer = ModelEvaluationWriter(
        model_id="st_gnn",
        model_name="Spatiotemporal Graph Neural Network",
        model_description=(
            "Two ST-Conv blocks (TGLU + Chebyshev spatial conv) operating on a "
            "fixed 30-electrode k-NN graph derived from MNE standard_1005."
        ),
        run_id=args.run_id or None,
    )
    run = writer.run

    artifacts: dict[str, str] = {}
    plots_dir = run.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    examples_path = plots_dir / "validation_examples.png"
    plot_examples(
        examples_path,
        noisy_center=val_noisy_center,
        clean_target=val_clean_center,
        artifact_target=val_artifact_demeaned,
        artifact_pred=pred_artifact,
    )
    artifacts["validation_examples"] = str(examples_path.relative_to(run.run_dir))

    rmse_path = plots_dir / "per_channel_rmse.png"
    plot_per_channel_rmse(
        rmse_path,
        artifact_target=val_artifact_demeaned,
        artifact_pred=pred_artifact,
    )
    artifacts["per_channel_rmse"] = str(rmse_path.relative_to(run.run_dir))

    training_jsonl = checkpoint.parent.parent / "training.jsonl"
    loss_path = plots_dir / "loss_log.png"
    plot_loss_curves(training_jsonl, loss_path)
    if loss_path.exists():
        artifacts["loss_log"] = str(loss_path.relative_to(run.run_dir))

    interpretation = (
        "Validation set: {n_val} examples (seed={seed}, val_ratio={vr}). "
        "Chebyshev order K=3, hidden=16, two ST-Conv blocks. "
        "Compare flat_metrics.synthetic.* against cascaded_context_dae and "
        "cascaded_dae on the same Niazy proof-fit split.".format(
            n_val=len(val_idx), seed=args.seed, vr=args.val_ratio,
        )
    )
    limitations = [
        "Niazy proof-fit dataset uses AAS-derived clean and artifact targets; "
        "absolute metrics overstate generalisation to independent recordings.",
        "Validation split is in-recording; the model has been exposed to the "
        "same artifact morphology family at training time.",
        "Per-electrode adjacency assumes the Niazy 30-channel layout in the "
        "exact stored order.",
    ]

    config = {
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "split": {"val_ratio": args.val_ratio, "seed": args.seed,
                   "n_train": int(len(train_idx)), "n_val": int(len(val_idx))},
        "device": args.device,
        "architecture": {
            "context_epochs": int(noisy_context.shape[1]),
            "n_channels": int(noisy_context.shape[2]),
            "samples": int(noisy_context.shape[3]),
            "hidden_channels": 16,
            "k_order": 3,
            "time_kernel": 3,
            "knn_k": 4,
        },
    }

    written = writer.write(
        metrics=metrics,
        config=config,
        artifacts=artifacts,
        interpretation=interpretation,
        limitations=limitations,
    )
    print("Evaluation written:")
    print(f"  manifest : {written.manifest_path}")
    print(f"  metrics  : {written.metrics_path}")
    print(f"  summary  : {written.summary_path}")


if __name__ == "__main__":
    main()
