"""Synthetic-supervised evaluation of a trained D4PM checkpoint.

Runs the conditional-diffusion artifact sampler on the Niazy proof-fit
``.npz`` bundle and writes metrics through :class:`ModelEvaluationWriter`.

Example::

    uv run python -m facet.models.d4pm.evaluate \\
        --checkpoint training_output/<run>/checkpoints/last.pt \\
        --dataset output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz \\
        --sample-steps 50 \\
        --max-channels 8 \\
        --max-examples 64
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from facet.evaluation import ModelEvaluationWriter
from facet.models.d4pm.training import D4PMTrainingModule

_DEFAULT_DATASET = Path(
    "./output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--epoch-samples", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--feats", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--data-consistency-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-channels", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demean-input", action="store_true", default=True)
    parser.add_argument("--remove-prediction-mean", action="store_true", default=True)
    parser.add_argument("--run-id", type=str, default=None)
    return parser.parse_args()


def _sample_artifact_batch(
    module: D4PMTrainingModule,
    noisy_y: torch.Tensor,
    *,
    sample_steps: int,
    data_consistency_weight: float,
) -> torch.Tensor:
    device = noisy_y.device
    h_t = torch.randn_like(noisy_y)

    step_indices = torch.linspace(
        module.num_steps - 1, 0, sample_steps, device=device
    ).long()

    for step_idx, t_int in enumerate(step_indices.tolist()):
        t_tensor = torch.full(
            (noisy_y.shape[0],), t_int, dtype=torch.long, device=device
        )
        noise_level = module.sqrt_alphas_cumprod[t_tensor]
        pred_noise = module.predictor(h_t, noisy_y, noise_level)

        sqrt_alpha = module.sqrt_alphas_cumprod[t_int]
        sqrt_one_minus = module.sqrt_one_minus_alphas_cumprod[t_int]
        h0_pred = (h_t - sqrt_one_minus * pred_noise) / sqrt_alpha

        if data_consistency_weight > 0.0:
            residual = noisy_y - h0_pred
            h0_pred = h0_pred + data_consistency_weight * residual

        if step_idx == len(step_indices) - 1:
            h_t = h0_pred
            break

        t_prev = step_indices[step_idx + 1].item()
        sqrt_alpha_prev = module.sqrt_alphas_cumprod[t_prev]
        sqrt_one_minus_prev = module.sqrt_one_minus_alphas_cumprod[t_prev]
        h_t = sqrt_alpha_prev * h0_pred + sqrt_one_minus_prev * torch.randn_like(h_t)

    return h_t


def _rms(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(values ** 2, axis=axis))


def _snr_db(signal: np.ndarray, noise: np.ndarray, axis: int = -1) -> np.ndarray:
    signal_power = np.mean(signal ** 2, axis=axis) + 1e-30
    noise_power = np.mean(noise ** 2, axis=axis) + 1e-30
    return 10.0 * np.log10(signal_power / noise_power)


def _correlation(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    a_centered = a - a.mean(axis=axis, keepdims=True)
    b_centered = b - b.mean(axis=axis, keepdims=True)
    num = np.sum(a_centered * b_centered, axis=axis)
    den = np.sqrt(
        np.sum(a_centered ** 2, axis=axis) * np.sum(b_centered ** 2, axis=axis)
    )
    return num / (den + 1e-30)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    bundle = np.load(args.dataset, allow_pickle=True)
    noisy = bundle["noisy_center"].astype(np.float32)
    artifact = bundle["artifact_center"].astype(np.float32)
    clean = bundle["clean_center"].astype(np.float32)
    ch_names = list(bundle["ch_names"])
    sfreq = float(bundle["sfreq"][0]) if "sfreq" in bundle else float("nan")

    n_examples_total, n_channels_total, n_samples = noisy.shape
    n_examples = min(args.max_examples, n_examples_total)
    n_channels = min(args.max_channels, n_channels_total)
    noisy = noisy[:n_examples, :n_channels]
    artifact = artifact[:n_examples, :n_channels]
    clean = clean[:n_examples, :n_channels]

    module = D4PMTrainingModule(
        epoch_samples=args.epoch_samples,
        num_steps=args.num_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        feats=args.feats,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        embed_dim=args.embed_dim,
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    module.load_state_dict(state_dict, strict=True)
    module.to(device)
    module.eval()

    flat_n = n_examples * n_channels
    noisy_flat = noisy.reshape(flat_n, n_samples).copy()
    artifact_flat = artifact.reshape(flat_n, n_samples).copy()
    clean_flat = clean.reshape(flat_n, n_samples).copy()

    if args.demean_input:
        noisy_flat -= noisy_flat.mean(axis=-1, keepdims=True)

    pred_artifact = np.zeros_like(artifact_flat)
    batch = 16
    inference_start = time.time()
    with torch.no_grad():
        for start in range(0, flat_n, batch):
            stop = min(start + batch, flat_n)
            y = torch.as_tensor(
                noisy_flat[start:stop, None, :], dtype=torch.float32, device=device
            )
            h = _sample_artifact_batch(
                module,
                y,
                sample_steps=args.sample_steps,
                data_consistency_weight=args.data_consistency_weight,
            )
            pred_artifact[start:stop] = h[:, 0, :].cpu().numpy()
    inference_seconds = time.time() - inference_start

    if args.remove_prediction_mean:
        pred_artifact -= pred_artifact.mean(axis=-1, keepdims=True)

    corrected = noisy_flat - pred_artifact

    artifact_centered = artifact_flat - artifact_flat.mean(axis=-1, keepdims=True)

    metrics = {
        "synthetic_supervised": {
            "clean_rms_before": float(_rms(noisy_flat - clean_flat).mean()),
            "clean_rms_after": float(_rms(corrected - clean_flat).mean()),
            "clean_snr_db_before": float(_snr_db(clean_flat, noisy_flat - clean_flat).mean()),
            "clean_snr_db_after": float(_snr_db(clean_flat, corrected - clean_flat).mean()),
            "artifact_rms": float(_rms(artifact_centered).mean()),
            "artifact_pred_rms_error": float(_rms(pred_artifact - artifact_centered).mean()),
            "artifact_corr": float(_correlation(pred_artifact, artifact_centered).mean()),
            "residual_rms_ratio": float(
                (_rms(corrected - clean_flat) / (_rms(noisy_flat - clean_flat) + 1e-30)).mean()
            ),
        },
        "runtime": {
            "inference_seconds_total": inference_seconds,
            "inference_seconds_per_segment": inference_seconds / max(flat_n, 1),
            "sample_steps": args.sample_steps,
            "num_steps_train": args.num_steps,
            "device": str(device),
        },
        "shape": {
            "n_examples": n_examples,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "sfreq_hz": sfreq,
        },
    }

    writer = ModelEvaluationWriter(
        model_id="d4pm",
        model_name="D4PM",
        model_description=(
            "Single-branch conditional DDPM gradient-artifact predictor "
            "(reduction of arXiv:2509.14302)."
        ),
        run_id=args.run_id or time.strftime("%Y%m%d_%H%M%S"),
    )
    plots_dir = writer.run.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_idx = 0
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(noisy_flat[plot_idx], label="noisy y", color="black")
    axes[0].plot(clean_flat[plot_idx], label="clean (target)", color="green", alpha=0.7)
    axes[0].set_title("noisy_center / clean_center example 0")
    axes[0].legend(loc="upper right")
    axes[1].plot(artifact_centered[plot_idx], label="true artifact", color="red", alpha=0.7)
    axes[1].plot(pred_artifact[plot_idx], label="predicted artifact", color="blue", alpha=0.7)
    axes[1].set_title("predicted vs true artifact")
    axes[1].legend(loc="upper right")
    axes[2].plot(corrected[plot_idx], label="corrected", color="purple", alpha=0.7)
    axes[2].plot(clean_flat[plot_idx], label="clean (target)", color="green", alpha=0.5)
    axes[2].set_title("corrected vs clean")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    plot_path = plots_dir / "example_correction.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    config = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "epoch_samples": args.epoch_samples,
        "num_steps": args.num_steps,
        "sample_steps": args.sample_steps,
        "data_consistency_weight": args.data_consistency_weight,
        "feats": args.feats,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "embed_dim": args.embed_dim,
        "device": str(device),
        "max_channels": n_channels,
        "max_examples": n_examples,
    }

    interpretation = (
        "Single-branch conditional DDPM evaluated on the supervised "
        "Niazy proof-fit pairs. Lower residual_rms_ratio is better. "
        "clean_snr_db_after - clean_snr_db_before measures correction gain."
    )
    limitations = [
        "Single-branch reduction of D4PM; dual-branch joint posterior not implemented.",
        "Iterative sampling cost grows linearly with sample_steps.",
        "Niazy proof-fit pairs are synthetic-supervised; real-recording proxy metrics not reported here.",
    ]

    run = writer.write(
        metrics=metrics,
        config=config,
        artifacts={"example_correction_plot": str(plot_path)},
        interpretation=interpretation,
        limitations=limitations,
    )
    print(f"Wrote evaluation manifest: {run.manifest_path}")
    print(f"metrics: {metrics['synthetic_supervised']}")


if __name__ == "__main__":
    main()
