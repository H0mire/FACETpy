#!/usr/bin/env python3
"""Unified-holdout cross-model re-evaluation for the Niazy proof-fit dataset.

Goal — produce directly comparable numbers across all 12 trained models by
evaluating every one of them on the SAME 167 held-out windows (5010
channel-windows). This addresses the test-split-size confound flagged in
docs/research/thesis_results_report.md §5 and docs/research/run_2_plan.md §5.1.

Holdout split:
    seed=42, val_ratio=0.2 at the WINDOW level
    → indices = sorted(np.random.default_rng(42).permutation(833)[:166])
    The same indices are used for every model. Per-channel models see
    167 * 30 = 5010 channel-windows; window-level models see 167 windows.

Metrics — replicated verbatim from examples/evaluate_conv_tasnet.py:
    clean_mse_before / after, clean_mae_before / after,
    clean_snr_db_before / after, clean_snr_improvement_db,
    artifact_mse / mae / corr / snr_db,
    residual_error_rms_ratio, clean_mse_reduction_pct.

Output layout:
    output/model_evaluations/<model_id>/holdout_v1/
        evaluation_manifest.json     (via ModelEvaluationWriter, holdout_split_hash in config)
        metrics.json                 (nested + flat_metrics)
        evaluation_summary.md
        plots/holdout_examples.png

Usage:
    uv run python tools/eval_unified_holdout.py                  # all 12 models
    uv run python tools/eval_unified_holdout.py --models dpae nested_gan
    uv run python tools/eval_unified_holdout.py --device cuda
    uv run python tools/eval_unified_holdout.py --dry-run        # split only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np

# Local repo root resolution — script lives in tools/.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from facet.evaluation.model_evaluation import ModelEvaluationWriter  # noqa: E402

DATASET_PATH = REPO_ROOT / "output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
EVAL_ROOT = REPO_ROOT / "output/model_evaluations"
TRAIN_ROOT = REPO_ROOT / "training_output"
RUN_ID = "holdout_v1"
HOLDOUT_SEED = 42
HOLDOUT_VAL_RATIO = 0.2

# ---------------------------------------------------------------------------
# Model registry — pointers to each model's trained TorchScript export, plus
# metadata needed by the unified eval driver. ts_path is relative to TRAIN_ROOT.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_name: str
    description: str
    ts_path: str  # relative to TRAIN_ROOT; "" if not used (e.g. d4pm)
    ts_path_cpu: str | None  # optional CPU-baked variant
    family: str
    notes: str


MODELS: dict[str, ModelSpec] = {
    "dpae": ModelSpec(
        model_id="dpae",
        model_name="DPAE",
        description="Dual-Pathway Autoencoder (per-channel single-epoch)",
        ts_path="dualpathwayautoencoderniazyprooffit_20260510_192929/exports/dpae.ts",
        ts_path_cpu=None,
        family="Discriminative",
        notes="(1,1,512) in → (1,1,512) artifact out. Per-segment demean.",
    ),
    "ic_unet": ModelSpec(
        model_id="ic_unet",
        model_name="IC-U-Net",
        description="ICA-augmented U-Net (multichannel, 7-epoch context concatenated along time)",
        ts_path="icunetniazyprooffit_20260510_223556/exports/ic_unet.ts",
        ts_path_cpu=None,
        family="Discriminative + ICA",
        notes="(1,30,7*512) in → (1,30,7*512) artifact out, center epoch sliced.",
    ),
    "denoise_mamba": ModelSpec(
        model_id="denoise_mamba",
        model_name="Denoise-Mamba",
        description="Selective state-space model (per-channel single-epoch)",
        ts_path="denoisemambaniazyprooffit_20260510_193847/exports/denoise_mamba.ts",
        ts_path_cpu=None,
        family="SSM",
        notes="(1,1,512) in → (1,1,512) artifact out. Per-segment demean.",
    ),
    "vit_spectrogram": ModelSpec(
        model_id="vit_spectrogram",
        model_name="ViT Spectrogram Inpainter",
        description="Vision Transformer spectrogram inpainter (per-channel 7-epoch context)",
        ts_path="vitspectrograminpainterniazyprooffit_20260510_211842/exports/vit_spectrogram.ts",
        ts_path_cpu="vitspectrograminpainterniazyprooffit_20260510_211842/exports/vit_spectrogram_cpu.ts",
        family="Vision (MAE)",
        notes="(1,7,1,512) in → (1,1,512) CLEAN out (not artifact). artifact = noisy_demeaned - pred_clean.",
    ),
    "st_gnn": ModelSpec(
        model_id="st_gnn",
        model_name="ST-GNN",
        description="Spatiotemporal Graph Neural Network (multichannel 7-epoch context)",
        ts_path="spatiotemporalgnnniazyprooffit_20260510_211512/exports/st_gnn.ts",
        ts_path_cpu=None,
        family="Graph (GNN)",
        notes="(1,7,30,512) in → (1,30,512) artifact out. Channel order is load-bearing.",
    ),
    "conv_tasnet": ModelSpec(
        model_id="conv_tasnet",
        model_name="Conv-TasNet",
        description="Audio source separation TCN (per-channel single-epoch)",
        ts_path="convtasnetniazyprooffit_20260510_202818/exports/conv_tasnet.ts",
        ts_path_cpu=None,
        family="Audio (TCN)",
        notes="(1,1,512) in → (1,2,512) sources out. source[1] = artifact.",
    ),
    "demucs": ModelSpec(
        model_id="demucs",
        model_name="Demucs",
        description="U-Net+LSTM source separation (per-channel 7-epoch concatenated)",
        ts_path="demucsniazyprooffit_20260510_224653/exports/demucs.ts",
        ts_path_cpu="demucsniazyprooffit_20260510_224653/exports/demucs_cpu.ts",
        family="Audio (U-Net+LSTM)",
        notes="(1,1,7*512=3584) in → (1,1,3584) artifact out. Center [3*512:4*512] sliced. Single-mean demean across all 7 epochs.",
    ),
    "sepformer": ModelSpec(
        model_id="sepformer",
        model_name="SepFormer",
        description="Transformer source separation (per-channel 7-epoch context)",
        ts_path="sepformerniazyprooffit_20260510_230104/exports/sepformer.ts",
        ts_path_cpu=None,
        family="Audio (Transformer)",
        notes="(1,7,1,512) in → (1,1,512) artifact out. Per-epoch demean.",
    ),
    "nested_gan": ModelSpec(
        model_id="nested_gan",
        model_name="Nested-GAN",
        description="Time-frequency + time-domain nested GAN (per-channel 7-epoch context)",
        ts_path="nestedganniazyprooffit_20260510_222546/exports/nested_gan.ts",
        ts_path_cpu=None,
        family="GAN (TF+Time)",
        notes="(1,7,1,512) in → (1,1,512) artifact out. Per-epoch demean.",
    ),
    "d4pm": ModelSpec(
        model_id="d4pm",
        model_name="D4PM",
        description="Diffusion-based artifact predictor (DDPM reverse, per-channel)",
        ts_path="d4pmartifactdiffusionniazyprooffit_20260510_201242/exports/d4pm.ts",
        ts_path_cpu=None,
        family="Diffusion",
        notes="DDPM reverse loop wrapped in TS. Slow on CPU — recommend --device cuda.",
    ),
    "dhct_gan": ModelSpec(
        model_id="dhct_gan",
        model_name="DHCT-GAN (v1, deprecated)",
        description="Hybrid CNN+Transformer GAN, single-epoch input (deprecated due to input-contract bug)",
        ts_path="dhctganniazyprooffit_20260510_213159/exports/dhct_gan.ts",
        ts_path_cpu=None,
        family="GAN (single-epoch input, failed)",
        notes="(1,1,512) in → (1,1,512) artifact out. Single epoch only, no context. Kept for completeness.",
    ),
    "dhct_gan_v2": ModelSpec(
        model_id="dhct_gan_v2",
        model_name="DHCT-GAN v2",
        description="Hybrid CNN+Transformer GAN with multi-epoch context (input-contract fixed)",
        ts_path="dhctganv2niazyprooffit_20260510_220534/exports/dhct_gan_v2.ts",
        ts_path_cpu=None,
        family="GAN (hybrid CNN+Transformer, ctx fix)",
        notes="(1,7,512) in — note flat 7-epoch packing, not (1,7,1,512). Per-epoch demean.",
    ),
}


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------


def compute_holdout_indices(
    n: int = 833, seed: int = HOLDOUT_SEED, val_ratio: float = HOLDOUT_VAL_RATIO
) -> np.ndarray:
    """Deterministic window-level holdout.

    Returns sorted np.ndarray of window indices (length = floor(n*val_ratio)).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    return np.asarray(sorted(perm[:n_val].tolist()), dtype=np.int64)


def holdout_split_hash(indices: np.ndarray) -> str:
    """Stable hash of the holdout index set for inclusion in eval manifests."""
    payload = ",".join(str(int(i)) for i in indices)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


# ---------------------------------------------------------------------------
# Metrics — verbatim from examples/evaluate_conv_tasnet.py (canonical formulas)
# ---------------------------------------------------------------------------

EPS = 1e-20


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _snr_db(reference: np.ndarray, error: np.ndarray) -> float:
    return float(
        10.0
        * np.log10(
            (np.mean(np.square(reference)) + EPS) / (np.mean(np.square(error)) + EPS)
        )
    )


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if np.std(a_flat) == 0.0 or np.std(b_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def compute_metrics(
    noisy: np.ndarray,
    clean: np.ndarray,
    artifact: np.ndarray,
    pred_artifact: np.ndarray,
    *,
    sfreq_hz: float,
) -> dict[str, float | int | bool]:
    """Compute the canonical metric set on (N, 30, 512) arrays.

    `corrected = noisy - pred_artifact` (the artifact-subtraction contract).
    Result fields match the existing per-model evaluate.py outputs so the
    flat_metrics dict is directly comparable to Run 1 artefacts.
    """
    if noisy.shape != clean.shape == artifact.shape == pred_artifact.shape:
        raise ValueError(
            f"shape mismatch: noisy={noisy.shape} clean={clean.shape} "
            f"artifact={artifact.shape} pred={pred_artifact.shape}"
        )

    corrected = noisy - pred_artifact
    before_error = noisy - clean
    after_error = corrected - clean

    metrics: dict[str, float | int | bool] = {
        "n_examples": int(noisy.shape[0]),
        "n_channels": int(noisy.shape[1]),
        "samples_per_epoch": int(noisy.shape[-1]),
        "sfreq_hz": float(sfreq_hz),
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
        "residual_error_rms_ratio": _rms(after_error) / (_rms(before_error) + EPS),
    }
    metrics["clean_mse_reduction_pct"] = 100.0 * (
        1.0 - metrics["clean_mse_after"] / (metrics["clean_mse_before"] + EPS)
    )
    metrics["clean_snr_improvement_db"] = (
        metrics["clean_snr_db_after"] - metrics["clean_snr_db_before"]
    )
    return metrics


# ---------------------------------------------------------------------------
# Inference infrastructure
# ---------------------------------------------------------------------------


def _load_torchscript(path: Path, device: str):
    """Load a TorchScript module; fall back from cuda to cpu on failure."""
    import torch

    try:
        m = torch.jit.load(str(path), map_location=device)
    except RuntimeError as e:
        if device != "cpu":
            print(f"[warn] failed to load on {device}: {e}; retrying on cpu")
            m = torch.jit.load(str(path), map_location="cpu")
        else:
            raise
    m.eval()
    return m


def _batched_forward(
    model, inputs: np.ndarray, *, batch_size: int, device: str
) -> np.ndarray:
    """Run model on inputs[N, ...] in batches; return concatenated numpy output."""
    import torch

    out_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = torch.as_tensor(
                inputs[start : start + batch_size], dtype=torch.float32, device=device
            )
            out = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            out_chunks.append(out)
    return np.concatenate(out_chunks, axis=0)


# ---------------------------------------------------------------------------
# Per-model inference functions.
#
# Contract for each: input arrays are already SLICED to the holdout windows
# (shape (167, 30, 512) for *_center, (167, 7, 30, 512) for noisy_context).
# Return: predicted_artifact of shape (167, 30, 512), float32.
# ---------------------------------------------------------------------------


def infer_dpae(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 128
) -> np.ndarray:
    """DPAE: per-channel single-epoch. Demean per segment, predict artifact, re-demean prediction."""
    ts_path = TRAIN_ROOT / spec.ts_path
    model = _load_torchscript(ts_path, device)
    noisy = ds["noisy_center"]  # (167, 30, 512)
    N, C, S = noisy.shape
    flat = noisy.reshape(N * C, 1, S).astype(np.float32, copy=True)
    flat -= flat.mean(axis=-1, keepdims=True)  # per-segment demean
    out = _batched_forward(model, flat, batch_size=batch_size, device=device)
    out = out.reshape(N * C, -1)[:, -S:]  # accept (N*C, 1, S) or (N*C, S)
    out -= out.mean(axis=-1, keepdims=True)  # remove prediction DC (matches adapter)
    return out.reshape(N, C, S)


def infer_denoise_mamba(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 128
) -> np.ndarray:
    """Per-channel single-epoch.

    The exported TorchScript has a CUDA device baked into the SSM scan
    (`torch.zeros(device='cuda:0')` in the Mamba block trace — see run_2_plan §3.5).
    To run on CPU we bypass the TS export entirely and rebuild the model from
    its Python source, loading the .pt checkpoint. The source code uses
    `device=x.device` correctly, so CPU inference works.
    """
    import torch

    from facet.models.denoise_mamba.training import build_model

    ts_path = TRAIN_ROOT / spec.ts_path
    ckpt_file = ts_path.parent.parent / "checkpoints" / "last.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"denoise_mamba checkpoint missing: {ckpt_file}")

    model = build_model(
        epoch_samples=512,
        d_model=64,
        d_state=16,
        expand=2,
        d_conv=4,
        n_blocks=4,
        dropout=0.1,
        input_kernel_size=7,
    )
    ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    noisy = ds["noisy_center"]
    N, C, S = noisy.shape
    flat = noisy.reshape(N * C, 1, S).astype(np.float32, copy=True)
    flat -= flat.mean(axis=-1, keepdims=True)

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            batch = torch.as_tensor(
                flat[start : start + batch_size], dtype=torch.float32, device=device
            )
            out = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(out)
    out = np.concatenate(predictions, axis=0)
    if out.ndim == 3 and out.shape[1] == 1:
        out = out.squeeze(1)
    out -= out.mean(axis=-1, keepdims=True)
    return out.reshape(N, C, S)


def infer_conv_tasnet(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 64
) -> np.ndarray:
    """Conv-TasNet outputs (1, n_sources>=2, S). source[1] is the artifact."""
    ts_path = TRAIN_ROOT / spec.ts_path
    model = _load_torchscript(ts_path, device)
    noisy = ds["noisy_center"]  # (167, 30, 512)
    N, C, S = noisy.shape
    flat = noisy.reshape(N * C, 1, S).astype(np.float32, copy=True)
    flat -= flat.mean(axis=-1, keepdims=True)
    out = _batched_forward(model, flat, batch_size=batch_size, device=device)
    # out shape: (N*C, n_sources, S) — pick source 1 as artifact (adapter convention)
    if out.ndim != 3 or out.shape[1] < 2:
        raise RuntimeError(f"conv_tasnet output shape unexpected: {out.shape}")
    art = out[:, 1, :]
    return art.reshape(N, C, S)


def infer_dhct_gan(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 128
) -> np.ndarray:
    """DHCT-GAN v1: single-epoch, same packing as DPAE."""
    return infer_dpae(spec, ds, device=device, batch_size=batch_size)


def _per_channel_7epoch_stack_predict(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int
) -> np.ndarray:
    """Shared inference for sepformer / nested_gan / vit_spectrogram packing.

    Input shape per item: (7, 1, 512), demeaned per epoch.
    Output shape per item: (1, 1, 512) — caller decides if it's artifact or clean.
    """
    ts_path = TRAIN_ROOT / spec.ts_path
    model = _load_torchscript(ts_path, device)
    ctx = ds["noisy_context"]  # (167, 7, 30, 512)
    N, T, C, S = ctx.shape
    # Per-channel-window flattening: (N, T, C, S) -> (N*C, T, 1, S)
    flat = ctx.transpose(0, 2, 1, 3).reshape(N * C, T, 1, S).astype(np.float32, copy=True)
    # Per-epoch demean (axis=-1 on the stack — same as evaluate scripts)
    flat -= flat.mean(axis=-1, keepdims=True)
    out = _batched_forward(model, flat, batch_size=batch_size, device=device)
    if out.ndim == 4:
        out = out.squeeze(2)
    if out.ndim != 3 or out.shape[-1] != S:
        raise RuntimeError(f"{spec.model_id} output shape unexpected: {out.shape}")
    # (N*C, 1, S) -> (N, C, S)
    out = out.reshape(N, C, 1, S).squeeze(2)
    return out


def infer_sepformer(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 32
) -> np.ndarray:
    return _per_channel_7epoch_stack_predict(spec, ds, device=device, batch_size=batch_size)


def infer_nested_gan(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 32
) -> np.ndarray:
    return _per_channel_7epoch_stack_predict(spec, ds, device=device, batch_size=batch_size)


def infer_vit_spectrogram(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 32
) -> np.ndarray:
    """ViT predicts CLEAN center, not artifact. Convert via: artifact = noisy_demeaned - pred_clean."""
    # Prefer the _cpu.ts variant when running on CPU
    use_spec = spec
    if device == "cpu" and spec.ts_path_cpu and (TRAIN_ROOT / spec.ts_path_cpu).exists():
        use_spec = ModelSpec(**{**spec.__dict__, "ts_path": spec.ts_path_cpu})
    pred_clean = _per_channel_7epoch_stack_predict(
        use_spec, ds, device=device, batch_size=batch_size
    )  # (N, C, S), demeaned-space prediction of clean

    noisy_center = ds["noisy_center"]  # (N, C, S)
    noisy_center_demeaned = noisy_center - noisy_center.mean(axis=-1, keepdims=True)
    pred_artifact = noisy_center_demeaned - pred_clean
    return pred_artifact.astype(np.float32, copy=False)


def infer_dhct_gan_v2(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 64
) -> np.ndarray:
    """DHCT-GAN v2: (1, 7, 512) flat packing (not (1, 7, 1, 512))."""
    ts_path = TRAIN_ROOT / spec.ts_path
    model = _load_torchscript(ts_path, device)
    ctx = ds["noisy_context"]  # (167, 7, 30, 512)
    N, T, C, S = ctx.shape
    # (N, T, C, S) -> (N*C, T, S)
    flat = ctx.transpose(0, 2, 1, 3).reshape(N * C, T, S).astype(np.float32, copy=True)
    flat -= flat.mean(axis=-1, keepdims=True)  # per-epoch demean
    out = _batched_forward(model, flat, batch_size=batch_size, device=device)
    if out.ndim == 3 and out.shape[1] == 1:
        out = out.squeeze(1)  # (N*C, S)
    elif out.ndim == 2:
        pass
    else:
        raise RuntimeError(f"dhct_gan_v2 output shape unexpected: {out.shape}")
    return out.reshape(N, C, S)


def infer_demucs(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 16
) -> np.ndarray:
    """Demucs: per-channel 7-epoch concatenated to (1, 1, 3584). Single-mean demean. Slice center."""
    use_spec = spec
    if device == "cpu" and spec.ts_path_cpu and (TRAIN_ROOT / spec.ts_path_cpu).exists():
        use_spec = ModelSpec(**{**spec.__dict__, "ts_path": spec.ts_path_cpu})
    ts_path = TRAIN_ROOT / use_spec.ts_path
    model = _load_torchscript(ts_path, device)
    ctx = ds["noisy_context"]  # (N, 7, 30, 512)
    N, T, C, S = ctx.shape
    # (N, T, C, S) -> (N*C, T, S) -> (N*C, T*S)
    stacked = ctx.transpose(0, 2, 1, 3).reshape(N * C, T, S)
    stacked = stacked.astype(np.float32, copy=True)
    # Single mean across all (T, S) — matches adapter's np.mean(epoch_stack)
    means = stacked.mean(axis=(-2, -1), keepdims=True)
    stacked = stacked - means
    flat = stacked.reshape(N * C, 1, T * S)
    out = _batched_forward(model, flat, batch_size=batch_size, device=device)
    if out.ndim == 3 and out.shape[1] == 1:
        out = out.squeeze(1)  # (N*C, T*S)
    elif out.ndim != 2:
        raise RuntimeError(f"demucs output shape unexpected: {out.shape}")
    # Slice center epoch: [3*S:4*S]
    center_start = (T // 2) * S
    center_stop = center_start + S
    center = out[:, center_start:center_stop]
    return center.reshape(N, C, S)


def infer_ic_unet(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 8
) -> np.ndarray:
    """IC-U-Net: multichannel context concatenated along time, (1, 30, 7*512)."""
    ts_path = TRAIN_ROOT / spec.ts_path
    model = _load_torchscript(ts_path, device)
    ctx = ds["noisy_context"]  # (N, T=7, C=30, S=512)
    N, T, C, S = ctx.shape
    # Stack along time: (N, T, C, S) -> (N, C, T*S) by interleaving epochs
    # ic_unet adapter: stack epochs end-to-end along axis=-1
    # Permute (N, T, C, S) -> (N, C, T, S) -> reshape (N, C, T*S)
    stacked = ctx.transpose(0, 2, 1, 3).reshape(N, C, T * S).astype(np.float32, copy=True)
    stacked -= stacked.mean(axis=-1, keepdims=True)  # per-channel demean
    out = _batched_forward(model, stacked, batch_size=batch_size, device=device)
    if out.ndim != 3 or out.shape[1] != C:
        raise RuntimeError(f"ic_unet output shape unexpected: {out.shape}")
    if out.shape[2] == S:
        # Some IC-U-Net exports already slice the center epoch internally.
        return out.astype(np.float32, copy=False)
    if out.shape[2] == T * S:
        center_start = (T // 2) * S
        return out[:, :, center_start : center_start + S].astype(np.float32, copy=False)
    raise RuntimeError(
        f"ic_unet output time dim {out.shape[2]} matches neither S={S} nor T*S={T*S}"
    )


def infer_st_gnn(
    spec: ModelSpec, ds: dict[str, np.ndarray], *, device: str, batch_size: int = 8
) -> np.ndarray:
    """ST-GNN: full multichannel context (1, 7, 30, 512). Channel order load-bearing."""
    ts_path = TRAIN_ROOT / spec.ts_path
    model = _load_torchscript(ts_path, device)
    ctx = ds["noisy_context"].astype(np.float32, copy=True)  # (N, 7, 30, 512)
    # Per-epoch & per-channel demean — matches adapter
    ctx -= ctx.mean(axis=-1, keepdims=True)
    out = _batched_forward(model, ctx, batch_size=batch_size, device=device)
    if out.ndim != 3 or out.shape[1] != ctx.shape[2] or out.shape[2] != ctx.shape[-1]:
        raise RuntimeError(f"st_gnn output shape unexpected: {out.shape}")
    return out.astype(np.float32, copy=False)


def infer_d4pm(
    spec: ModelSpec,
    ds: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int = 64,
    sample_steps: int = 50,
    data_consistency_weight: float = 0.5,
) -> np.ndarray:
    """D4PM: DDPM reverse loop via D4PMTrainingModule + .pt checkpoint.

    The shipped TorchScript export (d4pm.ts) is a zero stub — it does NOT contain
    the DDPM sampler. So we reconstruct the training module with default
    hyperparameters (matching src/facet/models/d4pm/evaluate.py defaults) and
    load the last.pt checkpoint, then run the same reverse sampler used in
    Run 1's per-model evaluation.
    """
    import torch

    from facet.models.d4pm.training import D4PMTrainingModule

    ckpt_path = TRAIN_ROOT / spec.ts_path  # ".../exports/d4pm.ts" — replace tail
    ckpt_dir = ckpt_path.parent.parent / "checkpoints"
    ckpt_file = ckpt_dir / "last.pt"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"d4pm checkpoint missing: {ckpt_file}")

    module = D4PMTrainingModule(
        epoch_samples=512,
        num_steps=200,
        beta_start=1e-4,
        beta_end=0.02,
        feats=64,
        d_model=128,
        d_ff=512,
        n_heads=2,
        n_layers=2,
        embed_dim=128,
    )
    ckpt = torch.load(str(ckpt_file), map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    module.load_state_dict(state_dict, strict=True)
    module = module.to(device)
    module.eval()

    noisy = ds["noisy_center"]  # (N, 30, 512)
    N, C, S = noisy.shape
    flat = noisy.reshape(N * C, 1, S).astype(np.float32, copy=True)
    flat -= flat.mean(axis=-1, keepdims=True)

    torch.manual_seed(0)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, flat.shape[0], batch_size):
            noisy_y = torch.as_tensor(
                flat[start : start + batch_size], dtype=torch.float32, device=device
            )
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
            predictions.append(h_t.detach().cpu().numpy().astype(np.float32, copy=False))
    out = np.concatenate(predictions, axis=0)
    if out.ndim == 3 and out.shape[1] == 1:
        out = out.squeeze(1)
    return out.reshape(N, C, S)


INFERENCE_FUNCS: dict[str, Callable] = {
    "dpae": infer_dpae,
    "ic_unet": infer_ic_unet,
    "denoise_mamba": infer_denoise_mamba,
    "vit_spectrogram": infer_vit_spectrogram,
    "st_gnn": infer_st_gnn,
    "conv_tasnet": infer_conv_tasnet,
    "demucs": infer_demucs,
    "sepformer": infer_sepformer,
    "nested_gan": infer_nested_gan,
    "d4pm": infer_d4pm,
    "dhct_gan": infer_dhct_gan,
    "dhct_gan_v2": infer_dhct_gan_v2,
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_examples(
    path: Path,
    noisy: np.ndarray,
    clean: np.ndarray,
    corrected: np.ndarray,
    artifact: np.ndarray,
    pred_artifact: np.ndarray,
    sfreq: float,
    model_id: str,
    *,
    seed: int = 7,
    n: int = 4,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    n = min(n, noisy.shape[0])
    idx = rng.choice(noisy.shape[0], size=n, replace=False)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    time = np.arange(noisy.shape[-1]) / sfreq
    for row, i in enumerate(idx):
        c = rng.integers(0, noisy.shape[1])  # random channel
        ax_l = axes[row, 0]
        ax_l.plot(time, noisy[i, c], label="noisy", linewidth=0.6, alpha=0.7, color="#999")
        ax_l.plot(time, corrected[i, c], label="corrected", linewidth=0.8, color="C0")
        ax_l.plot(time, clean[i, c], label="clean", linewidth=0.6, alpha=0.7, color="C2", linestyle="--")
        ax_l.set_title(f"window={i}, ch={c}: noisy / corrected / clean", fontsize=9)
        ax_l.legend(fontsize=7, loc="upper right")
        ax_l.set_xlabel("time [s]", fontsize=8)

        ax_r = axes[row, 1]
        ax_r.plot(time, artifact[i, c], label="true artifact", linewidth=0.6, color="#777")
        ax_r.plot(time, pred_artifact[i, c], label="pred artifact", linewidth=0.8, color="C3")
        ax_r.set_title(f"window={i}, ch={c}: predicted vs true artifact", fontsize=9)
        ax_r.legend(fontsize=7, loc="upper right")
        ax_r.set_xlabel("time [s]", fontsize=8)
    fig.suptitle(f"{model_id} on unified holdout (seed=42, n={noisy.shape[0]} windows)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def load_holdout(dataset_path: Path, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Load and slice the dataset to the holdout indices."""
    with np.load(dataset_path, allow_pickle=True) as data:
        return {
            "noisy_context": data["noisy_context"][indices].astype(np.float32, copy=False),
            "noisy_center": data["noisy_center"][indices].astype(np.float32, copy=False),
            "clean_center": data["clean_center"][indices].astype(np.float32, copy=False),
            "artifact_center": data["artifact_center"][indices].astype(np.float32, copy=False),
            "sfreq": float(data["sfreq"][0]) if "sfreq" in data.files else float("nan"),
        }


def run_model(
    spec: ModelSpec,
    ds: dict[str, np.ndarray],
    holdout_indices: np.ndarray,
    *,
    device: str,
) -> dict:
    """Run inference + metrics + write outputs for one model. Returns metrics dict."""
    print(f"\n=== {spec.model_id} ({spec.family}) ===")
    if not (TRAIN_ROOT / spec.ts_path).exists():
        print(f"[skip] export not found: {TRAIN_ROOT / spec.ts_path}")
        return {}

    infer = INFERENCE_FUNCS[spec.model_id]
    started = datetime.now(UTC)
    try:
        pred_artifact = infer(spec, ds, device=device)
    except Exception as e:
        print(f"[error] inference failed: {e!r}")
        return {"error": repr(e)}
    elapsed = (datetime.now(UTC) - started).total_seconds()

    if pred_artifact.shape != ds["artifact_center"].shape:
        print(
            f"[error] shape mismatch: pred={pred_artifact.shape} vs "
            f"true={ds['artifact_center'].shape}"
        )
        return {"error": "shape_mismatch"}

    metrics = compute_metrics(
        ds["noisy_center"],
        ds["clean_center"],
        ds["artifact_center"],
        pred_artifact,
        sfreq_hz=ds["sfreq"],
    )
    metrics["inference_seconds"] = float(elapsed)
    metrics["device"] = device  # string — auto-stripped by flat_metrics flattener

    # Write via ModelEvaluationWriter so the output matches the existing schema
    writer = ModelEvaluationWriter(
        model_id=spec.model_id,
        model_name=spec.model_name,
        model_description=spec.description,
        output_root=EVAL_ROOT,
        run_id=RUN_ID,
    )
    plot_path = writer.run.run_dir / "plots" / "holdout_examples.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    corrected = ds["noisy_center"] - pred_artifact
    _plot_examples(
        plot_path,
        ds["noisy_center"],
        ds["clean_center"],
        corrected,
        ds["artifact_center"],
        pred_artifact,
        ds["sfreq"],
        spec.model_id,
    )

    config = {
        "model_id": spec.model_id,
        "checkpoint": str(TRAIN_ROOT / spec.ts_path),
        "dataset": str(DATASET_PATH),
        "device": device,
        "holdout_split_hash": holdout_split_hash(holdout_indices),
        "holdout_seed": HOLDOUT_SEED,
        "holdout_val_ratio": HOLDOUT_VAL_RATIO,
        "holdout_n_windows": int(len(holdout_indices)),
        "family": spec.family,
        "notes": spec.notes,
    }
    writer.write(
        metrics={"unified_holdout": metrics},
        config=config,
        artifacts={"holdout_examples": str(plot_path)},
        interpretation=_format_interpretation(spec, metrics),
        limitations=[
            "Unified holdout split: window-level seed=42 val_ratio=0.2 → "
            f"{len(holdout_indices)} windows ({metrics['n_examples'] * metrics['n_channels']} channel-windows).",
            "Metrics use canonical formulas from examples/evaluate_conv_tasnet.py.",
            "Target is AAS-corrected 'clean' — fidelity to AAS, not absolute ground truth.",
        ],
    )

    print(
        f"  SNR  before={metrics['clean_snr_db_before']:+.2f} dB  "
        f"after={metrics['clean_snr_db_after']:+.2f} dB  "
        f"Δ={metrics['clean_snr_improvement_db']:+.2f} dB  "
        f"art_corr={metrics['artifact_corr']:+.4f}  "
        f"res_rms_ratio={metrics['residual_error_rms_ratio']:.3f}  "
        f"t={elapsed:.1f}s"
    )
    return metrics


def _format_interpretation(spec: ModelSpec, m: dict) -> str:
    return (
        f"{spec.model_id} on the unified holdout split: "
        f"SNR improvement = {m['clean_snr_improvement_db']:+.2f} dB "
        f"(before={m['clean_snr_db_before']:+.2f} dB → after={m['clean_snr_db_after']:+.2f} dB). "
        f"Artifact correlation with ground truth = {m['artifact_corr']:+.4f}. "
        f"Residual RMS ratio (after / before) = {m['residual_error_rms_ratio']:.3f}."
    )


def write_index_summary(
    results: dict[str, dict], holdout_indices: np.ndarray, output: Path
) -> None:
    """Write a holdout-only ranking markdown next to INDEX.md."""
    lines = [
        "# Unified Holdout Re-Evaluation (Run 2 §5.1)",
        "",
        f"Dataset: `{DATASET_PATH.relative_to(REPO_ROOT)}`",
        f"Holdout: seed={HOLDOUT_SEED}, val_ratio={HOLDOUT_VAL_RATIO} at window level "
        f"→ {len(holdout_indices)} windows × 30 channels = "
        f"{len(holdout_indices) * 30} channel-windows.",
        f"Split hash: `{holdout_split_hash(holdout_indices)}`",
        "",
        "All metrics computed identically via `tools/eval_unified_holdout.py`. "
        "See [docs/research/run_2_plan.md §5.1](../../docs/research/run_2_plan.md).",
        "",
        "| Rank | Model | Family | SNR↑ dB | SNR before | SNR after | art.corr | res.RMS ratio | t [s] |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        [(mid, m) for mid, m in results.items() if "clean_snr_improvement_db" in m],
        key=lambda kv: -kv[1]["clean_snr_improvement_db"],
    )
    for rank, (mid, m) in enumerate(ranked, 1):
        fam = MODELS[mid].family if mid in MODELS else "?"
        lines.append(
            f"| {rank} | {mid} | {fam} | "
            f"{m['clean_snr_improvement_db']:+.2f} | "
            f"{m['clean_snr_db_before']:+.2f} | "
            f"{m['clean_snr_db_after']:+.2f} | "
            f"{m['artifact_corr']:+.4f} | "
            f"{m['residual_error_rms_ratio']:.3f} | "
            f"{m.get('inference_seconds', 0):.1f} |"
        )

    failures = {mid: m for mid, m in results.items() if "error" in m}
    if failures:
        lines += ["", "## Failures", ""]
        for mid, m in failures.items():
            lines.append(f"- **{mid}**: {m['error']}")

    lines += [
        "",
        "## Per-model artifacts",
        "",
    ]
    for mid in MODELS:
        if mid in results and "clean_snr_improvement_db" in results[mid]:
            lines.append(
                f"- [{mid}/{RUN_ID}/](../{mid}/{RUN_ID}/)"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Subset of model IDs to run (default: all 12).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print the holdout split + indices, don't run inference.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the proof-fit context dataset npz.",
    )
    args = parser.parse_args()

    holdout = compute_holdout_indices()
    split_hash = holdout_split_hash(holdout)
    print(
        f"Holdout: seed={HOLDOUT_SEED}, val_ratio={HOLDOUT_VAL_RATIO} → "
        f"{len(holdout)} windows ({split_hash})"
    )
    print(f"  First 10 indices: {holdout[:10].tolist()}")
    print(f"  Last 10 indices: {holdout[-10:].tolist()}")

    # Persist split for reproducibility
    split_path = REPO_ROOT / "output/niazy_proof_fit_context_512/holdout_v1_indices.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": HOLDOUT_SEED,
                "val_ratio": HOLDOUT_VAL_RATIO,
                "n_windows_total": 833,
                "n_windows_holdout": int(len(holdout)),
                "indices": holdout.tolist(),
                "split_hash": split_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  Wrote {split_path.relative_to(REPO_ROOT)}")

    if args.dry_run:
        return 0

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print(f"[error] dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    ds = load_holdout(dataset_path, holdout)
    print(
        f"\nDataset slice loaded: "
        f"noisy_context={ds['noisy_context'].shape}  "
        f"noisy_center={ds['noisy_center'].shape}  "
        f"sfreq={ds['sfreq']:.1f} Hz"
    )

    targets = args.models if args.models else list(MODELS.keys())
    results: dict[str, dict] = {}
    for mid in targets:
        if mid not in MODELS:
            print(f"[skip] unknown model {mid!r}")
            continue
        results[mid] = run_model(MODELS[mid], ds, holdout, device=args.device)

    summary_path = EVAL_ROOT / "UNIFIED_HOLDOUT.md"
    write_index_summary(results, holdout, summary_path)
    print(f"\nSummary written to {summary_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
