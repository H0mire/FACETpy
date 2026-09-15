"""Measure the largest batch size a training config actually fits in.

Why this is a separate step. The DHCT-GAN encoder runs a Local-Global Transformer
Block in every one of its five encoding stages, and the *global* attention in the
first stage spans the full post-pooling window — 1792 tokens for a 7-epoch
context. Whether that costs O(L) or O(L^2) memory depends entirely on which
attention kernel PyTorch picks, and that differs by backend: on MPS the fallback
materialises the L x L matrix and the extended (multi-electrode) config needs
~5.3 GB *per example*; CUDA can dispatch to a memory-efficient kernel and may
need far less. Guessing either way wastes a pod.

So: run this on the target GPU before launching, and set ``batch_size`` from what
it reports rather than from the paper's 40.

Usage (on the GPU host)::

    .venv/bin/python tools/audit/measure_training_memory.py \\
        --config src/facet/models/masterthesis/dhct_gan/strict/training_weg_a_paper.yaml

    .venv/bin/python tools/audit/measure_training_memory.py \\
        --config src/facet/models/masterthesis/dhct_gan/strict/training_weg_a_extended.yaml \\
        --batch-sizes 2 4 8 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from facet.training.cli import TrainingCLIConfig, _build_dataset, _build_model, _build_wrapper


class _ShapeOnly:
    """Stand-in dataset carrying just the shapes the model factory reads."""

    def __init__(self, noisy: np.ndarray, target: np.ndarray) -> None:
        self.input_shape = tuple(np.shape(noisy))
        self.target_shape = tuple(np.shape(target))
        self.n_channels = self.input_shape[1]
        self.context_epochs = self.input_shape[0]
        self.epoch_samples = self.input_shape[2]

    def __len__(self) -> int:
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    p.add_argument("--device", default=None, help="Override the config's device")
    p.add_argument("--steps", type=int, default=2, help="Steps per batch size (the first allocates the most)")
    p.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=None,
        metavar=("EPOCHS", "CHANNELS", "SAMPLES"),
        help="Measure on synthetic tensors of this shape instead of loading the dataset. "
        "Memory and throughput depend only on the shape, so this answers the batch-size "
        "question before a multi-gigabyte dataset has finished copying to the host.",
    )
    p.add_argument(
        "--target-rows",
        type=int,
        default=2,
        help="Rows in the target tensor when --input-shape is used (DHCT-GAN strict: [artifact, clean]).",
    )
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args()


def _reset(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()


def _peak_gb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return float("nan")


def _sync(device: str) -> None:
    """GPU work is asynchronous; timing without a barrier measures the queue."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _total_gb(device: str) -> float:
    if device.startswith("cuda"):
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return float("nan")


def main() -> None:
    args = parse_args()
    cfg = TrainingCLIConfig.from_dict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    device = args.device or cfg.model.device or "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    # The wrapper factory reads the device off the config, not off this local, so
    # a --device override that only changes the local silently builds on the
    # config's device and fails with "Torch not compiled with CUDA enabled".
    cfg.model.device = device

    if args.input_shape is not None:
        dataset = None
        noisy = np.zeros(tuple(args.input_shape), dtype=np.float32)
        target = np.zeros((args.target_rows, args.input_shape[2]), dtype=np.float32)
        sfreq = 0.0
    else:
        dataset = _build_dataset([], cfg)
        noisy, target = dataset[0]
        sfreq = float(getattr(dataset, "sfreq", 0.0) or 0.0)
    total = _total_gb(device)
    print(f"config      {args.config}")
    print(f"device      {device}" + (f"  ({total:.1f} GB total)" if np.isfinite(total) else ""))
    print(f"input       {tuple(noisy.shape)}   target {tuple(target.shape)}")

    rows: list[dict] = []
    for batch_size in args.batch_sizes:
        _reset(device)
        model = _build_model(cfg, dataset or _ShapeOnly(noisy, target), sfreq)
        wrapper = _build_wrapper(cfg, model, dataset=dataset, sfreq=sfreq)
        if not rows:
            counts = getattr(wrapper, "parameter_counts", None)
            if callable(counts):
                print("parameters  " + "  ".join(f"{k}={v:,}" for k, v in counts().items()))
            print()
            print(f"{'batch':>6s} {'peak GB':>9s} {'GB/example':>11s} {'s/step':>8s} {'ex/s':>8s}  status")
            print("-" * 64)

        rng = np.random.default_rng(0)
        # Random rather than repeated: a batch of identical rows can take a
        # different BatchNorm path and understate the real allocation.
        x = rng.standard_normal((batch_size, *np.shape(noisy))).astype(np.float32)
        y = rng.standard_normal((batch_size, *np.shape(target))).astype(np.float32)
        row: dict = {"batch_size": batch_size}
        try:
            wrapper.train_step(x, y)          # warm-up: first step allocates and compiles
            start = time.perf_counter()
            for _ in range(args.steps):
                wrapper.train_step(x, y)
            _sync(device)
            per_step = (time.perf_counter() - start) / args.steps
            peak = _peak_gb(device)
            row |= {"peak_gb": peak, "ok": True, "seconds_per_step": per_step,
                    "examples_per_second": batch_size / per_step}
            print(f"{batch_size:>6d} {peak:>9.2f} {peak / batch_size:>11.2f} "
                  f"{per_step:>8.2f} {batch_size / per_step:>8.1f}  ok")
        except (RuntimeError, torch.OutOfMemoryError) as exc:  # noqa: PERF203
            row |= {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
            print(f"{batch_size:>6d} {'-':>9s} {'-':>11s}  FAILED: {row['error'][:60]}")
            rows.append(row)
            del wrapper, model
            _reset(device)
            break
        rows.append(row)
        del wrapper, model
        _reset(device)

    ok = [r for r in rows if r.get("ok")]
    if ok:
        best = max(ok, key=lambda r: r["batch_size"])
        print(f"\nlargest batch that fits: {best['batch_size']} at {best['peak_gb']:.2f} GB")
        if np.isfinite(total):
            headroom = total - best["peak_gb"]
            print(f"headroom on this device: {headroom:.1f} GB")
            if headroom < 0.15 * total:
                print("  -> less than 15 % free; drop one batch step for a safety margin")
        rate = best.get("examples_per_second")
        if rate and dataset is not None:
            n_train = int(0.8 * len(dataset))
            epoch_s = n_train / rate
            print(f"throughput: {rate:.1f} examples/s -> ~{epoch_s / 60:.1f} min per epoch "
                  f"over {n_train} training examples "
                  f"({epoch_s * 120 / 3600:.1f} h for 120 epochs)")
        elif rate:
            n_train = 19950   # Weg-A training split; only used when no dataset is loaded
            epoch_s = n_train / rate
            print(f"throughput: {rate:.1f} examples/s -> ~{epoch_s / 60:.1f} min per epoch "
                  f"over {n_train} training examples ({epoch_s * 120 / 3600:.1f} h for 120 epochs)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"config": str(args.config), "device": device, "rows": rows}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
