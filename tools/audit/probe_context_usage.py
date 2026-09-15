"""Measure what a paper-accurate edition's forward pass actually consumes.

Documentation can claim a 7-epoch, 30-channel context while the *dataset factory*
explodes the bundle into per-channel examples and the *forward pass* slices out
the centre epoch. Reading the code catches the obvious cases; it does not catch a
tensor that is consumed, reduced and then discarded. Gradients do: if
``d output / d input[i]`` is exactly zero for every sample, that slice provably
cannot influence the prediction.

For each edition this builds dataset *and* model exactly the way ``facet-train``
builds them (same factories, same YAML kwargs, same injected kwargs), takes one
real example, and reports the share of absolute input gradient on each slice of
every non-time axis.

Usage::

    .venv/bin/python tools/audit/probe_context_usage.py
    .venv/bin/python tools/audit/probe_context_usage.py --models demucs sepformer
    .venv/bin/python tools/audit/probe_context_usage.py --json output/audit/context_probe.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from facet.training.cli import TrainingCLIConfig, _import_object, _invoke_factory

MODELS_DIR = Path("src/facet/models")
CONFIG_NAME = "training_niazy_proof_fit.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="*", default=None, help="Edition name fragments; default: all")
    p.add_argument("--config", default=CONFIG_NAME)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument(
        "--receptive-field",
        action="store_true",
        help="Also measure the temporal receptive field of a single output sample",
    )
    return p.parse_args()


def editions(selected: list[str] | None, config_name: str) -> list[Path]:
    out = sorted(d for d in MODELS_DIR.glob("*_paper_accurate_edition") if (d / config_name).exists())
    return [d for d in out if not selected or any(s in d.name for s in selected)]


def build(edition: Path, config_name: str):
    cfg = TrainingCLIConfig.from_dict(yaml.safe_load((edition / config_name).read_text(encoding="utf-8")))
    ds = _invoke_factory(_import_object(cfg.data.dataset_factory), cfg.data.kwargs,
                         {"training_config": cfg.training, "chunk_size": cfg.training.chunk_size,
                          "target_type": cfg.training.target_type})
    noisy, target = ds[0]
    sfreq = float(getattr(ds, "sfreq", 0.0) or 0.0)
    injected = {
        "n_channels": getattr(ds, "n_channels", int(np.shape(noisy)[0])),
        "chunk_size": cfg.training.chunk_size,
        "sfreq": sfreq,
        "target_type": cfg.training.target_type,
        "training_config": cfg.training,
        "input_shape": getattr(ds, "input_shape", tuple(np.shape(noisy))),
        "target_shape": getattr(ds, "target_shape", tuple(np.shape(target))),
        "context_epochs": getattr(ds, "context_epochs", None),
        "epoch_samples": getattr(ds, "epoch_samples", None),
    }
    model = _invoke_factory(_import_object(cfg.model.factory), cfg.model.kwargs, injected)
    return ds, model, cfg, np.asarray(noisy), np.asarray(target)


def probe(model, example: np.ndarray, batch: int) -> dict:
    """Gradient share of the output over every slice of every non-time axis."""
    model.eval()
    x = torch.as_tensor(np.repeat(example[None], batch, axis=0), dtype=torch.float32).requires_grad_(True)
    out = model(x)
    if isinstance(out, dict):
        out = next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    out.abs().sum().backward()
    if x.grad is None:
        raise RuntimeError("no gradient reached the input")
    g = x.grad.abs().double()
    total = float(g.sum())
    if total <= 0:
        raise RuntimeError("input gradient is identically zero")

    axes = {}
    for axis in range(1, x.ndim - 1):          # every axis except batch and time
        other = tuple(i for i in range(x.ndim) if i != axis)
        share = (g.sum(dim=other) / total).numpy()
        axes[f"axis{axis}"] = {
            "size": int(share.size),
            "used": int((share > 1e-12).sum()),
            # Full precision on purpose. Rounding to six places turns a share of
            # 1e-8 into a printed 0.0, which reads as "this slice is unused" when
            # the honest statement is "this slice contributes a millionth".
            "share": [float(v) for v in share],
            "off_peak_share": float(share.sum() - share.max()),
        }
    return {"output_shape": list(out.shape), "axes": axes,
            "n_params": sum(p.numel() for p in model.parameters())}



def receptive_field(model, example: np.ndarray, sfreq: float) -> dict:
    """How much of the input one *single* output sample actually depends on.

    Per-axis gradient shares can look healthy while the model is still nearly
    blind: ST-GNN reaches three of seven epochs, but only because its output at
    time t depends on t +- 5 samples, which happens to straddle two epoch
    boundaries. Feeding it a 3584-sample context is then almost entirely wasted.
    Probing one output sample is what makes that visible.
    """
    model.eval()
    # A real example, not zeros: a masking model multiplies its estimate by the
    # input, so on an all-zero input the output is constant and the probe reports
    # "no dependence" for a model that in fact has one.
    x = torch.as_tensor(example[None], dtype=torch.float32).requires_grad_(True)
    out = model(x)
    if isinstance(out, dict):
        out = next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    flat = out.reshape(out.shape[0], -1, out.shape[-1])
    flat[0, 0, out.shape[-1] // 2].backward()
    if x.grad is None:
        return {"error": "no gradient reached the input"}

    grad = x.grad.abs()[0]
    time_axis = grad.reshape(-1, grad.shape[-1]) if grad.ndim == 2 else None
    if grad.ndim == 3:                                   # (epochs, channels, T) -> time
        epochs, channels, samples = grad.shape
        time_axis = grad.permute(1, 0, 2).reshape(channels, epochs * samples)
    elif grad.ndim == 2:
        time_axis = grad
    nz = np.flatnonzero(time_axis.sum(0).numpy() > 0)
    if nz.size == 0:
        return {"error": "output does not depend on the input"}
    span = int(nz.max() - nz.min() + 1)
    total = int(time_axis.shape[-1])
    return {
        "rf_samples": span,
        "rf_total_samples": total,
        "rf_fraction": round(span / total, 6),
        "rf_ms": round(span / sfreq * 1000, 3) if sfreq > 0 else None,
        "rf_channels_reached": int((time_axis.sum(1) > 0).sum()),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for edition in editions(args.models, args.config):
        name = edition.name.replace("_paper_accurate_edition", "")
        row: dict = {"model": name}
        try:
            ds, model, cfg, noisy, target = build(edition, args.config)
            row |= {
                "dataset_class": type(ds).__name__,
                "n_examples": len(ds),
                "example_input_shape": list(noisy.shape),
                "example_target_shape": list(target.shape),
                "dataset_context_epochs": getattr(ds, "context_epochs", None),
                "dataset_n_channels": getattr(ds, "n_channels", None),
                "target_type": cfg.training.target_type,
            }
            row |= probe(model, noisy, args.batch)
            if args.receptive_field:
                row |= receptive_field(model, noisy, float(getattr(ds, "sfreq", 0.0) or 0.0))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=5)
        rows.append(row)
        note = row.get("error", f"in {row.get('example_input_shape')} -> out {row.get('output_shape')}")
        print(f"{name:20s} {note}", flush=True)

    print()
    hdr = f"{'model':16s} {'dataset':38s} {'in':>16s} {'out':>14s} {'used per axis':>16s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['model']:16s} {'ERROR: ' + r['error'][:70]}")
            continue
        used = " ".join(f"{a['used']}/{a['size']}" for a in r["axes"].values()) or "-"
        line = (f"{r['model']:16s} {r['dataset_class']:38s} {str(r['example_input_shape']):>16s} "
                f"{str(r['output_shape'][1:]):>14s} {used:>16s}")
        if "rf_samples" in r:
            line += f"  rf {r['rf_samples']}/{r['rf_total_samples']} samples"
            if r.get("rf_ms") is not None:
                line += f" ({r['rf_ms']:.2f} ms)"
        print(line)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
