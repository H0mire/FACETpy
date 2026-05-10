"""Run FACET evaluation framework on the DL-corrected Niazy recording.

The script runs two pipelines on the Niazy EEG-fMRI recording:

* "DL only" — preprocessing + ``EpochContextDeepLearningCorrection``
* "DL + LP70" — same plus a 70 Hz lowpass filter on the corrected signal

For each pipeline it collects every metric provided by ``facet.evaluation``
(SNR, LegacySNR, RMS ratio, RMS residual, median artifact, FFT Allen, FFT
Niazy, spectral coherence, spike detection) and renders one comparison figure
plus a per-channel diagnostics figure.

Example:
    uv run python examples/evaluate_context_artifact_model_full_metrics.py \\
        --checkpoint training_output/sevenepochcontextartifactnet_20260430_134222/exports/seven_epoch_context_artifact_net.ts
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from facet.models.demo01 import EpochContextDeepLearningCorrection

from facet import (
    DownSample,
    DropChannels,
    HighPassFilter,
    LowPassFilter,
    TriggerAligner,
    TriggerDetector,
    UpSample,
    load,
)
from facet.evaluation import (
    FFTAllenCalculator,
    FFTNiazyCalculator,
    LegacySNRCalculator,
    MedianArtifactCalculator,
    MetricsReport,
    RMSCalculator,
    RMSResidualCalculator,
    SNRCalculator,
    SpectralCoherenceCalculator,
    SpikeDetectionRateCalculator,
)

DEFAULT_CHECKPOINT = Path(
    "./training_output/sevenepochcontextartifactnet_20260430_134222/"
    "exports/seven_epoch_context_artifact_net.ts"
)
DEFAULT_INPUT = Path("./examples/datasets/NiazyFMRI.edf")
DEFAULT_OUTPUT_DIR = Path("./output/context_artifact_model_full_metrics")
NON_EEG_CHANNELS = ["EMG", "ECG", "EOG", "EKG"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trigger-regex", default=r"\b1\b")
    parser.add_argument("--context-epochs", type=int, default=7)
    parser.add_argument("--epoch-samples", type=int, default=292)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--upsample-factor", type=int, default=10)
    parser.add_argument("--lowpass-hz", type=float, default=70.0)
    return parser.parse_args()


def _build_context(args: argparse.Namespace):
    ctx = load(str(args.input), preload=True, artifact_to_trigger_offset=-0.005)
    ctx = (
        ctx
        | DropChannels(channels=NON_EEG_CHANNELS)
        | TriggerDetector(regex=args.trigger_regex)
        | HighPassFilter(freq=1.0)
        | UpSample(factor=args.upsample_factor)
        | TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False)
        | DownSample(factor=args.upsample_factor)
    )
    return ctx


def _apply_dl(ctx, args: argparse.Namespace):
    return ctx | EpochContextDeepLearningCorrection(
        checkpoint_path=args.checkpoint,
        context_epochs=args.context_epochs,
        epoch_samples=args.epoch_samples,
        device=args.device,
    )


def _add_metrics(ctx, store: dict, label: str):
    return (
        ctx
        | SNRCalculator(verbose=False)
        | LegacySNRCalculator(verbose=False)
        | RMSCalculator(verbose=False)
        | RMSResidualCalculator(verbose=False)
        | MedianArtifactCalculator(verbose=False)
        | FFTAllenCalculator(verbose=False)
        | FFTNiazyCalculator(verbose=False)
        | SpectralCoherenceCalculator(verbose=False)
        | SpikeDetectionRateCalculator()
        | MetricsReport(name=label, store=store)
    )


def _flatten(metrics: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            flat[key] = float(value)
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float, np.integer, np.floating)):
                    flat[f"{key}.{sub_key}"] = float(sub_val)
                elif isinstance(sub_val, dict):
                    for inner_key, inner_val in sub_val.items():
                        if isinstance(inner_val, (int, float, np.integer, np.floating)):
                            flat[f"{key}.{sub_key}.{inner_key}"] = float(inner_val)
    return flat


def _format_value(v: float) -> str:
    if np.isnan(v) or np.isinf(v):
        return "nan"
    if abs(v) >= 1e3 or (abs(v) < 1e-2 and v != 0):
        return f"{v:.3e}"
    return f"{v:.3f}"


def _plot_full_metrics(
    flat_results: dict[str, dict[str, float]],
    save_path: Path,
    title: str,
) -> None:
    all_keys: list[str] = []
    for run_metrics in flat_results.values():
        for key in run_metrics:
            if key not in all_keys:
                all_keys.append(key)

    if not all_keys:
        return

    n_metrics = len(all_keys)
    cols = 4
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.0 * rows), squeeze=False)
    axes = axes.flatten()

    run_names = list(flat_results.keys())
    palette = plt.get_cmap("tab10").colors

    for idx, key in enumerate(all_keys):
        ax = axes[idx]
        values = [flat_results[run].get(key, np.nan) for run in run_names]
        bar_colors = [palette[i % len(palette)] for i in range(len(run_names))]
        bars = ax.bar(run_names, values, color=bar_colors)
        ax.set_title(key, fontsize=10)
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, val in zip(bars, values, strict=False):
            ax.annotate(
                _format_value(val),
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        finite = [v for v in values if np.isfinite(v)]
        if finite and max(finite) > 0 and (max(finite) / max(min(finite), 1e-12)) > 100:
            ax.set_yscale("symlog", linthresh=max(1e-6, min(abs(v) for v in finite if v != 0) * 0.1))

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_per_channel(
    raw_metrics: dict[str, dict],
    save_path: Path,
) -> None:
    keys_to_plot = [
        ("snr_per_channel", "SNR per channel"),
        ("legacy_snr_per_channel", "Legacy SNR per channel"),
        ("rms_ratio_per_channel", "RMS ratio per channel"),
        ("rms_residual_per_channel", "RMS residual per channel (target=1)"),
    ]
    available: list[tuple[str, str]] = []
    for key, title in keys_to_plot:
        if any(key in run for run in raw_metrics.values()):
            available.append((key, title))

    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.0 * n), squeeze=False)
    axes = axes.flatten()
    palette = plt.get_cmap("tab10").colors

    for ax, (key, title) in zip(axes, available, strict=False):
        for run_idx, (run_name, metrics) in enumerate(raw_metrics.items()):
            if key in metrics:
                values = np.asarray(metrics[key], dtype=float)
                ax.plot(
                    np.arange(len(values)),
                    values,
                    marker="o",
                    label=run_name,
                    color=palette[run_idx % len(palette)],
                    linewidth=1.0,
                )
        ax.set_title(title)
        ax.set_xlabel("Channel index")
        ax.grid(alpha=0.4, linestyle="--")
        ax.legend(fontsize=9)
        if "rms_residual" in key:
            ax.axhline(1.0, color="black", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_ctx = _build_context(args)
    raw_per_run: dict[str, dict] = {}
    flat_per_run: dict[str, dict[str, float]] = {}

    for label, with_lp in [("DL only", False), (f"DL + LP{int(args.lowpass_hz)}", True)]:
        store: dict[str, dict] = {}
        ctx = _apply_dl(base_ctx, args)
        if with_lp:
            ctx = ctx | LowPassFilter(freq=args.lowpass_hz)
        ctx = _add_metrics(ctx, store, label)

        run_metrics = ctx.metadata.custom.get("metrics", {})
        raw_per_run[label] = deepcopy(run_metrics)
        flat_per_run[label] = _flatten(run_metrics)

    metrics_path = args.output_dir / "full_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "input": str(args.input),
                "lowpass_hz": args.lowpass_hz,
                "metrics": {
                    run: {k: v for k, v in run_metrics.items() if not isinstance(v, list)}
                    for run, run_metrics in raw_per_run.items()
                },
                "flat_metrics": flat_per_run,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    full_plot_path = args.output_dir / "full_metrics_overview.png"
    _plot_full_metrics(
        flat_per_run,
        full_plot_path,
        title=f"FACET evaluation framework — {args.input.name} (DL vs DL+LP{int(args.lowpass_hz)})",
    )

    per_channel_path = args.output_dir / "full_metrics_per_channel.png"
    _plot_per_channel(raw_per_run, per_channel_path)

    framework_plot_path = args.output_dir / "full_metrics_framework_compare.png"
    MetricsReport.plot(
        {run: flat for run, flat in flat_per_run.items()},
        title="FACET MetricsReport.plot — all framework metrics",
        save_path=str(framework_plot_path),
        show=False,
    )

    print("Saved full FACET-framework evaluation:")
    print(f"  metrics json     : {metrics_path}")
    print(f"  overview plot    : {full_plot_path}")
    print(f"  per-channel plot : {per_channel_path}")
    print(f"  framework plot   : {framework_plot_path}")


if __name__ == "__main__":
    main()
