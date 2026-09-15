"""End-to-end FACETpy pipeline: FARM alone against FARM + Demucs cascade.

This is the deployment view of the run_6/run_7 results. Everything above them is
measured on prepared NPZ tensors; this runs the real thing — an EDF goes in, a
corrected EDF comes out, and the two corrections are compared on the same
recording.

The chain comes from :mod:`tools.pipeline_demo.reference_chain`, which reads it
from ``examples/complete_pipeline_example.py``. It is not rebuilt here, because
rebuilding it is how the two defects below got in.

1. **UpSample** — gives the trigger alignment sub-sample resolution.
2. **Trigger alignment** — artifact templates are position-sensitive. A template
   one sample off raises the reconstruction error by a factor of ~17
   (run_7 §5.8.2), so alignment is a precondition, not a refinement.
3. **FARM** — removes the epoch-repeatable component.
4. **Cascade (optional)** — predicts what FARM leaves.
5. **DownSample** — back to the native rate.
6. **Lowpass 70 Hz** — removes the residual artifact above the EEG band that
   subtraction leaves behind.

Two corrections against the first version
-----------------------------------------
**The trigger offset was missing.** The chain was assembled by hand and took the
``Loader`` default ``artifact_to_trigger_offset = 0.0``, but the gradient artifact
starts 5 ms *before* the trigger. Part of it therefore fell outside every epoch
window, and FARM removed 58 % of the power instead of ~99.7 %. Every number the
first version produced is void.

**The cascade's input arm was wrong.** ``farm_cascade`` ran after plain
``FARMCorrection``, but the template the cascade is trained against is
``FARM(cc=0.9) + PCA/OBS(4, 300 Hz)`` — the training bundle's primary correction.
So a fourth arm, ``farm_pca4``, now runs that stage *without* the model, and the
cascade's contribution is measured against it. Comparing the cascade to the
reference-chain FARM arm would have charged the extra PCA stage to the model.

Usage::

    .venv/bin/python tools/pipeline_demo/run_cascade_pipeline.py \\
        --input examples/datasets/NiazyFMRI.edf \\
        --checkpoint output/run6_grid_cascade/.../epoch0047_val_loss-1.7230.pt \\
        --out-dir output/pipeline_demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facet.core import Pipeline                                        # noqa: E402
from facet.correction import DeepLearningCorrection                     # noqa: E402
from pipeline_demo import reference_chain                               # noqa: E402
from pipeline_demo.cascade_adapter import CascadeDemucsAdapter          # noqa: E402

#: uncorrected → reference FARM → the cascade's own input → the cascade.
ARMS = ("uncorrected", "farm", "farm_pca4", "farm_cascade")
ARM_LABELS = {
    "uncorrected": "ohne jede Korrektur (auch ohne PCA)",
    "farm": "FARM (Referenzkette)",
    "farm_pca4": "FARM + PCA/OBS(4, 300 Hz) — Eingang der Kaskade",
    "farm_cascade": "FARM + PCA/OBS(4) + Kaskade",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("examples/datasets/NiazyFMRI.edf"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("output/pipeline_demo"))
    p.add_argument("--trigger-regex", default=r"\b1\b")
    p.add_argument("--no-pca", action="store_true",
                   help="Ablation: leave the cleanup PCACorrection out of every arm.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=None, help="Keep only the first N EEG channels.")
    p.add_argument("--replot-only", action="store_true",
                   help="Redraw the figures from a previous run's saved signals. Inference over the "
                        "whole recording costs minutes; a figure change should not.")
    p.add_argument("--no-align", action="store_true",
                   help="Skip slice alignment. Only for demonstrating what misalignment costs — "
                        "artifact templates are position-sensitive (run_7 §5.8.2).")
    return p.parse_args()


def build_pipeline(args: argparse.Namespace, arm: str) -> Pipeline:
    """One arm of the reference chain.

    ``--no-align`` drops both aligners, which is the demonstration of what
    misalignment costs, not a supported configuration.
    """
    steps = reference_chain.preprocessing(args.input, trigger_regex=args.trigger_regex)
    if args.no_align:
        steps = [st for st in steps
                 if type(st).__name__ not in ("TriggerAligner", "SubsampleAligner")]
    if arm == "farm":
        steps += reference_chain.farm()
    elif arm in ("farm_pca4", "farm_cascade"):
        steps += reference_chain.cascade_template_stage()
    if arm == "farm_cascade":
        steps.append(DeepLearningCorrection(
            model=CascadeDemucsAdapter(
                args.checkpoint, device=args.device, batch_size=args.batch_size,
            ),
        ))
    # The reference arm gets no corrector at all, and PCACorrection is one: on the
    # raw signal its OBS stage removes 16.7 % of the power by itself, so leaving it
    # in the reference would measure every other arm against an already-corrected
    # baseline.
    steps += reference_chain.postprocessing(
        include_pca=(not args.no_pca) and arm != "uncorrected")
    return Pipeline(steps, name=ARM_LABELS[arm])


def metrics(corrected: np.ndarray, raw: np.ndarray, sfreq: float) -> dict[str, float]:
    """Descriptive residual metrics. No ground truth exists on a real recording.

    On a real EDF there is no clean reference, so nothing here is an accuracy
    measure. These quantify how much of the original signal power is gone and
    where the remaining power sits — the two things a reader can check against
    the plots.
    """
    residual = raw - corrected
    band = np.fft.rfftfreq(corrected.shape[-1], d=1.0 / sfreq)
    spec = np.abs(np.fft.rfft(corrected, axis=-1)) ** 2
    eeg = (band >= 1.0) & (band <= 40.0)
    high = band > 70.0
    return {
        "rms_raw_uv": float(np.sqrt(np.mean(raw ** 2))) * 1e6,
        "rms_corrected_uv": float(np.sqrt(np.mean(corrected ** 2))) * 1e6,
        "rms_removed_uv": float(np.sqrt(np.mean(residual ** 2))) * 1e6,
        "power_removed_pct": float(100.0 * (1.0 - np.mean(corrected ** 2) / max(np.mean(raw ** 2), 1e-30))),
        "eeg_band_power_share_pct": float(100.0 * spec[..., eeg].sum() / max(spec.sum(), 1e-30)),
        "above_70hz_power_share_pct": float(100.0 * spec[..., high].sum() / max(spec.sum(), 1e-30)),
    }


SIGNALS_NPZ = "pipeline_demo_signals.npz"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    signals: dict[str, np.ndarray] = {}

    if args.replot_only:
        cached = args.out_dir / SIGNALS_NPZ
        if not cached.exists():
            raise SystemExit(f"--replot-only needs {cached}; run the pipeline once first")
        with np.load(cached) as b:
            signals = {k: b[k] for k in b.files if k != "sfreq_hz"}
            sfreq = float(b["sfreq_hz"])
        plot(args.out_dir, signals, sfreq, reference_chain.LOWPASS_HZ)
        print(f"redrew figures in {args.out_dir} from {cached.name}")
        return

    for label in ARMS:
        pipeline = build_pipeline(args, label)
        started = time.perf_counter()
        result = pipeline.run()
        elapsed = time.perf_counter() - started
        if not result.success:
            raise SystemExit(f"pipeline '{label}' failed: {result.error}")
        raw = result.context.get_raw()
        # EEG picks only. The trigger channel is a stim channel: MNE's filters
        # skip stim by design, so its square wave survives the lowpass and — at
        # an RMS of ~50 mV — swamps every spectral share computed over all
        # channels. Including it made an earlier version of this script report
        # 92 % of the power above a 70 Hz lowpass.
        data = raw.get_data(picks="eeg")
        if args.channels:
            data = data[: args.channels]
        signals[label] = data
        results[label] = {
            "pipeline": pipeline.name,
            "steps": [type(p).__name__ for p in pipeline.processors],
            "elapsed_seconds": round(elapsed, 2),
            "sfreq_hz": float(raw.info["sfreq"]),
            "n_eeg_channels": int(data.shape[0]),
            "channels_excluded_from_metrics": [
                raw.ch_names[i] for i, t in enumerate(raw.get_channel_types()) if t != "eeg"
            ],
            "n_samples": int(data.shape[-1]),
        }
        print(f"{label:14s} {elapsed:7.1f} s  EEG {data.shape}")

    raw_data = signals["uncorrected"]
    sfreq = results["uncorrected"]["sfreq_hz"]
    for label in ("farm", "farm_pca4", "farm_cascade"):
        n = min(signals[label].shape[-1], raw_data.shape[-1])
        results[label]["metrics"] = metrics(signals[label][..., :n], raw_data[..., :n], sfreq)
    # What the cascade contributes on top of *its own input*. Measuring it
    # against the reference FARM arm instead would credit the model with the
    # extra PCA/OBS stage that its training template already contains.
    base = "farm_pca4"
    n = min(signals[base].shape[-1], signals["farm_cascade"].shape[-1])
    delta = signals["farm_cascade"][..., :n] - signals[base][..., :n]
    results["farm_cascade"]["metrics"]["cascade_baseline_arm"] = base
    results["farm_cascade"]["metrics"]["cascade_change_vs_farm_rms_uv"] = (
        float(np.sqrt(np.mean(delta ** 2))) * 1e6
    )
    results["farm_cascade"]["metrics"]["cascade_change_share_of_farm_pct"] = float(
        100.0 * np.sqrt(np.mean(delta ** 2))
        / max(np.sqrt(np.mean((raw_data[..., :n] - signals[base][..., :n]) ** 2)), 1e-30)
    )

    (args.out_dir / "pipeline_demo_results.json").write_text(
        json.dumps({"input": str(args.input), "checkpoint": str(args.checkpoint),
                    "dropped_channels": list(reference_chain.NON_EEG_CHANNELS),
                    "highpass_hz": reference_chain.HIGHPASS_HZ,
                    "lowpass_hz": reference_chain.LOWPASS_HZ,
                    "upsample": reference_chain.UPSAMPLE,
                    "crop_seconds": list(reference_chain.CROP),
                    "artifact_to_trigger_offset_s": reference_chain.ARTIFACT_TO_TRIGGER_OFFSET,
                    "farm_kwargs": reference_chain.FARM_KWARGS,
                    "cascade_template_stage": {"farm": reference_chain.TRAINING_FARM_KWARGS,
                                               "pca": reference_chain.TRAINING_PCA_KWARGS},
                    "cleanup_pca": None if args.no_pca else reference_chain.PCA_KWARGS,
                    "arm_labels": ARM_LABELS,
                    "slice_alignment": not args.no_align,
                    "metric_caveat": "No clean reference exists on a real recording; these are "
                                     "descriptive residual and spectral figures, not accuracy.",
                    "runs": results}, indent=2), encoding="utf-8")
    # Cache the corrected signals so figures can be redrawn without repeating
    # the inference pass, which dominates the runtime.
    np.savez_compressed(args.out_dir / SIGNALS_NPZ, sfreq_hz=np.array(sfreq), **signals)
    plot(args.out_dir, signals, sfreq, reference_chain.LOWPASS_HZ)
    print(f"\nwrote {args.out_dir}/pipeline_demo_results.json")


def plot(out_dir: Path, signals: dict[str, np.ndarray], sfreq: float, lowpass_hz: float) -> None:
    colours = {"uncorrected": "#444444", "farm": "#D55E00",
               "farm_pca4": "#CC79A7", "farm_cascade": "#0072B2"}
    n = min(v.shape[-1] for v in signals.values())
    seconds = 6.0
    span = slice(0, min(int(seconds * sfreq), n))
    t = np.arange(span.stop) / sfreq

    fig, axes = plt.subplots(len(signals), 1, figsize=(12, 2.6 * len(signals)), sharex=True)
    for ax, (label, data) in zip(axes, signals.items()):
        ax.plot(t, data[0, span] * 1e6, color=colours[label], lw=0.6)
        ax.set_ylabel(f"{label}\nµV", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
    axes[0].set_title("Zeitverlauf, erster EEG-Kanal — identische Vorverarbeitung, "
                      "nur die Korrektur unterscheidet sich", fontsize=10)
    axes[-1].set_xlabel("Sekunden", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_pipeline_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Two panels. The absolute spectra alone are a poor figure: a factor of 2.4
    # in total power is a barely visible vertical shift on six log decades, so
    # three curves that genuinely differ look identical. The attenuation panel
    # states the same information as the quantity of interest — how much of each
    # frequency the correction removed.
    n_full = min(v.shape[-1] for v in signals.values())
    freqs = np.fft.rfftfreq(n_full, d=1.0 / sfreq)
    spectra = {
        label: (np.abs(np.fft.rfft(data[:, :n_full], axis=-1)) ** 2).mean(axis=0)
        for label, data in signals.items()
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for label, spec in spectra.items():
        ax1.semilogy(freqs, spec + 1e-30, color=colours[label], lw=0.7, label=label, alpha=0.85)
    ax1.axvline(lowpass_hz, color="#999999", ls="--", lw=1.0, label=f"Lowpass {lowpass_hz:g} Hz")
    ax1.set_ylabel("Leistung (a.u., log)", fontsize=9)
    ax1.set_title("Mittleres Leistungsspektrum über alle EEG-Kanäle (ganze Aufnahme)", fontsize=10)
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=8)

    base = spectra["uncorrected"] + 1e-30
    for label in ("farm", "farm_pca4", "farm_cascade"):
        att = 10.0 * np.log10((spectra[label] + 1e-30) / base)
        ax2.plot(freqs, att, color=colours[label], lw=0.7, label=label, alpha=0.9)
    ax2.axhline(0.0, color="#444444", lw=0.8)
    ax2.axvline(lowpass_hz, color="#999999", ls="--", lw=1.0)
    ax2.set_xlim(0, min(200, freqs[-1]))
    ax2.set_ylim(-40, 10)
    ax2.set_xlabel("Hz", fontsize=9)
    ax2.set_ylabel("Dämpfung gegen unkorrigiert (dB)", fontsize=9)
    ax2.set_title("Was die Korrektur je Frequenz entfernt — negativ = entfernt", fontsize=10)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_pipeline_spectrum.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Third figure: the cascade's own contribution, which is invisible next to
    # FARM's. Plotted as the difference between the two corrected signals.
    n = min(signals["farm_pca4"].shape[-1], signals["farm_cascade"].shape[-1])
    delta = signals["farm_cascade"][:, :n] - signals["farm_pca4"][:, :n]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(np.arange(span.stop) / sfreq, delta[0, span] * 1e6, color=colours["farm_cascade"], lw=0.6)
    ax1.set_ylabel("µV", fontsize=8)
    ax1.set_xlabel("Sekunden", fontsize=8)
    ax1.set_title("Beitrag der Kaskade: (FARM+PCA4+Kaskade) − (FARM+PCA4), erster EEG-Kanal",
                  fontsize=10)
    ax1.grid(alpha=0.2)
    ax1.tick_params(labelsize=7)
    dspec = (np.abs(np.fft.rfft(delta, axis=-1)) ** 2).mean(axis=0)
    dfreq = np.fft.rfftfreq(n, d=1.0 / sfreq)
    ax2.semilogy(dfreq, dspec + 1e-30, color=colours["farm_cascade"], lw=0.7, label="Kaskadenbeitrag")
    ax2.semilogy(freqs, spectra["farm_pca4"] + 1e-30, color=colours["farm_pca4"], lw=0.7,
                 alpha=0.7, label="Eingang der Kaskade (FARM+PCA4)")
    ax2.axvline(lowpass_hz, color="#999999", ls="--", lw=1.0)
    ax2.set_xlim(0, min(200, dfreq[-1]))
    ax2.set_xlabel("Hz", fontsize=8)
    ax2.set_ylabel("Leistung (a.u., log)", fontsize=8)
    ax2.grid(alpha=0.25, which="both")
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_pipeline_cascade_contribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
