"""Two pipelines on one recording, plotted over a chosen time window.

Arm A replaces the artifact with FARM's template average; arm B replaces FARM
itself with a direct deep-learning model. Both carry the same pre- and
post-processing, so the plot compares the corrector and nothing else:

    Loader -> DropChannels -> HighPass 1 Hz -> TriggerDetector -> UpSample x10
           -> SliceAligner -> [FARM | model] -> DownSample x10 -> LowPass 70 Hz

The uncorrected signal runs the identical chain minus the corrector and is drawn
as the reference, because "how much was removed" is only readable against it.

Usage::

    .venv/bin/python tools/pipeline_demo/plot_farm_vs_model.py \\
        --checkpoint training_output/demucsmc_.../epoch0039_val_loss0.0000.pt \\
        --start 25 --stop 35 --out output/pipeline_demo/farm_vs_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facet.correction import DeepLearningCorrection                   # noqa: E402
from pipeline_demo import reference_chain                             # noqa: E402
from pipeline_demo.cascade_adapter import CascadeDemucsAdapter        # noqa: E402
from pipeline_demo.direct_adapter import DirectDemucsAdapter          # noqa: E402

ARMS = ("uncorrected", "farm", "direct", "cascade")
LABELS = {
    "uncorrected": "unkorrigiert",
    "farm": "FARM",
    "direct": "Modell direkt (statt FARM)",
    "cascade": "FARM+PCA4 + Kaskade (stärkstes Verfahren)",
}
COLOURS = {"uncorrected": "#444444", "farm": "#D55E00",
           "direct": "#0072B2", "cascade": "#009E73"}


def build(arm: str, args: argparse.Namespace):
    """One arm of the reference chain, differing only in the corrector.

    The cleanup PCA runs after the learned stage in every arm. The **cascade**
    arm additionally carries the training bundle's primary correction
    (``FARM(cc=0.9) + PCA/OBS(4, 300 Hz)``) in front of the model, because that
    is the template it was trained to improve on — see reference_chain. Running
    plain FARM there, as an earlier version did, handed the model an input with a
    stage missing.
    """
    if arm == "uncorrected":
        correctors = []
    elif arm == "farm":
        correctors = reference_chain.farm()
    elif arm == "direct":
        # FARM replaced: the model predicts the whole artifact from the raw signal.
        correctors = [DeepLearningCorrection(
            model=DirectDemucsAdapter(args.direct_checkpoint, device=args.device,
                                      batch_size=args.batch_size))]
    elif arm == "cascade":
        # The deployed configuration: the bundle's primary correction removes the
        # repeatable part, the cascade predicts what it leaves.
        correctors = reference_chain.cascade_template_stage() + [DeepLearningCorrection(
            model=CascadeDemucsAdapter(args.cascade_checkpoint, device=args.device,
                                       batch_size=args.batch_size))]
    else:
        raise ValueError(arm)
    return reference_chain.build(args.input, correctors=correctors,
                                 include_pca=not args.no_pca, name=arm,
                                 trigger_regex=args.trigger_regex)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("examples/datasets/NiazyFMRI.edf"))
    p.add_argument("--direct-checkpoint", type=Path, required=True,
                   help="Model trained on the raw signal — takes FARM's place.")
    p.add_argument("--cascade-checkpoint", type=Path, required=True,
                   help="FARM-residual cascade — runs after FARM.")
    p.add_argument("--out", type=Path, default=Path("output/pipeline_demo/farm_vs_model"))
    p.add_argument("--start", type=float, default=25.0)
    p.add_argument("--stop", type=float, default=35.0)
    p.add_argument("--channels", nargs="*", default=None,
                   help="EEG channel names to plot (default: first four)")
    p.add_argument("--drop", nargs="*", default=["EMG", "ECG"])
    p.add_argument("--trigger-regex", default=r"\b1\b")
    p.add_argument("--upsample", type=int, default=10)
    p.add_argument("--highpass", type=float, default=1.0)
    p.add_argument("--lowpass", type=float, default=70.0)
    p.add_argument("--steady-from", type=float, default=30.0,
                   help="Set the y-scale from this second onwards, so the scan-onset "
                        "transient does not flatten the steady state.")
    p.add_argument("--zoom-from", type=float, default=31.0)
    p.add_argument("--zoom-seconds", type=float, default=2.0)
    p.add_argument("--no-pca", action="store_true",
                   help="Ablation: leave PCACorrection out of the chain entirely.")
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import mne
    raws, timing = {}, {}
    for arm in ARMS:
        import time
        t0 = time.perf_counter()
        res = build(arm, args).run()
        if not res.success:
            raise SystemExit(f"{arm} pipeline failed: {res.error}")
        raws[arm] = res.context.get_raw()
        timing[arm] = round(time.perf_counter() - t0, 2)
        print(f"{arm:12s} {timing[arm]:>7.2f} s", flush=True)

    ref = raws["uncorrected"]
    picks = mne.pick_types(ref.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
    names = [ref.ch_names[i] for i in picks]
    chosen = [c for c in (args.channels or names[:4]) if c in names]
    sf = ref.info["sfreq"]
    i0, i1 = int(args.start * sf), int(args.stop * sf)
    t = np.arange(i0, i1) / sf

    colours, labels = COLOURS, LABELS

    def seg(arm: str, k: int, a: int, b: int) -> np.ndarray:
        return raws[arm].get_data(picks=[picks[k]])[0, a:b] * 1e6

    # Scan onset: the first second of gradient artifact is an edge transient for
    # every corrector, because the context epochs around it are incomplete. It is
    # a real effect and stays in the plot, but it must not set the y-scale — with
    # it, the steady state collapses into a flat line and the figure shows nothing.
    steady0 = int(max(args.start, args.steady_from) * sf)

    def limit(arms, k: int) -> float:
        vals = [np.percentile(np.abs(seg(a, k, steady0, i1)), 99.5) for a in arms]
        return 1.2 * float(max(vals))

    # 1) Artifact scale — how much is removed at all.
    fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 2.1 * len(chosen)),
                             sharex=True, squeeze=False)
    for ax, ch in zip(axes[:, 0], chosen):
        k = names.index(ch)
        for arm in ARMS:
            ax.plot(t, seg(arm, k, i0, i1), lw=0.5, color=colours[arm],
                    label=labels[arm] if ch == chosen[0] else None)
        ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[-1, 0].set_xlabel("Zeit (s)", fontsize=9)
    fig.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{args.start:.0f}–{args.stop:.0f} s · Artefaktskala · "
                 f"gleiche Kette, nur der Korrektor getauscht", fontsize=11, y=1.03)
    out = args.out / f"eeg_{args.start:.0f}_{args.stop:.0f}s_artefaktskala.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # 2) EEG scale — the corrected arms only, one row per arm per channel so the
    #    traces do not hide each other.
    corr = ("farm", "direct", "cascade")
    fig, axes = plt.subplots(len(chosen) * len(corr), 1,
                             figsize=(13, 1.15 * len(chosen) * len(corr)),
                             sharex=True, squeeze=False)
    row = 0
    for ch in chosen:
        k = names.index(ch)
        lim = limit(corr, k)
        for arm in corr:
            ax = axes[row, 0]
            ax.plot(t, seg(arm, k, i0, i1), lw=0.5, color=colours[arm])
            ax.set_ylim(-lim, lim)
            ax.set_ylabel(f"{ch}\n{labels[arm].split(' (')[0]}", fontsize=7)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title("y-Skala je Kanal gemeinsam, aus dem eingeschwungenen Bereich "
                             f"ab {args.steady_from:.0f} s", fontsize=8)
            row += 1
    axes[-1, 0].set_xlabel("Zeit (s)", fontsize=9)
    fig.suptitle(f"{args.start:.0f}–{args.stop:.0f} s · EEG-Skala · "
                 f"Korrektoren getrennt übereinander", fontsize=11, y=1.005)
    out = args.out / f"eeg_{args.start:.0f}_{args.stop:.0f}s_eegskala.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # 3) Steady-state zoom — where the correctors can actually be compared.
    z0, z1 = int(args.zoom_from * sf), int((args.zoom_from + args.zoom_seconds) * sf)
    tz = np.arange(z0, z1) / sf
    fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 2.1 * len(chosen)),
                             sharex=True, squeeze=False)
    for ax, ch in zip(axes[:, 0], chosen):
        k = names.index(ch)
        for arm in corr:
            ax.plot(tz, seg(arm, k, z0, z1), lw=0.9, color=colours[arm],
                    label=labels[arm] if ch == chosen[0] else None)
        ax.set_ylim(-limit(corr, k), limit(corr, k))
        ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[-1, 0].set_xlabel("Zeit (s)", fontsize=9)
    fig.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{args.zoom_from:.1f}–{args.zoom_from + args.zoom_seconds:.1f} s · "
                 f"eingeschwungener Bereich, überlagert", fontsize=11, y=1.03)
    out = args.out / f"eeg_zoom_{args.zoom_from:.0f}s.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    stats = {}
    for arm in ARMS:
        d = raws[arm].get_data(picks=picks)[:, i0:i1] * 1e6
        ds = raws[arm].get_data(picks=picks)[:, steady0:i1] * 1e6
        stats[arm] = {"rms_uv": float(np.sqrt(np.mean(d ** 2))),
                      "rms_steady_uv": float(np.sqrt(np.mean(ds ** 2))),
                      "peak_to_peak_uv": float(np.ptp(d)),
                      "peak_to_peak_steady_uv": float(np.ptp(ds)),
                      "elapsed_seconds": timing[arm]}
    (args.out / "window_stats.json").write_text(json.dumps({
        "input": str(args.input), "window_s": [args.start, args.stop],
        "channels_plotted": chosen, "n_eeg_channels": len(picks),
        "chain": "Referenzkette aus examples/complete_pipeline_example.py: Loader"
                 f"(offset {reference_chain.ARTIFACT_TO_TRIGGER_OFFSET}) -> DropChannels -> "
                 "Crop -> HighPass -> TriggerDetector -> UpSample -> TriggerAligner -> "
                 "SubsampleAligner -> [FARM | Modell] -> PCA -> DownSample -> LowPass",
        "pca_included": not args.no_pca,
        "pca_position": "Aufräum-PCA nach der gelernten Stufe in jedem Arm; der "
                        "Kaskadenarm trägt zusätzlich die Primärkorrektur des "
                        "Trainings-Bundles (FARM cc=0.9 + PCA/OBS(4, 300 Hz)) davor, "
                        "weil genau die sein Template bildet",
        "direct_checkpoint": str(args.direct_checkpoint),
        "cascade_checkpoint": str(args.cascade_checkpoint),
        "caveat": "Kein sauberes Referenzsignal auf einer echten Aufnahme: die Zahlen sagen, "
                  "wie viel entfernt wurde, nicht ob das Richtige entfernt wurde.",
        "stats": stats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
