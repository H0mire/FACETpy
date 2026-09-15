"""Side-by-side plot of the reference chain, FACETpy 0.1.0 and the new editions.

Written to answer one question directly: is anything we trained better than the
cascaded denoising autoencoder that shipped with 0.1.0, on the same recording,
through the same pipeline, in the same window?

Two things make this comparable where a screenshot of each in isolation would
not be:

* **A shared y-scale.** Amplitudes that differ by a factor of two look identical
  when each panel is autoscaled, and that is exactly the failure mode -- an arm
  that deletes the signal draws a beautiful flat line on its own axis.
* **Residual gradient artifact in µV, over the full steady-state window, not the
  six seconds on screen.** The plot shows what the correction looks like; the
  number says how much artifact is left. RMS alone cannot: a corrector that
  removes the EEG along with the artifact scores best on it.

Usage::

    python tools/pipeline_demo/compare_against_legacy.py \
        --arms farm legacy_dl vit_spectrogram_deployment cascaded_dae_deployment \
        --channel Fp1 --start 4.0 --duration 6.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import mne
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from facet.core import ProcessingContext, ProcessingMetadata  # noqa: E402
from facet.evaluation import GradientArtifactResidualCalculator  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

#: Seconds into the *stored window* at which the gradient artifact begins. The
#: stored window starts at 25 s of the recording and the scan at ~28.7 s.
SCAN_ONSET_IN_WINDOW = 3.7

#: Where the steady state begins, for the residual measurement. Slightly after
#: the onset so the switch-on transient is not counted as a failure to correct.
STEADY_FROM_IN_WINDOW = 4.5

#: Human-readable names, so the panel labels say what the arm is rather than
#: what its cache file is called.
LABELS = {
    "farm": "FARM / AAS (Referenz)",
    "uncorrected": "unkorrigiert",
    "legacy_dl": "FACETpy 0.1.0 — FC-DAE-Kaskade",
}


def load(arm: str, arm_dir: Path) -> dict:
    path = arm_dir / f"{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} — erst plot_family_pipelines.py laufen lassen")
    with np.load(path, allow_pickle=True) as b:
        return {"data": b["data"], "ch_names": [str(n) for n in b["ch_names"]],
                "sfreq": float(b["sfreq"]), "triggers": b["triggers"],
                "window_s": b["window_s"]}


def residual_uv(arm: dict, tmin: float) -> float:
    """Residual gradient artifact in µV, on the epoch harmonics.

    The calculator wants volts and a context; the cache holds µV and arrays.
    """
    info = mne.create_info(arm["ch_names"], arm["sfreq"], "eeg")
    raw = mne.io.RawArray(arm["data"].astype(np.float64) * 1e-6, info, verbose=False)
    ctx = ProcessingContext(raw=raw,
                            metadata=ProcessingMetadata(triggers=list(arm["triggers"])))
    out = GradientArtifactResidualCalculator(tmin=tmin).execute(ctx)
    return float(out.metadata.custom["gradient_artifact_residual"]["comb_rms_uv"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--channel", default="Fp1")
    ap.add_argument("--start", type=float, default=4.0,
                    help="Sekunden in das gespeicherte Fenster (Scanbeginn: 3.7)")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--arm-dir", type=Path,
                    default=REPO / "output/pipeline_demo/legacy_vs_ours/arms")
    ap.add_argument("--out", type=Path,
                    default=REPO / "output/pipeline_demo/legacy_vs_ours/vergleich.png")
    args = ap.parse_args()

    if args.start + args.duration <= SCAN_ONSET_IN_WINDOW:
        raise SystemExit(f"Das Fenster liegt ganz vor dem Scanbeginn bei "
                         f"{SCAN_ONSET_IN_WINDOW} s — jeder Arm sähe perfekt aus.")

    arms = {a: load(a, args.arm_dir) for a in args.arms}
    sfreq = next(iter(arms.values()))["sfreq"]
    n0, n1 = int(args.start * sfreq), int((args.start + args.duration) * sfreq)
    t = np.arange(n1 - n0) / sfreq + args.start + 25.0

    segments, stats = {}, {}
    for name, arm in arms.items():
        seg = arm["data"][arm["ch_names"].index(args.channel)][n0:n1]
        segments[name] = seg
        stats[name] = (float(np.sqrt(np.mean(seg ** 2))),
                       residual_uv(arm, STEADY_FROM_IN_WINDOW))

    # One scale for every panel. Autoscaling each one hides the difference that
    # matters; see the module docstring.
    span = max(float(np.abs(s).max()) for s in segments.values()) * 1.05

    fig, axes = plt.subplots(len(arms), 1, figsize=(15, 2.0 * len(arms)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, name in zip(axes[:, 0], args.arms, strict=True):
        rms, comb = stats[name]
        ax.plot(t[:segments[name].size], segments[name], linewidth=0.6,
                color="#3b0764" if name == "farm" else "#c2410c")
        ax.set_ylabel(LABELS.get(name, name).replace(" — ", "\n"), fontsize=8)
        ax.set_ylim(-span, span)
        ax.grid(alpha=0.2)
        ax.text(0.995, 0.94, f"RMS {rms:.1f} µV     Gradientenartefakt-Rest {comb:.2f} µV",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    axes[-1, 0].set_xlabel("Zeit (s)")
    fig.suptitle(f"Kanal {args.channel} · {t[0]:.0f}–{t[-1]:.0f} s · gemeinsame y-Skala\n"
                 f"Rest gemessen über {25 + STEADY_FROM_IN_WINDOW:.1f}–160 s, "
                 f"nicht über den gezeigten Ausschnitt", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    plt.close(fig)

    print(f"{'Arm':34s} {'RMS µV':>9s} {'Rest µV':>10s} {'x FARM':>9s}")
    farm_comb = stats.get("farm", (0.0, float("nan")))[1]
    for name in args.arms:
        rms, comb = stats[name]
        print(f"{LABELS.get(name, name)[:34]:34s} {rms:9.2f} {comb:10.2f} "
              f"{comb / farm_comb:9.2f}")
    print(f"\ngeschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
