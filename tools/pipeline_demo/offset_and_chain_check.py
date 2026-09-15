"""Does the demo chain match ``examples/``, and what does the trigger offset do?

Two questions, one harness:

1. **The initial excursion.** The first artifact block after the scan onset shows
   a residual the steady state does not. If that is a defect rather than a
   property of the recording, changing the artifact-to-trigger offset or the
   missing processors should remove it.
2. **Chain fidelity.** ``examples/complete_pipeline_example.py`` is the reference
   chain. The pipeline demo left out three of its steps; this measures what each
   omission costs instead of arguing about it.

The metric is deliberately simple and local: the peak amplitude inside the onset
window against the RMS of the steady state. A corrector that handles the onset
like any other epoch has an onset peak comparable to the rest of the recording.

``examples/`` is read, never modified.

Usage::

    .venv/bin/python tools/pipeline_demo/offset_and_chain_check.py \\
        --out output/pipeline_demo/offset_check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facet.core import Pipeline                                        # noqa: E402
from facet.correction import FARMCorrection, PCACorrection             # noqa: E402
from facet.io import Loader                                            # noqa: E402
from facet.preprocessing import (                                      # noqa: E402
    Crop, DownSample, DropChannels, HighPassFilter, LowPassFilter, SliceAligner,
    TriggerAligner, TriggerDetector, UpSample,
)
from facet.preprocessing.alignment import SubsampleAligner             # noqa: E402

#: Constants copied from examples/complete_pipeline_example.py (read-only).
EX_NON_EEG = ["EKG", "EMG", "EOG", "ECG"]
EX_CROP = (0, 162)
EX_UPSAMPLE = 10
EX_FARM = dict(window_size=30, correlation_threshold=0.975, realign_after_averaging=True)
EX_PCA = dict(n_components=0.95, hp_freq=70.0)


def chain(variant: str, offset: float, args: argparse.Namespace) -> Pipeline:
    """Build one variant. ``demo`` is what the pipeline demo ran; ``example`` is
    the reference chain from ``examples/`` with its own processors and settings."""
    steps: list = [Loader(path=str(args.input), preload=True,
                          artifact_to_trigger_offset=offset)]
    if variant.startswith("example"):
        steps += [DropChannels(channels=EX_NON_EEG, on_missing="ignore"),
                  Crop(tmin=EX_CROP[0], tmax=EX_CROP[1])]
    else:
        steps.append(DropChannels(channels=["EMG", "ECG"], on_missing="ignore"))
    steps += [HighPassFilter(freq=1.0),
              TriggerDetector(regex=args.trigger_regex),
              UpSample(factor=EX_UPSAMPLE)]
    if variant.startswith("example"):
        steps += [TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),
                  SubsampleAligner()]
    else:
        steps.append(SliceAligner())
    steps.append(FARMCorrection(**(EX_FARM if variant.startswith("example") else {})))
    if variant == "example":
        steps.append(PCACorrection(**EX_PCA))
    steps += [DownSample(factor=EX_UPSAMPLE), LowPassFilter(freq=70.0)]
    return Pipeline(steps, name=f"{variant}_off{offset}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=Path("examples/datasets/NiazyFMRI.edf"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trigger-regex", default=r"\b1\b")
    p.add_argument("--onset", type=float, default=28.7, help="Approximate scan onset (s)")
    p.add_argument("--onset-window", type=float, default=1.0)
    p.add_argument("--steady", nargs=2, type=float, default=[30.0, 35.0])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import mne
    variants = [
        ("demo", 0.0, "Pipeline-Demo, wie bisher gelaufen"),
        ("demo", -0.005, "Pipeline-Demo + Offset aus examples/"),
        ("demo", -0.05, "Pipeline-Demo + zehnfacher Offset (Gegenprobe)"),
        ("example_nopca", -0.005, "examples/-Kette ohne PCA"),
        ("example", -0.005, "examples/-Kette vollständig (mit PCA)"),
    ]
    rows = []
    for variant, offset, label in variants:
        res = chain(variant, offset, args).run()
        if not res.success:
            rows.append({"variant": variant, "offset_s": offset, "label": label,
                         "status": f"FEHLER: {res.error}"})
            print(f"[FEHLER] {label}: {res.error}", flush=True)
            continue
        raw = res.context.get_raw()
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
        sf = raw.info["sfreq"]
        d = raw.get_data(picks=picks) * 1e6
        o0 = int((args.onset - args.onset_window / 2) * sf)
        o1 = int((args.onset + args.onset_window / 2) * sf)
        s0, s1 = int(args.steady[0] * sf), int(args.steady[1] * sf)
        onset_peak = float(np.abs(d[:, o0:o1]).max())
        steady_rms = float(np.sqrt(np.mean(d[:, s0:s1] ** 2)))
        steady_peak = float(np.abs(d[:, s0:s1]).max())
        rows.append({
            "variant": variant, "offset_s": offset, "label": label, "status": "ok",
            "n_eeg_channels": len(picks), "sfreq_hz": sf,
            "onset_peak_uv": round(onset_peak, 1),
            "steady_rms_uv": round(steady_rms, 2),
            "steady_peak_uv": round(steady_peak, 1),
            "onset_peak_over_steady_peak": round(onset_peak / max(steady_peak, 1e-9), 2),
            "onset_peak_over_steady_rms": round(onset_peak / max(steady_rms, 1e-9), 1),
        })
        print(f"{label:44s} Onset-Peak {onset_peak:>8.1f} µV   "
              f"eingeschwungen RMS {steady_rms:>6.2f} / Peak {steady_peak:>7.1f} µV   "
              f"Verhältnis {rows[-1]['onset_peak_over_steady_peak']:>6.2f}", flush=True)

    (args.out / "offset_and_chain_check.json").write_text(json.dumps({
        "input": str(args.input),
        "onset_window_s": [args.onset - args.onset_window / 2, args.onset + args.onset_window / 2],
        "steady_window_s": args.steady,
        "reference_chain": "examples/complete_pipeline_example.py (nur gelesen)",
        "substitution": "TriggerExplorer/TriggerEditor sind interaktiv; für den Skriptlauf "
                        "steht im Beispiel selbst, dass auto_select/Regex zu verwenden ist, "
                        "daher TriggerDetector.",
        "metric": "Spitzenamplitude im Onset-Fenster gegen RMS und Spitze im eingeschwungenen "
                  "Bereich. Ein Korrektor, der den Scanbeginn wie jede andere Epoche behandelt, "
                  "hat dort kein erhöhtes Verhältnis.",
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out / 'offset_and_chain_check.json'}")


if __name__ == "__main__":
    main()
