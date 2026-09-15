"""Messe die gespeicherten Pipeline-Arme und schreibe die Tabelle, die zitiert wird.

Erzeugt ``docs/research/run_7_pipeline_results.csv``. Die Arm-Archive selbst
liegen unter ``output/`` und sind gitignoriert; diese Tabelle ist die Ableitung,
die mit ins Repository wandert, und dieses Skript ist der Weg von der einen zur
anderen. Ohne es wäre die CSV eine Zahlenreihe ohne nachvollziehbare Herkunft.

**Zwei Kennzahlen, nicht eine.** Der Kammwert misst, wie viel epochenperiodischer
Artefakt übrig ist, und übersieht dabei zwei Fehlerarten, die in diesem Lauf
beide vorkamen: ein Arm, der eine Treppe ausgibt (``st_gnn`` -- wenig Leistung
auf den Harmonischen, weil die Amplitude klein ist), und einer, der einen
breitbandigen Zackenzug aufprägt (``ic_unet`` -- liegt nicht auf den
Harmonischen). Beide fängt der Nahtsprung. Deshalb stehen sie nebeneinander,
und deshalb steht in der Ausgabe eine Warnspalte.

Nutzung::

    uv run python tools/pipeline_demo/measure_arms.py
    uv run python tools/pipeline_demo/measure_arms.py \\
        --arm-dir output/pipeline_demo/legacy_vs_ours/arms --out /tmp/legacy.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import mne
import numpy as np

from facet.core import ProcessingContext, ProcessingMetadata
from facet.evaluation import (
    EpochSeamStepCalculator,
    GradientArtifactResidualCalculator,
)

REPO = Path(__file__).resolve().parents[2]

#: Analysefenster in Koordinaten des *gespeicherten* Fensters. Die Arme beginnen
#: bei 25 s der Aufnahme, der Scan bei ~28,7 s; 4,5 s hält den Einschaltvorgang
#: draußen. Der Kammwert hängt von der Fensterlänge ab -- dieselbe Aufnahme über
#: 5,5 s gemessen ergibt rund das Siebenfache -- deshalb gehört das Fenster in
#: die Metadatendatei und in jede zitierte Zahl.
T0_IN_WINDOW, T1_IN_WINDOW = 4.5, 135.0
WINDOW_OFFSET_S = 25.0

#: Ab diesem Nahtsprung ist ein Arm auffällig. FARM liegt bei 1,11, das
#: unkorrigierte Signal bei 5,69.
SEAM_SUSPECT = 1.8
SEAM_BROKEN = 2.5


def measure(path: Path, t0: float, t1: float, step_channel: str) -> dict:
    a = np.load(path, allow_pickle=True)
    names = [str(x) for x in a["ch_names"]]
    sfreq = float(a["sfreq"])
    raw = mne.io.RawArray(a["data"].astype(np.float64) * 1e-6,
                          mne.create_info(names, sfreq, "eeg"), verbose=False)
    ctx = ProcessingContext(
        raw=raw, metadata=ProcessingMetadata(triggers=list(map(int, a["triggers"]))))

    comb = GradientArtifactResidualCalculator(tmin=t0, tmax=t1).execute(ctx)\
        .metadata.custom["gradient_artifact_residual"]
    seam = EpochSeamStepCalculator(tmin=t0, tmax=t1).execute(ctx)\
        .metadata.custom["epoch_seam_step"]

    # Der mediane Abtastschritt trennt die Treppe vom Zackenzug: beide haben
    # einen hohen Nahtsprung, aber die Treppe steht dazwischen still.
    ch = step_channel if step_channel in names else names[0]
    diffs = np.abs(np.diff(a["data"][names.index(ch)][int(t0 * sfreq):int(t1 * sfreq)]
                           .astype(np.float64)))
    return {"arm": path.stem, "rms_uv": round(comb["rms_uv"], 3),
            "ga_rest_uv": round(comb["comb_rms_uv"], 3),
            "naht_ratio": round(seam["ratio"], 3),
            "median_sample_step_uv": round(float(np.median(diffs)), 4),
            "step_channel": ch, "n_harmonics": comb["n_harmonics"],
            "epoch_rate_hz": round(comb["epoch_rate_hz"], 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-dir", type=Path,
                    default=REPO / "output/pipeline_demo/deployment_first/arms")
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs/research/run_7_pipeline_results.csv")
    ap.add_argument("--reference", default="farm",
                    help="Arm, gegen den die ×-Spalte rechnet")
    ap.add_argument("--channel", default="Fp1", help="Kanal für den Abtastschritt")
    ap.add_argument("--start", type=float, default=T0_IN_WINDOW)
    ap.add_argument("--stop", type=float, default=T1_IN_WINDOW)
    args = ap.parse_args()

    paths = sorted(Path(p) for p in glob.glob(str(args.arm_dir / "*.npz")))
    if not paths:
        raise SystemExit(f"keine Arme unter {args.arm_dir} — erst "
                         f"plot_family_pipelines.py laufen lassen")

    rows = [measure(p, args.start, args.stop, args.channel) for p in paths]
    ref = next((r for r in rows if r["arm"] == args.reference), None)
    if ref is None:
        raise SystemExit(f"Referenzarm {args.reference!r} fehlt; vorhanden: "
                         f"{[r['arm'] for r in rows]}")
    for r in rows:
        r["x_reference"] = round(r["ga_rest_uv"] / ref["ga_rest_uv"], 2)
        # Nicht „bestanden/durchgefallen" -- das entscheidet der Augenschein.
        # Nur: hier lohnt das Hinsehen.
        r["hinweis"] = ("Naht gebrochen, ansehen" if r["naht_ratio"] >= SEAM_BROKEN
                        else "Naht auffällig" if r["naht_ratio"] >= SEAM_SUSPECT else "")
    rows.sort(key=lambda r: r["ga_rest_uv"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "arm_dir": os.path.relpath(args.arm_dir, REPO),
        "window_s": [WINDOW_OFFSET_S + args.start, WINDOW_OFFSET_S + args.stop],
        "reference_arm": args.reference, "step_channel": args.channel,
        "metrics": {
            "ga_rest_uv": "GradientArtifactResidualCalculator.comb_rms_uv, fmax 70 Hz",
            "naht_ratio": "EpochSeamStepCalculator.ratio; FARM = 1.11, unkorrigiert = 5.69",
            "median_sample_step_uv": "Median |diff| auf step_channel; trennt Treppe vom Zackenzug"},
        "caveat": "Der Kammwert ist fensterabhängig — dasselbe Signal über 5,5 s "
                  "gemessen ergibt rund das Siebenfache. Er sieht ausserdem nur "
                  "epochenperiodische Reste; breitbandig Aufgeprägtes findet erst "
                  "der Nahtsprung. Siehe run_7_stand_fuer_kapitel_5.md §2.2.",
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False))

    print(f"{'Arm':32s}{'RMS':>8s}{'GA-Rest':>9s}{'×ref':>7s}{'Naht':>7s}  Hinweis")
    for r in rows:
        print(f"{r['arm']:32s}{r['rms_uv']:8.2f}{r['ga_rest_uv']:9.2f}"
              f"{r['x_reference']:7.1f}{r['naht_ratio']:7.2f}  {r['hinweis']}")
    print(f"\ngeschrieben: {args.out}\n             {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
