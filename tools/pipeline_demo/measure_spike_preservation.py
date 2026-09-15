"""Miss, wie viel von einem gesetzten Spike die Korrektur übrig lässt.

Ergänzt den Augenschein um eine Zahl. Der Kammwert dieses Projekts misst
Restartefakt und belohnt damit indirekt jedes Modell, das transiente Ereignisse
glattbügelt — ein Spike ist breitbandig und liegt nicht auf den Epochenharmonischen.
Dieses Skript misst die Gegenrichtung: was von einem bekannten, injizierten
Ereignis nach der Korrektur noch da ist.

**Warum nicht einfach die Spitzenamplitude ablesen.** Der Spike sitzt auf einem
EEG, das an dieser Stelle selbst ±40 µV führt. Die rohe Spitze misst also Spike
plus Untergrund. Gemessen wird deshalb die Differenz zwischen dem Arm mit Spike
und demselben Arm ohne Spike — beide Läufe sind bis auf die Injektion identisch,
also ist die Differenz der durchgereichte Spike und sonst nichts.

Das setzt voraus, dass beide Läufe existieren: `--with-dir` und `--without-dir`.

Nutzung::

    uv run python tools/pipeline_demo/measure_spike_preservation.py \\
        --with-dir output/spike_test/arms \\
        --without-dir output/pipeline_demo/deployment_first/arms \\
        --truth output/spike_test/NiazyFMRI_spikes.truth.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: Halbes Fenster um den Spike-Gipfel, in Millisekunden. Der scharfe Transient
#: dauert 20-70 ms (IFCN), die Nachschwankung reicht weiter; 150 ms fassen den
#: ganzen Komplex, ohne benachbarte Sekunden mitzunehmen.
HALF_WINDOW_MS = 150.0


def load(arm_dir: Path, arm: str, channel: str) -> tuple[np.ndarray, float, float]:
    path = arm_dir / f"{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as b:
        names = [str(n) for n in b["ch_names"]]
        if channel not in names:
            raise ValueError(f"{arm}: Kanal {channel!r} fehlt; vorhanden {names[:6]}…")
        return (b["data"][names.index(channel)].astype(np.float64),
                float(b["sfreq"]), float(b["window_s"][0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-dir", type=Path, default=REPO / "output/spike_test/arms")
    ap.add_argument("--without-dir", type=Path,
                    default=REPO / "output/pipeline_demo/deployment_first/arms")
    ap.add_argument("--truth", type=Path,
                    default=REPO / "output/spike_test/NiazyFMRI_spikes.truth.json")
    ap.add_argument("--channel", default="Fp1")
    ap.add_argument("--out", type=Path, default=REPO / "output/spike_test/spike_preservation.csv")
    args = ap.parse_args()

    truth = json.loads(args.truth.read_text())
    times = [s["t_s"] for s in truth["spikes"]]
    injected = float(truth["amplitude_uv"])

    arms = sorted(p.stem for p in args.with_dir.glob("*.npz"))
    rows = []
    for arm in arms:
        try:
            with_sig, sfreq, offset = load(args.with_dir, arm, args.channel)
            without_sig, _, offset2 = load(args.without_dir, arm, args.channel)
        except (FileNotFoundError, ValueError) as e:
            print(f"  {arm}: übersprungen ({e.__class__.__name__})")
            continue
        if abs(offset - offset2) > 1e-6 or with_sig.size != without_sig.size:
            print(f"  {arm}: übersprungen — die beiden Läufe decken nicht dasselbe Fenster ab")
            continue

        half = int(HALF_WINDOW_MS * sfreq / 1000)
        peaks = []
        for t in times:
            i = int(round((t - offset) * sfreq))
            lo, hi = max(0, i - half), min(with_sig.size, i + half)
            # Differenz der beiden Läufe: der durchgereichte Spike ohne Untergrund.
            peaks.append(float(np.abs(with_sig[lo:hi] - without_sig[lo:hi]).max()))
        rows.append({"arm": arm,
                     **{f"spike_{t:g}s_uv": round(p, 2) for t, p in zip(times, peaks, strict=True)},
                     "mittel_uv": round(float(np.mean(peaks)), 2),
                     "erhalt_pct": round(100.0 * float(np.mean(peaks)) / injected, 1)})

    # Der unkorrigierte Arm ist der ehrlichere Nenner als die injizierten 100 µV:
    # er ist dieselbe Kette ohne Korrektor, hat also dasselbe Upsampling, dieselben
    # Filter und dasselbe Downsampling hinter sich. Gemessen liegt er bei rund
    # 106 % der Injektion -- der scharfe Transient gewinnt beim Resampling etwas
    # Überschwingen. Gegen 100 µV zu normieren hiesse, diesen Kettenanteil jedem
    # Modell als Verlust anzurechnen.
    referenz = next((r["mittel_uv"] for r in rows if r["arm"] == "uncorrected"), None)
    for r in rows:
        r["erhalt_vs_unkorrigiert_pct"] = (round(100.0 * r["mittel_uv"] / referenz, 1)
                                           if referenz else None)

    if not rows:
        raise SystemExit("keine vergleichbaren Arme gefunden")
    rows.sort(key=lambda r: -r["erhalt_pct"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"Injiziert: {injected:.0f} µV auf {args.channel}, "
          f"{len(times)} Stellen ({', '.join(f'{t:g} s' for t in times)})\n")
    print(f"{'Arm':32s}" + "".join(f"{f'{t:g}s':>9s}" for t in times)
          + f"{'Mittel':>9s}{'/100µV':>9s}{'/unkorr.':>10s}")
    for r in rows:
        vals = "".join(f"{r[k]:9.1f}" for k in r if k.startswith("spike_"))
        gegen = r.get("erhalt_vs_unkorrigiert_pct")
        print(f"{r['arm']:32s}{vals}{r['mittel_uv']:9.1f}{r['erhalt_pct']:8.1f}%"
              f"{(f'{gegen:9.1f}%' if gegen is not None else '        -')}")
    print(f"\ngeschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
