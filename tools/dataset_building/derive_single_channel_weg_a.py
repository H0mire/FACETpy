"""Leite aus einem Weg-A-Datensatz die einkanalige Fassung ab.

Weg A speichert je Beispiel die Zielelektrode **und ihre k nächsten Nachbarn**
(k = 2, also drei Kanäle). Für einen Vergleich gegen run 7 ist das eine Achse zu
viel: die dreizehn Familien dort sehen genau eine Elektrode, und ein Unterschied
im Ergebnis käme dann aus zwei Quellen gleichzeitig -- der unabhängigen
Clean-Quelle *und* dem räumlichen Kontext. Diese Ableitung nimmt die
Nachbarachse heraus, damit sich genau ein Hebel ändert.

**Abgeleitet, nicht neu gebaut.** Ein Neubau mit ``--k-neighbors 0`` liefe durch
dieselben Zufallsgeneratoren, aber nicht garantiert durch dieselben Ziehungen:
Spike-Injektion, Fehlermodi und Splitgrenze haingen an Seeds und an der Zahl der
Kanäle. Hier wird stattdessen die Nachbarachse der fertigen Datei
**abgeschnitten**. Damit sind Beispiele, Split, Spikes und Anreicherung
bitgleich zur dreikanaligen Fassung, und die beiden sind direkt vergleichbar.

Spalte 0 der Nachbarachse ist die Zielelektrode selbst -- ``select_neighbors``
sortiert nach Distanz, und die Distanz zu sich selbst ist 0. Das Skript prüft
das, statt es anzunehmen.

Die Quelldatei wird nicht angefasst.

Nutzung::

    uv run python tools/dataset_building/derive_single_channel_weg_a.py \\
        --input  output/weg_a_farm_v10_locked_512/weg_a_spatiotemporal_dataset.npz \\
        --output output/weg_a_farm_v10_locked_1ch/weg_a_spatiotemporal_dataset.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: Schlüssel mit einer Nachbarachse an Position 2: (N, Epochen, Kanäle, Samples).
KONTEXT_SCHLUESSEL = ("artifact_context", "artifact_context_template", "clean_context")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input", type=Path, default=REPO / "output/weg_a_farm_v10_locked_512/weg_a_spatiotemporal_dataset.npz"
    )
    ap.add_argument(
        "--output", type=Path, default=REPO / "output/weg_a_farm_v10_locked_1ch/weg_a_spatiotemporal_dataset.npz"
    )
    args = ap.parse_args()

    if args.output.resolve() == args.input.resolve():
        raise SystemExit("Ausgabe darf nicht die Eingabe sein -- die dreikanalige Fassung bleibt.")

    b = np.load(args.input, allow_pickle=True)
    nachbarn = b["neighbor_channel_indices"]
    ziel = b["target_channel_index"]
    if not bool((nachbarn[:, 0] == ziel).all()):
        raise SystemExit(
            "Spalte 0 der Nachbartabelle ist nicht die Zielelektrode -- "
            "die Annahme dieses Skripts gilt für diese Datei nicht."
        )

    neu: dict[str, np.ndarray] = {}
    for k in b.files:
        a = b[k]
        if k in KONTEXT_SCHLUESSEL:
            if a.ndim != 4:
                raise SystemExit(f"{k} hat {a.ndim} Achsen, erwartet wurden 4")
            neu[k] = np.ascontiguousarray(a[:, :, :1, :])
        elif k == "neighbor_channel_indices":
            neu[k] = np.ascontiguousarray(a[:, :1])
        elif k == "k_neighbors":
            neu[k] = np.asarray([0], dtype=a.dtype)
        else:
            neu[k] = a

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **neu)

    # Die Metadaten mitziehen, sonst behauptet die Kopie drei Kanäle.
    meta_ein = args.input.with_name(args.input.stem + "_metadata.json")
    if meta_ein.exists():
        m = json.loads(meta_ein.read_text())
        m["k_neighbors"] = 0
        if "input_shape" in m:
            m["input_shape"] = [m["input_shape"][0], 1, m["input_shape"][2]]
        m["derived_from"] = str(args.input.relative_to(REPO))
        m["derivation"] = (
            "Nachbarachse abgeschnitten (Spalte 0 = Zielelektrode). "
            "Beispiele, Split, Spikes und Fehlermodi sind bitgleich zur Quelle."
        )
        args.output.with_name(args.output.stem + "_metadata.json").write_text(
            json.dumps(m, indent=1, ensure_ascii=False)
        )

    for k in KONTEXT_SCHLUESSEL:
        if k in neu:
            print(f"  {k:28s} {b[k].shape} -> {neu[k].shape}")
    print(
        f"\ngeschrieben: {args.output}  "
        f"({args.output.stat().st_size / 2**30:.2f} GiB, Quelle "
        f"{args.input.stat().st_size / 2**30:.2f} GiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
