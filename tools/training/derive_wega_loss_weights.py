"""Leite die Gewichte der RecoveredCleanObjective für Weg A her.

Die Gewichte aus run 7 sind auf den Proof-Fit-Datensatz kalibriert und lassen
sich nicht übertragen. Zwei der Terme -- ``velocity`` und ``acceleration`` --
teilen durch die **Ableitungsenergie des sauberen Signals**, und die ist bei den
beiden Datensätzen völlig verschieden:

====================  ===========  ==========  =====================
Datensatz             RMS clean    RMS d/dt    Verhältnis
====================  ===========  ==========  =====================
Proof-Fit             334,9 µV     377,7 µV    **1,13**
Weg A                  37,1 µV       0,32 µV   **0,0087**
====================  ===========  ==========  =====================

Beim Proof-Fit-Datensatz ändert sich das "saubere" Signal von Sample zu Sample
um mehr als seine eigene Amplitude -- es ist die Ausgabe von AAS und trägt deren
Hochfrequenzrest. Weg As Clean stammt aus einer unabhängigen Quelle und ist
glatt. Derselbe Term liegt deshalb einmal bei 8 und einmal bei 2·10⁸; mit den
run-7-Gewichten wären über 99,99 % des Gradienten der Beschleunigungsterm.

**Methode.** Dieselbe, mit der in run 7 schon ``frequency_weight = 0,046``
hergeleitet wurde: die Terme werden am Identitätsstart gemessen und gegen den
Amplitudenterm angeglichen. Angeglichen werden nur die vier normierten
MSE-Terme. ``si_sdr`` und ``energy_ratio`` bleiben bei 1,0 -- sie sind
beschränkte Verhältnisterme, keine normierten MSEs, und wurden auch in run 7
nicht angeglichen.

Zusätzlich werden die drei Ankerwerte neu vermessen, weil die aus run 7
(perfekt −1,000 / Löschen 3,046 / nichts tun ≈43) mit den neuen Gewichten nicht
mehr gelten und ein Trainingsprotokoll ohne sie nicht lesbar ist:

perfekt
    die Vorhersage, für die ``clean_hat == clean`` gilt
Löschen
    die Vorhersage, für die ``clean_hat == 0`` -- der Arm, gegen den sich jedes
    Modell behaupten muss
nichts tun
    die Ausgabe am Identitätsstart

Nutzung::

    uv run python tools/training/derive_wega_loss_weights.py
    uv run python tools/training/derive_wega_loss_weights.py --n 512 --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]

FAMILIEN = ("nested_gan", "vit_spectrogram", "demucs", "dhct_gan")

#: Terme, die als normierte MSE gebaut sind und deshalb angeglichen werden.
ANGLEICHBAR = ("amplitude", "velocity", "acceleration", "frequency")
#: Bezugsterm: alle anderen werden auf seinen Wert gebracht.
BEZUG = "amplitude"


def terme(verlust, vorhersage, ziel) -> dict[str, float]:
    with torch.no_grad():
        verlust(vorhersage, ziel)
    roh = getattr(verlust, "last_terms", None) or getattr(verlust, "_last_terms", None)
    if not isinstance(roh, dict):
        raise RuntimeError("Der Verlust legt seine Einzelterme nicht offen "
                           "(erwartet wurde last_terms).")
    return {k: float(v) for k, v in roh.items()}


def gesamt(verlust, vorhersage, ziel) -> float:
    with torch.no_grad():
        v = verlust(vorhersage, ziel)
    return float(v if torch.is_tensor(v) else v[0])


def main() -> int:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from facet.training.cli import (_build_dataset, _build_model, _import_object,
                                    load_training_cli_config)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256, help="Beispiele für die Messung")
    ap.add_argument("--apply", action="store_true",
                    help="die hergeleiteten Gewichte in die configs/run8_wega_*.yaml schreiben")
    ap.add_argument("--out", type=Path, default=REPO / "docs/research/run_8_wega_loss_weights.json")
    args = ap.parse_args()

    ergebnis: dict[str, dict] = {}
    for fam in FAMILIEN:
        cfg_pfad = REPO / f"configs/run8_wega_{fam}.yaml"
        cfg = load_training_cli_config(cfg_pfad)
        cfg.model.device = "cpu"
        ds = _build_dataset([], cfg)
        modell = _build_model(cfg, ds, ds.sfreq)
        verlust = _import_object(cfg.model.loss_factory)(**cfg.model.loss_kwargs, sfreq=ds.sfreq)

        # Über den Trainingsteil streuen statt die ersten n zu nehmen: vorne
        # stehen zusammenhaengende Epochen, und die sind sich aehnlicher als der
        # Datensatz im Mittel.
        idx = np.linspace(0, 16000, args.n).astype(int)
        x = torch.from_numpy(np.stack([ds[i][0] for i in idx]))
        y = torch.from_numpy(np.stack([ds[i][1] for i in idx]))
        with torch.no_grad():
            identitaet = modell(x)

        artefakt = y[:, 0]
        clean = y[:, 1]
        noisy = y[:, 2]
        sagt_artefakt = getattr(verlust, "prediction_is", "artifact") == "artifact"
        perfekt = artefakt if sagt_artefakt else clean
        loeschen = noisy if sagt_artefakt else torch.zeros_like(clean)

        t = terme(verlust, identitaet, y)
        bezug = t[BEZUG]
        gewichte = {f"{k}_weight": (bezug / t[k] if t[k] else 0.0)
                    for k in ANGLEICHBAR if k in t}
        ergebnis[fam] = {
            "terme_bei_identitaetsstart": t,
            "hergeleitete_gewichte": gewichte,
            "anker_mit_run7_gewichten": {
                "perfekt": gesamt(verlust, perfekt, y),
                "loeschen": gesamt(verlust, loeschen, y),
                "nichts_tun": gesamt(verlust, identitaet, y),
            },
            "vorhersage_ist": "artifact" if sagt_artefakt else "clean",
        }

        # Dieselben Anker noch einmal, mit den neuen Gewichten.
        neue_kwargs = dict(cfg.model.loss_kwargs)
        neue_kwargs.update(gewichte)
        verlust_neu = _import_object(cfg.model.loss_factory)(**neue_kwargs, sfreq=ds.sfreq)
        ergebnis[fam]["anker_mit_neuen_gewichten"] = {
            "perfekt": gesamt(verlust_neu, perfekt, y),
            "loeschen": gesamt(verlust_neu, loeschen, y),
            "nichts_tun": gesamt(verlust_neu, identitaet, y),
        }

        print(f"--- {fam}  (Vorhersage ist {ergebnis[fam]['vorhersage_ist']})")
        print(f"    {'Term':14s}{'Wert':>14s}{'Gewicht':>14s}")
        for k, v in t.items():
            g = gewichte.get(f"{k}_weight")
            print(f"    {k:14s}{v:14.4g}" + (f"{g:14.4g}" if g is not None else f"{'1,0 (fest)':>14s}"))
        a, an = ergebnis[fam]["anker_mit_run7_gewichten"], ergebnis[fam]["anker_mit_neuen_gewichten"]
        print(f"    Anker  perfekt {a['perfekt']:+.4g} -> {an['perfekt']:+.4g} | "
              f"loeschen {a['loeschen']:+.4g} -> {an['loeschen']:+.4g} | "
              f"nichts tun {a['nichts_tun']:+.4g} -> {an['nichts_tun']:+.4g}", flush=True)

    # Ein gemeinsamer Satz, gemittelt im Logarithmus: die Gewichte spannen
    # Groessenordnungen, das arithmetische Mittel waere vom groessten dominiert.
    schluessel = sorted({k for f in ergebnis.values() for k in f["hergeleitete_gewichte"]})
    gemeinsam = {k: float(np.exp(np.mean([np.log(ergebnis[f]["hergeleitete_gewichte"][k])
                                          for f in FAMILIEN
                                          if ergebnis[f]["hergeleitete_gewichte"].get(k, 0) > 0])))
                 for k in schluessel}
    spanne = {k: (min(ergebnis[f]["hergeleitete_gewichte"][k] for f in FAMILIEN),
                  max(ergebnis[f]["hergeleitete_gewichte"][k] for f in FAMILIEN))
              for k in schluessel}
    print("\nGemeinsamer Satz (geometrisches Mittel ueber die vier Familien):")
    for k in schluessel:
        lo, hi = spanne[k]
        print(f"  {k:24s} {gemeinsam[k]:12.4g}   Spanne {lo:.4g} .. {hi:.4g}  (Faktor {hi/lo:.2f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n_beispiele": args.n, "je_familie": ergebnis,
         "gemeinsam": gemeinsam, "spanne": spanne,
         "methode": "Terme am Identitaetsstart gegen den Amplitudenterm angeglichen; "
                    "si_sdr und energy_ratio bleiben bei 1,0 (beschraenkte Verhaeltnisterme)."},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\ngeschrieben: {args.out}")

    if args.apply:
        for fam in FAMILIEN:
            p = REPO / f"configs/run8_wega_{fam}.yaml"
            text = p.read_text(encoding="utf-8")
            kopf, _, rest = text.partition("model:")
            c = yaml.safe_load("model:" + rest)
            c["model"]["loss_kwargs"].update(gemeinsam)
            p.write_text(kopf + yaml.safe_dump(c, sort_keys=False, allow_unicode=True),
                         encoding="utf-8")
            print(f"  aktualisiert: {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
