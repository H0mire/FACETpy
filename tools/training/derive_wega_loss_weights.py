"""Derive Weg-A loss scaling from the original identity-start protocol.

Evaluate amplitude, velocity, acceleration and frequency terms on 256 evenly
spaced indices from 0 through 16000. The latter bound is the recorded protocol,
not a generic holdout selector. Inputs must be resolved copies of the original
Weg-A configurations. Preserve the separate selection and locked holdout splits.
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

ANGLEICHBAR = ("amplitude", "velocity", "acceleration", "frequency")
BEZUG = "amplitude"


def terme(verlust, vorhersage, ziel) -> dict[str, float]:
    with torch.no_grad():
        verlust(vorhersage, ziel)
    roh = getattr(verlust, "last_terms", None) or getattr(verlust, "_last_terms", None)
    if not isinstance(roh, dict):
        raise RuntimeError("The loss does not expose its terms (expected last_terms).")
    return {k: float(v) for k, v in roh.items()}


def gesamt(verlust, vorhersage, ziel) -> float:
    with torch.no_grad():
        v = verlust(vorhersage, ziel)
    return float(v if torch.is_tensor(v) else v[0])


def main() -> int:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from facet.training.cli import _build_dataset, _build_model, _import_object, load_training_cli_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256, help="Number of sampled training examples")
    ap.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Resolved nested_gan.yaml, vit_spectrogram.yaml, demucs.yaml and dhct_gan.yaml",
    )
    ap.add_argument(
        "--write-configs", type=Path, help="Write new configurations with the derived weights to this directory"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ergebnis: dict[str, dict] = {}
    for fam in FAMILIEN:
        cfg_pfad = args.config_dir / f"{fam}.yaml"
        cfg = load_training_cli_config(cfg_pfad)
        cfg.model.device = "cpu"
        ds = _build_dataset([], cfg)
        modell = _build_model(cfg, ds, ds.sfreq)
        verlust = _import_object(cfg.model.loss_factory)(**cfg.model.loss_kwargs, sfreq=ds.sfreq)

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
        gewichte = {f"{k}_weight": (bezug / t[k] if t[k] else 0.0) for k in ANGLEICHBAR if k in t}
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

        neue_kwargs = dict(cfg.model.loss_kwargs)
        neue_kwargs.update(gewichte)
        verlust_neu = _import_object(cfg.model.loss_factory)(**neue_kwargs, sfreq=ds.sfreq)
        ergebnis[fam]["anker_mit_neuen_gewichten"] = {
            "perfekt": gesamt(verlust_neu, perfekt, y),
            "loeschen": gesamt(verlust_neu, loeschen, y),
            "nichts_tun": gesamt(verlust_neu, identitaet, y),
        }

        print(f"--- {fam}  (prediction target: {ergebnis[fam]['vorhersage_ist']})")
        print(f"    {'Term':14s}{'Value':>14s}{'Weight':>14s}")
        for k, v in t.items():
            g = gewichte.get(f"{k}_weight")
            print(f"    {k:14s}{v:14.4g}" + (f"{g:14.4g}" if g is not None else f"{'1.0 (fixed)':>14s}"))
        a, an = ergebnis[fam]["anker_mit_run7_gewichten"], ergebnis[fam]["anker_mit_neuen_gewichten"]
        print(
            f"    Anchors: perfect {a['perfekt']:+.4g} -> {an['perfekt']:+.4g} | "
            f"zero output {a['loeschen']:+.4g} -> {an['loeschen']:+.4g} | "
            f"identity {a['nichts_tun']:+.4g} -> {an['nichts_tun']:+.4g}",
            flush=True,
        )

    schluessel = sorted({k for f in ergebnis.values() for k in f["hergeleitete_gewichte"]})
    gemeinsam = {
        k: float(
            np.exp(
                np.mean(
                    [
                        np.log(ergebnis[f]["hergeleitete_gewichte"][k])
                        for f in FAMILIEN
                        if ergebnis[f]["hergeleitete_gewichte"].get(k, 0) > 0
                    ]
                )
            )
        )
        for k in schluessel
    }
    spanne = {
        k: (
            min(ergebnis[f]["hergeleitete_gewichte"][k] for f in FAMILIEN),
            max(ergebnis[f]["hergeleitete_gewichte"][k] for f in FAMILIEN),
        )
        for k in schluessel
    }
    print("\nShared weights (geometric mean across the four families):")
    for k in schluessel:
        lo, hi = spanne[k]
        print(f"  {k:24s} {gemeinsam[k]:12.4g}   range {lo:.4g} .. {hi:.4g}  (factor {hi / lo:.2f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "n_beispiele": args.n,
                "je_familie": ergebnis,
                "gemeinsam": gemeinsam,
                "spanne": spanne,
                "methode": "Identity-start terms scaled against the amplitude term; "
                "si_sdr and energy_ratio retain weight 1.0.",
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten: {args.out}")

    if args.write_configs:
        args.write_configs.mkdir(parents=True, exist_ok=True)
        for fam in FAMILIEN:
            config = yaml.safe_load((args.config_dir / f"{fam}.yaml").read_text())
            config["model"]["loss_kwargs"].update(gemeinsam)
            destination = args.write_configs / f"{fam}.yaml"
            if destination.exists():
                raise FileExistsError(destination)
            destination.write_text(yaml.safe_dump(config, sort_keys=False))
            print(f"Written: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
