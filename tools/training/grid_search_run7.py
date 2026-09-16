"""Run the recorded deployment grid search and evaluate each selected checkpoint.

Screening and confirmation preserve separate seeds and records. The proof-fit
protocol ranks residual artifact only after checking seam validity. Weg-A uses
its recorded selection split; the locked holdout is not used for selection.
Use --base-config with a config resolved by masterthesis_guide.reproduce."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
WINDOW_START_S = 25.0
WINDOW_STOP_S = 160.0
T0_IN_WINDOW, T1_IN_WINDOW = (4.5, 135.0)
SEAM_SUSPECT = 1.8
SEAM_BROKEN = 2.5
FARM_GA_REST_UV = 0.248


@dataclass(frozen=True)
class Achse:
    """A grid axis: the configuration field and its candidate values."""

    name: str
    pfad: tuple[str, ...]
    werte: tuple


AXES: dict[str, dict[str, Achse]] = {
    "nested_gan": {
        "lr": Achse("lr", ("training", "learning_rate"), (0.00015, 0.0005, 0.0015)),
        "kapazitaet": Achse("ch", ("model", "kwargs", "inner_channels"), (32, 48, 72)),
        "si_sdr": Achse("sisdr", ("model", "loss_kwargs", "si_sdr_weight"), (0.0, 1.0, 3.0)),
        "wd": Achse("wd", ("training", "weight_decay"), (0.0, 0.0001, 0.01)),
        "freq": Achse("freq", ("model", "loss_kwargs", "frequency_weight"), (0.015, 0.046, 0.14)),
        "bloecke": Achse("blk", ("model", "kwargs", "inner_blocks"), (2, 4, 6)),
    },
    "vit_spectrogram": {
        "lr": Achse("lr", ("training", "learning_rate"), (0.00015, 0.0003, 0.001)),
        "kapazitaet": Achse("dim", ("model", "kwargs", "embed_dim"), (120, 192, 288)),
        "si_sdr": Achse("sisdr", ("model", "loss_kwargs", "si_sdr_weight"), (0.0, 1.0, 3.0)),
        "wd": Achse("wd", ("training", "weight_decay"), (0.01, 0.05, 0.2)),
        "freq": Achse("freq", ("model", "loss_kwargs", "frequency_weight"), (0.015, 0.046, 0.14)),
        "tiefe": Achse("depth", ("model", "kwargs", "depth"), (4, 6, 10)),
    },
    "dhct_gan": {
        "lr": Achse("lr", ("training", "learning_rate"), (0.0001, 0.001, 0.003)),
        "kapazitaet": Achse("bc", ("model", "kwargs", "base_channels"), (8, 16, 32)),
        "si_sdr": Achse("sisdr", ("model", "loss_kwargs", "si_sdr_weight"), (0.0, 1.0, 3.0)),
        "wd": Achse("wd", ("training", "weight_decay"), (0.0, 0.0001, 0.01)),
        "freq": Achse("freq", ("model", "loss_kwargs", "frequency_weight"), (0.015, 0.046, 0.14)),
        "fenster": Achse("win", ("model", "kwargs", "window_size"), (8, 16, 32)),
    },
    "demucs": {
        "lr": Achse("lr", ("training", "learning_rate"), (0.0001, 0.0003, 0.001)),
        "kapazitaet": Achse("ic", ("model", "kwargs", "initial_channels"), (32, 64, 96)),
        "si_sdr": Achse("sisdr", ("model", "loss_kwargs", "si_sdr_weight"), (0.0, 1.0, 3.0)),
        "wd": Achse("wd", ("training", "weight_decay"), (0.0, 0.0001, 0.01)),
        "freq": Achse("freq", ("model", "loss_kwargs", "frequency_weight"), (0.015, 0.046, 0.14)),
        "tiefe": Achse("depth", ("model", "kwargs", "depth"), (3, 4, 5)),
    },
}
DEFAULT_AXES = ("lr", "kapazitaet", "si_sdr")
FAMILIES = {
    f: (REPO / f"masterthesis_guide/experiments/phase_2/deployment_{f}/config.yaml", f"{f}_deployment") for f in AXES
}


def familie_aufloesen(family: str, datensatz: str = "proof_fit") -> tuple[Path, str, dict[str, Any]]:
    from masterthesis_guide.reproduce import load_catalog

    catalog = load_catalog()
    if datensatz == "wega":
        eid = next(
            (
                e
                for e, record in catalog["experiments"].items()
                if e.startswith(f"wega_{family}_") and record.get("config")
            )
        )
    elif family == "dhct_gan":
        eid = "run8_dhct_gan_lr0_0001_bc8_sisdr0_s42"
    else:
        eid = f"deployment_{family}"
    override = (
        {"packing": "b1ts", "context": "stack", "grund": "Recorded seven-epoch context variant"}
        if family == "dhct_gan"
        else {}
    )
    return (REPO / catalog["experiments"][eid]["config"], f"{family}_deployment", override)


def lies(cfg: dict, pfad: tuple[str, ...]):
    d = cfg
    for k in pfad:
        d = d[k]
    return d


def setze(cfg: dict, pfad: tuple[str, ...], wert) -> None:
    d = cfg
    for k in pfad[:-1]:
        d = d[k]
    if pfad[-1] not in d:
        raise KeyError(f"{'.'.join(pfad)} is absent from the base configuration; available: {sorted(d)}")
    d[pfad[-1]] = wert


def baue_konfig(
    basis: dict, family: str, punkt: dict, seed: int, out_root: Path, device: str, max_epochs: int | None
) -> tuple[dict, str]:
    tag = "_".join(
        (
            f"{AXES[family][a].name}{v:g}" if isinstance(v, (int, float)) else f"{AXES[family][a].name}{v}"
            for a, v in punkt.items()
        )
    )
    tag = f"{tag}_s{seed}"
    cfg = copy.deepcopy(basis)
    for achse, wert in punkt.items():
        setze(cfg, AXES[family][achse].pfad, wert)
    cfg["model"]["device"] = device
    cfg["training"]["seed"] = int(seed)
    cfg.setdefault("checkpoint", {})["save_top_k"] = 1
    cfg["checkpoint"]["save_last"] = False
    cfg.setdefault("export", {})["enabled"] = False
    cfg["training"]["model_name"] = f"grid_{family}_{tag}"
    cfg["training"]["output_dir"] = str(out_root / family / tag)
    if max_epochs is not None:
        cfg["training"]["max_epochs"] = int(max_epochs)
    return (cfg, tag)


def _val_loss(p: Path) -> float:
    stem = p.stem
    marker = "_val_loss"
    if marker not in stem:
        return float("inf")
    try:
        return float(stem.split(marker, 1)[1])
    except ValueError:
        return float("inf")


def bester_checkpoint(run_dir: Path) -> Path:
    cks = [p for p in run_dir.glob("*/checkpoints/epoch*.pt")]
    if not cks:
        cks = [p for p in run_dir.glob("checkpoints/epoch*.pt")]
    if not cks:
        raise FileNotFoundError(f"No epoch checkpoint under {run_dir}")
    return min(cks, key=_val_loss)


def exportiere(cfg_pfad: Path, ckpt: Path, ziel: Path) -> Path:
    import torch

    from facet.training.cli import _build_dataset, _build_model, _load_contexts, load_training_cli_config

    cli_cfg = load_training_cli_config(cfg_pfad)
    contexts = _load_contexts(cli_cfg) if cli_cfg.data.context_factory else []
    dataset = _build_dataset(contexts, cli_cfg)
    sfreq = contexts[0].get_sfreq() if contexts else float(getattr(dataset, "sfreq", float("nan")))
    modell = _build_model(cli_cfg, dataset, sfreq)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    for schluessel in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and schluessel in state and isinstance(state[schluessel], dict):
            state = state[schluessel]
            break
    modell.load_state_dict(state)
    ziel_modul = modell
    hook = getattr(modell, "export_module", None)
    if callable(hook):
        ziel_modul = hook()
    ziel_modul = copy.deepcopy(ziel_modul).to("cpu").eval()
    form = getattr(dataset, "input_shape", None)
    if form is None:
        beispiel_ein = dataset[0][0]
        form = tuple(int(x) for x in beispiel_ein.shape)
    beispiel = torch.randn(1, *form)
    with torch.no_grad():
        scripted = torch.jit.trace(ziel_modul, beispiel)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(ziel))
    return ziel


def bewerte_in_pipeline(
    ts_pfad: Path, model_id: str, edf: Path, device: str, dc_mode: str, abweichung: dict | None = None
) -> dict:
    import mne
    import numpy as np

    from facet.core import ProcessingContext, ProcessingMetadata
    from facet.correction.deep_learning import DeepLearningCorrection
    from facet.evaluation import EpochSeamStepCalculator, GradientArtifactResidualCalculator
    from facet.models.masterthesis import pipeline as reference_chain
    from facet.models.masterthesis.adapters import FamilyAdapter

    benutzt = device
    for versuch in (device, "cpu"):
        adapter = FamilyAdapter(model_id, checkpoint=ts_pfad, device=versuch, dc_mode=dc_mode)
        if abweichung:
            import dataclasses

            felder = {k: v for k, v in abweichung.items() if k in ("packing", "context")}
            if felder:
                adapter.packing = dataclasses.replace(adapter.packing, **felder)
        pipe = reference_chain.build(
            edf, correctors=[DeepLearningCorrection(model=adapter)], include_pca=True, name=model_id
        )
        try:
            res = pipe.run()
        except Exception:
            if versuch == "cpu":
                raise
            print(f"    {model_id}: {versuch} cannot run this export; falling back to CPU", flush=True)
            continue
        if res.success:
            benutzt = versuch
            break
        if versuch == "cpu":
            raise RuntimeError(f"Pipeline failed: {res.error}")
        print(
            f"    {model_id}: {versuch} cannot run this export ({str(res.error)[:80]}); falling back to CPU", flush=True
        )
    raw = res.context.get_raw()
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
    sf = float(raw.info["sfreq"])
    i0, i1 = (int(WINDOW_START_S * sf), int(WINDOW_STOP_S * sf))
    data = raw.get_data(picks=picks)[:, i0:i1].astype(np.float64)
    namen = [raw.ch_names[i] for i in picks]
    trg = np.asarray(res.context.get_triggers() if res.context.has_triggers() else [], dtype=np.int64)
    trg = trg[(trg >= i0) & (trg < i1)] - i0
    fenster = mne.io.RawArray(data, mne.create_info(namen, sf, "eeg"), verbose=False)
    ctx = ProcessingContext(raw=fenster, metadata=ProcessingMetadata(triggers=list(map(int, trg))))
    comb = (
        GradientArtifactResidualCalculator(tmin=T0_IN_WINDOW, tmax=T1_IN_WINDOW)
        .execute(ctx)
        .metadata.custom["gradient_artifact_residual"]
    )
    seam = EpochSeamStepCalculator(tmin=T0_IN_WINDOW, tmax=T1_IN_WINDOW).execute(ctx).metadata.custom["epoch_seam_step"]
    return {
        "pipeline_device": benutzt,
        "ga_rest_uv": round(comb["comb_rms_uv"], 4),
        "rms_uv": round(comb["rms_uv"], 4),
        "naht_ratio": round(seam["ratio"], 4),
        "x_farm": round(comb["comb_rms_uv"] / FARM_GA_REST_UV, 2),
    }


def bewerte_auf_selektion(cfg_pfad: Path, ckpt: Path, device: str) -> dict:
    import numpy as np
    import torch

    from facet.training.cli import _build_dataset, _build_model, load_training_cli_config

    cli_cfg = load_training_cli_config(cfg_pfad)
    cli_cfg.model.device = device
    ds = _build_dataset([], cli_cfg)
    modell = _build_model(cli_cfg, ds, ds.sfreq)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and k in state and isinstance(state[k], dict):
            state = state[k]
            break
    modell.load_state_dict(state)
    modell = modell.to(device).eval()
    _, selektion = ds.train_val_split()
    from facet.training.cli import _import_object

    verlust = _import_object(cli_cfg.model.loss_factory)(**cli_cfg.model.loss_kwargs, sfreq=ds.sfreq)
    sagt_artefakt = getattr(verlust, "prediction_is", "artifact") == "artifact"
    fehler, fehler_null, cl, ch = ([], [], [], [])
    with torch.no_grad():
        for anfang in range(0, len(selektion), 64):
            paare = [selektion[i] for i in range(anfang, min(anfang + 64, len(selektion)))]
            x = torch.from_numpy(np.stack([p[0] for p in paare])).to(device)
            y = np.stack([p[1] for p in paare])
            pred = modell(x).cpu().numpy()
            clean, noisy = (y[:, 1], y[:, 2])
            clean_hat = noisy - pred if sagt_artefakt else pred
            fehler.append(clean_hat - clean)
            fehler_null.append(-clean)
            cl.append(clean)
            ch.append(clean_hat)
    f = np.concatenate([a.ravel() for a in fehler])
    fn = np.concatenate([a.ravel() for a in fehler_null])
    c = np.concatenate([a.ravel() for a in cl])
    h = np.concatenate([a.ravel() for a in ch])
    U = 1000000.0
    return {
        "err_uv": round(float(np.sqrt((f**2).mean())) * U, 4),
        "err_uv_null": round(float(np.sqrt((fn**2).mean())) * U, 4),
        "corr_clean": round(float(np.corrcoef(h, c)[0, 1]), 4),
        "besser_als_null": bool(np.sqrt((f**2).mean()) < np.sqrt((fn**2).mean())),
        "n_selektion": len(selektion),
    }


def hinweis(naht: float) -> str:
    if naht >= SEAM_BROKEN:
        return "verworfen: Nahtsprung"
    if naht >= SEAM_SUSPECT:
        return "auffaellig: Nahtsprung"
    return ""


def fuehre_punkt_aus(family: str, basis: dict, punkt: dict, seed: int, args) -> dict:
    cfg, tag = baue_konfig(basis, family, punkt, seed, args.out_root, args.device, args.max_epochs)
    out_root = args.out_root / family
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_pfad = out_root / f"{tag}.yaml"
    cfg_pfad.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    zeile = {"tag": tag, "seed": seed, **{a: punkt[a] for a in punkt}}
    if args.dry_run:
        return zeile | {"status": "dry-run"}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "facet.training.cli", "fit", "--config", str(cfg_pfad)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    zeile["train_s"] = round(time.perf_counter() - t0, 1)
    if proc.returncode != 0:
        return zeile | {"status": "train-fehler", "fehler": (proc.stderr or proc.stdout)[-600:]}
    try:
        ckpt = bester_checkpoint(Path(cfg["training"]["output_dir"]))
        zeile["val_loss"] = round(_val_loss(ckpt), 5)
        zeile["checkpoint"] = str(ckpt.relative_to(REPO)) if ckpt.is_relative_to(REPO) else str(ckpt)
    except Exception as e:
        return zeile | {"status": "checkpoint-fehler", "fehler": f"{type(e).__name__}: {e}"}
    if args.dataset == "wega":
        try:
            t1 = time.perf_counter()
            zeile |= bewerte_auf_selektion(cfg_pfad, ckpt, args.pipeline_device)
            zeile["bewertung_s"] = round(time.perf_counter() - t1, 1)
        except Exception as e:
            return zeile | {"status": "bewertung-fehler", "fehler": f"{type(e).__name__}: {e}"}
        zeile["hinweis"] = "" if zeile["besser_als_null"] else "schlechter als die Nullausgabe"
        zeile["status"] = "ok" if zeile["besser_als_null"] else "verworfen"
        return zeile
    try:
        ts = exportiere(cfg_pfad, ckpt, out_root / f"{tag}.ts")
    except Exception as e:
        return zeile | {"status": "export-fehler", "fehler": f"{type(e).__name__}: {e}"}
    try:
        t1 = time.perf_counter()
        _, model_id, abweichung = familie_aufloesen(family, args.dataset)
        zeile |= bewerte_in_pipeline(ts, model_id, args.edf, args.pipeline_device, args.dc_mode, abweichung)
        zeile["pipeline_s"] = round(time.perf_counter() - t1, 1)
    except Exception as e:
        return zeile | {"status": "pipeline-fehler", "fehler": f"{type(e).__name__}: {e}"}
    zeile["hinweis"] = hinweis(zeile["naht_ratio"])
    zeile["status"] = "verworfen" if zeile["naht_ratio"] >= SEAM_BROKEN else "ok"
    return zeile


def schreibe(pfad: Path, kopf: dict, zeilen: list[dict]) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps({**kopf, "zeilen": zeilen}, indent=2, ensure_ascii=False), encoding="utf-8")


def zeige(zeile: dict) -> None:
    if "err_uv" in zeile:
        print(
            f"    Error {zeile['err_uv']:7.2f} µV (zero baseline {zeile['err_uv_null']:7.2f}) | corr {zeile['corr_clean']:+.3f} | val_loss {zeile.get('val_loss', float('nan')):8.4f}{('  ' + zeile['hinweis'] if zeile.get('hinweis') else '')}",
            flush=True,
        )
        return
    if zeile.get("status") == "ok" or zeile.get("hinweis"):
        print(
            f"    Residual {zeile['ga_rest_uv']:6.2f} µV ({zeile['x_farm']:5.1f}x FARM) | seam {zeile['naht_ratio']:5.2f} | val_loss {zeile.get('val_loss', float('nan')):8.4f}{('  ' + zeile['hinweis'] if zeile.get('hinweis') else '')}",
            flush=True,
        )
    else:
        print(f"    {zeile.get('status')}: {str(zeile.get('fehler'))[:200]}", flush=True)


def rangliste(zeilen: list[dict]) -> list[dict]:
    ok = [z for z in zeilen if z.get("status") == "ok"]
    schluessel = "err_uv" if ok and "err_uv" in ok[0] else "ga_rest_uv"
    return sorted(ok, key=lambda z: z[schluessel])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("screen", "confirm"))
    p.add_argument("--base-config", type=Path, help="Resolved config produced by masterthesis_guide.reproduce config")
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--dataset", choices=("proof_fit", "wega"), default="proof_fit", help="Dataset.")
    p.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES), help="Axes.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seeds.")
    p.add_argument("--screen-seed", type=int, default=42)
    p.add_argument("--from", dest="quelle", type=Path, default=None, help="From.")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=None, help="Max epochs.")
    p.add_argument("--device", default="cuda", help="Training")
    p.add_argument("--pipeline-device", default="cuda", help="Pipeline device.")
    p.add_argument("--dc-mode", default="as_evaluated", choices=("as_evaluated", "reconcile", "segment_mean"))
    p.add_argument("--edf", type=Path, help="Original Niazy recording for proof-fit pipeline scoring")
    p.add_argument("--out-root", type=Path, default=REPO / "output/grids/run8")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Limit.")
    p.add_argument("--dry-run", action="store_true", help="Dry run.")
    args = p.parse_args()
    if args.n_shards < 1 or not 0 <= args.shard < args.n_shards:
        p.error("--shard must be within [0, --n-shards)")
    if not args.dry_run and not args.base_config:
        p.error("Supply --base-config from masterthesis_guide.reproduce config to resolve external inputs")
    family = args.family
    cfg_pfad, model_id, abweichung = familie_aufloesen(family, args.dataset)
    if args.base_config:
        cfg_pfad = args.base_config.resolve()
    basis = yaml.safe_load(cfg_pfad.read_text(encoding="utf-8"))
    if abweichung:
        packung = abweichung.get("packing")
        print(
            f"  Difference from Run 7: {abweichung.get('grund', '')}\n  Configuration {cfg_pfad}"
            + (f", packing {packung} instead of the Run-7 packing" if packung else "")
            + "\n"
        )
    if args.dataset == "proof_fit" and (args.edf is None or not args.edf.is_file()) and (not args.dry_run):
        raise SystemExit(f"{args.edf} is missing. Supply --edf for pipeline evaluation.")
    unbekannt = [a for a in args.axes if a not in AXES[family]]
    if unbekannt:
        raise SystemExit(f"Unknown axes {unbekannt} for {family}; available: {sorted(AXES[family])}")
    teil = f"_shard{args.shard}" if args.n_shards > 1 else ""
    if args.phase == "screen":
        werte = [AXES[family][a].werte for a in args.axes]
        gitter = [dict(zip(args.axes, kombi, strict=True)) for kombi in itertools.product(*werte)]
        aufgaben = [(g, args.screen_seed) for g in gitter]
        ziel = args.out_root / f"grid_{family}_screen{teil}.json"
    else:
        if args.quelle is None:
            raise SystemExit("confirm requires --from <screening.json>")
        vorher = json.loads(args.quelle.read_text(encoding="utf-8"))
        kandidaten = rangliste(vorher["zeilen"])[: args.top_k]
        if not kandidaten:
            raise SystemExit(f"{args.quelle} contains no evaluated grid point")
        achsen = vorher["achsen"]
        aufgaben = [({a: k[a] for a in achsen}, s) for k in kandidaten for s in args.seeds]
        args.axes = achsen
        ziel = args.out_root / f"grid_{family}_confirm{teil}.json"
        print("Screening candidates:")
        for k in kandidaten:
            print(f"  {k['tag']:40s} residual {k['ga_rest_uv']:.2f} µV  seam {k['naht_ratio']:.2f}")
        print()
    meine = [t for i, t in enumerate(aufgaben) if i % args.n_shards == args.shard]
    if args.limit is not None:
        meine = meine[: args.limit]
    basis_werte = {a: lies(basis, AXES[family][a].pfad) for a in AXES[family]}
    kopf = {
        "familie": family,
        "phase": args.phase,
        "datensatz": args.dataset,
        "achsen": list(args.axes),
        "basis_werte_run7": basis_werte,
        "gitterwerte": {a: list(AXES[family][a].werte) for a in args.axes},
        "fest_geblieben": {a: v for a, v in basis_werte.items() if a not in args.axes},
        "basis_konfig": str(cfg_pfad),
        "model_id": model_id,
        "abweichung_von_run7": {k: str(v) if isinstance(v, Path) else v for k, v in abweichung.items()} or None,
        "fenster_s": [WINDOW_START_S, WINDOW_STOP_S],
        "analyse_s": [T0_IN_WINDOW, T1_IN_WINDOW],
        "farm_ga_rest_uv": FARM_GA_REST_UV,
        "naht_verworfen_ab": SEAM_BROKEN,
        "dc_mode": args.dc_mode,
        "bewertung": "Residual artifact in the complete correction chain; reject discontinuous seams before ranking.",
    }
    print(f"{family}: {len(aufgaben)} points in total, shard {args.shard}/{args.n_shards} -> {len(meine)}")
    print(f"  {'Axis':12s} {'Configuration path':40s} {'run 7':>10s}   Grid")
    for a in args.axes:
        ach = AXES[family][a]
        vorher = basis_werte[a]
        print(f"  {a:12s} {'.'.join(ach.pfad):40s} {vorher!s:>10s}   {list(ach.werte)}")
    fest = [f"{a}={basis_werte[a]}" for a in AXES[family] if a not in args.axes]
    if fest:
        print(f"  Unchanged from the base configuration: {', '.join(fest)}")
    if args.dataset == "wega":
        print(
            "  Score reconstruction error in µV on the selection split against zero output.\n  The locked holdout is excluded from selection.\n",
            flush=True,
        )
    else:
        print(f"  Recorded FARM residual: {FARM_GA_REST_UV:.3f} µV.\n", flush=True)
    zeilen: list[dict] = []
    for i, (punkt, seed) in enumerate(meine, start=1):
        beschriftung = ", ".join(
            f"{a}={punkt[a]:g}" if isinstance(punkt[a], (int, float)) else f"{a}={punkt[a]}" for a in args.axes
        )
        print(f"[{i}/{len(meine)}] {beschriftung}, seed {seed}", flush=True)
        zeile = fuehre_punkt_aus(family, basis, punkt, seed, args)
        zeige(zeile)
        zeilen.append(zeile)
        schreibe(ziel, kopf, zeilen)
    beste = rangliste(zeilen)
    if beste:
        b = beste[0]
        print(
            f"\nBest evaluated point: {b['tag']} -> residual {b['ga_rest_uv']:.2f} µV ({b['x_farm']:.1f}x FARM), seam {b['naht_ratio']:.2f}, val_loss {b.get('val_loss', float('nan')):.4f}"
        )
        nach_loss = min((z for z in zeilen if "val_loss" in z), key=lambda z: z["val_loss"], default=None)
        if nach_loss and nach_loss["tag"] != b["tag"]:
            print(
                f"Selection by validation loss would choose {nach_loss['tag']} (residual {nach_loss.get('ga_rest_uv', float('nan')):.2f} µV); the two rankings differ."
            )
    else:
        print("\nNo evaluated point; inspect status/error fields in " + str(ziel))
    print(f"Written: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
