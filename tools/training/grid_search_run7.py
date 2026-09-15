"""Gridsuche für die vier besten Deployment-Editionen — bewertet in der Pipeline.

Run 7 hat dreizehn Familien mit **je einer** Hyperparameterwahl gemessen. Diese
Wahl stammt aus den jeweiligen Papers und wurde nie an dieser Aufgabe getunt. Die
Rangfolge trägt deshalb zwei Anteile, die man nicht auseinanderhalten kann: wie
gut die Architektur ist, und wie gut ihre geerbten Konstanten zufällig passen.
Dieses Skript trennt sie für die vier aussichtsreichsten Familien.

**Bewertet wird in der Korrekturkette, nicht am Trainingsverlust.** Das ist die
zentrale Konsequenz aus run 7: über die zwölf messbaren Familien erklärt der
Trainingsverlust nur 37 % der Pipeline-Rangvarianz (Spearman ρ = 0,61, p = 0,036);
Demucs springt zwischen beiden Ordnungen um sechs Ränge nach unten, DHCT-GAN um
sechs nach oben. Eine Gridsuche, die nach ``val_loss`` auswählt, würde genau den
Fehler wiederholen, den das Kapitel beschreibt. Jeder Gitterpunkt läuft deshalb
nach dem Training durch dieselbe Kette wie die Hauptmessung und wird mit
denselben zwei Kennzahlen bewertet: Kammrest (GA-Rest in µV) und Nahtsprung.

**Zwei Kennzahlen, nicht eine.** Der Kammrest allein hat in run 7 zweimal in die
Irre geführt: ``st_gnn`` stand nach ihm auf Platz zwei und gibt eine Treppe aus,
``ic_unet`` hat nichts entfernt, sondern einen Zackenzug aufgeprägt. Der
Nahtsprung fängt beides. Hier ist er ein **Ausschlusskriterium**, kein Summand:
ein Punkt mit Nahtsprung ≥ 2,5 wird nicht gewertet, egal wie gut sein Kammrest
ist. FARM liegt bei 1,11, das unkorrigierte Signal bei 5,69.

**Zwei Phasen, weil ein Gitterpunkt nicht misst, was er zu messen scheint.**
Die Seed-Streuung in run 7 lag bei σ ≈ 0,06–0,08 im Trainingsverlust —
vergleichbar mit den Effekten, die hier gesucht werden. Die beste von 27
Einzelläufen zu nehmen heißt größtenteils, den günstigsten Seed zu nehmen.
Darum:

``screen``
    das volle Gitter, ein Seed je Punkt. Findet die Region, nicht den Sieger.
``confirm``
    die besten ``--top-k`` aus einem Screening-Ergebnis, je ``--seeds`` Läufe.
    Erst die Streuung dieser Wiederholungen sagt, ob der Unterschied echt ist.

**Der Export kommt aus dem besten Checkpoint, nicht aus der letzten Epoche.**
``facet.training.cli`` exportiert das Modell im Zustand nach dem letzten
Trainingsschritt; nichts stellt vorher die besten Gewichte wieder her. Bei
``patience: 20`` liegen dazwischen bis zu zwanzig Epochen. Für eine Auswahl nach
``val_loss`` wäre das ein anderer Gegenstand als der gemessene, deshalb lädt
dieses Skript den Checkpoint mit dem besten Wert selbst und traced ihn auf der
CPU neu.

Jeder Lauf schreibt sein eigenes Verzeichnis; ``grid_<familie>_<phase>.json``
sammelt die bewerteten Zeilen fortlaufend, sodass ein abgebrochener Durchlauf
verwertbar bleibt.

Nutzung (auf dem Pod, eine Familie je Pod)::

    .venv/bin/python tools/training/grid_search_run7.py screen \\
        --family nested_gan --device cuda --out-root grids/run7

    .venv/bin/python tools/training/grid_search_run7.py confirm \\
        --family nested_gan --from grids/run7/grid_nested_gan_screen.json \\
        --top-k 3 --seeds 42 43 44

Voraussetzungen auf dem Pod: der Datensatz unter
``output/niazy_proof_fit_context_512/`` (1,16 GB, mit ``runpodctl send``
übertragen) und ``examples/datasets/NiazyFMRI.edf`` aus dem Repository — ohne
die EDF gibt es keine Pipeline-Bewertung und damit keinen Sinn in diesem Skript.
"""

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

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "pipeline_demo"))

#: Analysefenster der Hauptmessung, in Sekunden der Aufnahme. Identisch zu
#: ``tools/pipeline_demo/measure_arms.py``, damit die Gitterzahlen direkt gegen
#: ``docs/research/run_7_pipeline_results.csv`` lesbar sind. Der Kammwert hängt
#: von der Fensterlänge ab — dieselbe Aufnahme über 5,5 s ergibt rund das
#: Siebenfache —, also ist ein abweichendes Fenster ein anderer Maßstab.
WINDOW_START_S = 25.0
WINDOW_STOP_S = 160.0
T0_IN_WINDOW, T1_IN_WINDOW = 4.5, 135.0

#: Nahtsprung als Ausschluss, nicht als Abzug. FARM = 1,11; unkorrigiert = 5,69.
SEAM_SUSPECT = 1.8
SEAM_BROKEN = 2.5

#: Der Kammrest von FARM über demselben Fenster. Bezugspunkt jeder Zeile: die
#: Gridsuche ist erst dann interessant, wenn ein Punkt hier deutlich näher
#: herankommt als die 1,40 µV des besten ungetunten Modells.
FARM_GA_REST_UV = 0.248


@dataclass(frozen=True)
class Achse:
    """Eine Gitterachse: wohin sie in der YAML greift und welche Werte sie hat."""

    name: str                 # kurzer Name für den Lauf-Tag
    pfad: tuple[str, ...]     # Pfad in der Konfiguration
    werte: tuple


#: Die Achsen je Familie. Drei davon sind vorausgewählt (``DEFAULT_AXES``), die
#: übrigen stehen für einen zweiten Durchgang bereit — das Gitter wächst
#: multiplikativ, vier Achsen zu drei Stufen sind 81 Läufe und damit mehr als
#: eine Pod-Nacht.
#:
#: Warum diese drei:
#:
#: * ``lr`` — die vier Basiskonfigurationen benutzen 3e-4, 5e-4 und 1e-3 ohne
#:   erkennbaren Grund; keiner dieser Werte wurde an dieser Aufgabe geprüft.
#: * ``kapazitaet`` — in run 7 brachte die Verbreiterung von ``bn4`` auf ``bn128``
#:   +0,125 im Verlust. Kapazität ist an dieser Aufgabe nachweislich eine echte
#:   Achse und keine Formalie.
#: * ``si_sdr`` — der skaleninvariante Term gegen den absoluten Amplitudenanker.
#:   In run 6 war genau dieses Gewicht das, was die Amplitude festhält; das
#:   typische Versagen hier ist ein ``energy_ratio`` weit weg von 1.
AXES: dict[str, dict[str, Achse]] = {
    "nested_gan": {
        "lr": Achse("lr", ("training", "learning_rate"), (1.5e-4, 5e-4, 1.5e-3)),
        "kapazitaet": Achse("ch", ("model", "kwargs", "inner_channels"), (32, 48, 72)),
        "si_sdr": Achse("sisdr", ("model", "loss_kwargs", "si_sdr_weight"), (0.0, 1.0, 3.0)),
        "wd": Achse("wd", ("training", "weight_decay"), (0.0, 1e-4, 1e-2)),
        "freq": Achse("freq", ("model", "loss_kwargs", "frequency_weight"), (0.015, 0.046, 0.14)),
        "bloecke": Achse("blk", ("model", "kwargs", "inner_blocks"), (2, 4, 6)),
    },
    "vit_spectrogram": {
        # embed_dim muss durch n_heads (6) teilbar bleiben: 120/6=20, 192/6=32,
        # 288/6=48. Ein nicht teilbarer Wert bricht die Attention beim Bauen.
        # 1,5e-4 ist die Basisrate des MAE-Papers; die Konfiguration von run 7
        # benutzt das Doppelte, ohne Warmup und ohne Cosine.
        "lr": Achse("lr", ("training", "learning_rate"), (1.5e-4, 3e-4, 1e-3)),
        "kapazitaet": Achse("dim", ("model", "kwargs", "embed_dim"), (120, 192, 288)),
        "si_sdr": Achse("sisdr", ("model", "loss_kwargs", "si_sdr_weight"), (0.0, 1.0, 3.0)),
        "wd": Achse("wd", ("training", "weight_decay"), (0.01, 0.05, 0.2)),
        "freq": Achse("freq", ("model", "loss_kwargs", "frequency_weight"), (0.015, 0.046, 0.14)),
        "tiefe": Achse("depth", ("model", "kwargs", "depth"), (4, 6, 10)),
    },
    "dhct_gan": {
        # Das Paper nennt 1e-4 (oder 1e-3 abfallend auf 1e-4); run 7 fährt 1e-3
        # konstant. Beide Werte stehen im Gitter.
        "lr": Achse("lr", ("training", "learning_rate"), (1e-4, 1e-3, 3e-3)),
        # Der Stamm des Papers ist 32 Kanäle breit, run 7 fährt die Hälfte.
        "kapazitaet": Achse("bc", ("model", "kwargs", "base_channels"), (8, 16, 32)),
        "si_sdr": Achse("sisdr", ("model", "loss_kwargs", "si_sdr_weight"), (0.0, 1.0, 3.0)),
        "wd": Achse("wd", ("training", "weight_decay"), (0.0, 1e-4, 1e-2)),
        "freq": Achse("freq", ("model", "loss_kwargs", "frequency_weight"), (0.015, 0.046, 0.14)),
        "fenster": Achse("win", ("model", "kwargs", "window_size"), (8, 16, 32)),
    },
    "demucs": {
        "lr": Achse("lr", ("training", "learning_rate"), (1e-4, 3e-4, 1e-3)),
        # Die Kanäle verdoppeln sich je Ebene, 96 bei depth 4 heißt 96-192-384-768.
        "kapazitaet": Achse("ic", ("model", "kwargs", "initial_channels"), (32, 64, 96)),
        "si_sdr": Achse("sisdr", ("model", "loss_kwargs", "si_sdr_weight"), (0.0, 1.0, 3.0)),
        "wd": Achse("wd", ("training", "weight_decay"), (0.0, 1e-4, 1e-2)),
        "freq": Achse("freq", ("model", "loss_kwargs", "frequency_weight"), (0.015, 0.046, 0.14)),
        "tiefe": Achse("depth", ("model", "kwargs", "depth"), (3, 4, 5)),
    },
}

DEFAULT_AXES = ("lr", "kapazitaet", "si_sdr")

#: Familie -> (Konfigurationspfad, Modell-ID der Pipeline). Die Modell-ID ist
#: der Schlüssel in ``DEPLOYMENT_SPECS``; sie entscheidet über Packung, Kontext
#: und Ausgabedeutung und darf hier nicht neu erfunden werden.
FAMILIES = {
    f: (REPO / f"src/facet/models/{f}_deployment_edition/training_niazy_proof_fit.yaml",
        f"{f}_deployment")
    for f in AXES
}

#: Abweichungen von der run-7-Konfiguration, je Familie: eigene Basiskonfiguration
#: und die Packung, mit der die Bewertung den Export lesen muss.
#:
#: ``dhct_gan`` faehrt in run 7 ``b1s`` -- eine Epoche, ein Kanal, also ohne jede
#: Vergleichsachse. Run 7 hat selbst gemessen, was das kostet: die Epochenachse lag
#: mit -0,473 +- 0,057 deutlich ueber der kontextlosen Basis mit -0,254 +- 0,082.
#: Ein Gitter ueber eine Konfiguration ohne Vergleichsachse optimiert die falsche
#: Sache, deshalb laeuft run 8 hier mit sieben Kontextepochen.
#:
#: **Die Packung muss mitwandern.** Trainiert der Lauf ``b1ts`` und liest die
#: Bewertung ihn als ``b1s``, dann schlaegt nichts fehl -- der Adapter reicht dem
#: Modell einfach die falsche Form und misst ein Ergebnis, das nichts bedeutet.
ABWEICHUNGEN: dict[str, dict[str, Any]] = {
    "dhct_gan": {"config": REPO / "configs/run8_dhct_gan_deployment_ctx_epochs.yaml",
                 "packing": "b1ts", "context": "stack",
                 "grund": "run 7 fuhr b1s (1 Epoche, 1 Kanal) -- keine Vergleichsachse"},
}

#: Dieselben vier Familien auf Weg A. Die Achsen und Stufen bleiben identisch,
#: damit sich die beiden Gitter direkt gegeneinander lesen lassen: derselbe
#: Suchraum, ein anderer Datensatz. Auch ``dhct_gan`` behaelt seine
#: ``batch_size 16`` -- sie war auf der 24-GB-Karte erzwungen und waere auf den
#: 48-GB-Slices nicht noetig, aber eine andere Batchgroesse waere eine zweite
#: Aenderung und machte den Vergleich der beiden Gitter unlesbar.
WEGA_ABWEICHUNGEN: dict[str, dict[str, Any]] = {
    f: {"config": REPO / f"configs/run8_wega_{f}.yaml",
        "grund": "Weg A: unabhaengige Clean-Quelle, Verlustgewichte neu hergeleitet"}
    for f in AXES
}
WEGA_ABWEICHUNGEN["dhct_gan"] |= {"packing": "b1ts", "context": "stack"}


def familie_aufloesen(family: str, datensatz: str = "proof_fit") -> tuple[Path, str, dict[str, Any]]:
    cfg, model_id = FAMILIES[family]
    tabelle = WEGA_ABWEICHUNGEN if datensatz == "wega" else ABWEICHUNGEN
    ab = tabelle.get(family, {})
    return Path(ab.get("config", cfg)), model_id, ab


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------


def lies(cfg: dict, pfad: tuple[str, ...]):
    """Der Wert, der vor der Gridsuche an dieser Stelle stand."""
    d = cfg
    for k in pfad:
        d = d[k]
    return d


def setze(cfg: dict, pfad: tuple[str, ...], wert) -> None:
    d = cfg
    for k in pfad[:-1]:
        d = d[k]
    if pfad[-1] not in d:
        raise KeyError(f"{'.'.join(pfad)} steht nicht in der Basiskonfiguration — "
                       f"vorhanden: {sorted(d)}")
    d[pfad[-1]] = wert


def baue_konfig(basis: dict, family: str, punkt: dict, seed: int,
                out_root: Path, device: str, max_epochs: int | None) -> tuple[dict, str]:
    tag = "_".join(f"{AXES[family][a].name}{v:g}" if isinstance(v, (int, float))
                   else f"{AXES[family][a].name}{v}" for a, v in punkt.items())
    tag = f"{tag}_s{seed}"
    cfg = copy.deepcopy(basis)
    for achse, wert in punkt.items():
        setze(cfg, AXES[family][achse].pfad, wert)
    cfg["model"]["device"] = device
    cfg["training"]["seed"] = int(seed)
    # Ein Pod hat 20 GB. save_top_k 2 plus last sind drei Dateien je Lauf und
    # bei 27 Läufen das, was zuerst volllaeuft -- gebraucht wird genau einer.
    cfg.setdefault("checkpoint", {})["save_top_k"] = 1
    cfg["checkpoint"]["save_last"] = False
    # Der Export der CLI traced die *letzte* Epoche und hängt am CPU-Trace-Fix,
    # der noch nicht im Repository steht. Beides umgeht dieses Skript, indem es
    # selbst aus dem besten Checkpoint exportiert.
    cfg.setdefault("export", {})["enabled"] = False
    cfg["training"]["model_name"] = f"grid_{family}_{tag}"
    cfg["training"]["output_dir"] = str(out_root / family / tag)
    if max_epochs is not None:
        cfg["training"]["max_epochs"] = int(max_epochs)
    # Der CLI-Export bleibt an, wird aber nicht benutzt: er traced die letzte
    # Epoche. Bewertet wird der beste Checkpoint, den dieses Skript selbst lädt.
    return cfg, tag


# ---------------------------------------------------------------------------
# Export aus dem besten Checkpoint
# ---------------------------------------------------------------------------


def _val_loss(p: Path) -> float:
    """Der Wert aus dem Dateinamen ``epoch0007_val_loss-0.7270.pt``.

    Nach Namen zu sortieren nähme die letzte Epoche, nicht die beste — und die
    Verluste sind hier negativ, das Minuszeichen gehört zur Zahl.
    """
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
        raise FileNotFoundError(f"kein Epochen-Checkpoint unter {run_dir}")
    return min(cks, key=_val_loss)


def exportiere(cfg_pfad: Path, ckpt: Path, ziel: Path) -> Path:
    """Baue das Modell so, wie der Trainer es gebaut hat, lade den Checkpoint, trace auf CPU.

    Gebaut wird über die Bausteine der CLI selbst — derselbe Datensatzaufbau,
    dieselbe Kwarg-Injektion. Die Eingabeform je Beispiel steht nicht im
    Datensatzarchiv, sie entsteht erst in der ``dataset_factory``; sie hier zu
    raten heißt, bei einer der vier Familien danebenzuliegen und es nicht zu
    merken, weil ``torch.jit.trace`` jede Form annimmt.

    Und gebaut wird aus den Kwargs *dieses* Laufs, nicht aus den Fabrikvorgaben:
    eine Vorgabe, die von der trainierten Architektur abweicht, macht den
    Checkpoint still unladbar. Genau das ist in run 7 passiert, als die
    Kontextkonfigurationen die ``model.kwargs`` der Familie verloren haben.
    """
    import torch

    from facet.training.cli import (_build_dataset, _build_model, _load_contexts,
                                    load_training_cli_config)

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

    # Ein Modell mit mehreren Köpfen lässt sich nicht unverändert tracen -- und
    # das Modell, nicht der Exporter, weiß welcher Kopf der ausgelieferte ist.
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
        # Ohne Autograd: nn.MultiheadAttention nimmt sonst einen anderen internen
        # Pfad und torch.jit's eigene Graphprüfung schlägt scheinbar fehl.
        scripted = torch.jit.trace(ziel_modul, beispiel)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(ziel))
    return ziel


# ---------------------------------------------------------------------------
# Bewertung in der Kette
# ---------------------------------------------------------------------------


def bewerte_in_pipeline(ts_pfad: Path, model_id: str, edf: Path, device: str,
                        dc_mode: str, abweichung: dict | None = None) -> dict:
    """Eine volle Korrekturkette mit genau diesem Export, dann die zwei Kennzahlen.

    Der Adapter ist die geprüfte Implementierung aus ``family_adapters``; nur das
    Laden des Modells wird umgebogen, weil dort ein Glob auf die Läufe von run 7
    steht und nicht auf ein Gitterverzeichnis.
    """
    import mne
    import numpy as np
    import torch

    import reference_chain
    from family_adapters import FamilyAdapter
    from facet.core import ProcessingContext, ProcessingMetadata
    from facet.correction.deep_learning import DeepLearningCorrection
    from facet.evaluation import (EpochSeamStepCalculator,
                                  GradientArtifactResidualCalculator)

    class GitterAdapter(FamilyAdapter):
        def _load_model(self):
            if self._model is None:
                m = torch.jit.load(str(ts_pfad), map_location=self.device)
                m.eval()
                self._model = m
            return self._model, torch

    # Erst auf dem gewuenschten Geraet, bei Fehlschlag auf der CPU. Die Spur
    # entsteht auf der CPU und laeuft bei den meisten Familien trotzdem auf der
    # GPU -- aber nicht bei allen: Demucs' LSTM traegt seinen Startzustand als
    # CPU-Konstante im Graphen, und `torch.lstm` bekommt dann CUDA-Eingaben und
    # einen CPU-Zustand. Der Rueckfall geht auf die *genauere* Seite, denn die
    # CPU ist der Bezug und die GPU die Naeherung (gemessen 1,2e-5 relativ).
    # Welches Geraet gerechnet hat, steht in der Ergebniszeile -- ein stiller
    # Wechsel des Messgeraets waere sonst nirgends nachlesbar.
    benutzt = device
    for versuch in (device, "cpu"):
        adapter = GitterAdapter(model_id, device=versuch, dc_mode=dc_mode)
        if abweichung:
            import dataclasses
            felder = {k: v for k, v in abweichung.items() if k in ("packing", "context")}
            if felder:
                adapter.packing = dataclasses.replace(adapter.packing, **felder)
        pipe = reference_chain.build(edf, correctors=[DeepLearningCorrection(model=adapter)],
                                     include_pca=True, name=model_id)
        try:
            res = pipe.run()
        except Exception:
            if versuch == "cpu":
                raise
            print(f"    {model_id}: {versuch} traegt diesen Export nicht, weiche auf die CPU aus",
                  flush=True)
            continue
        if res.success:
            benutzt = versuch
            break
        if versuch == "cpu":
            raise RuntimeError(f"Pipeline fehlgeschlagen: {res.error}")
        print(f"    {model_id}: {versuch} traegt diesen Export nicht ({str(res.error)[:80]}), "
              f"weiche auf die CPU aus", flush=True)

    raw = res.context.get_raw()
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
    sf = float(raw.info["sfreq"])
    i0, i1 = int(WINDOW_START_S * sf), int(WINDOW_STOP_S * sf)
    data = raw.get_data(picks=picks)[:, i0:i1].astype(np.float64)
    namen = [raw.ch_names[i] for i in picks]
    trg = np.asarray(res.context.get_triggers() if res.context.has_triggers() else [],
                     dtype=np.int64)
    trg = trg[(trg >= i0) & (trg < i1)] - i0

    fenster = mne.io.RawArray(data, mne.create_info(namen, sf, "eeg"), verbose=False)
    ctx = ProcessingContext(raw=fenster,
                            metadata=ProcessingMetadata(triggers=list(map(int, trg))))
    comb = GradientArtifactResidualCalculator(tmin=T0_IN_WINDOW, tmax=T1_IN_WINDOW)\
        .execute(ctx).metadata.custom["gradient_artifact_residual"]
    seam = EpochSeamStepCalculator(tmin=T0_IN_WINDOW, tmax=T1_IN_WINDOW)\
        .execute(ctx).metadata.custom["epoch_seam_step"]
    return {"pipeline_device": benutzt,
            "ga_rest_uv": round(comb["comb_rms_uv"], 4),
            "rms_uv": round(comb["rms_uv"], 4),
            "naht_ratio": round(seam["ratio"], 4),
            "x_farm": round(comb["comb_rms_uv"] / FARM_GA_REST_UV, 2)}


def bewerte_auf_selektion(cfg_pfad: Path, ckpt: Path, device: str) -> dict:
    """Bewerte einen Weg-A-Lauf auf dem **Selektionssplit**, in Mikrovolt.

    Nicht am Trainingsverlust und nicht in der Korrekturkette. Weg A bringt seinen
    eigenen Massstab mit, und er ist physikalisch lesbar:

    ``err_uv``
        RMS des Rekonstruktionsfehlers ``clean_hat - clean``.
    ``corr_clean``
        Korrelation des wiederhergestellten mit dem wahren sauberen Signal.
    ``spike_ratio``
        Median von ``|clean_hat| / |clean|`` an den markierten Spike-Spitzen.
        1,0 ist richtig; darunter wird die Spitze gedaempft, darueber ueberhoeht.
        Der Kammwert dieses Projekts kann das nicht sehen -- ein Modell, das
        Transienten glattbuegelt, sieht dort gut aus.
    ``err_uv_null``
        derselbe Fehler fuer die **Nullausgabe**. Bei einem Artefakt, das 41-mal
        ueber dem EEG liegt, ist "alles wegwerfen" ein starker Arm; eine Zahl
        ohne diesen Vergleich sagt nichts.

    **Der Selektionssplit, nicht der gesperrte Holdout.** ``example_split`` kennt
    beide: 1 ist zum Auswaehlen da, 2 wird einmal angefasst, wenn feststeht, was
    berichtet wird. Eine Konfiguration auf derselben Menge auszuwaehlen und zu
    berichten, gibt eine Zahl zurueck, die die Auswahl bereits optimiert hat --
    genau dafuer wurde der Holdout herausgeschnitten.
    """
    import numpy as np
    import torch

    from facet.training.cli import (_build_dataset, _build_model,
                                    load_training_cli_config)

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
    # **Den Verlust fragen, nicht die Konfiguration.** ``target_type`` sagt, was der
    # *Kern* ausgibt; der Deployment-Wrapper wandelt das um, und alle Editionen
    # liefern am Ende das Artefakt. ``vit_spectrogram`` steht auf
    # ``target_type: clean``, sein Verlust aber auf ``prediction_is: artifact``.
    # Wer sich auf die Konfiguration verlaesst, rechnet dort ``clean_hat`` aus der
    # falschen Groesse -- ohne Fehlermeldung, mit einem Ergebnis, das nach einem
    # kaputten Modell aussieht statt nach einer kaputten Messung.
    from facet.training.cli import _import_object
    verlust = _import_object(cli_cfg.model.loss_factory)(
        **cli_cfg.model.loss_kwargs, sfreq=ds.sfreq)
    sagt_artefakt = getattr(verlust, "prediction_is", "artifact") == "artifact"

    fehler, fehler_null, cl, ch, quoten = [], [], [], [], []
    with torch.no_grad():
        for anfang in range(0, len(selektion), 64):
            paare = [selektion[i] for i in range(anfang, min(anfang + 64, len(selektion)))]
            x = torch.from_numpy(np.stack([p[0] for p in paare])).to(device)
            y = np.stack([p[1] for p in paare])
            pred = modell(x).cpu().numpy()
            artefakt, clean, noisy = y[:, 0], y[:, 1], y[:, 2]
            clean_hat = (noisy - pred) if sagt_artefakt else pred
            fehler.append(clean_hat - clean)
            fehler_null.append(-clean)          # Nullausgabe: clean_hat = 0
            cl.append(clean); ch.append(clean_hat)

    f = np.concatenate([a.ravel() for a in fehler])
    fn = np.concatenate([a.ravel() for a in fehler_null])
    c = np.concatenate([a.ravel() for a in cl])
    h = np.concatenate([a.ravel() for a in ch])
    U = 1e6
    return {"err_uv": round(float(np.sqrt((f ** 2).mean())) * U, 4),
            "err_uv_null": round(float(np.sqrt((fn ** 2).mean())) * U, 4),
            "corr_clean": round(float(np.corrcoef(h, c)[0, 1]), 4),
            "besser_als_null": bool(np.sqrt((f ** 2).mean()) < np.sqrt((fn ** 2).mean())),
            "n_selektion": len(selektion)}


def hinweis(naht: float) -> str:
    if naht >= SEAM_BROKEN:
        return "verworfen: Nahtsprung"
    if naht >= SEAM_SUSPECT:
        return "auffaellig: Nahtsprung"
    return ""


# ---------------------------------------------------------------------------
# Durchlauf
# ---------------------------------------------------------------------------


def fuehre_punkt_aus(family: str, basis: dict, punkt: dict, seed: int, args) -> dict:
    cfg, tag = baue_konfig(basis, family, punkt, seed, args.out_root, args.device,
                           args.max_epochs)
    out_root = args.out_root / family
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_pfad = out_root / f"{tag}.yaml"
    cfg_pfad.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    zeile = {"tag": tag, "seed": seed, **{a: punkt[a] for a in punkt}}
    if args.dry_run:
        return zeile | {"status": "dry-run"}

    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "facet.training.cli", "fit",
                           "--config", str(cfg_pfad)],
                          capture_output=True, text=True, cwd=str(REPO))
    zeile["train_s"] = round(time.perf_counter() - t0, 1)
    if proc.returncode != 0:
        return zeile | {"status": "train-fehler",
                        "fehler": (proc.stderr or proc.stdout)[-600:]}

    try:
        ckpt = bester_checkpoint(Path(cfg["training"]["output_dir"]))
        zeile["val_loss"] = round(_val_loss(ckpt), 5)
        zeile["checkpoint"] = str(ckpt.relative_to(REPO)) if ckpt.is_relative_to(REPO) else str(ckpt)
    except Exception as e:                       # noqa: BLE001 - die Zeile soll die Ursache tragen
        return zeile | {"status": "checkpoint-fehler", "fehler": f"{type(e).__name__}: {e}"}

    if args.dataset == "wega":
        # Kein TorchScript-Export: bewertet wird direkt aus dem Checkpoint. Der
        # Export ist fuer die Pipeline noetig, und die laeuft hier nicht -- Weg A
        # bringt seinen eigenen Massstab mit. Weniger Schritte, weniger Fehlerarten.
        try:
            t1 = time.perf_counter()
            zeile |= bewerte_auf_selektion(cfg_pfad, ckpt, args.pipeline_device)
            zeile["bewertung_s"] = round(time.perf_counter() - t1, 1)
        except Exception as e:                   # noqa: BLE001
            return zeile | {"status": "bewertung-fehler", "fehler": f"{type(e).__name__}: {e}"}
        zeile["hinweis"] = "" if zeile["besser_als_null"] else "schlechter als die Nullausgabe"
        zeile["status"] = "ok" if zeile["besser_als_null"] else "verworfen"
        return zeile

    try:
        ts = exportiere(cfg_pfad, ckpt, out_root / f"{tag}.ts")
    except Exception as e:                       # noqa: BLE001
        return zeile | {"status": "export-fehler", "fehler": f"{type(e).__name__}: {e}"}

    try:
        t1 = time.perf_counter()
        _, model_id, abweichung = familie_aufloesen(family, args.dataset)
        zeile |= bewerte_in_pipeline(ts, model_id, args.edf,
                                     args.pipeline_device, args.dc_mode, abweichung)
        zeile["pipeline_s"] = round(time.perf_counter() - t1, 1)
    except Exception as e:                       # noqa: BLE001
        return zeile | {"status": "pipeline-fehler", "fehler": f"{type(e).__name__}: {e}"}

    zeile["hinweis"] = hinweis(zeile["naht_ratio"])
    zeile["status"] = "verworfen" if zeile["naht_ratio"] >= SEAM_BROKEN else "ok"
    return zeile


def schreibe(pfad: Path, kopf: dict, zeilen: list[dict]) -> None:
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps({**kopf, "zeilen": zeilen}, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def zeige(zeile: dict) -> None:
    if "err_uv" in zeile:
        print(f"    Fehler {zeile['err_uv']:7.2f} µV (Null {zeile['err_uv_null']:7.2f}) | "
              f"corr {zeile['corr_clean']:+.3f} | val_loss {zeile.get('val_loss', float('nan')):8.4f}"
              f"{'  ' + zeile['hinweis'] if zeile.get('hinweis') else ''}", flush=True)
        return
    if zeile.get("status") == "ok" or zeile.get("hinweis"):
        print(f"    GA-Rest {zeile['ga_rest_uv']:6.2f} µV ({zeile['x_farm']:5.1f}x FARM) | "
              f"Naht {zeile['naht_ratio']:5.2f} | val_loss {zeile.get('val_loss', float('nan')):8.4f}"
              f"{'  ' + zeile['hinweis'] if zeile.get('hinweis') else ''}", flush=True)
    else:
        print(f"    {zeile.get('status')}: {str(zeile.get('fehler'))[:200]}", flush=True)


def rangliste(zeilen: list[dict]) -> list[dict]:
    """Nur gewertete Punkte, nach der Zielgroesse des jeweiligen Datensatzes.

    Proof-Fit: Kammrest, Nahtsprung >= 2,5 fliegt raus. Weg A: Rekonstruktions-
    fehler in µV, und wer schlechter ist als die Nullausgabe, fliegt raus.
    """
    ok = [z for z in zeilen if z.get("status") == "ok"]
    schluessel = "err_uv" if ok and "err_uv" in ok[0] else "ga_rest_uv"
    return sorted(ok, key=lambda z: z[schluessel])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("screen", "confirm"))
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--dataset", choices=("proof_fit", "wega"), default="proof_fit",
                   help="proof_fit: run-7-Datensatz, bewertet in der Korrekturkette. "
                        "wega: unabhaengige Clean-Quelle, bewertet auf dem Selektionssplit "
                        "in µV -- der gesperrte Holdout bleibt unangetastet.")
    p.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES),
                   help=f"Achsen aus AXES[family]; Vorgabe {' '.join(DEFAULT_AXES)}")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="nur für confirm: Wiederholungen je Kandidat")
    p.add_argument("--screen-seed", type=int, default=42)
    p.add_argument("--from", dest="quelle", type=Path, default=None,
                   help="confirm: das Screening-JSON, aus dem die Kandidaten kommen")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="überschreibt die Basis; für ein Probegitter sinnvoll, sonst nicht")
    p.add_argument("--device", default="cuda", help="Training")
    p.add_argument("--pipeline-device", default="cuda",
                   help="Bewertung des Exports. Der Export wird auf der CPU getraced "
                        "(portabel), laeuft aber auch auf der GPU: gemessen betraegt der "
                        "Unterschied beider Wege 1,2e-5 relativ, also Float-Rauschen weit "
                        "unter der Aufloesung des Kammwerts. Auf der CPU kostet die "
                        "Bewertung rund 15 min je Punkt, auf der GPU rund 2,5 -- bei 27 "
                        "Punkten ist das der Unterschied zwischen zwei Naechten und einer.")
    p.add_argument("--dc-mode", default="as_evaluated",
                   choices=("as_evaluated", "reconcile", "segment_mean"))
    p.add_argument("--edf", type=Path, default=REPO / "examples/datasets/NiazyFMRI.edf")
    p.add_argument("--out-root", type=Path, default=REPO / "grids/run7")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None,
                   help="nur die ersten N Punkte -- fuer einen Probelauf auf einem frischen Pod")
    p.add_argument("--dry-run", action="store_true",
                   help="schreibt nur die Konfigurationen und zeigt das Gitter")
    args = p.parse_args()

    family = args.family
    cfg_pfad, model_id, abweichung = familie_aufloesen(family, args.dataset)
    basis = yaml.safe_load(cfg_pfad.read_text(encoding="utf-8"))
    if abweichung:
        packung = abweichung.get("packing")
        print(f"  ABWEICHUNG von run 7: {abweichung.get('grund','')}\n"
              f"  Konfiguration {cfg_pfad.relative_to(REPO)}"
              + (f", Packung {packung} statt der run-7-Packung" if packung else "") + "\n")
    if args.dataset == "proof_fit" and not args.edf.exists() and not args.dry_run:
        raise SystemExit(f"{args.edf} fehlt — ohne die Aufnahme gibt es keine "
                         f"Pipeline-Bewertung, und dann ist diese Suche sinnlos.")

    unbekannt = [a for a in args.axes if a not in AXES[family]]
    if unbekannt:
        raise SystemExit(f"unbekannte Achsen {unbekannt} für {family}; "
                         f"bekannt: {sorted(AXES[family])}")

    # Jeder Shard schreibt seine eigene Datei. Vier Prozesse auf denselben Pfad
    # heisst: der zuletzt Schreibende gewinnt, und die Ergebnisse der anderen drei
    # sind weg -- ohne Fehlermeldung, denn jeder einzelne Schreibvorgang gelingt.
    teil = f"_shard{args.shard}" if args.n_shards > 1 else ""

    if args.phase == "screen":
        werte = [AXES[family][a].werte for a in args.axes]
        gitter = [dict(zip(args.axes, kombi, strict=True))
                  for kombi in itertools.product(*werte)]
        aufgaben = [(g, args.screen_seed) for g in gitter]
        ziel = args.out_root / f"grid_{family}_screen{teil}.json"
    else:
        if args.quelle is None:
            raise SystemExit("confirm braucht --from <screening.json>")
        vorher = json.loads(args.quelle.read_text(encoding="utf-8"))
        kandidaten = rangliste(vorher["zeilen"])[:args.top_k]
        if not kandidaten:
            raise SystemExit(f"{args.quelle} enthält keinen gewerteten Punkt")
        achsen = vorher["achsen"]
        aufgaben = [({a: k[a] for a in achsen}, s)
                    for k in kandidaten for s in args.seeds]
        args.axes = achsen
        ziel = args.out_root / f"grid_{family}_confirm{teil}.json"
        print("Kandidaten aus dem Screening:")
        for k in kandidaten:
            print(f"  {k['tag']:40s} GA-Rest {k['ga_rest_uv']:.2f} µV  Naht {k['naht_ratio']:.2f}")
        print()

    meine = [t for i, t in enumerate(aufgaben) if i % args.n_shards == args.shard]
    if args.limit is not None:
        meine = meine[:args.limit]
    # Was vorher stand, wandert mit in die Ergebnisdatei: ohne den Ausgangswert
    # ist später nicht mehr entscheidbar, ob ein Gitterpunkt eine Verbesserung
    # gegenüber run 7 ist oder nur ein anderer Punkt.
    basis_werte = {a: lies(basis, AXES[family][a].pfad) for a in AXES[family]}
    kopf = {"familie": family, "phase": args.phase, "datensatz": args.dataset,
            "achsen": list(args.axes),
            "basis_werte_run7": basis_werte,
            "gitterwerte": {a: list(AXES[family][a].werte) for a in args.axes},
            "fest_geblieben": {a: v for a, v in basis_werte.items() if a not in args.axes},
            "basis_konfig": str(cfg_pfad.relative_to(REPO)), "model_id": model_id,
            "abweichung_von_run7": {k: (str(v) if isinstance(v, Path) else v)
                                    for k, v in abweichung.items()} or None,
            "fenster_s": [WINDOW_START_S, WINDOW_STOP_S],
            "analyse_s": [T0_IN_WINDOW, T1_IN_WINDOW],
            "farm_ga_rest_uv": FARM_GA_REST_UV,
            "naht_verworfen_ab": SEAM_BROKEN, "dc_mode": args.dc_mode,
            "bewertung": "Kammrest in der vollen Korrekturkette; Nahtsprung als Ausschluss. "
                         "Nicht val_loss — der erklaert in run 7 nur 37 % der Pipeline-Rangvarianz."}

    print(f"{family}: {len(aufgaben)} Punkte gesamt, Shard {args.shard}/{args.n_shards} "
          f"-> {len(meine)}")
    print(f"  {'Achse':12s} {'Pfad in der Konfiguration':40s} {'run 7':>10s}   Gitter")
    for a in args.axes:
        ach = AXES[family][a]
        vorher = basis_werte[a]
        print(f"  {a:12s} {'.'.join(ach.pfad):40s} {vorher!s:>10s}   {list(ach.werte)}")
    fest = [f"{a}={basis_werte[a]}" for a in AXES[family] if a not in args.axes]
    if fest:
        print(f"  unveraendert aus run 7: {', '.join(fest)}")
    if args.dataset == "wega":
        print("  Bewertet wird der Rekonstruktionsfehler in µV auf dem Selektionssplit, "
              "gegen die Nullausgabe.\n  Der gesperrte Holdout bleibt unberuehrt.\n", flush=True)
    else:
        print(f"  FARM liegt bei {FARM_GA_REST_UV:.2f} µV, das beste ungetunte Modell "
              f"(nested_gan) bei 1,40 µV\n", flush=True)

    zeilen: list[dict] = []
    for i, (punkt, seed) in enumerate(meine, start=1):
        beschriftung = ", ".join(f"{a}={punkt[a]:g}" if isinstance(punkt[a], (int, float))
                                 else f"{a}={punkt[a]}" for a in args.axes)
        print(f"[{i}/{len(meine)}] {beschriftung}, seed {seed}", flush=True)
        zeile = fuehre_punkt_aus(family, basis, punkt, seed, args)
        zeige(zeile)
        zeilen.append(zeile)
        schreibe(ziel, kopf, zeilen)

    beste = rangliste(zeilen)
    if beste:
        b = beste[0]
        print(f"\nbestes gewertetes: {b['tag']} -> GA-Rest {b['ga_rest_uv']:.2f} µV "
              f"({b['x_farm']:.1f}x FARM), Naht {b['naht_ratio']:.2f}, "
              f"val_loss {b.get('val_loss', float('nan')):.4f}")
        nach_loss = min((z for z in zeilen if "val_loss" in z), key=lambda z: z["val_loss"], default=None)
        if nach_loss and nach_loss["tag"] != b["tag"]:
            print(f"nach val_loss haette man {nach_loss['tag']} genommen "
                  f"(GA-Rest {nach_loss.get('ga_rest_uv', float('nan')):.2f} µV) — "
                  f"die beiden Ordnungen stimmen hier nicht ueberein.")
    else:
        print("\nkein Punkt gewertet — siehe die Fehlerfelder in " + str(ziel))
    print(f"geschrieben: {ziel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
