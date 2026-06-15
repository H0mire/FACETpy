# Research / Run-Pläne — Überblick

Konsolidierter Index der Run-Pläne (Stand 2026-06-11). Zwei Stränge:

- **Engineering** — Modelle korrekt bauen & trainieren.
- **Wissenschaft** — von AAS lösen und AAS schlagen (das Thesis-Ziel).

## Aktive Pläne

| Run | Titel | Strang | Status | Datei |
|---|---|---|---|---|
| — | Run 1: Exploration der 12 Originale | Engineering | ✅ ausgeführt (Mai 2026) | `thesis_results_report.md` |
| **run_2** | Bugfixes der Run-1-Originale | Engineering | ⛔ **superseded** (durch Editionen + run_4) | [`run_2_plan.md`](run_2_plan.md) |
| **run_3** | Weg A — von AAS entkoppelter Datensatz | Wissenschaft | 📋 geplant (Daten-Fundament) | [`run_3_decoupled_dataset_weg_a.md`](run_3_decoupled_dataset_weg_a.md) |
| **run_4** | Fleet-Training der paper-accurate Editionen | Engineering | 📋 geplant (braucht Pods) | [`run_4_paper_accurate_editions_fleet.md`](run_4_paper_accurate_editions_fleet.md) |
| **run_5** | Weg B — self-supervised, label-frei | Wissenschaft | 📋 geplant | [`run_5_self_supervised_weg_b.md`](run_5_self_supervised_weg_b.md) |
| **run_6** | Beat-AAS: Spike-Preservation | Wissenschaft | 📋 geplant (Zielpunkt; baut auf run_3) | [`run_6_beat_aas_spike_preservation.md`](run_6_beat_aas_spike_preservation.md) |

**Abhängigkeiten:** run_3 (Weg A, inkl. Spike-Injektions-Modus) ist Voraussetzung
für run_5 (Weg B) **und** run_6 (Spike). run_4 (Fleet) liefert die trainierten
DL-Modelle, die in run_6 gegen AAS antreten. run_3 und run_5 sind die zwei
Decoupling-Wege (A = unabhängiges clean; B = ganz ohne Target).

## Umnummerierung (2026-06-11)

Die Serie war verheddert (alter Arc 2/3 + Ops 4 + Decoupling 5/6). Neu sortiert
nach Abhängigkeit; `run_2`/`run_4`-Dateinamen blieben stabil (historische
Report-Links bzw. 24 Modell-YAMLs zeigen darauf):

| früher | jetzt |
|---|---|
| run_3_plan.md (Spike) | run_6_beat_aas_spike_preservation.md |
| run_5_decoupled_dataset_weg_a.md (Weg A) | run_3_decoupled_dataset_weg_a.md |
| run_6_self_supervised_weg_b.md (Weg B) | run_5_self_supervised_weg_b.md |
| run_2_plan.md (Bugfix) | unverändert, als superseded markiert |
| run_4_…fleet.md | unverändert |

## Begleitende Dokumente

- [`dl_training_lessons.md`](dl_training_lessons.md) — stehende Engineering-Lehren
  + Sanity-Checklisten (aus run_2 §3/§5/§7 ausgelagert).
- [`thesis_results_report.md`](thesis_results_report.md) — was in Run 1 passiert ist.
- [`architecture_catalog.md`](architecture_catalog.md) — Menü der Modellfamilien.
- `../deep_learning_parallel_runpod_workflow.md` — GPU-Fleet-Operator-Guide.
