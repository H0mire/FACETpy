# Deep-Learning Artifact Correction — Requirements

**Status:** Draft
**Owner:** Janik Müller
**Context:** `feature/add-deeplearning` branch, master's thesis
**Last updated:** 2026-06-09

---

## 1. Purpose

This document specifies the requirements needed to take the deep-learning (DL)
artifact-correction subsystem of FACETpy from its current state — a well-built
contract layer plus a fleet of proof-of-concept models — to a state where it can
**substantiate the thesis claim** that DL model families are a better alternative
to Averaged Artifact Subtraction (AAS) and linear methods for correcting
EEG-fMRI artifacts (gradient, BCG, and future artifact types).

The architecture and contracts are already strong. These requirements therefore
focus on the **experimental rigor and contributor tooling** gaps that currently
prevent the thesis claim from being evidenced, not on rebuilding the framework.

## 2. Background (current state)

- A stable contract layer exists in `src/facet/correction/deep_learning.py`
  (`DeepLearningModelSpec`, `DeepLearningModelAdapter`,
  `EpochContextArtifactAdapter`, `DeepLearningCorrection`).
- A framework-agnostic training stack exists in `src/facet/training/`
  (`Trainer`, `TrainableModelWrapper`, dataset, losses, config, callbacks, CLI).
- 15 model packages live under `src/facet/models/<id>/`, all conforming to the
  prescribed layout and routing correction through `DeepLearningCorrection`.
- Evaluation outputs are standardized via `ModelEvaluationWriter`
  (`evaluation_manifest.json`, `metrics.json`, `evaluation_summary.md`).
- GPU-fleet (RunPod) tooling exists under `tools/gpu_fleet/` for parallel training.

**Key gap:** the headline `niazy_proof_fit` dataset uses AAS-corrected EEG as the
"clean" target (`noisy = AAS_clean + AAS_artifact`), so models are trained to
*reproduce* AAS and are upper-bounded by it. No evaluation currently compares a
DL model head-to-head against an AAS/classical baseline on a target that is
independent of the method being beaten.

## 3. Goals

- **G1** — Produce evidence that can support or refute "DL > AAS" on a benchmark
  with a true, method-independent clean reference.
- **G2** — Make every reported metric traceable to an explicit ground-truth
  provenance, so no number can be read without knowing what "clean" meant.
- **G3** — Guarantee reproducibility of reported results (seeding, resumability,
  run manifests).
- **G4** — Lower the cost for future contributors to add and *correctly evaluate*
  a new model, while preventing unvalidated models from being mistaken for
  validated ones.

## 4. Non-goals

- **NG1** — Adding new model architectures. The existing fleet is sufficient to
  search the method space; no 16th architecture is required to evidence the thesis.
- **NG2** — Real-time / streaming inference performance work.
- **NG3** — Changing the core `DeepLearningModelAdapter` / `DeepLearningModelSpec`
  contract surface, which is considered stable and adequate.
- **NG4** — Clinical validation or regulatory concerns.

---

## 5. Requirements

Priority: **P0** (blocks the thesis claim) · **P1** (materially strengthens it) ·
**P2** (quality / future contributors).

### 5.1 Evaluation & scientific evidence

- **R1 (P0) — Baseline-comparison evaluation.**
  The evaluation path MUST compute metrics for AAS (and SHOULD include OBS and PCA)
  on the *same* evaluation pairs as the DL model. `ModelEvaluationWriter` output
  MUST include a "vs. classical baseline" section so every `evaluation_summary.md`
  reports DL and baseline metrics side by side.
  - *Acceptance:* a single command produces a table with `clean_snr_db_after`,
    `residual_rms_ratio`, and `artifact_corr` for {DL model, AAS, OBS, PCA} on one
    benchmark, with the delta (DL − AAS) highlighted.

- **R2 (P0) — True-ground-truth benchmark.**
  At least one evaluation benchmark MUST use a clean reference that is independent
  of AAS (synthetic-spike path: real clean EEG + independently-sourced artifact,
  where the underlying clean signal is genuinely known). Thesis claims of
  outperforming AAS MUST be reported on this benchmark, not on `niazy_proof_fit`.
  - *Acceptance:* a documented benchmark dataset whose metadata marks the clean
    source as method-independent, with at least one model + baseline comparison run.

- **R3 (P1) — Ground-truth provenance as a first-class field.**
  Dataset metadata MUST carry a mandatory provenance enum (e.g.
  `surrogate_aas` / `synthetic_true_clean` / `real_reference`) and the evaluation
  manifest MUST surface it. The summary MUST refuse to omit it.
  - *Acceptance:* loading a dataset without a provenance value fails fast; every
    `evaluation_manifest.json` contains the provenance of its evaluation set.

- **R4 (P1) — Per-artifact-type coverage.**
  The evaluation framework MUST distinguish results by artifact type (gradient vs.
  BCG vs. future) so the thesis can report per-artifact correction gain rather than
  a single aggregate.

- **R5 (P2) — Honest result framing for proof-fit runs.**
  Any evaluation run against a surrogate (AAS-derived) target MUST be labeled as a
  proof-fit / can-it-learn sanity check in its summary, never presented as a
  generalization result.

### 5.2 Reproducibility

- **R6 (P0) — Framework-level seeding.**
  `config.seed` MUST seed the framework RNGs (`torch.manual_seed`,
  `torch.cuda.manual_seed_all`, `tf.random.set_seed`) before model instantiation,
  in addition to the existing numpy batch-shuffle seeding. Weight init and dropout
  MUST be reproducible from config.
  - *Acceptance:* two runs with identical config and seed produce identical
    initial weights and matching loss curves (within nondeterministic-kernel
    tolerance, documented).

- **R7 (P1) — Resume-from-checkpoint.**
  The trainer/CLI MUST support resuming an interrupted run from a saved checkpoint
  (model + optimizer + scheduler + epoch counter), so long GPU-fleet jobs survive
  interruption.
  - *Acceptance:* `facet-train ... --resume <checkpoint>` continues from the
    correct epoch with restored optimizer/scheduler state.

- **R8 (P1) — Leakage-safe split default.**
  The train/val split MUST default to a leakage-safe mode. `split_mode="random"`
  with overlapping windows MUST NOT be the default; when only one recording is
  present, random splitting of overlapping windows MUST be refused or warned loudly.
  - *Acceptance:* default config on a single recording produces a contiguous,
    guard-banded split; choosing `random` requires explicit opt-in.

- **R9 (P2) — Run manifest completeness.**
  The training `summary.json` SHOULD record the resolved seed, dataset provenance,
  git commit, and (if used) the W&B run id, so a local run links unambiguously to
  its tracked experiment.

### 5.3 Contributor tooling & model fleet

- **R10 (P1) — Shared evaluation harness.**
  A reusable evaluation harness MUST exist so a model author supplies only "how to
  run my checkpoint," not a re-implemented arg-parser, device setup, SNR/RMS/corr
  metrics, and writer plumbing. The per-model `evaluate.py` boilerplate
  (~10–15% duplication today) SHOULD be removed in favor of this harness.
  - *Acceptance:* a new model's evaluation script is < ~40 lines and delegates all
    metric computation and writing to the shared harness.

- **R11 (P2) — "New model" template + canonical dataset.**
  A reference end-to-end template (adapter + training factory + config + minimal
  docs) MUST exist alongside one canonical small dataset, enabling a contributor to
  stand up a new model against the contract quickly.

- **R12 (P1) — Supported vs. experimental model tiers.**
  Models MUST be tiered. A small "supported" set (validated against a baseline on a
  true-ground-truth benchmark) and an "experimental / proof-fit only" set (the
  remainder) MUST be explicitly marked, following the existing frozen `demo01`
  pattern, so unvalidated models are not mistaken for safe templates.
  - *Acceptance:* each model README states its tier; the models index lists tiers.

- **R13 (P2) — Keep the contract layer generous.**
  The shared core (specs, adapters, base classes, helpers) SHOULD continue to
  absorb genuinely-reusable logic, but model-specific assumptions MUST remain in
  model folders per the existing `models/README.md` governance.

---

## 6. Acceptance criteria (thesis-level)

The subsystem meets its purpose when:

1. There exists at least one benchmark with a method-independent clean reference
   (R2) carrying explicit provenance (R3).
2. A single reproducible command (R6) produces a head-to-head table of the
   supported models vs. AAS/OBS/PCA on that benchmark (R1, R12), broken down by
   artifact type (R4).
3. Every reported number is traceable to its ground-truth provenance and the git
   commit / seed that produced it (R3, R9).
4. A new contributor can add and correctly evaluate a model without copying
   evaluation boilerplate (R10, R11).

## 7. Priority summary

| Priority | Requirements |
|----------|--------------|
| **P0** — blocks thesis claim | R1, R2, R6 |
| **P1** — materially strengthens | R3, R4, R7, R8, R10, R12 |
| **P2** — quality / contributors | R5, R9, R11, R13 |

## 8. Suggested sequencing

1. **R6 + R8** (seeding, leakage-safe split) — small, mechanical, unblock trustworthy numbers.
2. **R2 + R3** (true-GT benchmark + provenance) — establish what "clean" means.
3. **R1 + R10** (baseline comparison via shared harness) — the core evidence deliverable.
4. **R12 + R4** (tiering + per-artifact breakdown) — frame the result set.
5. **R7, R9, R5, R11, R13** — robustness and contributor polish.

---

## 9. Open questions

- Which independently-sourced fMRI artifact library backs the synthetic true-clean
  benchmark, and is its montage/sfreq compatible with the supported models?
- For BCG, is there a comparable true-ground-truth path, or only a surrogate?
- Which 3–5 models enter the "supported" tier, and on what selection metric?
