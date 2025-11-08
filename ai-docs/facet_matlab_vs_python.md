# Implementation Design: Bringing FACETpy to MATLAB Parity

## Background

FACETpy is a refactoring effort that aims to reproduce the mature EEG-fMRI artifact correction workflow of the legacy MATLAB toolbox. The MATLAB edition has served as the benchmark for signal quality and diagnostic coverage since 2012, while the current Python codebase (January 2025 snapshot) is still converging on feature parity. This document captures a comparative analysis between both editions and enumerates the implementation work needed for FACETpy to reach the MATLAB benchmark in terms of correction quality, robustness, and evaluation depth.

## MATLAB Edition Capabilities That Drive Superior Results

- **Declarative multi-stage pipeline** – The `RASequence` orchestrates eleven ordered steps that cover cutting, upsampling, multi-level alignment, volume-artifact suppression, templating, PCA, reinsertion, filtering, and ANC (`facet matlab edition/facet_matlab/src/+FACET/@FACET/FACET.m#L688-L706`). Each step exposes listeners for progress reporting and guarantees data invariants between stages.
- **Volume-artifact modelling and interpolation** – `RARemoveVolumeArtifact` models volume transitions, leverages logistic weighting, and backfills volume-gap samples by interpolation, preventing large residuals around volume triggers (`.../FACET.m#L1174-L1227`).
- **Adaptive artifact templating** – `CalcAvgArt` aligns triggers to averaged templates, computes per-epoch amplitude scaling (`Alpha`), and interpolates gaps when needed, yielding channel-aware residuals (`.../CalcAvgArt.m#L60-L116`).
- **Template selection strategies** – Multiple weighting functions (e.g., `AvgArtWghtFARM` for correlation-ranked selection, `AvgArtWghtCorrespondingSlice` for slice-wise pairing) are plug-ins in `RASequence` (`.../AvgArtWghtFARM.m#L1-L134`, `.../AvgArtWghtCorrespondingSlice.m#L1-L63`).
- **Automatic OBS/PCA configuration** – `DoPCA` derives the optimal number of components via slope, cumulative variance, and variance explained heuristics before calling `FitOBS`, which reuses factorized matrices for speed (`.../DoPCA.m#L1-L84`, `.../FitOBS.m#L1-L42`).
- **Trigger management toolkit** – Functions such as `FindTriggers`, `CompleteTriggers`, and `GenerateSliceTriggers` supply volume/slice aware reconstruction, ensuring consistent artifact lengths even with missing hardware markers (`.../FACET.m#L440-L532`, `.../FACET.m#L1174-L1227`).
- **Rich evaluation suite** – `FACET.Eval` produces peak-to-peak, RMS, SNR, and spectral metrics (Allen, Niazy) with per-channel statistics, allowing quantitative comparison of correction methods (`facet matlab edition/facet_matlab/src/+FACET/@Eval/Eval.m#L230-L355`).

## FACETpy Snapshot (Jan 2025)

- **Simplified default pipeline** – `create_standard_pipeline` wires only loader → trigger detection → upsample → coarse trigger alignment → AAS → optional ANC/PCA → downsample/filter/export (`src/facet/__init__.py#L162-L213`). Missing steps include slice/volume specific alignment, volume artifact removal, templated paste, and automated low-pass decisions.
- **AAS limitations** – `AASCorrection` implements a sliding-window correlation heuristic across epochs but lacks per-epoch amplitude scaling, advanced weighting plug-ins, and volume-gap interpolation (`src/facet/correction/aas.py#L86-L205`).
- **PCA requires manual tuning** – `PCACorrection` accepts a static component count or variance fraction; it does not auto-select components per channel or reuse OBS heuristics (`src/facet/correction/pca.py#L57-L165`).
- **Sub-sample alignment stub** – `helpers/alignsubsample.py` still contains a partially translated `YourClass.AlignSubSample` stub that is not wired into any processor (`src/facet/helpers/alignsubsample.py#L1-L118`).
- **Trigger recovery heuristics** – `MissingTriggerDetector` only uses a simple correlation check around large gaps and lacks explicit volume/slice modelling (`src/facet/preprocessing/trigger_detection.py#L246-L332`).
- **Evaluation coverage gap** – The evaluation module currently exposes SNR (legacy + reference-based) without RMS or spectral diagnostics (`src/facet/evaluation/metrics.py#L18-L219`), while documentation claims a broader metric suite (`docs/source/project_overview.rst#L60-L121`).
- **Documentation/code drift** – Public docs advertise deep-learning pipelines and broad frameworks that do not exist in `src/facet/correction` or elsewhere, signalling feature gaps to close for user trust (`docs/source/project_overview.rst#L60-L123`).

## Impact on Correction Quality

### Alignment Fidelity
- Without slice-level `RAAlignSlices` and sub-sample refinement, FACETpy relies on a single template channel for alignment, which is brittle when artifacts drift across slices. Implementing full RA alignment is critical for maintaining inter-slice consistency.

### Artifact Template Accuracy
- Lack of per-slice weighting strategies, `Alpha` scaling, and gap interpolation causes averaged templates to underfit long sessions or drift-heavy recordings, leading to residual ringing and volume-edge artifacts that the MATLAB pipeline suppresses.

### Residual Suppression
- Static PCA configuration risks under- or over-subtraction. MATLAB’s OBS heuristics adapt to changing spectral content per channel, producing cleaner residuals before ANC. FACETpy should adopt similar data-driven selection.

### Diagnostics & Feedback Loop
- The MATLAB evaluation suite closes the loop between correction choices and objective metrics; FACETpy’s single-metric reporting provides limited guidance for tuning pipelines or regression testing.

### Documentation Credibility
- Divergence between promised features (deep learning, comprehensive evaluation) and actual code increases onboarding friction and obscures true capabilities, slowing adoption and external validation.

## Recommended Implementation Steps

1. **Recreate RASequence as composable processors** – Introduce processors for cut/paste, slice alignment, subsample alignment, volume artifact removal, averaged template calculation (with Alpha scaling), and integrate them into a default pipeline mirroring MATLAB ordering.
2. **Port averaging strategies** – Implement a registry of averaging weight calculators (`AvgArtWghtFARM`, slice-corresponding, Allen-style) and expose them as configurable options on `AASCorrection`.
3. **Adopt OBS heuristics in PCA** – Translate `DoPCA` thresholds to Python, retaining per-channel auto-selection and caching the pseudo-inverse as done in `FitOBS`.
4. **Finish and wire sub-sample alignment** – Replace `helpers/alignsubsample.YourClass` with a real processor and run it after coarse alignment in the pipeline.
5. **Enhance trigger management** – Provide processors for volume/slice detection, trigger completion, and slice-trigger generation with clear metadata updates.
6. **Match evaluation breadth** – Port the MATLAB evaluation metrics (RMS ratios, spectral measures) into FACETpy’s `evaluation` package and surface structured results in `ProcessingContext`.
7. **Synchronize documentation** – Update `docs/source/project_overview.rst` once the above features ship or adjust claims now to avoid misleading users.

## TODO Backlog

- [ ] **P0**: Implement `Cut`, `Paste`, `AlignSlices`, `AlignSubSample`, and `RemoveVolumeArtifact` processors; update `create_standard_pipeline` to include them in the canonical order (`facet matlab edition/facet_matlab/src/+FACET/@FACET/FACET.m#L688-L706`).
- [ ] **P0**: Add per-epoch scaling (`Alpha`) and gap interpolation to `AASCorrection`, using MATLAB’s `CalcAvgArt` as the reference logic (`facet matlab edition/facet_matlab/src/+FACET/@FACET/CalcAvgArt.m#L60-L116`).
- [ ] **P0**: Replace `helpers/alignsubsample.py#L1-L118` stub with a functional processor and corresponding unit tests.
- [ ] **P1**: Port correlation-based weighting strategies (`AvgArtWghtFARM`, slice-paired, Allen, Moosmann) and expose configuration on AAS (`facet matlab edition/facet_matlab/src/+FACET/AvgArtWghtFARM.m#L1-L134`).
- [ ] **P1**: Reimplement OBS heuristics (slope, cumulative variance, variance explained) for PCA component selection and caching (`facet matlab edition/facet_matlab/src/+FACET/@FACET/DoPCA.m#L18-L84`, `.../FitOBS.m#L1-L42`).
- [ ] **P1**: Expand `MissingTriggerDetector` with explicit volume/slice reconstruction akin to `CompleteTriggers` (`facet matlab edition/facet_matlab/src/+FACET/@FACET/FACET.m#L440-L532`).
- [ ] **P1**: Port MATLAB evaluation metrics (peak-to-peak, RMS residual/correction, Allen/Niazy spectra) into `src/facet/evaluation`, returning structured results comparable to `EvalResult` (`facet matlab edition/facet_matlab/src/+FACET/@Eval/Eval.m#L230-L355`).
- [ ] **P2**: Align documentation claims with implemented functionality or schedule the missing deep-learning/evaluation features (`docs/source/project_overview.rst#L60-L123`).
- [ ] **P2**: Establish regression tests that compare Python outputs against MATLAB golden datasets for shared pipelines (`facet matlab edition/facet_matlab/src/CleanEx*.m` as scenario definitions).

## Key Reference Files

| Topic | MATLAB Reference | Python Reference |
|-------|------------------|------------------|
| Pipeline orchestration | `facet matlab edition/facet_matlab/src/+FACET/@FACET/FACET.m#L688-L706` | `src/facet/__init__.py#L162-L213` |
| Volume artifact handling | `facet matlab edition/facet_matlab/src/+FACET/@FACET/FACET.m#L1174-L1227` | *(missing)* |
| Averaging matrix strategies | `facet matlab edition/facet_matlab/src/+FACET/AvgArtWghtFARM.m#L1-L134` | `src/facet/correction/aas.py#L138-L205` |
| PCA/OBS heuristics | `facet matlab edition/facet_matlab/src/+FACET/@FACET/DoPCA.m#L18-L84` | `src/facet/correction/pca.py#L57-L165` |
| Missing trigger recovery | `facet matlab edition/facet_matlab/src/+FACET/@FACET/FACET.m#L440-L532` | `src/facet/preprocessing/trigger_detection.py#L246-L332` |
| Evaluation metrics | `facet matlab edition/facet_matlab/src/+FACET/@Eval/Eval.m#L230-L355` | `src/facet/evaluation/metrics.py#L18-L219` |

This roadmap should provide the next engineer with clear context, priorities, and source anchors for closing the MATLAB ↔ Python parity gap.
