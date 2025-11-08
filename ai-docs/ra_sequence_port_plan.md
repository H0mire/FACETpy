# Plan: Porting MATLAB RASequence Steps into FACETpy

## Goals

- Mirror the MATLAB correction chain (Cut → UpSample → AlignSlices → AlignSubSample → RemoveVolumeArt → CalcAvgArt → PCA → DownSample → Paste → LowPass → ANC) inside FACETpy.
- Preserve the MATLAB data contract: `RAEEGAll/RAEEGAcq` separation, trigger bookkeeping, averaging matrices, and estimated noise tracking.
- Keep compatibility with the existing `ProcessingContext` and pipeline abstractions (`src/facet/core/pipeline.py`).

## Data Model Updates

| Requirement | Proposed Change | Notes |
|-------------|-----------------|-------|
| Acquisition window (`AcqStart`, `AcqEnd`) | Extend `ProcessingMetadata` with `acq_start_sample` / `acq_end_sample` | Needed by Cut/Paste; default derived from triggers (`src/facet/preprocessing/trigger_detection.py#L67`). |
| Trigger offsets (`PreTrig`, `PostTrig`) | Store in `ProcessingMetadata` as `pre_trigger_samples` / `post_trigger_samples` | Keep current `artifact_to_trigger_offset` for time-based interfaces. |
| Upsampled triggers (`TriggersUp`) | Cache under `metadata.custom['upsampled_triggers']` | Calculated by new `Cut` processor after upsampling; consumed by alignment and averaging steps. |
| Estimated noise separation | Allow `ProcessingContext` to hold both acquisition-window noise and full-length noise (current `estimated_noise` covers full-length). | Introduce helper methods such as `context.get_noise_window(start, end)` for reuse. |

## Processor Mapping

| MATLAB Step (`facet matlab edition/facet_matlab/src/+FACET/@FACET/FACET.m#L1040-L1255`) | FACETpy Plan |
|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `RACut` (`#L1072-L1079`) | `CutAcquisitionWindow` processor: crops raw/noise arrays to `[acq_start:acq_end]`, caches original segments for `Paste`. Requires metadata fields above. |
| `RAUpSample` (`#L1090-L1113`) | Reuse existing `UpSample` processor; ensure it records upsampled triggers (`metadata.custom['upsampled_triggers']`) and zero-fills noise efficiently. |
| `RAAlignSlices` (`#L1126-L1156`) | `SliceAligner` processor: uses reference epoch to align using cross-correlation, updates `metadata.custom['upsampled_triggers']` & `artifact_length`. Needs MNE epoch access similar to `TriggerAligner` but on upsampled data. |
| `RAAlignSubSample` (`#L1158-L1172`) | Finish `helpers/alignsubsample.py` into `SubsampleAligner` processor (already partially ported). Integrate with metadata to reapply shifts on raw data. |
| `RARemoveVolumeArtifact` (`#L1174-L1227`) | `VolumeArtifactSuppressor` processor: detects volume trigger boundaries via diffs, builds logistic weighting, interpolates gaps. Requires access to upsampled data and metadata `volume_gaps`. |
| Averaging Setup (`AvgArtWght*` functions) | Parameterize `AASCorrection` with strategy classes; start with FARM correlation selection; others follow in separate TODOs. |
| `RACalcAvgArt` (`#L1232-L1240` + `CalcAvgArt.m#L60-L116`) | Extend `AASCorrection` to apply Alpha scaling, trigger alignment, volume-gap interpolation; expose intermediate artifacts for downstream PCA and ANC. |
| `RAPCA` (`#L1242-L1259`) | Already mapped to `PCACorrection`—add auto OBS heuristics (see `DoPCA.m#L18-L84`). |
| `RADownSample`, `RAPaste`, `RALowPass`, `RAANC` | Reuse existing processors (`DownSample`, planned `PasteAcquisitionWindow`, `LowPassFilter`, `ANCCorrection`) once acquisition window handling is in place. |

## Integration Strategy

1. **Phase 1 (current sprint)**  
   - Implement `CutAcquisitionWindow`, `SliceAligner`, `SubsampleAligner` (finished), `PasteAcquisitionWindow`.  
   - Update `create_standard_pipeline` to insert new processors before AAS/ANC (guarded by feature flags).
   - Add unit tests covering: acquisition window extraction, alignment effects on synthetic triggers, round-trip Cut+Paste.
   - Include a runnable example so downstream teams can exercise the updated pipeline without diving into processor wiring.

   **Phase 1 usage example**

   ```python
   from facet import create_standard_pipeline

   pipeline = create_standard_pipeline(
       input_path="example_simple_bids/sub-muellerj/ses-01/eeg/sub-muellerj_ses-01_task-restingstate_eeg.edf",
       output_path="output/sub-muellerj_task-restingstate_corrected.edf",
       trigger_regex=r"\b5\b",
       upsample_factor=10,
       use_anc=False,
       use_pca=False,
   )

   result = pipeline.run()
   context = result.get_context()
   print("acq window:", context.metadata.acq_start_sample, context.metadata.acq_end_sample)
   print("upsampled triggers:", context.metadata.custom["upsampled_triggers"].shape)
   ```

   Running this example exercises the full Phase 1 chain (`EDFLoader → TriggerDetector → Cut → UpSample → SliceAligner → SubsampleAligner → AAS → DownSample → Paste → HighPass → EDFExporter`). It verifies that acquisition-window metadata is populated automatically and that `metadata.custom["upsampled_triggers"]` is ready for Phase 2 processors.

2. **Phase 2**  
   - Port `RARemoveVolumeArtifact` and tie into pipeline when `metadata.volume_gaps` is True.  
   - Introduce Alpha scaling in `AASCorrection` and ensure estimated noise matches MATLAB behaviour.

3. **Phase 3**  
   - Integrate OBS heuristics, additional averaging strategies, and expanded evaluation metrics as tracked in `ai-docs/facet_matlab_vs_python.md`.

## Dependencies & Risks

- **Metadata compatibility**: Existing processors ignore acquisition fields; ensure defaults prevent crashes when processors run standalone.
- **Performance**: Slice/subsample alignment requires FFT-based operations; reuse numpy/SciPy (already available via `scipy.signal`) and consider caching to avoid repeated FFTs.
- **Testing**: Need synthetic Raw objects with deterministic triggers. Use MNE `create_info` + numpy arrays; follow patterns in `tests/test_preprocessing.py`.
- **Documentation**: Update pipeline docs (`docs/source/getting_started/quickstart.rst`) once new processors are production-ready.

This plan establishes the scaffold for Step 1 of the backlog: making FACETpy’s pipeline structure align with the MATLAB RASequence while keeping changes incremental and testable.
