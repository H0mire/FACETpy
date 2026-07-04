Changelog
=========

All notable changes to FACETpy are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

[2.1.0] - 2026-07-04
--------------------

Correctness fixes and MATLAB-faithful improvements to the classical
correction, preprocessing, and evaluation stack (independent of the
deep-learning subsystem). Notably, this release fixes two correctness bugs in
``PCACorrection`` (subtracting the residual instead of the fitted artifact) and
``SubsampleAligner`` (a constant shift bias) — see *Fixed* below.

Added
~~~~~

- ``TriggerEditor`` - interactive processor for aligning the artifact window to
  the trigger; supersedes ``ArtifactOffsetFinder``
- Fast/quality sub-sample alignment modes with session-cached sub-sample shift
  and full multi-channel alignment in fractional modes
- Detection of multiple missing triggers within wide gaps
- ``RawPlotter`` prediction-source overlay for residual diagnostics
- ``examples/`` reorganized into a ``pipelines/`` subdirectory; new
  volume→slice large-dataset example

Changed
~~~~~~~

- ``PCACorrection`` OBS high-pass redesigned to be MATLAB/FASTR-faithful: a
  short ``firls`` FIR (order derived from cutoff + a ±10 Hz transition band)
  replaces the order-5 Butterworth, with a 70 Hz default and padding capped to
  the acquisition window so it no longer silently disables itself on short
  windows
- ``PCACorrection`` epochs are now mean-centered only (MATLAB
  ``detrend('constant')``) instead of z-scored, so the singular-value variance
  ranking (and OBS auto-selection) is no longer distorted
- ``ANCCorrection`` fastranc matches the MATLAB FACET implementation exactly
- ``SNRCalculator`` / ``LegacySNRCalculator`` residual handling made
  MATLAB-faithful (over-corrected channels are dropped, not clamped)
- Example pipelines updated to ``TriggerEditor`` and retuned correction steps

Fixed
~~~~~

- **``PCACorrection`` subtracted the wrong quantity.** The OBS routine removed
  the post-fit *residual* (signal minus fitted artifact — i.e. the clean EEG)
  instead of the OBS-fitted artifact reconstruction, so it effectively stripped
  the neural signal and left the residual artifact in place. It now subtracts
  the fitted artifact, matching MATLAB FACET (``FACET.m:1266-1267``). This was a
  Python-port-only regression: the MATLAB reference is correct — its
  ``fitted_res = papc * pinv(papc) * Ipca`` (``FitOBS.m``) is the OBS projection
  (the reconstruction), which it subtracts from the EEG; the Python port had
  additionally formed ``X - reconstruction`` and subtracted that instead.
- **``SubsampleAligner`` applied a constant ``+search_radius`` bias to every
  shift.** The reference epoch was extracted over the inner window
  (``window_length``) while the per-trigger search segments used the extended
  window (``window_length + 2*search_radius``); ``crosscorrelation`` then padded
  the shorter reference and displaced the zero-lag point, so a zero-offset
  artifact yielded ``shift = search_radius`` instead of ``0``. The reference is
  now extracted over the same extended window.
- ``MetricsReport`` no longer crashes when SNR is undefined (all channels
  over-corrected); it now renders ``n/a`` instead

Removed
~~~~~~~

- Unused MATLAB sub-sample alignment reference port (``alignsubsample.py``)

[2.0.0] - 2025-10-31
--------------------

Major refactoring and modernization of FACETpy.

Added
~~~~~

**Core Architecture**

- New processor-based architecture with ``Processor`` base class
- ``ProcessingContext`` for immutable data flow
- ``Pipeline`` for composing processing workflows
- ``ProcessorRegistry`` for plugin discovery
- ``ParallelExecutor`` for automatic parallelization

**Processors**

*I/O:*

- ``Loader`` - Load EEG data with automatic format detection (EDF, BDF, GDF, BrainVision, EEGLAB, FIF)
- ``BIDSLoader`` - Load BIDS format data
- ``EDFExporter`` - Export to EDF format
- ``BIDSExporter`` - Export to BIDS format

*Preprocessing:*

- ``HighPassFilter``, ``LowPassFilter``, ``BandPassFilter``, ``NotchFilter``
- ``UpSample``, ``DownSample``, ``Resample``
- ``TriggerDetector``, ``QRSTriggerDetector``, ``MissingTriggerDetector``
- ``TriggerAligner``, ``SubsampleAligner``

*Correction:*

- ``AASCorrection`` - Averaged Artifact Subtraction (refactored)
- ``ANCCorrection`` - Adaptive Noise Cancellation (refactored)
- ``PCACorrection`` - PCA-based correction (refactored)

*Evaluation:*

- ``SNRCalculator`` - Signal-to-noise ratio
- ``RMSCalculator`` - RMS ratio
- ``MedianArtifactCalculator`` - Median artifact amplitude
- ``MetricsReport`` - Formatted metrics output

*Composite:*

- ``SequenceProcessor`` - Sequential execution
- ``ConditionalProcessor`` - Conditional execution
- ``SwitchProcessor`` - Switch between processors
- ``NoOpProcessor`` - No-op placeholder
- ``LambdaProcessor`` - Lambda function wrapper

**Features**

- Full type hints throughout codebase
- Parallel processing support (channel-wise parallelization)
- Plugin system with decorator-based registration
- Processing history tracking
- Immutable context pattern
- First-class MNE integration
- Detailed logging with loguru

**Documentation**

- Complete API reference
- User guide (architecture, pipelines, processors, parallel processing, custom processors)
- Migration notes in examples and updated API docstrings
- Tutorial and examples
- Comprehensive docstrings (NumPy style)

**Testing**

- Unit tests for all processors
- Integration tests for workflows
- Test fixtures and utilities
- Coverage tracking
- Pytest markers for test organization
