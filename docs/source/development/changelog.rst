Changelog
=========

All notable changes to FACETpy are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

[2.1.0] - 2026-07-04
--------------------

MATLAB-faithful improvements to the classical correction, preprocessing, and
evaluation stack (independent of the deep-learning subsystem).

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

- ``PCACorrection`` OBS high-pass is now MATLAB/FASTR-faithful (70 Hz default)
- ``ANCCorrection`` fastranc matches the MATLAB FACET implementation exactly
- ``SNRCalculator`` / ``LegacySNRCalculator`` residual handling made
  MATLAB-faithful (over-corrected channels are dropped, not clamped)
- Example pipelines updated to ``TriggerEditor`` and retuned correction steps

Fixed
~~~~~

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
