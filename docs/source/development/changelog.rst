Changelog
=========

All notable changes to FACETpy are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

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
- Migration guide from v1.x
- Tutorial and examples
- Comprehensive docstrings (NumPy style)

**Testing**

- Unit tests for all processors
- Integration tests for workflows
- Test fixtures and utilities
- Coverage tracking
- Pytest markers for test organization