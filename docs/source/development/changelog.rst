Changelog
=========

All notable changes to FACETpy are documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/>`_.

[2.0.0] - 2025-01-12
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

- ``EDFLoader`` - Load EDF/EDF+ files
- ``BIDSLoader`` - Load BIDS format data
- ``GDFLoader`` - Load GDF format data
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
- Comprehensive test suite (150+ tests, >90% coverage)
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

Changed
~~~~~~~

- **API**: Functional API → Object-oriented API
- **Data Flow**: Mutable operations → Immutable contexts
- **Configuration**: JSON files → Python objects
- **Error Handling**: Mixed exceptions → Specific exception types
- **Module Structure**: Flat → Hierarchical organization

Deprecated
~~~~~~~~~~

- Legacy functional API (``correct_fmri_artifact``, etc.)
- JSON configuration files
- Global settings
- Command-line interface (replaced by Python API)

Removed
~~~~~~~

- Built-in plotting functions (use MNE instead)
- Automatic parameter tuning
- Batch processing scripts (use Python loops)

Fixed
~~~~~

- Memory leaks in AAS correction
- Trigger alignment precision issues
- Edge cases in artifact detection
- Parallel processing race conditions

[1.2.3] - 2024-06-15
--------------------

Last release of v1.x series.

Fixed
~~~~~

- Bug in trigger detection for short recordings
- Memory usage optimization
- Documentation typos

[1.2.2] - 2024-03-20
--------------------

Fixed
~~~~~

- ANC convergence issues
- Edge case in upsampling

[1.2.1] - 2024-01-10
--------------------

Fixed
~~~~~

- Compatibility with MNE 1.5.0
- Installation issues on Windows

[1.2.0] - 2023-11-05
--------------------

Added
~~~~~

- PCA-based correction option
- Batch processing support
- Configuration file validation

Changed
~~~~~~~

- Improved AAS performance
- Better error messages

[1.1.0] - 2023-08-15
--------------------

Added
~~~~~

- QRS trigger detection
- Missing trigger interpolation
- Export to BIDS format

Fixed
~~~~~

- Edge cases in trigger alignment
- Memory efficiency

[1.0.0] - 2023-05-01
--------------------

Initial release.

Added
~~~~~

- AAS correction algorithm
- ANC correction algorithm
- Trigger detection
- EDF I/O
- Basic documentation

Upgrade Guide
-------------

From v1.x to v2.0
~~~~~~~~~~~~~~~~~

See :doc:`../migration/v2_migration_guide` for detailed migration instructions.

**Quick Summary:**

1. Replace function calls with Pipeline API
2. Update configuration to use Python objects
3. Update imports (``facet.utils`` → ``facet.core``)
4. Update error handling for new exception types
5. Update tests to use new API

**Example:**

.. code-block:: python

   # v1.x
   from facet import correct_fmri_artifact
   result = correct_fmri_artifact("in.edf", "out.edf")

   # v2.0
   from facet import create_standard_pipeline
   pipeline = create_standard_pipeline("in.edf", "out.edf")
   result = pipeline.run()

Versioning Policy
-----------------

FACETpy follows `Semantic Versioning <https://semver.org/>`_:

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

Release Schedule
----------------

- **Major releases**: Yearly
- **Minor releases**: Quarterly
- **Patch releases**: As needed

Support Policy
--------------

- **Latest version**: Full support
- **Previous major**: Security fixes for 1 year
- **Older versions**: No support

Deprecation Policy
------------------

Features marked as deprecated will:

1. Show deprecation warnings for at least 6 months
2. Be removed in the next major version
3. Have migration path documented

Reporting Issues
----------------

Found a bug? Report it on `GitHub Issues <https://github.com/your-org/facetpy/issues>`_.

Contributing
------------

See :doc:`contributing` for how to contribute to FACETpy.
