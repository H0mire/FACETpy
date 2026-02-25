Quick Start
===========

This guide will get you up and running with FACETpy in 5 minutes.

Before running examples, install dependencies from the repository root:

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
   cd facetpy

Basic Correction Pipeline
--------------------------

The simplest way to correct fMRI artifacts:

.. code-block:: python

   from facet import create_standard_pipeline

   # Create pipeline
   pipeline = create_standard_pipeline(
       input_path="my_data.edf",
       output_path="corrected.edf",
       trigger_regex=r"\b1\b",  # Pattern to match triggers
       evaluate=True
   )

   # Run correction
   result = pipeline.run()

   # Check results
   if result.success:
       print(f"✓ Correction completed in {result.execution_time:.2f}s")
       metrics = result.context.metadata.custom['metrics']
       print(f"  SNR: {metrics['snr']:.2f}")
       print(f"  RMS improvement: {metrics['rms_ratio']:.2f}x")
   else:
       print(f"✗ Failed: {result.error}")

That's it! Your corrected data is saved to ``corrected.edf``.

Understanding the Standard Pipeline
------------------------------------

The standard pipeline performs these steps:

1. **Load** EDF file
2. **Detect** triggers using regex pattern
3. **Upsample** to 10x for precise alignment
4. **Align** triggers using cross-correlation
5. **Correct** artifacts with AAS (Averaged Artifact Subtraction)
6. **Downsample** back to original sampling rate
7. **Optional** PCA correction (enabled by default if available)
8. **Downsample** back to original sampling rate
9. **Restore** previously cut acquisition windows
10. **Filter** with lowpass filter (70 Hz)
11. **Optional** ANC correction (enabled by default if available)
12. **Export** corrected data

Custom Pipeline
---------------

For more control, build a custom pipeline:

.. code-block:: python

   from facet.core import Pipeline
   from facet.io import Loader, EDFExporter
   from facet.preprocessing import TriggerDetector, UpSample, DownSample
   from facet.correction import AASCorrection
   from facet.evaluation import SNRCalculator, MetricsReport

   pipeline = Pipeline([
       # Load your data
       Loader(path="data.edf", preload=True),

       # Detect fMRI triggers
       TriggerDetector(regex=r"\b1\b"),

       # Upsample for precision
       UpSample(factor=10),

       # Main artifact correction
       AASCorrection(
           window_size=30,              # Size of sliding window
           correlation_threshold=0.975  # Correlation threshold
       ),

       # Downsample back
       DownSample(factor=10),

       # Evaluate correction quality
       SNRCalculator(),
       MetricsReport(),

       # Save corrected data
       EDFExporter(path="corrected.edf", overwrite=True)
   ], name="My fMRI Correction Pipeline")

   # Run it
   result = pipeline.run()

Step-by-Step Processing
------------------------

For maximum control, process step by step:

.. code-block:: python

   from facet.core import ProcessingContext
   from facet.io import Loader
   from facet.preprocessing import TriggerDetector, UpSample
   from facet.correction import AASCorrection

   # 1. Load data
   loader = Loader(path="data.edf", preload=True)
   context = loader.execute(ProcessingContext())

   # 2. Detect triggers
   detector = TriggerDetector(regex=r"\b1\b")
   context = detector.execute(context)
   print(f"Found {len(context.get_triggers())} triggers")

   # 3. Upsample
   upsampler = UpSample(factor=10)
   context = upsampler.execute(context)

   # 4. Apply correction
   aas = AASCorrection(window_size=30)
   context = aas.execute(context)

   # 5. Access results
   corrected_raw = context.get_raw()
   corrected_raw.save("corrected.fif")

Pipe Operator (``|``)
---------------------

You can apply processors directly to a ``ProcessingContext`` using the pipe
operator (``__or__``):

.. code-block:: python

   from facet import load, HighPassFilter, TriggerDetector, UpSample, AASCorrection

   ctx = load("data.edf", preload=True)
   ctx = (
       ctx
       | HighPassFilter(1.0)
       | TriggerDetector(r"\b1\b")
       | UpSample(10)
       | AASCorrection(window_size=30)
   )

Parallel Processing
-------------------

Speed up processing with parallel execution:

.. code-block:: python

   # Use all CPU cores
   result = pipeline.run(parallel=True, n_jobs=-1)

   # Use specific number of cores
   result = pipeline.run(parallel=True, n_jobs=4)

Processors automatically parallelize by channel when safe to do so.

Common Patterns
---------------

BCG Correction
~~~~~~~~~~~~~~

For ballistocardiogram (BCG) artifact correction:

.. code-block:: python

   from facet.preprocessing import QRSTriggerDetector
   from facet.correction import AASCorrection

   pipeline = Pipeline([
       Loader(path="data.edf", preload=True),
       QRSTriggerDetector(),  # Detect QRS complexes
       AASCorrection(window_size=20),  # Smaller window for BCG
       EDFExporter(path="corrected.edf")
   ])

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple files with the same pipeline:

.. code-block:: python

   from facet.core import Pipeline, ProcessingContext
   from facet.io import Loader, EDFExporter

   # Define reusable correction pipeline
   correction = Pipeline([
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       AASCorrection(window_size=30),
       DownSample(factor=10)
   ])

   # Process multiple files
   files = ["subject1.edf", "subject2.edf", "subject3.edf"]

   for input_file in files:
       print(f"Processing {input_file}...")

       # Load
       loader = Loader(path=input_file, preload=True)
       context = loader.execute(ProcessingContext())

       # Correct
       result = correction.run(initial_context=context)

       if result.success:
           # Save
           output = input_file.replace('.edf', '_corrected.edf')
           exporter = EDFExporter(path=output, overwrite=True)
           exporter.execute(result.context)
           print(f"  ✓ Saved to {output}")

Accessing Results
-----------------

Get data and metrics from the result:

.. code-block:: python

   result = pipeline.run()

   # Corrected MNE Raw object
   corrected_raw = result.context.get_raw()

   # Trigger positions
   triggers = result.context.get_triggers()

   # Quality metrics
   metrics = result.context.metadata.custom.get('metrics', {})
   snr = metrics.get('snr')
   rms_ratio = metrics.get('rms_ratio')

   # Processing history
   for entry in result.context.get_history():
       print(f"{entry['processor']} at {entry['timestamp']}")

Next Steps
----------

* Read the :doc:`tutorial` for detailed examples
* See :doc:`examples` for more use cases
* Check the :doc:`../user_guide/architecture` to understand the design
* Browse the :doc:`../api/core` for complete API reference
