Examples
========

This page provides various examples of using FACETpy for different scenarios.

Basic Correction
----------------

Minimal pipeline for basic correction:

.. code-block:: python

   from facet import create_standard_pipeline

   pipeline = create_standard_pipeline(
       "data.edf",
       "corrected.edf",
       trigger_regex=r"\b1\b"
   )

   result = pipeline.run()

BCG Correction
--------------

Correct ballistocardiogram (BCG) artifacts:

.. code-block:: python

   from facet.core import Pipeline
   from facet.io import Loader, EDFExporter
   from facet.preprocessing import QRSTriggerDetector
   from facet.correction import AASCorrection

   pipeline = Pipeline([
       Loader(path="data.edf", preload=True),
       QRSTriggerDetector(),  # Detect QRS complexes
       AASCorrection(window_size=20),  # Smaller window for BCG
       EDFExporter(path="corrected.edf")
   ])

   result = pipeline.run()

Batch Processing
----------------

Process multiple files:

.. code-block:: python

   from facet.core import Pipeline, ProcessingContext
   from facet.io import Loader, EDFExporter

   # Define correction pipeline
   correction = Pipeline([
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       AASCorrection(window_size=30),
       DownSample(factor=10)
   ])

   # Process files
   files = ["s1.edf", "s2.edf", "s3.edf"]

   for file in files:
       loader = Loader(path=file, preload=True)
       context = loader.execute(ProcessingContext())

       result = correction.run(initial_context=context)

       if result.success:
           output = file.replace('.edf', '_corrected.edf')
           exporter = EDFExporter(path=output)
           exporter.execute(result.context)

Conditional Processing
----------------------

Apply processors conditionally:

.. code-block:: python

   from facet.core import ConditionalProcessor

   def needs_pca(context):
       metrics = context.metadata.custom.get('metrics', {})
       return metrics.get('snr', float('inf')) < 10

   pipeline = Pipeline([
       # ... preprocessing ...
       AASCorrection(window_size=30),
       SNRCalculator(),
       ConditionalProcessor(
           condition=needs_pca,
           processor=PCACorrection(n_components=0.95)
       ),
       # ... export ...
   ])

Custom Processor
----------------

Create your own processor:

.. code-block:: python

   from facet.core import Processor, register_processor

   @register_processor
   class CustomDenoiser(Processor):
       name = "custom_denoiser"
       description = "My custom denoising algorithm"

       def __init__(self, threshold=0.5):
           self.threshold = threshold
           super().__init__()

       def process(self, context):
           raw = context.get_raw()

           # Your custom logic here
           denoised_data = my_denoising_algorithm(
               raw.get_data(copy=False),
               threshold=self.threshold
           )

           # Create new raw with denoised data
           raw_denoised = raw.copy()
           raw_denoised._data = denoised_data

           return context.with_raw(raw_denoised)

   # Use it
   pipeline = Pipeline([
       Loader(path="data.edf", preload=True),
       CustomDenoiser(threshold=0.5),
       EDFExporter(path="denoised.edf")
   ])

Accessing Metrics
-----------------

Get quality metrics:

.. code-block:: python

   from facet.evaluation import SNRCalculator, RMSCalculator

   pipeline = Pipeline([
       # ... correction steps ...
       SNRCalculator(),
       RMSCalculator()
   ])

   result = pipeline.run()

   if result.success:
       metrics = result.context.metadata.custom['metrics']
       print(f"SNR: {metrics['snr']:.2f}")
       print(f"RMS improvement: {metrics['rms_ratio']:.2f}x")
       print(f"Per-channel SNR: {metrics['snr_per_channel']}")

Manual Trigger Specification
-----------------------------

Provide triggers manually instead of detecting:

.. code-block:: python

   import numpy as np
   from facet.core import ProcessingContext, ProcessingMetadata

   # Load data
   loader = Loader(path="data.edf", preload=True)
   context = loader.execute(ProcessingContext())

   # Manually set triggers
   triggers = np.array([1000, 2000, 3000, 4000])  # Sample positions
   metadata = context.metadata.copy()
   metadata.triggers = triggers
   metadata.artifact_length = 950  # Samples

   context = context.with_metadata(metadata)

   # Continue with correction
   aas = AASCorrection(window_size=30)
   context = aas.execute(context)

Working with Different File Formats
------------------------------------

BIDS Format
~~~~~~~~~~~

.. code-block:: python

   from facet.io import BIDSLoader, BIDSExporter

   pipeline = Pipeline([
       BIDSLoader(
           bids_root="/path/to/bids",
           subject="01",
           session="01",
           task="rest"
       ),
       # ... correction ...
       BIDSExporter(
           bids_root="/path/to/bids_corrected",
           subject="01",
           session="01",
           task="rest"
       )
   ])

GDF Format
~~~~~~~~~~

.. code-block:: python

   from facet.io import Loader

   pipeline = Pipeline([
       Loader(path="data.gdf", preload=True),
       # ... correction ...
       EDFExporter(path="corrected.edf")
   ])

Converting Between Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the convenience functions :func:`facet.load` and :func:`facet.export` to convert between supported formats (EDF, GDF, BDF, SET, MFF, etc.):

.. code-block:: python

   from facet import load, export

   INPUT_FILE = "./examples/datasets/NiazyFMRI.set"
   OUTPUT_FILE = "./examples/datasets/NiazyFMRI.bdf"

   ctx = load(INPUT_FILE)
   export(ctx, OUTPUT_FILE)

Performance Optimization
------------------------

Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Maximum parallelization
   result = pipeline.run(parallel=True, n_jobs=-1)

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For very large files, process in chunks (future feature)
   # Currently, use preload=True and ensure sufficient RAM

Debugging
---------

Check Processing History
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = pipeline.run()

   if result.success:
       for entry in result.context.get_history():
           print(f"{entry['processor']}")
           print(f"  Timestamp: {entry['timestamp']}")
           print(f"  Parameters: {entry['parameters']}")

Save Intermediate Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from facet.core import Pipeline

   # Create sub-pipelines to save intermediate results
   preprocessing = Pipeline([
       Loader(path="data.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10)
   ])

   result1 = preprocessing.run()
   result1.context.get_raw().save("after_preprocessing.fif")

   # Continue from there
   correction = Pipeline([
       AASCorrection(window_size=30),
       DownSample(factor=10)
   ])

   result2 = correction.run(initial_context=result1.context)

More Examples
-------------

See the ``examples/`` directory in the repository for more complete examples:

* ``complete_pipeline_example.py`` - Full correction workflow
* ``complete_pipeline_example_mff.py`` - End-to-end workflow using MFF input
* ``advanced_workflows.py`` - Conditional steps, parallel execution, factory shortcut
* ``channelwise_execution.py`` - Channel-wise execution: flag inspection, backend comparison, custom processor
* ``batch_processing.py`` - Batch correction across multiple files with ``Pipeline.map()``
* ``evaluation.py`` - SNR, RMS, and other quality metrics
* ``convert_types.py`` - Convert between file formats (e.g. SET to BDF) using ``load`` and ``export``
* ``inline_steps.py`` - Inline callable steps and ``ProcessingContext`` pipe operator
* ``memory_efficient_pipeline.py`` - Streaming-style workflow for large recordings
* ``quickstart.py`` - Minimal runnable correction example
* ``eeg_generation_visualization_example.py`` - Synthetic EEG generation and plotting
