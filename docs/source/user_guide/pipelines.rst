Pipeline Guide
==============

Pipelines are the primary way to compose and execute processing workflows in FACETpy.
This guide covers everything you need to know about creating and using pipelines.

What is a Pipeline?
-------------------

A Pipeline is a container that executes a sequence of processors in order. Each processor
receives the output context from the previous processor, processes it, and passes its output
to the next processor.

.. code-block:: python

   from facet.core import Pipeline
   from facet.preprocessing import TriggerDetector, UpSample
   from facet.correction import AASCorrection

   pipeline = Pipeline([
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       AASCorrection(window_size=30)
   ])

   result = pipeline.run(initial_context=context)

Basic Pipeline Usage
--------------------

Creating a Pipeline
~~~~~~~~~~~~~~~~~~~

Create a pipeline by passing a list of processors:

.. code-block:: python

   from facet.core import Pipeline
   from facet.io import EDFLoader, EDFExporter
   from facet.preprocessing import TriggerDetector
   from facet.correction import AASCorrection

   pipeline = Pipeline([
       EDFLoader(path="data.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       AASCorrection(window_size=30),
       EDFExporter(path="output.edf", overwrite=True)
   ], name="My Pipeline")

Running a Pipeline
~~~~~~~~~~~~~~~~~~

Execute the pipeline with the ``run()`` method:

.. code-block:: python

   result = pipeline.run()

   if result.success:
       print(f"Pipeline completed in {result.execution_time:.2f}s")
       final_context = result.context
   else:
       print(f"Pipeline failed: {result.error}")

With Initial Context
~~~~~~~~~~~~~~~~~~~~

Start with an existing context:

.. code-block:: python

   from facet.core import ProcessingContext

   # Create initial context
   initial_context = ProcessingContext(raw=raw, metadata=metadata)

   # Run pipeline starting from this context
   result = pipeline.run(initial_context=initial_context)

Pipeline Results
----------------

The ``PipelineResult`` object contains:

- ``success`` (bool) - Whether pipeline completed successfully
- ``context`` (ProcessingContext) - Final processing context
- ``error`` (Exception) - Exception if pipeline failed
- ``execution_time`` (float) - Total execution time in seconds

.. code-block:: python

   result = pipeline.run()

   print(f"Success: {result.success}")
   print(f"Time: {result.execution_time:.2f}s")

   if result.success:
       # Access final data
       corrected_raw = result.context.get_raw()
       metrics = result.context.metadata.custom.get('metrics', {})

       # Access processing history
       history = result.context.get_history()
       for step in history:
           print(f"- {step.name}: {step.parameters}")
   else:
       print(f"Error: {result.error}")

Advanced Pipeline Features
--------------------------

Parallel Execution
~~~~~~~~~~~~~~~~~~

Enable parallel processing for compatible processors:

.. code-block:: python

   # Use all CPU cores
   result = pipeline.run(parallel=True, n_jobs=-1)

   # Use specific number of cores
   result = pipeline.run(parallel=True, n_jobs=4)

Processors marked with ``parallel_safe = True`` will be automatically
parallelized by channel when this is enabled.

Fluent API
~~~~~~~~~~

Modify pipelines using the fluent API:

.. code-block:: python

   pipeline = Pipeline([processor1, processor2])

   # Add processor
   pipeline.add(processor3)

   # Insert at position
   pipeline.insert(1, new_processor)

   # Remove processor
   pipeline.remove(2)

   # Method chaining
   pipeline.add(processor4).add(processor5)

Pipeline Inspection
~~~~~~~~~~~~~~~~~~~

Inspect pipeline structure:

.. code-block:: python

   # Get number of processors
   print(f"Pipeline has {len(pipeline)} processors")

   # Get processor by index
   first_processor = pipeline[0]

   # Get human-readable description
   print(pipeline.describe())

   # Serialize to dictionary
   pipeline_dict = pipeline.to_dict()

Validation
~~~~~~~~~~

Validate all processors before running:

.. code-block:: python

   # Check if pipeline can run with given context
   errors = pipeline.validate_all(context)

   if errors:
       print("Validation errors:")
       for error in errors:
           print(f"  - {error}")
   else:
       print("Pipeline is valid")
       result = pipeline.run(initial_context=context)

Pipeline Builder
----------------

The ``PipelineBuilder`` provides a fluent interface for constructing pipelines:

.. code-block:: python

   from facet.core import PipelineBuilder

   builder = PipelineBuilder(name="My Workflow")

   pipeline = (builder
       .add(EDFLoader(path="data.edf"))
       .add(TriggerDetector(regex=r"\b1\b"))
       .add(UpSample(factor=10))
       .add(AASCorrection(window_size=30))
       .add(DownSample(factor=10))
       .add(EDFExporter(path="output.edf"))
       .build())

Conditional Addition
~~~~~~~~~~~~~~~~~~~~

Add processors conditionally:

.. code-block:: python

   use_anc = True
   use_pca = False

   pipeline = (PipelineBuilder()
       .add(TriggerDetector(regex=r"\b1\b"))
       .add(AASCorrection(window_size=30))
       .add_if(use_anc, ANCCorrection(filter_order=5))
       .add_if(use_pca, PCACorrection(n_components=0.95))
       .build())

Common Pipeline Patterns
------------------------

Standard Correction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from facet import create_standard_pipeline

   pipeline = create_standard_pipeline(
       input_path="data.edf",
       output_path="corrected.edf",
       trigger_regex=r"\b1\b",
       upsample_factor=10,
       use_anc=True,
       use_pca=False
   )

   result = pipeline.run()

Custom Correction Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   pipeline = Pipeline([
       EDFLoader(path="data.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       TriggerAligner(ref_trigger_index=0),
       AASCorrection(window_size=30, correlation_threshold=0.975),
       ANCCorrection(filter_order=5, hp_freq=1.0),
       DownSample(factor=10),
       HighPassFilter(freq=0.5),
       SNRCalculator(),
       RMSCalculator(),
       MetricsReport(),
       EDFExporter(path="corrected.edf", overwrite=True)
   ], name="Custom Correction")

Batch Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

Process multiple files:

.. code-block:: python

   import glob
   from pathlib import Path

   # Create pipeline template
   def create_pipeline(input_path, output_path):
       return Pipeline([
           EDFLoader(path=input_path, preload=True),
           TriggerDetector(regex=r"\b1\b"),
           AASCorrection(window_size=30),
           EDFExporter(path=output_path, overwrite=True)
       ])

   # Process all files
   input_files = glob.glob("data/*.edf")

   for input_file in input_files:
       output_file = f"corrected/{Path(input_file).stem}_corrected.edf"
       pipeline = create_pipeline(input_file, output_file)
       result = pipeline.run()
       print(f"{input_file}: {'✓' if result.success else '✗'}")

Evaluation-Only Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate already corrected data:

.. code-block:: python

   pipeline = Pipeline([
       EDFLoader(path="corrected.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       SNRCalculator(),
       RMSCalculator(),
       MedianArtifactCalculator(),
       MetricsReport()
   ], name="Evaluation")

   result = pipeline.run()
   metrics = result.context.metadata.custom['metrics']

Error Handling
--------------

Handling Pipeline Failures
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = pipeline.run()

   if not result.success:
       print(f"Pipeline failed: {result.error}")

       # Check which processor failed
       if result.context:
           history = result.context.get_history()
           print(f"Completed {len(history)} steps before failure")

       # Re-raise exception if needed
       raise result.error

Try-Except Pattern
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   try:
       result = pipeline.run()
       if result.success:
           print("Success!")
       else:
           print(f"Failed: {result.error}")
   except Exception as e:
       print(f"Unexpected error: {e}")

Performance Tips
----------------

1. **Use Parallel Processing**
   Enable parallelization for multi-channel data:

   .. code-block:: python

      result = pipeline.run(parallel=True, n_jobs=-1)

2. **Preload Data**
   Load data into memory for faster processing:

   .. code-block:: python

      EDFLoader(path="data.edf", preload=True)

3. **Optimize Window Size**
   Larger AAS window sizes are faster but less adaptive:

   .. code-block:: python

      AASCorrection(window_size=30)  # Good balance

4. **Reduce Upsampling**
   Use lower factors if precision allows:

   .. code-block:: python

      UpSample(factor=5)  # Instead of 10

5. **Profile Pipeline**
   Check execution times:

   .. code-block:: python

      result = pipeline.run()
      print(f"Total time: {result.execution_time:.2f}s")

      for step in result.context.get_history():
          print(f"{step.name}: {step.timestamp}")

Best Practices
--------------

1. **Name Your Pipelines**

   .. code-block:: python

      Pipeline([...], name="Descriptive Name")

2. **Validate Before Running**

   .. code-block:: python

      errors = pipeline.validate_all(context)
      if not errors:
          result = pipeline.run(initial_context=context)

3. **Check Results**

   .. code-block:: python

      if result.success:
          # Process results
      else:
          # Handle error

4. **Use Type Hints**

   .. code-block:: python

      from facet.core import Pipeline, PipelineResult

      def process_file(path: str) -> PipelineResult:
          pipeline = Pipeline([...])
          return pipeline.run()

5. **Log Progress**

   FACETpy still records every detail with ``loguru`` (see ``logs/*.log``), but the
   console now streams a Rich-powered dashboard that tracks processor states,
   durations, and a live progress bar. This interactive view is enabled by
   default—just run your pipeline and watch the terminal update in place.

   Prefer the legacy line-by-line console output? Export
   ``FACET_CONSOLE_MODE=classic`` (or ``legacy``) before starting Python and
   you'll get the traditional loguru sink back while file logging remains
   untouched.

   .. code-block:: bash

      FACET_CONSOLE_MODE=classic python my_pipeline.py

   You can still log explicitly from processors using loguru:

   .. code-block:: python

      from loguru import logger

      logger.info("Starting pipeline")
      result = pipeline.run()
      logger.info(f"Completed in {result.execution_time:.2f}s")

Next Steps
----------

- Learn about :doc:`processors` in detail
- Explore :doc:`parallel_processing` capabilities
- Create :doc:`custom_processors` for your needs
- Check out :doc:`../getting_started/examples` for complete workflows
