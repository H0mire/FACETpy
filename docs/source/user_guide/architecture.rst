Architecture Overview
=====================

FACETpy 2.0 is built on a modular, pipeline-based architecture that provides flexibility,
extensibility, and ease of use.

Core Concepts
-------------

The architecture consists of four main components:

1. **Processors** - Individual processing steps
2. **Context** - Data container passed between processors
3. **Pipeline** - Workflow orchestrator
4. **Registry** - Plugin discovery system

.. image:: ../_static/architecture_diagram.png
   :align: center
   :width: 600px
   :alt: Architecture diagram (if available)

Processors
----------

Processors are the building blocks of FACETpy. Each processor:

* Performs a single, well-defined operation
* Receives a ``ProcessingContext`` as input
* Returns a new ``ProcessingContext`` as output
* Is independently testable and reusable

Example Processor
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from facet.core import Processor, register_processor

   @register_processor
   class MyProcessor(Processor):
       name = "my_processor"
       description = "Does something useful"

       def __init__(self, param1, param2=default):
           self.param1 = param1
           self.param2 = param2
           super().__init__()

       def validate(self, context):
           """Validate prerequisites before processing."""
           super().validate(context)
           if not context.has_triggers():
               raise ProcessorValidationError("Triggers required")

       def process(self, context):
           """Main processing logic."""
           raw = context.get_raw()

           # Do something with the data
           # ...

           return context.with_raw(modified_raw)

Processor Lifecycle
~~~~~~~~~~~~~~~~~~~

When ``processor.execute(context)`` is called:

1. **Validate** - Check prerequisites
2. **Process** - Execute main logic
3. **Record** - Add history entry
4. **Return** - Return new context

.. code-block:: python

   result_context = processor.execute(input_context)

Processing Context
------------------

The ``ProcessingContext`` is a container that holds:

* **Raw Data** - MNE Raw object with EEG data
* **Metadata** - Triggers, artifact info, parameters
* **Estimated Noise** - Accumulated artifact estimates
* **Processing History** - Record of all operations

Context is Immutable
~~~~~~~~~~~~~~~~~~~~

Context follows an immutable-by-default pattern:

.. code-block:: python

   # Creating new contexts
   context1 = ProcessingContext(raw=raw, metadata=metadata)
   context2 = context1.with_raw(new_raw)  # context1 unchanged
   context3 = context2.with_metadata(new_metadata)  # context2 unchanged

This prevents accidental modifications and makes debugging easier.

Accessing Data
~~~~~~~~~~~~~~

.. code-block:: python

   # Get data
   raw = context.get_raw()  # Current processed data
   raw_orig = context.get_raw_original()  # Original data
   triggers = context.get_triggers()  # Trigger positions
   noise = context.get_estimated_noise()  # Artifact estimates

   # Check availability
   if context.has_triggers():
       triggers = context.get_triggers()

   if context.has_estimated_noise():
       noise = context.get_estimated_noise()

Processing Metadata
~~~~~~~~~~~~~~~~~~~

Metadata tracks processing parameters:

.. code-block:: python

   metadata = context.metadata

   # Standard fields
   triggers = metadata.triggers
   artifact_length = metadata.artifact_length
   upsampling_factor = metadata.upsampling_factor

   # Custom data
   metadata.custom['my_key'] = my_value
   my_value = metadata.custom.get('my_key')

Pipeline
--------

Pipeline orchestrates processor execution:

.. code-block:: python

   from facet.core import Pipeline

   pipeline = Pipeline([
       processor1,
       processor2,
       processor3
   ], name="My Pipeline")

   result = pipeline.run()

Pipeline Features
~~~~~~~~~~~~~~~~~

**Sequential Execution**

.. code-block:: python

   result = pipeline.run()  # Runs processors in order

**Parallel Execution**

.. code-block:: python

   result = pipeline.run(parallel=True, n_jobs=-1)

**Initial Context**

.. code-block:: python

   initial_context = ProcessingContext(raw=raw)
   result = pipeline.run(initial_context=initial_context)

**Error Handling**

.. code-block:: python

   result = pipeline.run()

   if result.success:
       final_context = result.context
       print(f"Completed in {result.execution_time:.2f}s")
   else:
       print(f"Failed at: {result.failed_processor}")
       print(f"Error: {result.error}")

Composite Processors
~~~~~~~~~~~~~~~~~~~~

Build complex workflows with composite processors:

.. code-block:: python

   from facet.core import SequenceProcessor, ConditionalProcessor

   # Run sequence of processors
   correction_sequence = SequenceProcessor([
       AASCorrection(window_size=30),
       ANCCorrection()
   ])

   # Conditional execution
   conditional_pca = ConditionalProcessor(
       condition=lambda ctx: ctx.metadata.custom.get('needs_pca', False),
       processor=PCACorrection(n_components=0.95)
   )

Registry
--------

The registry provides plugin discovery and management.

Registration
~~~~~~~~~~~~

Register processors with a decorator:

.. code-block:: python

   from facet.core import register_processor

   @register_processor
   class MyProcessor(Processor):
       name = "my_processor"  # Unique identifier

Discovery
~~~~~~~~~

.. code-block:: python

   from facet.core import get_processor, list_processors

   # Get processor class by name
   ProcessorClass = get_processor("aas_correction")
   processor = ProcessorClass(window_size=30)

   # List all registered processors
   all_processors = list_processors()
   for name, proc_class in all_processors.items():
       print(f"{name}: {proc_class.__name__}")

Parallel Execution
------------------

FACETpy supports two types of parallelization:

Pipeline-Level Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute multiple pipelines concurrently:

.. code-block:: python

   import concurrent.futures

   pipelines = [create_pipeline(file) for file in files]

   with concurrent.futures.ProcessPoolExecutor() as executor:
       results = executor.map(lambda p: p.run(), pipelines)

Processor-Level Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Processors marked as ``parallel_safe`` can parallelize internally:

.. code-block:: python

   @register_processor
   class MyProcessor(Processor):
       parallel_safe = True  # Can be parallelized
       parallelize_by_channels = True  # Split by channels

       def process(self, context):
           # This will run in parallel when pipeline.run(parallel=True)
           ...

Channel-Wise Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ParallelExecutor`` automatically splits by channels:

.. code-block:: python

   pipeline = Pipeline([
       Loader(path="data.edf"),
       TriggerDetector(regex=r"\b1\b"),
       AASCorrection(window_size=30)  # Will parallelize by channel
   ])

   result = pipeline.run(parallel=True, n_jobs=-1)

Data Flow
---------

Typical data flow through FACETpy:

.. code-block:: text

   ┌─────────────┐
   │ Load Data   │ Loader
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Detect      │ TriggerDetector
   │ Triggers    │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Upsample    │ UpSample
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Align       │ TriggerAligner
   │ Triggers    │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ AAS         │ AASCorrection
   │ Correction  │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ ANC         │ ANCCorrection
   │ Correction  │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Downsample  │ DownSample
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Export      │ EDFExporter
   └─────────────┘

Each arrow represents a ``ProcessingContext`` being passed between processors.

Design Principles
-----------------

1. **Single Responsibility**
   Each processor does one thing well

2. **Immutability**
   Contexts are not modified in-place

3. **Composability**
   Processors can be combined in any order

4. **Explicit over Implicit**
   Clear validation and error messages

5. **MNE Integration**
   First-class support for MNE objects

6. **Extensibility**
   Easy to add custom processors

7. **Testability**
   Each component independently testable

Benefits
--------

This architecture provides:

* **Flexibility** - Build any workflow
* **Reusability** - Share processors across projects
* **Maintainability** - Clear separation of concerns
* **Debuggability** - Track data flow with history
* **Performance** - Built-in parallelization
* **Extensibility** - Plugin system for custom needs

Next Steps
----------

* Check out the :doc:`../getting_started/tutorial` for hands-on examples
* Review the :doc:`../api/core` documentation for detailed API reference
* Learn about creating custom processors in the tutorial
* Explore the example workflows in :doc:`../getting_started/examples`
