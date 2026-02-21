Parallel Processing
===================

FACETpy supports parallel processing to speed up computation on multi-channel data.
This guide explains how to use parallelization effectively.

Overview
--------

FACETpy offers two main approaches to parallelization:

1. **Pipeline-level parallelization** - Process multiple files concurrently
2. **Processor-level parallelization** - Split channels across CPU cores

Both can be combined for maximum performance on large datasets.

Quick Start
-----------

Enable parallel processing in your pipeline:

.. code-block:: python

   from facet.core import Pipeline
   from facet.preprocessing import TriggerDetector
   from facet.correction import AASCorrection

   pipeline = Pipeline([
       TriggerDetector(regex=r"\b1\b"),
       AASCorrection(window_size=30)
   ])

   # Use all CPU cores
   result = pipeline.run(parallel=True, n_jobs=-1)

Processor-Level Parallelization
--------------------------------

How It Works
~~~~~~~~~~~~

When a processor is marked as ``parallel_safe = True``, the ``ParallelExecutor``
can automatically split processing by channels:

.. code-block:: text

   Original Data (4 channels)
   ┌─────────────────────────┐
   │ Ch1, Ch2, Ch3, Ch4      │
   └───────────┬─────────────┘
               │
      Split by channels
               │
   ┌───────────┴─────────────┐
   │                         │
   │   ┌──────┐   ┌──────┐  │
   │   │ Ch1  │   │ Ch3  │  │
   │   │ Ch2  │   │ Ch4  │  │
   │   └──────┘   └──────┘  │
   │                         │
   │   Process in parallel   │
   │                         │
   │   ┌──────┐   ┌──────┐  │
   │   │ Ch1' │   │ Ch3' │  │
   │   │ Ch2' │   │ Ch4' │  │
   │   └──────┘   └──────┘  │
   │                         │
   └───────────┬─────────────┘
               │
       Merge channels
               │
   ┌───────────▼─────────────┐
   │ Ch1', Ch2', Ch3', Ch4'  │
   └─────────────────────────┘

Enabling Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from facet.core import Pipeline
   from facet.io import Loader
   from facet.preprocessing import TriggerDetector
   from facet.correction import AASCorrection

   pipeline = Pipeline([
       Loader(path="data.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       AASCorrection(window_size=30)  # parallel_safe = True
   ])

   # Enable parallel processing
   result = pipeline.run(
       parallel=True,  # Enable parallelization
       n_jobs=-1       # Use all CPU cores
   )

Specifying Number of Jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use all cores
   result = pipeline.run(parallel=True, n_jobs=-1)

   # Use 4 cores
   result = pipeline.run(parallel=True, n_jobs=4)

   # Use half of available cores
   import multiprocessing
   n_cores = multiprocessing.cpu_count() // 2
   result = pipeline.run(parallel=True, n_jobs=n_cores)

Which Processors Support Parallelization?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check the ``parallel_safe`` flag:

.. code-block:: python

   from facet.correction import AASCorrection
   from facet.preprocessing import TriggerDetector

   aas = AASCorrection(window_size=30)
   detector = TriggerDetector(regex=r"\b1\b")

   print(f"AAS parallel safe: {aas.parallel_safe}")  # True
   print(f"Detector parallel safe: {detector.parallel_safe}")  # False

Processors marked ``parallel_safe = True``:

- ``AASCorrection``
- ``ANCCorrection``
- ``PCACorrection``
- ``HighPassFilter``
- ``LowPassFilter``
- ``BandPassFilter``
- ``NotchFilter``

Processors that are NOT parallel safe:

- ``TriggerDetector`` (operates on annotations)
- ``TriggerAligner`` (requires cross-channel information)
- ``Loader``/``EDFExporter`` (I/O operations)

ParallelExecutor
~~~~~~~~~~~~~~~~

The ``ParallelExecutor`` handles channel splitting automatically:

.. code-block:: python

   from facet.core import ParallelExecutor

   # Manual usage (usually not needed)
   executor = ParallelExecutor(n_jobs=4)
   result_context = executor.execute(processor, input_context)

Pipeline-Level Parallelization
-------------------------------

Process Multiple Files
~~~~~~~~~~~~~~~~~~~~~~

Use Python's ``concurrent.futures`` to process multiple files in parallel:

.. code-block:: python

   import concurrent.futures
   from pathlib import Path

   def process_file(input_path):
       """Process a single file."""
       output_path = f"corrected/{Path(input_path).stem}_corrected.edf"

       pipeline = Pipeline([
           Loader(path=input_path, preload=True),
           TriggerDetector(regex=r"\b1\b"),
           AASCorrection(window_size=30),
           EDFExporter(path=output_path, overwrite=True)
       ])

       return pipeline.run()

   # Process multiple files in parallel
   input_files = list(Path("data").glob("*.edf"))

   with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
       results = list(executor.map(process_file, input_files))

   # Check results
   for input_file, result in zip(input_files, results):
       status = "✓" if result.success else "✗"
       print(f"{status} {input_file.name}")

Thread Pool vs Process Pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**ThreadPoolExecutor** - Faster startup, shared memory:

.. code-block:: python

   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
       results = executor.map(process_file, files)

**ProcessPoolExecutor** - True parallelism, isolated memory:

.. code-block:: python

   with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
       results = executor.map(process_file, files)

Use ``ProcessPoolExecutor`` for CPU-intensive work (recommended).

Batch Processing Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor, as_completed
   from tqdm import tqdm

   def process_with_progress(input_files, n_workers=4):
       """Process files with progress bar."""
       results = {}

       with ProcessPoolExecutor(max_workers=n_workers) as executor:
           # Submit all jobs
           future_to_file = {
               executor.submit(process_file, f): f
               for f in input_files
           }

           # Process as they complete
           for future in tqdm(as_completed(future_to_file),
                             total=len(input_files),
                             desc="Processing"):
               input_file = future_to_file[future]
               try:
                   result = future.result()
                   results[input_file] = result
               except Exception as e:
                   print(f"Error processing {input_file}: {e}")
                   results[input_file] = None

       return results

   # Use it
   input_files = list(Path("data").glob("*.edf"))
   results = process_with_progress(input_files, n_workers=4)

Performance Considerations
--------------------------

When to Use Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Good use cases:**

- Multi-channel data (>10 channels)
- Long recordings (>10 minutes)
- Batch processing multiple files
- CPU-intensive operations (AAS, ANC, PCA)

**Not beneficial:**

- Single-channel data
- Very short recordings (<1 minute)
- Single file processing with few channels
- I/O-bound operations

Optimal Number of Workers
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import multiprocessing

   # Get CPU count
   n_cpus = multiprocessing.cpu_count()
   print(f"Available CPUs: {n_cpus}")

   # General guidelines:
   # - Processor-level: Use all cores (n_jobs=-1)
   # - Pipeline-level: Use n_cpus - 1 (leave one for system)

   # For mixed workload
   n_pipeline_workers = max(1, n_cpus - 1)
   n_processor_jobs = 2  # Use fewer per-file to allow more files

Memory Usage
~~~~~~~~~~~~

Parallel processing increases memory usage:

.. code-block:: python

   # Memory per worker ≈ raw data size + processing overhead

   # Example: 32 channels, 5000 Hz, 10 min recording
   # Memory ≈ 32 * 5000 * 600 * 8 bytes ≈ 750 MB per worker

   # With 4 workers: 4 * 750 MB = 3 GB

   # Monitor memory
   import psutil
   print(f"Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB")

Overhead
~~~~~~~~

Parallelization has overhead:

- Starting worker processes
- Splitting/merging data
- Inter-process communication

Only worth it if processing time >> overhead.

.. code-block:: python

   # Rule of thumb:
   # Parallelize if processing time > 10 seconds per file

Benchmarking
------------

Compare Performance
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time

   # Sequential processing
   start = time.time()
   result = pipeline.run(parallel=False)
   sequential_time = time.time() - start

   # Parallel processing
   start = time.time()
   result = pipeline.run(parallel=True, n_jobs=-1)
   parallel_time = time.time() - start

   # Calculate speedup
   speedup = sequential_time / parallel_time
   print(f"Sequential: {sequential_time:.2f}s")
   print(f"Parallel:   {parallel_time:.2f}s")
   print(f"Speedup:    {speedup:.2f}x")

Scaling Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   def benchmark_scaling(pipeline, context, max_workers=8):
       """Benchmark scaling with different worker counts."""
       results = {}

       for n_workers in range(1, max_workers + 1):
           start = time.time()
           result = pipeline.run(
               initial_context=context,
               parallel=True,
               n_jobs=n_workers
           )
           elapsed = time.time() - start
           results[n_workers] = elapsed

       return results

   # Run benchmark
   scaling = benchmark_scaling(pipeline, context)

   # Plot results
   import matplotlib.pyplot as plt

   workers = list(scaling.keys())
   times = list(scaling.values())

   plt.plot(workers, times, marker='o')
   plt.xlabel('Number of Workers')
   plt.ylabel('Processing Time (s)')
   plt.title('Parallel Scaling')
   plt.grid(True)
   plt.show()

Best Practices
--------------

1. **Profile First**

   Measure before optimizing:

   .. code-block:: python

      # Time your pipeline
      import time
      start = time.time()
      result = pipeline.run()
      print(f"Time: {time.time() - start:.2f}s")

2. **Start Conservative**

   Begin with fewer workers:

   .. code-block:: python

      # Start with 2-4 workers
      result = pipeline.run(parallel=True, n_jobs=4)

      # Scale up if beneficial
      result = pipeline.run(parallel=True, n_jobs=-1)

3. **Monitor Resources**

   Watch CPU and memory:

   .. code-block:: python

      import psutil

      # Before processing
      cpu_percent = psutil.cpu_percent(interval=1)
      mem = psutil.virtual_memory()
      print(f"CPU: {cpu_percent}%, Memory: {mem.percent}%")

4. **Handle Errors**

   Parallel processing can hide errors:

   .. code-block:: python

      try:
          result = pipeline.run(parallel=True, n_jobs=-1)
          if not result.success:
              print(f"Pipeline failed: {result.error}")
      except Exception as e:
          print(f"Unexpected error: {e}")

5. **Batch Appropriately**

   Group files by size:

   .. code-block:: python

      # Process large files one at a time
      large_files = [f for f in files if f.stat().st_size > 1e9]

      # Process small files in parallel
      small_files = [f for f in files if f.stat().st_size <= 1e9]

Troubleshooting
---------------

Parallel Processing Not Working
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check if processors support parallelization:

.. code-block:: python

   for proc in pipeline.processors:
       if not proc.parallel_safe:
           print(f"{proc.name} is not parallel safe")

Out of Memory Errors
~~~~~~~~~~~~~~~~~~~~

Reduce number of workers:

.. code-block:: python

   # Use fewer workers
   result = pipeline.run(parallel=True, n_jobs=2)

   # Or disable parallelization
   result = pipeline.run(parallel=False)

Slower with Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overhead may exceed benefit:

.. code-block:: python

   # Disable parallelization for small datasets
   if n_channels < 10 or duration < 60:
       result = pipeline.run(parallel=False)
   else:
       result = pipeline.run(parallel=True, n_jobs=-1)

Advanced Topics
---------------

Custom Parallel Executors
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom parallel execution:

.. code-block:: python

   from facet.core import ParallelExecutor

   class CustomExecutor(ParallelExecutor):
       def __init__(self, n_jobs=-1, backend='loky'):
           super().__init__(n_jobs=n_jobs)
           self.backend = backend

       def execute(self, processor, context):
           # Custom parallel execution logic
           return super().execute(processor, context)

GPU Acceleration
~~~~~~~~~~~~~~~~

FACETpy doesn't currently support GPU acceleration, but you can
create custom processors that use GPU libraries:

.. code-block:: python

   from facet.core import Processor
   import torch  # or cupy, jax, etc.

   class GPUProcessor(Processor):
       name = "gpu_processor"
       parallel_safe = False  # GPU handles parallelization

       def process(self, context):
           raw = context.get_raw()
           data = torch.from_numpy(raw._data).cuda()

           # GPU processing
           result = self.gpu_function(data)

           # Copy back to CPU
           raw._data = result.cpu().numpy()
           return context.with_raw(raw)

Distributed Processing
~~~~~~~~~~~~~~~~~~~~~~

For cluster environments, use Dask:

.. code-block:: python

   from dask.distributed import Client
   from dask import delayed

   # Connect to cluster
   client = Client('scheduler-address:8786')

   # Create delayed tasks
   tasks = [delayed(process_file)(f) for f in files]

   # Compute in parallel
   results = client.compute(tasks)

Next Steps
----------

- Learn about :doc:`pipelines` for workflow composition
- Explore :doc:`processors` for available operations
- Check :doc:`custom_processors` for creating parallel-safe processors
- See :doc:`../getting_started/examples` for complete examples
