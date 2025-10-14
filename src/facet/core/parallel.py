"""
Parallel Execution Module

This module provides multiprocessing support for pipeline execution.

Author: FACETpy Team
Date: 2025-01-12
"""

import multiprocessing as mp
from typing import List, Optional, Callable
from loguru import logger
import numpy as np
from .processor import Processor
from .context import ProcessingContext


def _worker_function(processor_config: dict, context_data: dict) -> dict:
    """
    Worker function for multiprocessing.

    This function runs in a separate process and must be picklable.

    Args:
        processor_config: Serialized processor configuration
        context_data: Serialized context data

    Returns:
        Serialized result context
    """
    # Reconstruct processor
    processor_class = processor_config['class']
    processor_params = processor_config['parameters']
    processor = processor_class(**processor_params)

    # Reconstruct context
    context = ProcessingContext.from_dict(context_data)

    # Process
    result = processor.execute(context)

    # Serialize result
    return result.to_dict()


class ParallelExecutor:
    """
    Executor for parallel processing of channels or epochs.

    This class handles multiprocessing for processors that support it,
    typically for channel-wise or epoch-wise operations.

    Example:
        executor = ParallelExecutor(n_jobs=4)
        result_context = executor.execute(processor, context)
    """

    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = "multiprocessing",
        verbose: bool = True
    ):
        """
        Initialize parallel executor.

        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs, -2 for all but one)
            backend: Parallel backend ("multiprocessing", "threading", or "serial")
            verbose: Show progress messages
        """
        self.n_jobs = self._determine_n_jobs(n_jobs)
        self.backend = backend
        self.verbose = verbose

        if backend not in ["multiprocessing", "threading", "serial"]:
            raise ValueError(
                f"Invalid backend: {backend}. "
                "Choose from: multiprocessing, threading, serial"
            )

    def _determine_n_jobs(self, n_jobs: int) -> int:
        """Determine actual number of jobs."""
        if n_jobs == -1:
            return mp.cpu_count()
        elif n_jobs == -2:
            return max(1, mp.cpu_count() - 1)
        elif n_jobs < -2:
            return max(1, mp.cpu_count() + n_jobs + 1)
        elif n_jobs == 0:
            raise ValueError("n_jobs cannot be 0")
        else:
            return n_jobs

    def execute(
        self,
        processor: Processor,
        context: ProcessingContext
    ) -> ProcessingContext:
        """
        Execute processor in parallel if possible.

        This method attempts to parallelize the processor execution.
        If parallelization is not applicable, falls back to serial execution.

        Args:
            processor: Processor to execute
            context: Input context

        Returns:
            Output context
        """
        if not processor.parallel_safe:
            logger.warning(
                f"Processor {processor.name} is not parallel-safe, "
                "executing serially"
            )
            return processor.execute(context)

        # Check if we can parallelize by channels
        if hasattr(processor, 'parallelize_by_channels') and processor.parallelize_by_channels:
            return self._execute_channel_wise(processor, context)

        # Check if we can parallelize by epochs
        if hasattr(processor, 'parallelize_by_epochs') and processor.parallelize_by_epochs:
            return self._execute_epoch_wise(processor, context)

        # Fall back to serial execution
        logger.debug(
            f"No parallelization strategy found for {processor.name}, "
            "executing serially"
        )
        return processor.execute(context)

    def _execute_channel_wise(
        self,
        processor: Processor,
        context: ProcessingContext
    ) -> ProcessingContext:
        """
        Execute processor channel-wise in parallel.

        Args:
            processor: Processor to execute
            context: Input context

        Returns:
            Output context with processed channels
        """
        logger.info(
            f"Executing {processor.name} in parallel across {self.n_jobs} jobs"
        )

        raw = context.get_raw()
        n_channels = len(raw.ch_names)

        # Split channels into chunks
        channel_chunks = self._split_into_chunks(
            list(range(n_channels)),
            self.n_jobs
        )

        if self.backend == "multiprocessing":
            results = self._execute_multiprocessing(
                processor,
                context,
                channel_chunks
            )
        elif self.backend == "threading":
            results = self._execute_threading(
                processor,
                context,
                channel_chunks
            )
        else:  # serial
            results = self._execute_serial(
                processor,
                context,
                channel_chunks
            )

        # Merge results
        return self._merge_channel_results(context, results)

    def _execute_epoch_wise(
        self,
        processor: Processor,
        context: ProcessingContext
    ) -> ProcessingContext:
        """
        Execute processor epoch-wise in parallel.

        Args:
            processor: Processor to execute
            context: Input context

        Returns:
            Output context with processed epochs
        """
        logger.info(
            f"Executing {processor.name} epoch-wise in parallel "
            f"across {self.n_jobs} jobs"
        )

        # Get epochs
        if not context.has_triggers():
            raise ValueError("Context has no triggers for epoch-wise processing")

        triggers = context.get_triggers()
        n_epochs = len(triggers)

        # Split epochs into chunks
        epoch_chunks = self._split_into_chunks(
            list(range(n_epochs)),
            self.n_jobs
        )

        if self.backend == "multiprocessing":
            results = self._execute_multiprocessing_epochs(
                processor,
                context,
                epoch_chunks
            )
        elif self.backend == "threading":
            results = self._execute_threading_epochs(
                processor,
                context,
                epoch_chunks
            )
        else:  # serial
            results = self._execute_serial_epochs(
                processor,
                context,
                epoch_chunks
            )

        # Merge results
        return self._merge_epoch_results(context, results)

    def _execute_multiprocessing(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: List[List[int]]
    ) -> List[ProcessingContext]:
        """Execute using multiprocessing."""
        # Prepare processor config
        processor_config = {
            'class': processor.__class__,
            'parameters': processor._parameters
        }

        # Create contexts for each chunk
        chunk_contexts = []
        for chunk in channel_chunks:
            # Create a context with only selected channels
            chunk_context = self._create_channel_subset_context(context, chunk)
            chunk_contexts.append(chunk_context.to_dict())

        # Execute in parallel
        with mp.Pool(processes=self.n_jobs) as pool:
            results = pool.starmap(
                _worker_function,
                [(processor_config, ctx_data) for ctx_data in chunk_contexts]
            )

        # Convert results back to contexts
        return [ProcessingContext.from_dict(result) for result in results]

    def _execute_threading(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: List[List[int]]
    ) -> List[ProcessingContext]:
        """Execute using threading (GIL-limited)."""
        from concurrent.futures import ThreadPoolExecutor

        results = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for chunk in channel_chunks:
                chunk_context = self._create_channel_subset_context(context, chunk)
                future = executor.submit(processor.execute, chunk_context)
                futures.append(future)

            results = [future.result() for future in futures]

        return results

    def _execute_serial(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: List[List[int]]
    ) -> List[ProcessingContext]:
        """Execute serially (for debugging/comparison)."""
        results = []
        for chunk in channel_chunks:
            chunk_context = self._create_channel_subset_context(context, chunk)
            result = processor.execute(chunk_context)
            results.append(result)
        return results

    def _create_channel_subset_context(
        self,
        context: ProcessingContext,
        channel_indices: List[int]
    ) -> ProcessingContext:
        """Create context with subset of channels."""
        raw = context.get_raw()
        picks = [raw.ch_names[i] for i in channel_indices]
        subset_raw = raw.copy().pick(picks)

        return context.with_raw(subset_raw, copy_metadata=True)

    def _merge_channel_results(
        self,
        original_context: ProcessingContext,
        results: List[ProcessingContext]
    ) -> ProcessingContext:
        """Merge channel-wise results back into single context."""
        # Get original raw structure
        raw = original_context.get_raw().copy()

        # Merge data from all chunks
        for i, result_ctx in enumerate(results):
            result_raw = result_ctx.get_raw()
            result_data = result_raw.get_data(copy=False)
            result_channels = result_raw.ch_names

            # Find channel indices in original
            for j, ch_name in enumerate(result_channels):
                ch_idx = raw.ch_names.index(ch_name)
                raw._data[ch_idx] = result_data[j]

        return original_context.with_raw(raw)

    def _merge_epoch_results(
        self,
        original_context: ProcessingContext,
        results: List[ProcessingContext]
    ) -> ProcessingContext:
        """Merge epoch-wise results."""
        # Implementation depends on how epochs are stored
        # This is a placeholder
        logger.warning("Epoch-wise merging not fully implemented yet")
        return original_context

    def _split_into_chunks(self, items: List, n_chunks: int) -> List[List]:
        """Split list into approximately equal chunks."""
        chunk_size = len(items) // n_chunks
        remainder = len(items) % n_chunks

        chunks = []
        start = 0
        for i in range(n_chunks):
            # Distribute remainder across first chunks
            size = chunk_size + (1 if i < remainder else 0)
            if size > 0:
                chunks.append(items[start:start + size])
                start += size

        return chunks

    # Epoch-wise methods (placeholders for now)
    def _execute_multiprocessing_epochs(self, processor, context, epoch_chunks):
        """Execute epoch-wise using multiprocessing."""
        raise NotImplementedError("Epoch-wise multiprocessing not yet implemented")

    def _execute_threading_epochs(self, processor, context, epoch_chunks):
        """Execute epoch-wise using threading."""
        raise NotImplementedError("Epoch-wise threading not yet implemented")

    def _execute_serial_epochs(self, processor, context, epoch_chunks):
        """Execute epoch-wise serially."""
        raise NotImplementedError("Epoch-wise serial execution not yet implemented")
