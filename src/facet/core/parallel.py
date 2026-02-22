"""
Parallel Execution Module

This module provides multiprocessing support for pipeline execution.

Author: FACETpy Team
Date: 2025-01-12
"""

import functools
import multiprocessing as mp
import sys
from typing import Callable, List, Optional
from loguru import logger
import numpy as np
import mne
from .processor import Processor
from .context import ProcessingContext
from facet.logging_config import suppress_stdout
from facet.console.progress import processor_progress

# Use "spawn" so child processes start clean without inheriting threads
# (e.g. the Textual TUI thread).  "fork" is unsafe in multithreaded
# processes and can deadlock when the forked child inherits held locks.
# Workers already serialise everything via to_dict/from_dict, so spawn
# is a drop-in replacement.
if sys.platform != "win32":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


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
    processor_class = processor_config['class']
    processor_params = processor_config['parameters']
    processor = processor_class(**processor_params)

    context = ProcessingContext.from_dict(context_data)

    result = processor.execute(context)

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

        if getattr(processor, 'channel_wise', False):
            return self._execute_channel_wise(processor, context)

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

        if n_channels == 0:
            logger.warning("No channels available for parallel execution")
            return context

        channel_chunks = self._split_into_chunks(
            list(range(n_channels)),
            self.n_jobs
        )

        progress_total = n_channels if n_channels > 0 else None
        with processor_progress(
            total=progress_total,
            message=f"{processor.name}: channels",
        ) as progress:

            def _update_progress(processed: int) -> None:
                if processed <= 0:
                    return
                next_value = progress.current + processed
                progress.advance(
                    processed,
                    message=(
                        f"{int(next_value)}/{n_channels} channels"
                        if n_channels
                        else "channels"
                    ),
                )

            if self.backend == "multiprocessing":
                results = self._execute_multiprocessing(
                    processor,
                    context,
                    channel_chunks,
                    progress_callback=_update_progress,
                )
            elif self.backend == "threading":
                results = self._execute_threading(
                    processor,
                    context,
                    channel_chunks,
                    progress_callback=_update_progress,
                )
            else:  # serial
                results = self._execute_serial(
                    processor,
                    context,
                    channel_chunks,
                    progress_callback=_update_progress,
                )

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

        if not context.has_triggers():
            raise ValueError("Context has no triggers for epoch-wise processing")

        triggers = context.get_triggers()
        n_epochs = len(triggers)

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

        return self._merge_epoch_results(context, results)

    def _execute_multiprocessing(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: List[List[int]],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[ProcessingContext]:
        """Execute using multiprocessing."""
        processor_config = {
            'class': processor.__class__,
            'parameters': processor._parameters
        }

        chunk_contexts = []
        for chunk in channel_chunks:
            chunk_context = self._create_channel_subset_context(context, chunk)
            chunk_contexts.append(chunk_context.to_dict())

        worker = functools.partial(_worker_function, processor_config)
        contexts: List[ProcessingContext] = []
        with mp.Pool(processes=self.n_jobs) as pool:
            for idx, result in enumerate(pool.imap(worker, chunk_contexts)):
                contexts.append(ProcessingContext.from_dict(result))
                if progress_callback:
                    chunk_size = len(channel_chunks[idx])
                    progress_callback(chunk_size)

        return contexts

    def _execute_threading(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: List[List[int]],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[ProcessingContext]:
        """Execute using threading (GIL-limited)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[ProcessingContext] = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            for chunk in channel_chunks:
                chunk_context = self._create_channel_subset_context(context, chunk)
                future = executor.submit(processor.execute, chunk_context)
                futures[future] = len(chunk)

            for future in as_completed(futures):
                results.append(future.result())
                if progress_callback:
                    progress_callback(futures[future])

        return results

    def _execute_serial(
        self,
        processor: Processor,
        context: ProcessingContext,
        channel_chunks: List[List[int]],
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[ProcessingContext]:
        """Execute serially (for debugging/comparison)."""
        results = []
        for chunk in channel_chunks:
            chunk_context = self._create_channel_subset_context(context, chunk)
            result = processor.execute(chunk_context)
            results.append(result)
            if progress_callback:
                progress_callback(len(chunk))
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

        subset_ctx = context.with_raw(subset_raw, copy_metadata=True)

        if context.has_estimated_noise():
            noise = context.get_estimated_noise()
            if noise is not None and noise.ndim == 2:
                subset_noise = noise[channel_indices, :]
                subset_ctx.set_estimated_noise(subset_noise.copy())

        return subset_ctx

    def _merge_channel_results(
        self,
        original_context: ProcessingContext,
        results: List[ProcessingContext]
    ) -> ProcessingContext:
        """Merge channel-wise results back into single context."""
        if not results:
            return original_context

        original_raw = original_context.get_raw()
        template_raw = results[0].get_raw()
        template_data = results[0].get_data(copy=False)

        new_sfreq = template_raw.info['sfreq']
        n_times = template_data.shape[1]
        dtype = template_data.dtype

        merged_data = np.zeros(
            (len(original_raw.ch_names), n_times),
            dtype=dtype
        )
        channel_index = {
            name: idx for idx, name in enumerate(original_raw.ch_names)
        }

        for result_ctx in results:
            result_raw = result_ctx.get_raw()
            result_data = result_ctx.get_data(copy=False)
            for j, ch_name in enumerate(result_raw.ch_names):
                ch_idx = channel_index[ch_name]
                merged_data[ch_idx] = result_data[j]

        # Build new RawArray at the upsampled rate to avoid mutating protected info
        info = original_raw.info.copy()
        if hasattr(info, "_unlock"):
            with info._unlock():
                info['sfreq'] = new_sfreq
        else:
            info['sfreq'] = new_sfreq

        with suppress_stdout():
            new_raw = mne.io.RawArray(
                data=merged_data,
                info=info
            )

        merged_context = original_context.with_raw(new_raw)
        merged_context._metadata = results[0].metadata.copy()

        if any(result_ctx.has_estimated_noise() for result_ctx in results):
            # Estimated noise is stored channel-wise; merge similarly
            noise_data = np.zeros_like(merged_data)
            for result_ctx in results:
                if not result_ctx.has_estimated_noise():
                    continue
                result_noise = result_ctx.get_estimated_noise()
                result_raw = result_ctx.get_raw()
                for j, ch_name in enumerate(result_raw.ch_names):
                    ch_idx = channel_index[ch_name]
                    noise_data[ch_idx] = result_noise[j]
            merged_context.set_estimated_noise(noise_data)

        return merged_context

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
