"""
Pipeline Module

This module defines the Pipeline class for executing sequences of processors.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import List, Optional, Dict, Any, Tuple, Callable, Union
from loguru import logger
from .processor import Processor
from .context import ProcessingContext
from .parallel import ParallelExecutor
from ..console import get_console
from ..console.progress import set_current_step_index


class PipelineError(Exception):
    """Base exception for pipeline-related errors."""
    pass


class PipelineResult:
    """
    Result of pipeline execution.

    Contains the final context and metadata about execution.
    """

    def __init__(
        self,
        context: Optional[ProcessingContext],
        success: bool = True,
        error: Optional[Exception] = None,
        execution_time: float = 0.0,
        failed_processor: Optional[str] = None,
        failed_processor_index: Optional[int] = None
    ):
        """
        Initialize pipeline result.

        Args:
            context: Final processing context
            success: Whether pipeline completed successfully
            error: Exception if pipeline failed
            execution_time: Total execution time in seconds
        """
        self.context = context
        self.success = success
        self.error = error
        self.execution_time = execution_time
        self.failed_processor = failed_processor
        self.failed_processor_index = failed_processor_index

    def get_context(self) -> ProcessingContext:
        """Get final processing context."""
        return self.context

    def get_raw(self):
        """Get final raw data (convenience method)."""
        return self.context.get_raw()

    def get_history(self):
        """Get processing history."""
        return self.context.get_history()

    def was_successful(self) -> bool:
        """Check if pipeline succeeded."""
        return self.success

    @property
    def metrics(self) -> Dict[str, Any]:
        """
        Shortcut to evaluation metrics stored in context.

        Returns the ``metrics`` dict from ``context.metadata.custom``, or an
        empty dict if no metrics have been calculated yet.

        Example::

            result = pipeline.run()
            print(result.metrics['snr'])
        """
        if self.context is None:
            return {}
        return self.context.metadata.custom.get('metrics', {})

    @property
    def metrics_df(self):
        """
        Return scalar evaluation metrics as a ``pandas.Series``.

        Nested dicts (e.g. ``fft_allen``) are flattened with ``_`` separators.
        Returns ``None`` if pandas is not available.

        Example::

            result = pipeline.run()
            print(result.metrics_df)
        """
        try:
            import pandas as pd
        except ImportError:
            return None

        flat: Dict[str, Any] = {}
        for k, v in self.metrics.items():
            if isinstance(v, (int, float)):
                flat[k] = v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        flat[f"{k}_{sub_k}"] = sub_v
        return pd.Series(flat, name=self.context.metadata.custom.get('pipeline_name', 'metrics'))

    def plot(self, **kwargs):
        """
        Plot the corrected data using ``RawPlotter`` defaults.

        Accepts any keyword arguments supported by ``RawPlotter``.

        Example::

            result = pipeline.run()
            result.plot(channel="Fp1", start=5.0, duration=10.0)
        """
        from ..evaluation import RawPlotter
        plotter = RawPlotter(**kwargs)
        plotter.execute(self.context)

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"PipelineResult({status}, time={self.execution_time:.2f}s)"


class Pipeline:
    """
    Pipeline for executing sequences of processors.

    A pipeline orchestrates the execution of multiple processors in sequence,
    handles errors, provides progress tracking, and supports parallelization.

    Example::

        pipeline = Pipeline([
            EDFLoader("data.edf"),
            HighPassFilter(freq=1.0),
            UpSample(factor=10),
            TriggerDetector(regex=r"\\btrigger\\b"),
            AASCorrection(),
            EDFExporter("output.edf")
        ])

        result = pipeline.run()
        if result.was_successful():
            print(f"Completed in {result.execution_time:.2f}s")

    Attributes:
        processors: List of processors to execute
        name: Optional pipeline name
    """

    def __init__(
        self,
        processors: List[Union[Processor, Callable]],
        name: Optional[str] = None
    ):
        """
        Initialize pipeline.

        Plain callables (``Callable[[ProcessingContext], ProcessingContext]``)
        are automatically wrapped in a :class:`~facet.core.LambdaProcessor` so
        they can be used as inline steps without ceremony::

            pipeline = Pipeline([
                EDFLoader("data.edf"),
                HighPassFilter(1.0),
                lambda ctx: (print(ctx.get_sfreq()) or ctx),
                AASCorrection(),
            ])

        Args:
            processors: List of :class:`~facet.core.Processor` instances **or**
                plain callables to execute in order.
            name: Optional pipeline name (for logging)
        """
        self.processors = self._normalise_processors(processors)
        self.name = name or "Pipeline"

    @staticmethod
    def _normalise_processors(
        items: List[Union[Processor, Callable]],
        _index_offset: int = 0,
    ) -> List[Processor]:
        """
        Coerce each item to a :class:`Processor`.

        Plain callables are wrapped in a :class:`~facet.core.LambdaProcessor`.
        Anything else that is not a :class:`Processor` raises :exc:`TypeError`.
        """
        from .processor import LambdaProcessor
        result: List[Processor] = []
        for i, p in enumerate(items):
            if isinstance(p, Processor):
                result.append(p)
            elif callable(p):
                display_name = getattr(p, '__name__', None) or f"step_{_index_offset + i}"
                result.append(LambdaProcessor(name=display_name, func=p))
            else:
                raise TypeError(
                    f"Item at index {_index_offset + i} must be a Processor instance "
                    f"or a callable, got {type(p)}"
                )
        return result

    def _validate_pipeline(self) -> None:
        """No-op — validation now happens in _normalise_processors."""

    def run(
        self,
        initial_context: Optional[ProcessingContext] = None,
        parallel: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            initial_context: Initial context (if None, first processor creates it)
            parallel: Enable parallel execution for compatible processors
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            show_progress: Show progress bar

        Returns:
            PipelineResult containing final context and metadata
        """
        import time

        start_time = time.time()
        console = get_console()

        console.set_pipeline_metadata(
            {
                "execution_mode": "parallel" if parallel else "serial",
                "n_jobs": str(n_jobs),
            }
        )
        step_names = [processor.name for processor in self.processors]
        console.start_pipeline(self.name, len(self.processors), step_names=step_names)

        logger.info(f"Starting pipeline: {self.name}")
        logger.info(f"Number of processors: {len(self.processors)}")

        try:
            context = initial_context
            current_processor: Optional[Tuple[int, Processor]] = None

            # Execute processors
            for i, processor in enumerate(self.processors):
                current_processor = (i, processor)
                logger.info(
                    f"[{i+1}/{len(self.processors)}] Executing: {processor.name}"
                )

                console.step_started(i, processor.name)
                set_current_step_index(i)

                step_start = time.time()
                executed_in_parallel = parallel and processor.parallel_safe

                try:
                    if executed_in_parallel:
                        executor = ParallelExecutor(n_jobs=n_jobs)
                        context = executor.execute(processor, context)
                    else:
                        context = processor.execute(context)
                finally:
                    set_current_step_index(None)

                duration = time.time() - step_start
                console.step_completed(
                    i,
                    processor.name,
                    duration,
                    metrics={
                        "execution_mode": "parallel" if executed_in_parallel else "serial",
                        "last_duration": f"{duration:.2f}s",
                    },
                )

            # Success
            execution_time = time.time() - start_time
            logger.info(
                f"Pipeline completed successfully in {execution_time:.2f}s"
            )
            console.pipeline_complete(True, execution_time)

            return PipelineResult(
                context=context,
                success=True,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            if current_processor:
                failed_index, failed_proc = current_processor
                logger.error(
                    f"Pipeline failed after {execution_time:.2f}s during processor "
                    f"{failed_proc.name} (step {failed_index + 1}/{len(self.processors)}): {e}"
                )
            else:
                logger.error(
                    f"Pipeline failed after {execution_time:.2f}s before executing any "
                    f"processor: {e}"
                )
            logger.opt(exception=e).debug("Exception details")

            console.pipeline_failed(
                execution_time,
                e,
                current_processor[0] if current_processor else None,
                current_processor[1].name if current_processor else None,
            )

            return PipelineResult(
                context=context if context else None,
                success=False,
                error=e,
                execution_time=execution_time,
                failed_processor=current_processor[1].name if current_processor else None,
                failed_processor_index=current_processor[0] if current_processor else None
            )

    def add(self, processor: Union[Processor, Callable]) -> 'Pipeline':
        """
        Add a processor or callable to the pipeline (fluent API).

        Args:
            processor: :class:`~facet.core.Processor` instance or callable.

        Returns:
            Self for chaining
        """
        [normalised] = self._normalise_processors([processor], _index_offset=len(self.processors))
        self.processors.append(normalised)
        return self

    def extend(self, processors: List[Union[Processor, Callable]]) -> 'Pipeline':
        """
        Extend pipeline with multiple processors or callables.

        Args:
            processors: List of processors or callables to add.

        Returns:
            Self for chaining
        """
        self.processors.extend(
            self._normalise_processors(processors, _index_offset=len(self.processors))
        )
        return self

    def insert(self, index: int, processor: Union[Processor, Callable]) -> 'Pipeline':
        """
        Insert a processor or callable at a specific position.

        Args:
            index: Position to insert
            processor: :class:`~facet.core.Processor` instance or callable.

        Returns:
            Self for chaining
        """
        [normalised] = self._normalise_processors([processor], _index_offset=index)
        self.processors.insert(index, normalised)
        return self

    def remove(self, index: int) -> 'Pipeline':
        """
        Remove processor at index.

        Args:
            index: Index to remove

        Returns:
            Self for chaining
        """
        self.processors.pop(index)
        return self

    def validate_all(self, context: ProcessingContext) -> List[str]:
        """
        Validate all processors against a context.

        Useful for checking if a pipeline can run before actually running it.

        Args:
            context: Context to validate against

        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        for i, processor in enumerate(self.processors):
            try:
                processor.validate(context)
            except Exception as e:
                errors.append(f"Processor {i} ({processor.name}): {str(e)}")
        return errors

    def describe(self) -> str:
        """
        Get human-readable pipeline description.

        Returns:
            Multi-line string describing pipeline
        """
        lines = [f"Pipeline: {self.name}", "=" * 50]

        for i, processor in enumerate(self.processors):
            lines.append(f"{i+1}. {processor.name} ({processor.__class__.__name__})")
            if hasattr(processor, '_parameters'):
                for key, value in processor._parameters.items():
                    lines.append(f"   - {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize pipeline to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'name': self.name,
            'processors': [
                {
                    'class': proc.__class__.__name__,
                    'name': proc.name,
                    'parameters': proc._parameters if hasattr(proc, '_parameters') else {}
                }
                for proc in self.processors
            ]
        }

    def map(
        self,
        inputs: List[Union[str, ProcessingContext]],
        loader_factory: Optional[Callable[[str], 'Processor']] = None,
        parallel: bool = False,
        n_jobs: int = -1,
        on_error: str = "continue",
    ) -> List[PipelineResult]:
        """
        Run the pipeline on multiple inputs and return a result per input.

        Each input can be:

        - A ``ProcessingContext`` — passed directly as ``initial_context``.
        - A **file path string** — a loader is looked up or created automatically.
          The pipeline must contain a loader as its first processor **or** you
          must supply *loader_factory* so the method can create one per file.

        Args:
            inputs: List of file paths or ``ProcessingContext`` objects.
            loader_factory: ``Callable[[path], Processor]`` that creates a fresh
                loader for each path string.  When not provided the first
                processor in the pipeline is expected to be a loader that exposes
                a ``path`` attribute; a shallow copy of it is made for each file.
            parallel: Whether to pass ``parallel=True`` to each ``pipeline.run()``.
            n_jobs: Passed through to ``pipeline.run()``.
            on_error: ``"continue"`` (default) — log failures and keep going;
                      ``"raise"`` — re-raise the first error encountered.

        Returns:
            List of ``PipelineResult`` objects, one per input, in the same order.

        Example::

            pipeline = Pipeline([
                TriggerDetector(regex=r"\\b1\\b"),
                UpSample(factor=10),
                AASCorrection(window_size=30),
                DownSample(factor=10),
            ])

            results = pipeline.map(
                ["sub-01.edf", "sub-02.edf", "sub-03.edf"],
                loader_factory=lambda p: EDFLoader(path=p, preload=True),
            )

            for path, result in zip(files, results):
                if result.success:
                    print(f"{path}: SNR={result.metrics.get('snr', 'N/A')}")
        """
        results: List[PipelineResult] = []

        for item in inputs:
            if isinstance(item, ProcessingContext):
                initial_ctx = item
                label = repr(item)
            else:
                label = str(item)
                initial_ctx = None

                if loader_factory is not None:
                    loader = loader_factory(item)
                    try:
                        initial_ctx = loader.execute(None)
                    except Exception as exc:
                        logger.error(f"Loader failed for '{label}': {exc}")
                        result = PipelineResult(
                            context=None,
                            success=False,
                            error=exc,
                            failed_processor=getattr(loader, 'name', 'loader'),
                        )
                        results.append(result)
                        if on_error == "raise":
                            raise
                        continue
                else:
                    # Attempt to patch the path attribute of the first processor
                    first = self.processors[0] if self.processors else None
                    if first is not None and hasattr(first, 'path'):
                        import copy
                        patched = copy.copy(first)
                        patched.path = item
                        try:
                            initial_ctx = patched.execute(None)
                        except Exception as exc:
                            logger.error(f"Loader failed for '{label}': {exc}")
                            result = PipelineResult(
                                context=None,
                                success=False,
                                error=exc,
                                failed_processor=getattr(patched, 'name', 'loader'),
                            )
                            results.append(result)
                            if on_error == "raise":
                                raise
                            continue
                    else:
                        raise ValueError(
                            f"Cannot load '{label}': supply a loader_factory or "
                            "ensure the first pipeline processor has a 'path' attribute."
                        )

            logger.info(f"Pipeline.map: processing '{label}'")

            # Skip the first processor if we already ran it as a loader above
            skip_first = not isinstance(item, ProcessingContext) and initial_ctx is not None
            if skip_first and len(self.processors) > 1:
                tail_pipeline = Pipeline(self.processors[1:], name=self.name)
                result = tail_pipeline.run(
                    initial_context=initial_ctx,
                    parallel=parallel,
                    n_jobs=n_jobs,
                )
            else:
                result = self.run(
                    initial_context=initial_ctx,
                    parallel=parallel,
                    n_jobs=n_jobs,
                )

            results.append(result)

            if not result.success and on_error == "raise":
                raise result.error

        return results

    def __len__(self) -> int:
        """Get number of processors."""
        return len(self.processors)

    def __getitem__(self, index: int) -> Processor:
        """Get processor by index."""
        return self.processors[index]

    def __repr__(self) -> str:
        """String representation."""
        return f"Pipeline(name='{self.name}', n_processors={len(self.processors)})"


class PipelineBuilder:
    """
    Fluent builder for constructing pipelines.

    Example::

        pipeline = (PipelineBuilder()
            .load_edf("data.edf")
            .highpass(1.0)
            .upsample(10)
            .detect_triggers(r"\\btrigger\\b")
            .aas_correction()
            .export_edf("output.edf")
            .build())
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize builder.

        Args:
            name: Optional pipeline name
        """
        self._processors: List[Processor] = []
        self._name = name

    def add(self, processor: Processor) -> 'PipelineBuilder':
        """
        Add custom processor.

        Args:
            processor: Processor to add

        Returns:
            Self for chaining
        """
        self._processors.append(processor)
        return self

    def add_if(
        self,
        condition: bool,
        processor: Processor
    ) -> 'PipelineBuilder':
        """
        Add processor conditionally.

        Args:
            condition: Whether to add processor
            processor: Processor to add

        Returns:
            Self for chaining
        """
        if condition:
            self._processors.append(processor)
        return self

    def build(self) -> Pipeline:
        """
        Build the pipeline.

        Returns:
            Constructed Pipeline instance
        """
        return Pipeline(self._processors, name=self._name)

    # Convenience methods can be added here for common processors
    # These will be populated as we implement specific processors
