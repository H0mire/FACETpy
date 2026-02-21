"""
Pipeline Module

This module defines the Pipeline class for executing sequences of processors.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import List, Optional, Dict, Any, Tuple
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
        processors: List[Processor],
        name: Optional[str] = None
    ):
        """
        Initialize pipeline.

        Args:
            processors: List of processors to execute in order
            name: Optional pipeline name (for logging)
        """
        self.processors = processors
        self.name = name or "Pipeline"
        self._validate_pipeline()

    def _validate_pipeline(self) -> None:
        """Validate pipeline structure."""
        for i, proc in enumerate(self.processors):
            if not isinstance(proc, Processor):
                raise TypeError(
                    f"Processor at index {i} must be a Processor instance, "
                    f"got {type(proc)}"
                )

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

    def add(self, processor: Processor) -> 'Pipeline':
        """
        Add processor to pipeline (fluent API).

        Args:
            processor: Processor to add

        Returns:
            Self for chaining
        """
        self.processors.append(processor)
        return self
    
    def extend(self, processors: List[Processor]) -> 'Pipeline':
        """
        Extend pipeline with multiple processors.

        Args:
            processors: List of processors to add

        Returns:
            Self for chaining
        """
        self.processors.extend(processors)
        return self

    def insert(self, index: int, processor: Processor) -> 'Pipeline':
        """
        Insert processor at specific position.

        Args:
            index: Position to insert
            processor: Processor to insert

        Returns:
            Self for chaining
        """
        self.processors.insert(index, processor)
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
