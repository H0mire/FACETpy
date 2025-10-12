"""
Processor Base Module

This module defines the base Processor interface and related abstractions
for building processing pipelines.

Author: FACETpy Team
Date: 2025-01-12
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from loguru import logger
from .context import ProcessingContext


class ProcessorValidationError(Exception):
    """Raised when processor validation fails."""
    pass


class Processor(ABC):
    """
    Base class for all processors in the pipeline.

    Processors are the fundamental building blocks of FACETpy pipelines.
    Each processor:
    - Takes a ProcessingContext as input
    - Performs a specific operation
    - Returns a new ProcessingContext (immutable by default)
    - Can validate prerequisites before processing
    - Tracks its execution in the context history

    Subclasses must implement:
    - process(): Main processing logic
    - validate(): Check prerequisites (optional)

    Example:
        class HighPassFilter(Processor):
            name = "highpass_filter"

            def __init__(self, freq: float):
                self.freq = freq

            def validate(self, context: ProcessingContext) -> None:
                if not context.has_raw():
                    raise ProcessorValidationError("No raw data available")

            def process(self, context: ProcessingContext) -> ProcessingContext:
                raw = context.get_raw().copy()
                raw.filter(l_freq=self.freq, h_freq=None)
                return context.with_raw(raw)
    """

    # Class attributes (can be overridden in subclasses)
    name: str = "base_processor"
    description: str = "Base processor"
    version: str = "1.0.0"

    # Processing flags
    requires_triggers: bool = False
    requires_raw: bool = True
    modifies_raw: bool = True
    parallel_safe: bool = True  # Can be parallelized

    def __init__(self):
        """Initialize processor."""
        self._parameters = self._get_parameters()

    def __call__(self, context: ProcessingContext) -> ProcessingContext:
        """
        Make processor callable.

        Args:
            context: Input processing context

        Returns:
            Output processing context
        """
        return self.execute(context)

    def execute(self, context: ProcessingContext) -> ProcessingContext:
        """
        Execute the processor with validation and history tracking.

        Args:
            context: Input processing context

        Returns:
            Output processing context

        Raises:
            ProcessorValidationError: If validation fails
        """
        # Validate prerequisites
        logger.debug(f"Executing processor: {self.name}")
        self.validate(context)

        # Process
        result = self.process(context)

        # Add to history
        result.add_history_entry(
            name=self.name,
            processor_type=self.__class__.__name__,
            parameters=self._parameters
        )

        logger.debug(f"Completed processor: {self.name}")
        return result

    def validate(self, context: ProcessingContext) -> None:
        """
        Validate that prerequisites are met.

        Override this method to add custom validation logic.

        Args:
            context: Processing context

        Raises:
            ProcessorValidationError: If validation fails
        """
        if self.requires_raw and not context.has_raw():
            raise ProcessorValidationError(
                f"{self.name} requires raw data, but none is available"
            )

        if self.requires_triggers and not context.has_triggers():
            raise ProcessorValidationError(
                f"{self.name} requires triggers, but none are available"
            )

    @abstractmethod
    def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Process the context.

        This is the main method to implement in subclasses.

        Args:
            context: Input processing context

        Returns:
            Output processing context
        """
        pass

    def _get_parameters(self) -> Dict[str, Any]:
        """
        Get processor parameters for history tracking.

        Override this to customize parameter tracking.

        Returns:
            Dictionary of parameters
        """
        # Get all instance variables that don't start with _
        params = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
        return params

    def __repr__(self) -> str:
        """String representation."""
        params_str = ", ".join(f"{k}={v}" for k, v in self._parameters.items())
        return f"{self.__class__.__name__}({params_str})"


class SequenceProcessor(Processor):
    """
    Processor that executes multiple processors in sequence.

    This is a composite processor that allows grouping multiple
    processing steps into a single unit.

    Example:
        preprocessing = SequenceProcessor([
            HighPassFilter(freq=1.0),
            UpSample(factor=10),
            Notch Filter(freqs=[50])
        ])
    """

    name = "sequence"
    requires_raw = False  # Depends on child processors

    def __init__(self, processors: List[Processor]):
        """
        Initialize sequence processor.

        Args:
            processors: List of processors to execute in order
        """
        self.processors = processors
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate all child processors."""
        for processor in self.processors:
            processor.validate(context)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute all processors in sequence."""
        result = context
        for processor in self.processors:
            result = processor.execute(result)
        return result


class ConditionalProcessor(Processor):
    """
    Processor that executes conditionally based on context.

    Example:
        ConditionalProcessor(
            condition=lambda ctx: ctx.metadata.custom.get("needs_upsampling"),
            processor=UpSample(factor=10),
            else_processor=None  # Skip if condition is False
        )
    """

    name = "conditional"
    requires_raw = False

    def __init__(
        self,
        condition: callable,
        processor: Processor,
        else_processor: Optional[Processor] = None
    ):
        """
        Initialize conditional processor.

        Args:
            condition: Callable that takes context and returns bool
            processor: Processor to execute if condition is True
            else_processor: Processor to execute if condition is False (optional)
        """
        self.condition = condition
        self.processor = processor
        self.else_processor = else_processor
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate based on condition."""
        if self.condition(context):
            self.processor.validate(context)
        elif self.else_processor is not None:
            self.else_processor.validate(context)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute conditionally."""
        if self.condition(context):
            logger.debug(f"Condition met, executing {self.processor.name}")
            return self.processor.execute(context)
        elif self.else_processor is not None:
            logger.debug(f"Condition not met, executing {self.else_processor.name}")
            return self.else_processor.execute(context)
        else:
            logger.debug("Condition not met, skipping")
            return context


class SwitchProcessor(Processor):
    """
    Processor that switches between multiple processors based on selector.

    Example:
        SwitchProcessor(
            selector=lambda ctx: "motion" if ctx.has_motion_data() else "aas",
            cases={
                "aas": AASCorrection(),
                "motion": MotionBasedCorrection()
            },
            default=AASCorrection()
        )
    """

    name = "switch"
    requires_raw = False

    def __init__(
        self,
        selector: callable,
        cases: Dict[str, Processor],
        default: Optional[Processor] = None
    ):
        """
        Initialize switch processor.

        Args:
            selector: Callable that takes context and returns case key
            cases: Dictionary mapping case keys to processors
            default: Default processor if selector returns unknown key
        """
        self.selector = selector
        self.cases = cases
        self.default = default
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate selected processor."""
        case_key = self.selector(context)
        processor = self.cases.get(case_key, self.default)
        if processor is None:
            raise ProcessorValidationError(
                f"No processor for case '{case_key}' and no default provided"
            )
        processor.validate(context)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute selected processor."""
        case_key = self.selector(context)
        processor = self.cases.get(case_key, self.default)

        if processor is None:
            raise ProcessorValidationError(
                f"No processor for case '{case_key}' and no default provided"
            )

        logger.debug(f"Selected case '{case_key}', executing {processor.name}")
        return processor.execute(context)


class NoOpProcessor(Processor):
    """Processor that does nothing (useful for testing or placeholders)."""

    name = "noop"
    requires_raw = False
    modifies_raw = False

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Return context unchanged."""
        return context


class LambdaProcessor(Processor):
    """
    Processor that executes a lambda function.

    Useful for quick custom operations without creating a full processor class.

    Example:
        LambdaProcessor(
            name="remove_bad_channels",
            func=lambda ctx: ctx.with_raw(
                ctx.get_raw().copy().drop_channels(["EKG"])
            )
        )
    """

    requires_raw = False

    def __init__(self, name: str, func: callable):
        """
        Initialize lambda processor.

        Args:
            name: Processor name
            func: Function that takes context and returns new context
        """
        self.name = name
        self.func = func
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Execute lambda function."""
        return self.func(context)
