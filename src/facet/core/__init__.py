"""
Core Module

This module contains the core infrastructure for FACETpy pipelines.

Author: FACETpy Team
Date: 2025-01-12
"""

from .context import ProcessingContext, ProcessingMetadata, ProcessingStep
from .processor import (
    Processor,
    ProcessorValidationError,
    SequenceProcessor,
    ConditionalProcessor,
    SwitchProcessor,
    NoOpProcessor,
    LambdaProcessor
)
from .pipeline import Pipeline, PipelineBuilder, PipelineResult
from .registry import (
    ProcessorRegistry,
    register_processor,
    get_processor,
    list_processors
)
from .parallel import ParallelExecutor

__all__ = [
    # Context
    'ProcessingContext',
    'ProcessingMetadata',
    'ProcessingStep',

    # Processor
    'Processor',
    'ProcessorValidationError',
    'SequenceProcessor',
    'ConditionalProcessor',
    'SwitchProcessor',
    'NoOpProcessor',
    'LambdaProcessor',

    # Pipeline
    'Pipeline',
    'PipelineBuilder',
    'PipelineResult',

    # Registry
    'ProcessorRegistry',
    'register_processor',
    'get_processor',
    'list_processors',

    # Parallel
    'ParallelExecutor',
]
