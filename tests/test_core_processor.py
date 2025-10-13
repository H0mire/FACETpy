"""
Tests for Processor base class and composite processors.
"""

import pytest
import numpy as np

from facet.core import (
    Processor,
    ProcessingContext,
    SequenceProcessor,
    ConditionalProcessor,
    SwitchProcessor,
    ProcessorError,
    ProcessorValidationError,
    register_processor,
    get_processor
)
from tests.conftest import create_mock_processor


@pytest.mark.unit
class TestProcessorBase:
    """Tests for Processor base class."""

    def test_processor_must_be_subclassed(self):
        """Test that Processor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Processor()

    def test_custom_processor_execute(self, sample_context):
        """Test custom processor execution."""
        # Create a simple processor
        class DoubleDataProcessor(Processor):
            name = "double_data"

            def process(self, context):
                raw = context.get_raw()
                raw_doubled = raw.copy()
                raw_doubled._data *= 2
                return context.with_raw(raw_doubled)

        processor = DoubleDataProcessor()
        result = processor.execute(sample_context)

        # Check result is new context
        assert result is not sample_context

        # Check data was doubled
        original_data = sample_context.get_raw()._data
        result_data = result.get_raw()._data
        np.testing.assert_allclose(result_data, original_data * 2)

    def test_processor_validation(self, sample_context):
        """Test processor validation."""
        class RequiresTriggersProcessor(Processor):
            name = "requires_triggers"
            requires_triggers = True

            def process(self, context):
                return context

        processor = RequiresTriggersProcessor()

        # Should pass with triggers
        processor.execute(sample_context)

        # Should fail without triggers
        context_no_triggers = ProcessingContext(raw=sample_context.get_raw())
        with pytest.raises(ProcessorValidationError):
            processor.execute(context_no_triggers)

    def test_processor_history_tracking(self, sample_context):
        """Test that processors add history entries."""
        processor = create_mock_processor("test")
        result = processor.execute(sample_context)

        history = result.get_history()
        assert len(history) == 1
        assert history[0]['processor'] == "test"

    def test_processor_get_parameters(self):
        """Test _get_parameters method."""
        class ParamProcessor(Processor):
            name = "param_proc"

            def __init__(self, param1, param2=10):
                self.param1 = param1
                self.param2 = param2
                super().__init__()

            def process(self, context):
                return context

        processor = ParamProcessor(param1=5, param2=20)
        params = processor._get_parameters()

        assert params['param1'] == 5
        assert params['param2'] == 20


@pytest.mark.unit
class TestSequenceProcessor:
    """Tests for SequenceProcessor."""

    def test_sequence_execution(self, sample_context):
        """Test sequential execution of processors."""
        proc1 = create_mock_processor("proc1")
        proc2 = create_mock_processor("proc2")
        proc3 = create_mock_processor("proc3")

        sequence = SequenceProcessor([proc1, proc2, proc3])
        result = sequence.execute(sample_context)

        # Check all processors were called
        assert proc1.call_count == 1
        assert proc2.call_count == 1
        assert proc3.call_count == 1

        # Check history has all three
        history = result.get_history()
        assert len(history) >= 3

    def test_sequence_with_modifications(self, sample_context):
        """Test sequence where processors modify data."""
        class MultiplyProcessor(Processor):
            def __init__(self, factor):
                self.name = f"multiply_{factor}"
                self.factor = factor
                super().__init__()

            def process(self, context):
                raw = context.get_raw()
                raw_modified = raw.copy()
                raw_modified._data *= self.factor
                return context.with_raw(raw_modified)

        proc1 = MultiplyProcessor(2)
        proc2 = MultiplyProcessor(3)

        sequence = SequenceProcessor([proc1, proc2])
        result = sequence.execute(sample_context)

        # Data should be multiplied by 2*3=6
        original_data = sample_context.get_raw()._data
        result_data = result.get_raw()._data
        np.testing.assert_allclose(result_data, original_data * 6)

    def test_empty_sequence(self, sample_context):
        """Test empty sequence processor."""
        sequence = SequenceProcessor([])
        result = sequence.execute(sample_context)

        # Should return unchanged context
        assert result.get_raw() is sample_context.get_raw()


@pytest.mark.unit
class TestConditionalProcessor:
    """Tests for ConditionalProcessor."""

    def test_condition_true(self, sample_context):
        """Test conditional processor when condition is True."""
        mock_proc = create_mock_processor("conditional")

        conditional = ConditionalProcessor(
            condition=lambda ctx: True,
            processor=mock_proc
        )

        result = conditional.execute(sample_context)

        # Processor should have been called
        assert mock_proc.call_count == 1

    def test_condition_false(self, sample_context):
        """Test conditional processor when condition is False."""
        mock_proc = create_mock_processor("conditional")

        conditional = ConditionalProcessor(
            condition=lambda ctx: False,
            processor=mock_proc
        )

        result = conditional.execute(sample_context)

        # Processor should NOT have been called
        assert mock_proc.call_count == 0

        # Context should be unchanged
        assert result.get_raw() is sample_context.get_raw()

    def test_condition_based_on_context(self, sample_context):
        """Test condition that checks context properties."""
        mock_proc = create_mock_processor("conditional")

        # Condition: only run if more than 5 triggers
        conditional = ConditionalProcessor(
            condition=lambda ctx: len(ctx.get_triggers()) > 5,
            processor=mock_proc
        )

        result = conditional.execute(sample_context)

        # Should execute since we have 10 triggers
        assert mock_proc.call_count == 1


@pytest.mark.unit
class TestSwitchProcessor:
    """Tests for SwitchProcessor."""

    def test_switch_selection(self, sample_context):
        """Test switch processor selects correct processor."""
        proc_a = create_mock_processor("proc_a")
        proc_b = create_mock_processor("proc_b")

        switch = SwitchProcessor(
            selector=lambda ctx: "a",
            processors={"a": proc_a, "b": proc_b}
        )

        result = switch.execute(sample_context)

        # Only proc_a should be called
        assert proc_a.call_count == 1
        assert proc_b.call_count == 0

    def test_switch_with_different_selections(self, sample_context):
        """Test switch with context-based selection."""
        proc_low = create_mock_processor("low")
        proc_high = create_mock_processor("high")

        # Select based on number of triggers
        def selector(ctx):
            return "high" if len(ctx.get_triggers()) > 5 else "low"

        switch = SwitchProcessor(
            selector=selector,
            processors={"low": proc_low, "high": proc_high}
        )

        result = switch.execute(sample_context)

        # Should select high (we have 10 triggers)
        assert proc_low.call_count == 0
        assert proc_high.call_count == 1

    def test_switch_with_default(self, sample_context):
        """Test switch processor with default case."""
        proc_a = create_mock_processor("proc_a")
        proc_default = create_mock_processor("default")

        switch = SwitchProcessor(
            selector=lambda ctx: "unknown",
            processors={"a": proc_a},
            default=proc_default
        )

        result = switch.execute(sample_context)

        # Should use default
        assert proc_a.call_count == 0
        assert proc_default.call_count == 1

    def test_switch_no_default_raises_error(self, sample_context):
        """Test switch without default raises error for unknown key."""
        proc_a = create_mock_processor("proc_a")

        switch = SwitchProcessor(
            selector=lambda ctx: "unknown",
            processors={"a": proc_a}
        )

        with pytest.raises(ProcessorError):
            switch.execute(sample_context)


@pytest.mark.unit
class TestProcessorRegistry:
    """Tests for processor registry."""

    def test_register_processor(self):
        """Test registering a processor."""
        @register_processor
        class TestRegistryProcessor(Processor):
            name = "test_registry_proc"
            description = "Test processor for registry"

            def process(self, context):
                return context

        # Should be retrievable
        ProcessorClass = get_processor("test_registry_proc")
        assert ProcessorClass is TestRegistryProcessor

    def test_get_nonexistent_processor(self):
        """Test getting a processor that doesn't exist."""
        with pytest.raises(KeyError):
            get_processor("nonexistent_processor_xyz")

    def test_processor_auto_registration(self):
        """Test that processors are auto-registered."""
        # Built-in processors should be registered
        from facet.preprocessing import TriggerDetector

        ProcessorClass = get_processor("trigger_detector")
        assert ProcessorClass is TriggerDetector
