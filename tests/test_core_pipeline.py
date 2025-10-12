"""
Tests for Pipeline class.
"""

import pytest
import time

from facet.core import Pipeline, ProcessingContext, PipelineError
from tests.conftest import create_mock_processor


@pytest.mark.unit
class TestPipeline:
    """Tests for Pipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        proc1 = create_mock_processor("proc1")
        proc2 = create_mock_processor("proc2")

        pipeline = Pipeline([proc1, proc2], name="Test Pipeline")

        assert pipeline.name == "Test Pipeline"
        assert len(pipeline.processors) == 2

    def test_pipeline_execution(self, sample_context):
        """Test basic pipeline execution."""
        proc1 = create_mock_processor("proc1")
        proc2 = create_mock_processor("proc2")
        proc3 = create_mock_processor("proc3")

        pipeline = Pipeline([proc1, proc2, proc3])
        result = pipeline.run(initial_context=sample_context)

        # Check success
        assert result.success is True
        assert result.error is None
        assert result.context is not None

        # Check all processors were called
        assert proc1.call_count == 1
        assert proc2.call_count == 1
        assert proc3.call_count == 1

    def test_pipeline_with_loader(self, sample_edf_file):
        """Test pipeline with loader creating initial context."""
        from facet.io import EDFLoader

        loader = EDFLoader(path=str(sample_edf_file), preload=True)
        proc1 = create_mock_processor("proc1")

        pipeline = Pipeline([loader, proc1])
        result = pipeline.run()

        # Should create context from loader
        assert result.success is True
        assert result.context is not None
        assert proc1.call_count == 1

    def test_pipeline_error_handling(self, sample_context):
        """Test pipeline handles processor errors."""
        proc1 = create_mock_processor("proc1")

        # Processor that raises error
        def error_fn(ctx):
            raise ValueError("Test error")

        proc2 = create_mock_processor("error_proc", process_fn=error_fn)
        proc3 = create_mock_processor("proc3")

        pipeline = Pipeline([proc1, proc2, proc3])
        result = pipeline.run(initial_context=sample_context)

        # Check failure
        assert result.success is False
        assert result.error is not None
        assert result.failed_processor == "error_proc"

        # Only proc1 should have been called
        assert proc1.call_count == 1
        assert proc3.call_count == 0

    def test_pipeline_timing(self, sample_context):
        """Test pipeline records execution time."""
        proc1 = create_mock_processor("proc1")

        pipeline = Pipeline([proc1])
        result = pipeline.run(initial_context=sample_context)

        assert result.execution_time > 0
        assert isinstance(result.execution_time, float)

    def test_pipeline_history_tracking(self, sample_context):
        """Test pipeline tracks processing history."""
        proc1 = create_mock_processor("proc1")
        proc2 = create_mock_processor("proc2")

        pipeline = Pipeline([proc1, proc2])
        result = pipeline.run(initial_context=sample_context)

        history = result.context.get_history()
        assert len(history) >= 2

        # Check processors are recorded
        processor_names = [h['processor'] for h in history]
        assert "proc1" in processor_names
        assert "proc2" in processor_names

    def test_empty_pipeline(self, sample_context):
        """Test empty pipeline."""
        pipeline = Pipeline([])
        result = pipeline.run(initial_context=sample_context)

        # Should succeed with unchanged context
        assert result.success is True
        assert result.context.get_raw() is sample_context.get_raw()

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        proc1 = create_mock_processor("proc1")
        pipeline = Pipeline([proc1], name="My Pipeline")

        repr_str = repr(pipeline)
        assert "My Pipeline" in repr_str
        assert "1 processors" in repr_str


@pytest.mark.unit
class TestPipelineResult:
    """Tests for PipelineResult class."""

    def test_result_attributes(self, sample_context):
        """Test PipelineResult attributes."""
        pipeline = Pipeline([create_mock_processor("proc1")])
        result = pipeline.run(initial_context=sample_context)

        # Check all attributes exist
        assert hasattr(result, 'success')
        assert hasattr(result, 'context')
        assert hasattr(result, 'error')
        assert hasattr(result, 'execution_time')
        assert hasattr(result, 'failed_processor')

    def test_successful_result(self, sample_context):
        """Test attributes of successful result."""
        pipeline = Pipeline([create_mock_processor("proc1")])
        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        assert result.context is not None
        assert result.error is None
        assert result.failed_processor is None
        assert result.execution_time > 0

    def test_failed_result(self, sample_context):
        """Test attributes of failed result."""
        def error_fn(ctx):
            raise ValueError("Test error")

        proc_error = create_mock_processor("error_proc", process_fn=error_fn)
        pipeline = Pipeline([proc_error])
        result = pipeline.run(initial_context=sample_context)

        assert result.success is False
        assert result.error is not None
        assert "Test error" in str(result.error)
        assert result.failed_processor == "error_proc"


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for pipeline with real processors."""

    def test_simple_correction_pipeline(self, sample_edf_file):
        """Test a simple correction pipeline."""
        from facet.io import EDFLoader, EDFExporter
        from facet.preprocessing import TriggerDetector, UpSample, DownSample
        from facet.correction import AASCorrection
        import tempfile

        # Create output path
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
            output_path = tmp.name

        pipeline = Pipeline([
            EDFLoader(path=str(sample_edf_file), preload=True),
            TriggerDetector(regex=r"\b1\b"),
            UpSample(factor=2),  # Small factor for speed
            AASCorrection(window_size=5),  # Small window for speed
            DownSample(factor=2),
            EDFExporter(path=output_path, overwrite=True)
        ])

        result = pipeline.run()

        # Check success
        assert result.success is True
        assert result.context is not None

        # Check file was created
        import os
        assert os.path.exists(output_path)

        # Cleanup
        os.unlink(output_path)

    @pytest.mark.slow
    def test_full_correction_pipeline(self, sample_edf_file):
        """Test a full correction pipeline with multiple steps."""
        from facet.io import EDFLoader
        from facet.preprocessing import TriggerDetector, UpSample, DownSample
        from facet.correction import AASCorrection
        from facet.evaluation import SNRCalculator, RMSCalculator

        pipeline = Pipeline([
            EDFLoader(path=str(sample_edf_file), preload=True),
            TriggerDetector(regex=r"\b1\b"),
            UpSample(factor=2),
            AASCorrection(window_size=5),
            DownSample(factor=2),
            SNRCalculator(),
            RMSCalculator()
        ])

        result = pipeline.run()

        assert result.success is True

        # Check metrics were calculated
        metrics = result.context.metadata.custom.get('metrics', {})
        assert 'snr' in metrics
        assert 'rms_ratio' in metrics
