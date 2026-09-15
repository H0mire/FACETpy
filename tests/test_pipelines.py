"""Tests for pipeline factory helpers."""

import mne
import numpy as np
import pytest

from facet.core import Pipeline, ProcessingContext
from facet.correction import (
    AASCorrection,
    DeepLearningArchitecture,
    DeepLearningCorrection,
    DeepLearningExecutionGranularity,
    DeepLearningModelAdapter,
    DeepLearningModelSpec,
    DeepLearningOutputType,
    DeepLearningPrediction,
    DeepLearningRuntime,
)
from facet.pipelines import create_standard_pipeline
from facet.preprocessing import DownSample
from tests.conftest import create_mock_processor


class _ChannelSequentialDLModel(DeepLearningModelAdapter):
    """Simple channel-wise artifact predictor for executor regression tests."""

    spec = DeepLearningModelSpec(
        name="ChannelSequentialDLModel",
        architecture=DeepLearningArchitecture.AUTOENCODER,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
    )

    def predict(self, context):
        n_channels, n_samples = context.get_raw()._data.shape
        return DeepLearningPrediction(
            artifact_data=np.full((n_channels, n_samples), 0.1),
            metadata={"n_channels_seen": n_channels},
        )


class _MultichannelDLModel(DeepLearningModelAdapter):
    """Simple multichannel model used to validate execution mode handling."""

    spec = DeepLearningModelSpec(
        name="MultichannelDLModel",
        architecture=DeepLearningArchitecture.UNET,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        supports_multichannel=True,
    )

    def predict(self, context):
        n_channels, n_samples = context.get_raw()._data.shape
        return DeepLearningPrediction(artifact_data=np.full((n_channels, n_samples), 0.1))


@pytest.mark.unit
class TestPipelineFactories:
    """Tests for pre-built pipeline factories."""

    def test_standard_pipeline_accepts_additional_corrections(self):
        extra = create_mock_processor("learned_hook")

        pipeline = create_standard_pipeline(
            input_path="input.edf",
            output_path="output.edf",
            use_anc=False,
            use_pca=False,
            additional_corrections=[extra],
        )

        extra_index = pipeline.processors.index(extra)

        assert isinstance(pipeline.processors[extra_index - 1], AASCorrection)
        assert isinstance(pipeline.processors[extra_index + 1], DownSample)


@pytest.mark.unit
class TestDeepLearningExecutionModes:
    """Regression tests for pipeline execution mode interaction."""

    def test_channel_sequential_runs_channel_wise_deep_learning_processor(self):
        data = np.vstack(
            [
                np.full(128, 1.0),
                np.full(128, 2.0),
                np.full(128, 3.0),
            ]
        )
        info = mne.create_info(
            ch_names=["EEG001", "EEG002", "EEG003"],
            sfreq=250.0,
            ch_types=["eeg", "eeg", "eeg"],
        )
        raw = mne.io.RawArray(data.copy(), info, verbose=False)
        context = ProcessingContext(raw=raw, raw_original=raw.copy())

        processor = DeepLearningCorrection(_ChannelSequentialDLModel())
        pipeline = Pipeline([processor])

        serial = pipeline.run(initial_context=context, channel_sequential=False, show_progress=False)
        channel_seq = pipeline.run(initial_context=context, channel_sequential=True, show_progress=False)

        expected = data - 0.1
        np.testing.assert_allclose(serial.context.get_data(copy=False), expected)
        np.testing.assert_allclose(channel_seq.context.get_data(copy=False), expected)

        serial_run = serial.context.metadata.custom["deep_learning_runs"][0]
        channel_seq_run = channel_seq.context.metadata.custom["deep_learning_runs"][0]

        assert processor.channel_wise is True
        assert serial_run["execution_granularity"] == DeepLearningExecutionGranularity.CHANNEL.value
        assert channel_seq_run["execution_granularity"] == DeepLearningExecutionGranularity.CHANNEL.value

    def test_channel_sequential_rejects_multichannel_deep_learning_processor(self):
        data = np.vstack(
            [
                np.full(64, 1.0),
                np.full(64, 2.0),
            ]
        )
        info = mne.create_info(
            ch_names=["EEG001", "EEG002"],
            sfreq=250.0,
            ch_types=["eeg", "eeg"],
        )
        raw = mne.io.RawArray(data.copy(), info, verbose=False)
        context = ProcessingContext(raw=raw, raw_original=raw.copy())

        pipeline = Pipeline([DeepLearningCorrection(_MultichannelDLModel())])
        result = pipeline.run(initial_context=context, channel_sequential=True, show_progress=False)

        assert result.success is False
        assert result.error is not None
        assert "cannot run with channel_sequential=True" in str(result.error)
