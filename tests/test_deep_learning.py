"""Tests for deep-learning integration helpers."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

import facet.correction.deep_learning as dl_module
from facet.core import ProcessorValidationError
from facet.correction import (
    DeepLearningArchitecture,
    DeepLearningChannelGroupingStrategy,
    DeepLearningCorrection,
    DeepLearningDualOutputPolicy,
    DeepLearningExecutionGranularity,
    DeepLearningModelAdapter,
    DeepLearningModelRegistry,
    DeepLearningModelSpec,
    DeepLearningOutputType,
    DeepLearningPrediction,
    DeepLearningRuntime,
    get_deep_learning_model,
    list_deep_learning_blueprints,
    list_deep_learning_models,
    register_deep_learning_model,
)


class ArtifactOnlyModel(DeepLearningModelAdapter):
    """Return an explicit artifact estimate for two channels."""

    spec = DeepLearningModelSpec(
        name="ArtifactOnlyModel",
        architecture=DeepLearningArchitecture.AUTOENCODER,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
        uses_triggers=True,
        requires_artifact_length=True,
    )

    def predict(self, context):
        return DeepLearningPrediction(
            artifact_data=np.full((2, 6), 0.25e-6),
            start_sample=3,
            stop_sample=9,
            channel_indices=[0, 2],
            metadata={"mode": "artifact_only"},
        )


class CleanSignalModel(DeepLearningModelAdapter):
    """Return a cleaned signal for a single channel."""

    spec = DeepLearningModelSpec(
        name="CleanSignalModel",
        architecture=DeepLearningArchitecture.STATE_SPACE,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.CLEAN,
    )

    def predict(self, context):
        original = context.get_raw()._data[[1], 4:8]
        return DeepLearningPrediction(
            clean_data=original * 0.5,
            start_sample=4,
            stop_sample=8,
            channel_indices=[1],
            metadata={"mode": "clean_only"},
        )


class MissingRuntimeModel(DeepLearningModelAdapter):
    """Require a runtime that the test hides via monkeypatch."""

    spec = DeepLearningModelSpec(
        name="MissingRuntimeModel",
        architecture=DeepLearningArchitecture.UNET,
        runtime=DeepLearningRuntime.TENSORFLOW,
    )

    def predict(self, context):
        return DeepLearningPrediction(artifact_data=np.zeros((context.get_n_channels(), context.get_raw().n_times)))


class StrictDualOutputModel(DeepLearningModelAdapter):
    """Return inconsistent clean and artifact predictions."""

    spec = DeepLearningModelSpec(
        name="StrictDualOutputModel",
        architecture=DeepLearningArchitecture.HYBRID,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.BOTH,
        dual_output_policy=DeepLearningDualOutputPolicy.STRICT,
    )

    def predict(self, context):
        original = context.get_raw()._data[[0], 2:6]
        artifact = np.full((1, 4), 0.1e-6)
        clean = original - 0.3e-6
        return DeepLearningPrediction(
            clean_data=clean,
            artifact_data=artifact,
            start_sample=2,
            stop_sample=6,
            channel_indices=[0],
        )


class PreferArtifactModel(DeepLearningModelAdapter):
    """Resolve inconsistent outputs by trusting artifact_data."""

    spec = DeepLearningModelSpec(
        name="PreferArtifactModel",
        architecture=DeepLearningArchitecture.HYBRID,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.BOTH,
        dual_output_policy=DeepLearningDualOutputPolicy.PREFER_ARTIFACT,
    )

    def predict(self, context):
        original = context.get_raw()._data[[0], 2:6]
        artifact = np.full((1, 4), 0.2e-6)
        inconsistent_clean = original - 0.5e-6
        return DeepLearningPrediction(
            clean_data=inconsistent_clean,
            artifact_data=artifact,
            start_sample=2,
            stop_sample=6,
            channel_indices=[0],
        )


class ChunkedArtifactModel(DeepLearningModelAdapter):
    """Predict a constant artifact over each execution chunk."""

    spec = DeepLearningModelSpec(
        name="ChunkedArtifactModel",
        architecture=DeepLearningArchitecture.AUTOENCODER,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
        supports_chunking=True,
        chunk_size_samples=400,
        chunk_overlap_samples=100,
    )

    def __init__(self):
        self.seen_chunks = []
        super().__init__()

    def predict(self, context):
        self.seen_chunks.append(
            (
                context.metadata.custom["chunk_start_sample"],
                context.metadata.custom["chunk_stop_sample"],
            )
        )
        n_channels, n_samples = context.get_raw()._data.shape
        return DeepLearningPrediction(
            artifact_data=np.full((n_channels, n_samples), 0.05e-6),
            metadata={"chunk": self.seen_chunks[-1]},
        )


class ChannelWiseArtifactModel(DeepLearningModelAdapter):
    """Predict a constant artifact and support single-channel execution."""

    spec = DeepLearningModelSpec(
        name="ChannelWiseArtifactModel",
        architecture=DeepLearningArchitecture.AUTOENCODER,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
    )

    def predict(self, context):
        n_channels, n_samples = context.get_raw()._data.shape
        return DeepLearningPrediction(
            artifact_data=np.full((n_channels, n_samples), 0.02e-6),
            metadata={"n_channels_seen": n_channels},
        )


class ChannelGroupArtifactModel(DeepLearningModelAdapter):
    """Predict only the target channel in a target-first channel group."""

    spec = DeepLearningModelSpec(
        name="ChannelGroupArtifactModel",
        architecture=DeepLearningArchitecture.HYBRID,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL_GROUP,
        channel_group_size=3,
        channel_grouping_strategy=DeepLearningChannelGroupingStrategy.INDEX,
        supports_multichannel=True,
    )

    def __init__(self):
        self.group_sizes = []
        self.target_names = []
        super().__init__()

    def predict(self, context):
        n_channels, n_samples = context.get_raw()._data.shape
        self.group_sizes.append(n_channels)
        self.target_names.append(context.metadata.custom["channel_group_target_name"])
        artifact = np.zeros((n_channels, n_samples))
        artifact[0] = 0.03e-6
        return DeepLearningPrediction(
            artifact_data=artifact,
            metadata={"n_channels_seen": n_channels, "target": self.target_names[-1]},
        )


@pytest.mark.unit
class TestDeepLearningRegistry:
    """Tests for adapter registration and research blueprints."""

    def test_registry_registers_and_resolves_adapter(self):
        registry = DeepLearningModelRegistry.get_instance()
        model_name = "unit_test_dl_model"

        with contextlib.suppress(KeyError):
            registry.unregister(model_name)

        @register_deep_learning_model(name=model_name, force=True)
        class RegisteredModel(DeepLearningModelAdapter):
            spec = DeepLearningModelSpec(
                name="RegisteredModel",
                architecture=DeepLearningArchitecture.CUSTOM,
                runtime=DeepLearningRuntime.NUMPY,
            )

            def predict(self, context):
                return DeepLearningPrediction(
                    artifact_data=np.zeros((context.get_n_channels(), context.get_raw().n_times))
                )

        try:
            assert get_deep_learning_model(model_name) is RegisteredModel
            assert model_name in list_deep_learning_models()
        finally:
            registry.unregister(model_name)

    def test_research_blueprints_cover_document_families(self):
        blueprints = list_deep_learning_blueprints()

        assert "dar" in blueprints
        assert "d4pm" in blueprints
        assert "denoise_mamba" in blueprints
        assert "st_gnn" in blueprints
        assert blueprints["st_gnn"].requires_channel_positions is True
        assert blueprints["d4pm"].device_preference == "cuda"
        assert blueprints["d4pm"].supports_chunking is True
        assert blueprints["d4pm"].chunk_size_samples == 4096
        assert blueprints["dar"].execution_granularity == DeepLearningExecutionGranularity.CHANNEL
        assert blueprints["dhct_gan"].execution_granularity == DeepLearningExecutionGranularity.MULTICHANNEL

    def test_model_spec_validates_chunking_configuration(self):
        with pytest.raises(ValueError, match="supports_chunking=True"):
            DeepLearningModelSpec(
                name="InvalidChunkingModel",
                architecture=DeepLearningArchitecture.CUSTOM,
                runtime=DeepLearningRuntime.NUMPY,
                chunk_size_samples=1024,
            )

    def test_model_spec_rejects_multichannel_execution_without_multichannel_support(self):
        with pytest.raises(ValueError, match="multichannel execution requires"):
            DeepLearningModelSpec(
                name="InvalidMultichannelModel",
                architecture=DeepLearningArchitecture.CUSTOM,
                runtime=DeepLearningRuntime.NUMPY,
                execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
                supports_multichannel=False,
            )

    def test_model_spec_requires_checkpoint_path_for_checkpoint_format(self):
        with pytest.raises(ValueError, match="requires checkpoint_path"):
            DeepLearningModelSpec(
                name="InvalidCheckpointModel",
                architecture=DeepLearningArchitecture.CUSTOM,
                runtime=DeepLearningRuntime.NUMPY,
                checkpoint_format="npy",
            )

    def test_model_spec_requires_group_size_for_channel_group_execution(self):
        with pytest.raises(ValueError, match="channel_group_size >= 2"):
            DeepLearningModelSpec(
                name="InvalidChannelGroupModel",
                architecture=DeepLearningArchitecture.CUSTOM,
                runtime=DeepLearningRuntime.NUMPY,
                execution_granularity=DeepLearningExecutionGranularity.CHANNEL_GROUP,
                supports_multichannel=True,
            )


@pytest.mark.unit
class TestDeepLearningCorrection:
    """Tests for the pipeline-facing correction processor."""

    def test_artifact_prediction_updates_raw_noise_and_metadata(self, sample_context):
        original = sample_context.get_raw()._data.copy()

        processor = DeepLearningCorrection(ArtifactOnlyModel())
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_raw()._data[0, 3:9], original[0, 3:9] - 0.25e-6)
        np.testing.assert_allclose(result.get_raw()._data[2, 3:9], original[2, 3:9] - 0.25e-6)
        np.testing.assert_allclose(result.get_raw()._data[1], original[1])

        noise = result.get_estimated_noise()
        assert noise.shape == original.shape
        np.testing.assert_allclose(noise[0, 3:9], 0.25e-6)
        np.testing.assert_allclose(noise[2, 3:9], 0.25e-6)
        assert np.allclose(noise[1], 0.0)

        runs = result.metadata.custom["deep_learning_runs"]
        assert len(runs) == 1
        assert runs[0]["model"] == "ArtifactOnlyModel"
        assert runs[0]["channel_indices"] == [0, 2]
        assert runs[0]["prediction_metadata"]["mode"] == "artifact_only"

    def test_clean_prediction_is_converted_into_noise_estimate(self, sample_context):
        original = sample_context.get_raw()._data.copy()

        processor = DeepLearningCorrection(CleanSignalModel())
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_raw()._data[1, 4:8], original[1, 4:8] * 0.5)
        np.testing.assert_allclose(result.get_estimated_noise()[1, 4:8], original[1, 4:8] * 0.5)
        np.testing.assert_allclose(result.get_raw()._data[0], original[0])

    def test_missing_runtime_fails_validation(self, sample_context, monkeypatch):
        original_find_spec = dl_module.importlib.util.find_spec

        def fake_find_spec(module_name):
            if module_name == "tensorflow":
                return None
            return original_find_spec(module_name)

        monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)

        processor = DeepLearningCorrection(MissingRuntimeModel())
        with pytest.raises(ProcessorValidationError, match="tensorflow"):
            processor.execute(sample_context)

    def test_missing_checkpoint_fails_validation(self, sample_context, tmp_path):
        class MissingCheckpointPathModel(DeepLearningModelAdapter):
            spec = DeepLearningModelSpec(
                name="MissingCheckpointPathModel",
                architecture=DeepLearningArchitecture.CUSTOM,
                runtime=DeepLearningRuntime.NUMPY,
                checkpoint_path=str(tmp_path / "missing_model.npy"),
            )

            def predict(self, context):
                return DeepLearningPrediction(artifact_data=np.zeros((context.get_n_channels(), context.get_raw().n_times)))

        processor = DeepLearningCorrection(MissingCheckpointPathModel())
        with pytest.raises(ProcessorValidationError, match="checkpoint does not exist"):
            processor.execute(sample_context)

    def test_incompatible_checkpoint_format_fails_validation(self, sample_context, tmp_path):
        checkpoint = tmp_path / "weights.pt"
        checkpoint.write_bytes(b"placeholder")

        class IncompatibleCheckpointFormatModel(DeepLearningModelAdapter):
            spec = DeepLearningModelSpec(
                name="IncompatibleCheckpointFormatModel",
                architecture=DeepLearningArchitecture.CUSTOM,
                runtime=DeepLearningRuntime.NUMPY,
                checkpoint_path=str(checkpoint),
            )

            def predict(self, context):
                return DeepLearningPrediction(artifact_data=np.zeros((context.get_n_channels(), context.get_raw().n_times)))

        processor = DeepLearningCorrection(IncompatibleCheckpointFormatModel())
        with pytest.raises(ProcessorValidationError, match="incompatible with runtime"):
            processor.execute(sample_context)

    def test_inconsistent_dual_outputs_fail_in_strict_mode(self, sample_context):
        processor = DeepLearningCorrection(StrictDualOutputModel())

        with pytest.raises(ProcessorValidationError, match="inconsistent clean_data and artifact_data"):
            processor.execute(sample_context)

    def test_dual_outputs_can_be_reconciled_by_policy(self, sample_context):
        original = sample_context.get_raw()._data.copy()

        processor = DeepLearningCorrection(PreferArtifactModel())
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_estimated_noise()[0, 2:6], 0.2e-6)
        np.testing.assert_allclose(result.get_raw()._data[0, 2:6], original[0, 2:6] - 0.2e-6)

        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["dual_output_policy"] == DeepLearningDualOutputPolicy.PREFER_ARTIFACT.value

    def test_chunked_execution_processes_multiple_chunks(self, sample_context):
        original = sample_context.get_raw()._data.copy()
        model = ChunkedArtifactModel()

        processor = DeepLearningCorrection(model)
        result = processor.execute(sample_context)

        assert len(model.seen_chunks) > 1
        np.testing.assert_allclose(result.get_raw()._data, original - 0.05e-6)
        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.05e-6))

        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["execution_mode"] == "chunked"
        assert run["chunk_count"] == len(model.seen_chunks)
        assert len(run["prediction_metadata"]["chunks"]) == len(model.seen_chunks)

    def test_channel_granularity_sets_channel_wise_processor_flag(self):
        processor = DeepLearningCorrection(ChannelWiseArtifactModel())

        assert processor.channel_wise is True

    def test_channel_group_execution_updates_only_target_channel_from_each_group(self, sample_context):
        original = sample_context.get_raw()._data.copy()
        model = ChannelGroupArtifactModel()

        processor = DeepLearningCorrection(model)
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_raw()._data, original - 0.03e-6)
        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.03e-6))
        assert all(size == 3 for size in model.group_sizes)
        assert len(model.target_names) == sample_context.get_raw()._data.shape[0]

        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["execution_mode"] == "channel_group"
        assert run["group_count"] == sample_context.get_raw()._data.shape[0]
        assert len(run["prediction_metadata"]["groups"]) == sample_context.get_raw()._data.shape[0]
