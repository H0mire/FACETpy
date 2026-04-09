"""Tests for deep-learning integration helpers."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import facet.correction.deep_learning as dl_module
from facet.core import ProcessingContext, ProcessorValidationError
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
    NumpyInferenceAdapter,
    OnnxInferenceAdapter,
    OnnxTensorLayout,
    PyTorchTensorLayout,
    TensorFlowTensorLayout,
    get_deep_learning_model,
    list_deep_learning_blueprints,
    list_deep_learning_models,
    load_deep_learning_config,
    register_deep_learning_model,
    save_deep_learning_config,
    spec_from_dict,
    spec_to_dict,
)


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def numpy(self):
        return self._array

    def detach(self):
        return self

    def cpu(self):
        return self


class _FakeDeviceContext:
    def __init__(self, device_name, device_calls):
        self.device_name = device_name
        self.device_calls = device_calls

    def __enter__(self):
        self.device_calls.append(self.device_name)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTensorFlowModule:
    def __init__(self, *, keras_model=None, saved_model=None, gpu_available=False):
        self._keras_model = keras_model
        self._saved_model = saved_model
        self.gpu_available = gpu_available
        self.loaded_model_paths = []
        self.device_calls = []
        self.keras = SimpleNamespace(models=SimpleNamespace(load_model=self._load_model))
        self.saved_model = SimpleNamespace(load=self._load_saved_model)
        self.config = SimpleNamespace(list_logical_devices=self._list_logical_devices)

    def _load_model(self, path, compile=False):
        self.loaded_model_paths.append((path, compile))
        return self._keras_model

    def _load_saved_model(self, path):
        self.loaded_model_paths.append((path, None))
        return self._saved_model

    def _list_logical_devices(self, device_type):
        if device_type == "GPU" and self.gpu_available:
            return ["GPU:0"]
        return []

    def convert_to_tensor(self, value):
        return _FakeTensor(value)

    def device(self, device_name):
        return _FakeDeviceContext(device_name, self.device_calls)


class _FakeSavedModelSignature:
    def __init__(self, output_factory, *, keyword_names=None, reject_positional=False):
        self.output_factory = output_factory
        self.keyword_names = tuple(keyword_names or ())
        self.reject_positional = reject_positional
        self.structured_input_signature = (
            (),
            {name: object() for name in self.keyword_names},
        )

    def __call__(self, *args, **kwargs):
        if kwargs:
            return self.output_factory(next(iter(kwargs.values())))
        if self.reject_positional:
            raise TypeError("positional invocation not supported")
        if len(args) != 1:
            raise TypeError("expected exactly one positional input")
        return self.output_factory(args[0])


class _FakeSavedModel:
    def __init__(self, signatures):
        self.signatures = signatures


class _FakeNoGradContext:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        self.calls.append("enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.calls.append("exit")
        return False


class _FakeTorchModule:
    def __init__(self, *, load_return=None, jit_load_return=None, cuda_available=False):
        self._load_return = load_return
        self._jit_load_return = jit_load_return
        self.load_calls = []
        self.jit_load_calls = []
        self.as_tensor_calls = []
        self.no_grad_calls = []
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.jit = SimpleNamespace(load=self._jit_load)

    def _jit_load(self, path, map_location=None):
        self.jit_load_calls.append((path, map_location))
        return self._jit_load_return

    def load(self, path, map_location=None):
        self.load_calls.append((path, map_location))
        return self._load_return

    def as_tensor(self, value, device=None):
        self.as_tensor_calls.append(device)
        return _FakeTensor(value)

    def no_grad(self):
        return _FakeNoGradContext(self.no_grad_calls)


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


class PositionChannelGroupArtifactModel(DeepLearningModelAdapter):
    """Same as channel-group model, but using position-based neighbors."""

    spec = DeepLearningModelSpec(
        name="PositionChannelGroupArtifactModel",
        architecture=DeepLearningArchitecture.HYBRID,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL_GROUP,
        channel_group_size=3,
        channel_grouping_strategy=DeepLearningChannelGroupingStrategy.POSITION,
        supports_multichannel=True,
        requires_channel_positions=True,
    )

    def predict(self, context):
        n_channels, n_samples = context.get_raw()._data.shape
        artifact = np.zeros((n_channels, n_samples))
        artifact[0] = 0.01e-6
        return DeepLearningPrediction(artifact_data=artifact)


class ChunkedChannelGroupArtifactModel(DeepLearningModelAdapter):
    """Run chunked inference inside channel-group execution."""

    spec = DeepLearningModelSpec(
        name="ChunkedChannelGroupArtifactModel",
        architecture=DeepLearningArchitecture.HYBRID,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL_GROUP,
        channel_group_size=3,
        channel_grouping_strategy=DeepLearningChannelGroupingStrategy.INDEX,
        supports_multichannel=True,
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
                context.metadata.custom["channel_group_target_name"],
                context.metadata.custom["chunk_start_sample"],
                context.metadata.custom["chunk_stop_sample"],
            )
        )
        n_channels, n_samples = context.get_raw()._data.shape
        artifact = np.zeros((n_channels, n_samples))
        artifact[0] = 0.04e-6
        return DeepLearningPrediction(
            artifact_data=artifact,
            metadata={"chunk": self.seen_chunks[-1]},
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

    def test_channel_group_context_does_not_use_with_raw_full_noise_copy(
        self,
        sample_context_with_noise,
        monkeypatch,
    ):
        original_with_raw = ProcessingContext.with_raw

        def fail_on_group_with_raw(self, raw, copy_metadata=True):
            if raw.get_data().shape[0] < self.get_n_channels():
                raise AssertionError("channel-group context should not be built through with_raw()")
            return original_with_raw(self, raw, copy_metadata=copy_metadata)

        monkeypatch.setattr(ProcessingContext, "with_raw", fail_on_group_with_raw)

        processor = DeepLearningCorrection(ChannelGroupArtifactModel())
        result = processor.execute(sample_context_with_noise)

        assert result.get_estimated_noise().shape == sample_context_with_noise.get_estimated_noise().shape

    def test_position_channel_grouping_computes_channel_positions_once(self, sample_context, monkeypatch):
        raw = sample_context.get_raw()
        coordinates = [
            (0.0, 0.1, 0.1),
            (0.1, 0.0, 0.1),
            (0.2, 0.1, 0.0),
            (0.3, 0.1, 0.1),
        ]
        for ch, xyz in zip(raw.info["chs"], coordinates, strict=True):
            ch["loc"][:3] = xyz

        call_count = 0
        original_channel_positions = DeepLearningCorrection._channel_positions

        def counting_channel_positions(raw_arg, channel_indices_arg):
            nonlocal call_count
            call_count += 1
            return original_channel_positions(raw_arg, channel_indices_arg)

        monkeypatch.setattr(DeepLearningCorrection, "_channel_positions", staticmethod(counting_channel_positions))

        processor = DeepLearningCorrection(PositionChannelGroupArtifactModel())
        result = processor.execute(sample_context)

        assert call_count == 1
        assert result.metadata.custom["deep_learning_runs"][0]["execution_mode"] == "channel_group"

    def test_channel_group_execution_supports_chunked_local_inference(self, sample_context):
        original = sample_context.get_raw()._data.copy()
        model = ChunkedChannelGroupArtifactModel()

        processor = DeepLearningCorrection(model)
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_raw()._data, original - 0.04e-6)
        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.04e-6))
        assert len(model.seen_chunks) > sample_context.get_n_channels()

        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["execution_mode"] == "channel_group"
        assert run["group_count"] == sample_context.get_n_channels()
        assert all(group["local_execution_mode"] == "chunked" for group in run["prediction_metadata"]["groups"])
        assert all(len(group["prediction_metadata"]["chunks"]) > 1 for group in run["prediction_metadata"]["groups"])

    def test_position_channel_grouping_requires_valid_electrode_positions(self, sample_context):
        processor = DeepLearningCorrection(PositionChannelGroupArtifactModel())

        with pytest.raises(ProcessorValidationError, match="requires electrode positions"):
            processor.execute(sample_context)

    def test_tensorflow_inference_adapter_runs_single_output_model_via_registry(
        self,
        sample_context,
        monkeypatch,
        tmp_path,
    ):
        checkpoint = tmp_path / "tf_model.keras"
        checkpoint.write_bytes(b"placeholder")

        class FakeKerasArtifactModel:
            def __call__(self, input_tensor, training=False):
                arr = input_tensor.numpy()
                return _FakeTensor(np.full_like(arr, 0.15))

        fake_tf = _FakeTensorFlowModule(keras_model=FakeKerasArtifactModel())
        original_find_spec = dl_module.importlib.util.find_spec
        original_import_module = dl_module.importlib.import_module

        def fake_find_spec(module_name):
            if module_name == "tensorflow":
                return object()
            return original_find_spec(module_name)

        def fake_import_module(module_name, package=None):
            if module_name == "tensorflow":
                return fake_tf
            return original_import_module(module_name, package)

        monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(dl_module.importlib, "import_module", fake_import_module)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            "tensorflow_inference",
            model_kwargs={
                "checkpoint_path": str(checkpoint),
                "input_layout": TensorFlowTensorLayout.BATCH_TIME_CHANNELS,
                "spec_overrides": {
                    "name": "TensorFlowArtifactRuntimeModel",
                    "architecture": DeepLearningArchitecture.AUTOENCODER,
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            },
        )
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.15))
        np.testing.assert_allclose(result.get_raw()._data, original - 0.15)
        assert fake_tf.loaded_model_paths == [(str(checkpoint), False)]
        assert fake_tf.device_calls == ["/CPU:0"]

        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["model"] == "TensorFlowArtifactRuntimeModel"
        assert run["runtime"] == DeepLearningRuntime.TENSORFLOW.value
        assert run["prediction_metadata"]["runtime_backend"] == "tensorflow"
        assert run["prediction_metadata"]["input_layout"] == TensorFlowTensorLayout.BATCH_TIME_CHANNELS.value

    def test_tensorflow_inference_adapter_maps_dual_outputs(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "tf_dual.keras"
        checkpoint.write_bytes(b"placeholder")

        class FakeKerasDualOutputModel:
            def __call__(self, input_tensor, training=False):
                arr = input_tensor.numpy().astype(np.float64)
                artifact = np.full_like(arr, 0.2e-6)
                clean = arr - artifact
                return {
                    "artifact": _FakeTensor(artifact),
                    "clean": _FakeTensor(clean),
                }

        fake_tf = _FakeTensorFlowModule(keras_model=FakeKerasDualOutputModel(), gpu_available=True)
        original_find_spec = dl_module.importlib.util.find_spec
        original_import_module = dl_module.importlib.import_module

        def fake_find_spec(module_name):
            if module_name == "tensorflow":
                return object()
            return original_find_spec(module_name)

        def fake_import_module(module_name, package=None):
            if module_name == "tensorflow":
                return fake_tf
            return original_import_module(module_name, package)

        monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(dl_module.importlib, "import_module", fake_import_module)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            "tensorflow_inference",
            model_kwargs={
                "checkpoint_path": str(checkpoint),
                "spec_overrides": {
                    "name": "TensorFlowDualRuntimeModel",
                    "architecture": DeepLearningArchitecture.HYBRID,
                    "output_type": DeepLearningOutputType.BOTH,
                    "device_preference": "cuda",
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            },
        )
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.2e-6), atol=1e-12)
        np.testing.assert_allclose(result.get_raw()._data, original - 0.2e-6, atol=1e-12)
        assert fake_tf.device_calls == ["/GPU:0"]
        assert result.metadata.custom["deep_learning_runs"][0]["prediction_metadata"]["device"] == "/GPU:0"

    def test_tensorflow_saved_model_uses_only_available_signature(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "saved_model"
        checkpoint.mkdir()

        def output_factory(input_tensor):
            arr = input_tensor.numpy()
            return {"artifact": _FakeTensor(np.full_like(arr, 0.05))}

        saved_model = _FakeSavedModel(
            signatures={
                "predict_artifact": _FakeSavedModelSignature(output_factory),
            }
        )
        fake_tf = _FakeTensorFlowModule(saved_model=saved_model)
        original_find_spec = dl_module.importlib.util.find_spec
        original_import_module = dl_module.importlib.import_module

        def fake_find_spec(module_name):
            if module_name == "tensorflow":
                return object()
            return original_find_spec(module_name)

        def fake_import_module(module_name, package=None):
            if module_name == "tensorflow":
                return fake_tf
            return original_import_module(module_name, package)

        monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(dl_module.importlib, "import_module", fake_import_module)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            "tensorflow_inference",
            model_kwargs={
                "checkpoint_path": str(checkpoint),
                "output_key": "artifact",
                "spec_overrides": {
                    "name": "TensorFlowSavedModelArtifact",
                    "architecture": DeepLearningArchitecture.AUTOENCODER,
                },
            },
        )
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.05))
        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["prediction_metadata"]["serving_signature_name"] == "predict_artifact"

    def test_tensorflow_saved_model_supports_explicit_input_key(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "saved_model_keyword"
        checkpoint.mkdir()

        def output_factory(input_tensor):
            arr = input_tensor.numpy()
            return {"artifact": _FakeTensor(np.full_like(arr, 0.07))}

        saved_model = _FakeSavedModel(
            signatures={
                "serving_default": _FakeSavedModelSignature(
                    output_factory,
                    keyword_names=("waveform",),
                    reject_positional=True,
                ),
            }
        )
        fake_tf = _FakeTensorFlowModule(saved_model=saved_model)
        original_find_spec = dl_module.importlib.util.find_spec
        original_import_module = dl_module.importlib.import_module

        def fake_find_spec(module_name):
            if module_name == "tensorflow":
                return object()
            return original_find_spec(module_name)

        def fake_import_module(module_name, package=None):
            if module_name == "tensorflow":
                return fake_tf
            return original_import_module(module_name, package)

        monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(dl_module.importlib, "import_module", fake_import_module)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            "tensorflow_inference",
            model_kwargs={
                "checkpoint_path": str(checkpoint),
                "input_key": "waveform",
                "output_key": "artifact",
                "spec_overrides": {
                    "name": "TensorFlowSavedModelKeywordInput",
                    "architecture": DeepLearningArchitecture.AUTOENCODER,
                },
            },
        )
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.07))
        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["prediction_metadata"]["serving_signature_name"] == "serving_default"

    def test_pytorch_inference_adapter_runs_single_output_model_via_registry(
        self,
        sample_context,
        monkeypatch,
        tmp_path,
    ):
        checkpoint = tmp_path / "torch_model.pt"
        checkpoint.write_bytes(b"placeholder")

        class FakeTorchArtifactModule:
            def __init__(self):
                self.eval_called = False
                self.device = None

            def to(self, device_name):
                self.device = device_name
                return self

            def eval(self):
                self.eval_called = True
                return self

            def __call__(self, input_tensor):
                arr = input_tensor.numpy()
                return _FakeTensor(np.full_like(arr, 0.12))

        fake_model = FakeTorchArtifactModule()
        fake_torch = _FakeTorchModule(load_return=fake_model)
        original_find_spec = dl_module.importlib.util.find_spec
        original_import_module = dl_module.importlib.import_module

        def fake_find_spec(module_name):
            if module_name == "torch":
                return object()
            return original_find_spec(module_name)

        def fake_import_module(module_name, package=None):
            if module_name == "torch":
                return fake_torch
            return original_import_module(module_name, package)

        monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(dl_module.importlib, "import_module", fake_import_module)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            "pytorch_inference",
            model_kwargs={
                "checkpoint_path": str(checkpoint),
                "input_layout": PyTorchTensorLayout.BATCH_TIME_CHANNELS,
                "spec_overrides": {
                    "name": "PyTorchArtifactRuntimeModel",
                    "architecture": DeepLearningArchitecture.AUTOENCODER,
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            },
        )
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.12))
        np.testing.assert_allclose(result.get_raw()._data, original - 0.12)
        assert fake_torch.load_calls == [(str(checkpoint), "cpu")]
        assert fake_torch.as_tensor_calls == ["cpu"]
        assert fake_torch.no_grad_calls == ["enter", "exit"]
        assert fake_model.eval_called is True
        assert fake_model.device == "cpu"

        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["model"] == "PyTorchArtifactRuntimeModel"
        assert run["runtime"] == DeepLearningRuntime.PYTORCH.value
        assert run["prediction_metadata"]["runtime_backend"] == "pytorch"
        assert run["prediction_metadata"]["input_layout"] == PyTorchTensorLayout.BATCH_TIME_CHANNELS.value

    def test_pytorch_inference_adapter_maps_dual_outputs(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "torch_dual.pt"
        checkpoint.write_bytes(b"placeholder")

        class FakeTorchDualModule:
            def __init__(self):
                self.device = None

            def to(self, device_name):
                self.device = device_name
                return self

            def eval(self):
                return self

            def __call__(self, input_tensor):
                arr = input_tensor.numpy().astype(np.float64)
                artifact = np.full_like(arr, 0.25e-6)
                clean = arr - artifact
                return {
                    "artifact": _FakeTensor(artifact),
                    "clean": _FakeTensor(clean),
                }

        fake_model = FakeTorchDualModule()
        fake_torch = _FakeTorchModule(load_return=fake_model, cuda_available=True)
        original_find_spec = dl_module.importlib.util.find_spec
        original_import_module = dl_module.importlib.import_module

        def fake_find_spec(module_name):
            if module_name == "torch":
                return object()
            return original_find_spec(module_name)

        def fake_import_module(module_name, package=None):
            if module_name == "torch":
                return fake_torch
            return original_import_module(module_name, package)

        monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(dl_module.importlib, "import_module", fake_import_module)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            "pytorch_inference",
            model_kwargs={
                "checkpoint_path": str(checkpoint),
                "spec_overrides": {
                    "name": "PyTorchDualRuntimeModel",
                    "architecture": DeepLearningArchitecture.HYBRID,
                    "output_type": DeepLearningOutputType.BOTH,
                    "device_preference": "cuda",
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            },
        )
        result = processor.execute(sample_context)

        np.testing.assert_allclose(result.get_estimated_noise(), np.full_like(original, 0.25e-6), atol=1e-12)
        np.testing.assert_allclose(result.get_raw()._data, original - 0.25e-6, atol=1e-12)
        assert fake_torch.load_calls == [(str(checkpoint), "cuda:0")]
        assert fake_torch.as_tensor_calls == ["cuda:0"]


# ---------------------------------------------------------------------------
# Helpers for ONNX tests
# ---------------------------------------------------------------------------


class _FakeOnnxOutput:
    """Minimal stand-in for onnxruntime NodeArg."""

    def __init__(self, name: str):
        self.name = name


class _FakeOnnxInput:
    """Minimal stand-in for onnxruntime NodeArg (input side)."""

    def __init__(self, name: str):
        self.name = name


class _FakeOnnxSession:
    """Stand-in for onnxruntime.InferenceSession."""

    def __init__(self, path: str, *, providers: list[str], output_factory, input_name: str = "input"):
        self.path = path
        self._providers = list(providers)
        self._output_factory = output_factory
        self._input_name = input_name
        self.run_calls: list[tuple] = []

    def get_inputs(self):
        return [_FakeOnnxInput(self._input_name)]

    def get_outputs(self):
        return [o for o in self._output_factory(None, only_names=True)]

    def get_providers(self):
        return self._providers

    def run(self, output_names, input_feed):
        self.run_calls.append((output_names, input_feed))
        return self._output_factory(input_feed, only_names=False)


class _SingleArtifactOnnxSession(_FakeOnnxSession):
    """Returns a single artifact output of constant value 0.15."""

    _ARTIFACT_VALUE = 0.15

    def __init__(self, path, *, providers):
        super().__init__(path, providers=providers, output_factory=self._make_outputs)

    def _make_outputs(self, input_feed, only_names):
        if only_names:
            return [_FakeOnnxOutput("output")]
        arr = next(iter(input_feed.values()))
        return [np.full_like(arr, self._ARTIFACT_VALUE, dtype=np.float32)]


class _SingleCleanOnnxSession(_FakeOnnxSession):
    """Returns a single clean-signal output (original * 0.5)."""

    def __init__(self, path, *, providers):
        super().__init__(path, providers=providers, output_factory=self._make_outputs)

    def _make_outputs(self, input_feed, only_names):
        if only_names:
            return [_FakeOnnxOutput("output")]
        arr = next(iter(input_feed.values())).astype(np.float64)
        return [(arr * 0.5).astype(np.float32)]


class _DualOutputOnnxSession(_FakeOnnxSession):
    """Returns both 'artifact' and 'clean' outputs."""

    _ARTIFACT_VALUE = 0.3e-6

    def __init__(self, path, *, providers):
        super().__init__(path, providers=providers, output_factory=self._make_outputs)

    def _make_outputs(self, input_feed, only_names):
        if only_names:
            return [_FakeOnnxOutput("artifact"), _FakeOnnxOutput("clean")]
        arr = next(iter(input_feed.values())).astype(np.float64)
        artifact = np.full_like(arr, self._ARTIFACT_VALUE, dtype=np.float32)
        clean = (arr - artifact).astype(np.float32)
        return [artifact, clean]


def _make_onnx_monkeypatch(monkeypatch, session_factory, *, cuda_available: bool = False):
    """Patch importlib so that 'onnxruntime' resolves to a fake module."""
    original_find_spec = dl_module.importlib.util.find_spec
    original_import_module = dl_module.importlib.import_module

    available_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda_available else ["CPUExecutionProvider"]

    class _FakeOrtModule:
        SessionOptions = None  # no-op

        @staticmethod
        def get_available_providers():
            return available_providers

        @staticmethod
        def InferenceSession(path, *, sess_options=None, providers=None):
            return session_factory(path, providers=providers or [])

    fake_ort = _FakeOrtModule()

    def fake_find_spec(module_name):
        if module_name == "onnxruntime":
            return object()
        return original_find_spec(module_name)

    def fake_import_module(module_name, package=None):
        if module_name == "onnxruntime":
            return fake_ort
        return original_import_module(module_name, package)

    monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(dl_module.importlib, "import_module", fake_import_module)
    return fake_ort


@pytest.mark.unit
class TestOnnxInferenceAdapter:
    """Tests for OnnxInferenceAdapter end-to-end inference via DeepLearningCorrection."""

    # ------------------------------------------------------------------
    # Construction and spec validation
    # ------------------------------------------------------------------

    def test_onnx_adapter_default_spec_has_onnx_runtime(self, tmp_path):
        path = tmp_path / "model.onnx"
        path.write_bytes(b"placeholder")
        adapter = OnnxInferenceAdapter(checkpoint_path=str(path))
        assert adapter.spec.runtime == DeepLearningRuntime.ONNX

    def test_onnx_adapter_rejects_non_onnx_runtime_override(self, tmp_path):
        path = tmp_path / "model.onnx"
        path.write_bytes(b"placeholder")
        with pytest.raises(ValueError, match="runtime='onnx'"):
            OnnxInferenceAdapter(
                checkpoint_path=str(path),
                spec_overrides={"runtime": DeepLearningRuntime.PYTORCH},
            )

    def test_onnx_adapter_output_layout_defaults_to_input_layout(self, tmp_path):
        path = tmp_path / "model.onnx"
        path.write_bytes(b"placeholder")
        adapter = OnnxInferenceAdapter(
            checkpoint_path=str(path),
            input_layout=OnnxTensorLayout.BATCH_TIME_CHANNELS,
        )
        assert adapter.output_layout == OnnxTensorLayout.BATCH_TIME_CHANNELS

    def test_onnx_adapter_registered_in_registry(self):
        registry = DeepLearningModelRegistry.get_instance()
        cls = registry.get("onnx_inference")
        assert cls is OnnxInferenceAdapter

    # ------------------------------------------------------------------
    # Single artifact output
    # ------------------------------------------------------------------

    def test_onnx_single_artifact_output_corrects_signal(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        sessions = []

        def session_factory(path, *, providers):
            s = _SingleArtifactOnnxSession(path, providers=providers)
            sessions.append(s)
            return s

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            OnnxInferenceAdapter(
                checkpoint_path=str(checkpoint),
                spec_overrides={
                    "name": "OnnxArtifactModel",
                    "architecture": DeepLearningArchitecture.AUTOENCODER,
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            )
        )
        result = processor.execute(sample_context)

        expected_artifact = np.full_like(original, _SingleArtifactOnnxSession._ARTIFACT_VALUE)
        np.testing.assert_allclose(result.get_estimated_noise(), expected_artifact, atol=1e-6)
        np.testing.assert_allclose(result.get_raw()._data, original - expected_artifact, atol=1e-6)
        assert len(sessions) == 1
        assert len(sessions[0].run_calls) == 1

    # ------------------------------------------------------------------
    # Single clean output
    # ------------------------------------------------------------------

    def test_onnx_single_clean_output_corrects_signal(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        def session_factory(path, *, providers):
            return _SingleCleanOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            OnnxInferenceAdapter(
                checkpoint_path=str(checkpoint),
                spec_overrides={
                    "name": "OnnxCleanModel",
                    "architecture": DeepLearningArchitecture.UNET,
                    "output_type": DeepLearningOutputType.CLEAN,
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            )
        )
        result = processor.execute(sample_context)

        # clean = original * 0.5  →  artifact = original * 0.5
        np.testing.assert_allclose(
            result.get_raw()._data,
            original * 0.5,
            atol=1e-6,
        )

    # ------------------------------------------------------------------
    # Dual output (artifact + clean)
    # ------------------------------------------------------------------

    def test_onnx_dual_output_corrects_signal(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        def session_factory(path, *, providers):
            return _DualOutputOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            OnnxInferenceAdapter(
                checkpoint_path=str(checkpoint),
                spec_overrides={
                    "name": "OnnxDualModel",
                    "architecture": DeepLearningArchitecture.HYBRID,
                    "output_type": DeepLearningOutputType.BOTH,
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            )
        )
        result = processor.execute(sample_context)

        expected_artifact = np.full_like(original, _DualOutputOnnxSession._ARTIFACT_VALUE)
        np.testing.assert_allclose(result.get_estimated_noise(), expected_artifact, atol=1e-11)
        np.testing.assert_allclose(result.get_raw()._data, original - expected_artifact, atol=1e-11)

    # ------------------------------------------------------------------
    # Provider / device resolution
    # ------------------------------------------------------------------

    def test_onnx_cpu_provider_selected_without_cuda(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        providers_used: list[list[str]] = []

        def session_factory(path, *, providers):
            providers_used.append(list(providers))
            return _SingleArtifactOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory, cuda_available=False)

        processor = DeepLearningCorrection(
            OnnxInferenceAdapter(
                checkpoint_path=str(checkpoint),
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            )
        )
        processor.execute(sample_context)
        assert providers_used[0] == ["CPUExecutionProvider"]

    def test_onnx_cuda_provider_selected_when_available(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        providers_used: list[list[str]] = []

        def session_factory(path, *, providers):
            providers_used.append(list(providers))
            return _SingleArtifactOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory, cuda_available=True)

        processor = DeepLearningCorrection(
            OnnxInferenceAdapter(
                checkpoint_path=str(checkpoint),
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                    "device_preference": "auto",
                },
            )
        )
        processor.execute(sample_context)
        assert providers_used[0][0] == "CUDAExecutionProvider"

    def test_onnx_explicit_cuda_preference_raises_without_cuda(self, sample_context, monkeypatch, tmp_path):
        from facet.core import ProcessorValidationError

        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        def session_factory(path, *, providers):
            return _SingleArtifactOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory, cuda_available=False)

        adapter = OnnxInferenceAdapter(
            checkpoint_path=str(checkpoint),
            spec_overrides={"device_preference": "cuda"},
        )
        with pytest.raises(ProcessorValidationError, match="CUDAExecutionProvider"):
            adapter._load_session()

    # ------------------------------------------------------------------
    # Input / output layout
    # ------------------------------------------------------------------

    def test_onnx_batch_time_channels_input_layout(self, sample_context, monkeypatch, tmp_path):
        """Model receives (1, time, channels) when layout=BATCH_TIME_CHANNELS."""
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        received_shapes: list[tuple] = []

        class _ShapeCapturingSession(_FakeOnnxSession):
            def __init__(self, path, *, providers):
                super().__init__(path, providers=providers, output_factory=self._factory)

            def _factory(self, input_feed, only_names):
                if only_names:
                    return [_FakeOnnxOutput("output")]
                arr = next(iter(input_feed.values()))
                received_shapes.append(arr.shape)
                # return zeros in BATCH_TIME_CHANNELS layout
                return [np.zeros_like(arr)]

        def session_factory(path, *, providers):
            return _ShapeCapturingSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        raw = sample_context.get_raw()
        n_channels, n_samples = raw._data.shape

        processor = DeepLearningCorrection(
            OnnxInferenceAdapter(
                checkpoint_path=str(checkpoint),
                input_layout=OnnxTensorLayout.BATCH_TIME_CHANNELS,
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            )
        )
        processor.execute(sample_context)

        assert len(received_shapes) == 1
        assert received_shapes[0] == (1, n_samples, n_channels)

    # ------------------------------------------------------------------
    # Named output_key selection
    # ------------------------------------------------------------------

    def test_onnx_output_key_selects_named_output(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        class _NamedOutputSession(_FakeOnnxSession):
            _VALUE = 0.05e-6

            def __init__(self, path, *, providers):
                super().__init__(path, providers=providers, output_factory=self._factory)

            def _factory(self, input_feed, only_names):
                if only_names:
                    return [_FakeOnnxOutput("noise"), _FakeOnnxOutput("artifact")]
                arr = next(iter(input_feed.values()))
                return [
                    np.zeros_like(arr),
                    np.full_like(arr, self._VALUE, dtype=np.float32),
                ]

        def session_factory(path, *, providers):
            return _NamedOutputSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            OnnxInferenceAdapter(
                checkpoint_path=str(checkpoint),
                output_key="artifact",
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            )
        )
        result = processor.execute(sample_context)

        np.testing.assert_allclose(
            result.get_raw()._data,
            original - _NamedOutputSession._VALUE,
            atol=1e-11,
        )

    def test_onnx_missing_output_key_raises(self, sample_context, monkeypatch, tmp_path):
        from facet.core import ProcessorValidationError

        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        def session_factory(path, *, providers):
            return _SingleArtifactOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        adapter = OnnxInferenceAdapter(
            checkpoint_path=str(checkpoint),
            output_key="nonexistent",
            spec_overrides={
                "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                "supports_multichannel": True,
            },
        )
        adapter._load_session()  # prime session cache
        with pytest.raises(ProcessorValidationError, match="nonexistent"):
            adapter.predict(sample_context)

    # ------------------------------------------------------------------
    # Session caching (lazy load)
    # ------------------------------------------------------------------

    def test_onnx_session_created_once_across_multiple_predict_calls(
        self, sample_context, monkeypatch, tmp_path
    ):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        creation_count = [0]

        def session_factory(path, *, providers):
            creation_count[0] += 1
            return _SingleArtifactOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        adapter = OnnxInferenceAdapter(
            checkpoint_path=str(checkpoint),
            spec_overrides={
                "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                "supports_multichannel": True,
            },
        )
        adapter.predict(sample_context)
        adapter.predict(sample_context)
        adapter.predict(sample_context)

        assert creation_count[0] == 1

    # ------------------------------------------------------------------
    # Metadata in pipeline result
    # ------------------------------------------------------------------

    def test_onnx_prediction_metadata_contains_backend_and_providers(
        self, sample_context, monkeypatch, tmp_path
    ):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        def session_factory(path, *, providers):
            return _SingleArtifactOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        processor = DeepLearningCorrection(
            OnnxInferenceAdapter(
                checkpoint_path=str(checkpoint),
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            ),
            store_run_metadata=True,
        )
        result = processor.execute(sample_context)

        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["runtime"] == DeepLearningRuntime.ONNX.value
        pred_meta = run["prediction_metadata"]
        assert pred_meta["runtime_backend"] == "onnxruntime"
        assert "providers" in pred_meta
        assert "input_layout" in pred_meta

    # ------------------------------------------------------------------
    # Via registry name
    # ------------------------------------------------------------------

    def test_onnx_adapter_resolves_via_registry_name(self, sample_context, monkeypatch, tmp_path):
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        def session_factory(path, *, providers):
            return _SingleArtifactOnnxSession(path, providers=providers)

        original_find_spec = dl_module.importlib.util.find_spec
        original_import_module = dl_module.importlib.import_module

        class _FakeOrtModule:
            SessionOptions = None

            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

            @staticmethod
            def InferenceSession(path, *, sess_options=None, providers=None):
                return session_factory(path, providers=providers or [])

        fake_ort = _FakeOrtModule()

        def fake_find_spec(module_name):
            if module_name == "onnxruntime":
                return object()
            return original_find_spec(module_name)

        def fake_import_module(module_name, package=None):
            if module_name == "onnxruntime":
                return fake_ort
            return original_import_module(module_name, package)

        monkeypatch.setattr(dl_module.importlib.util, "find_spec", fake_find_spec)
        monkeypatch.setattr(dl_module.importlib, "import_module", fake_import_module)

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            "onnx_inference",
            model_kwargs={
                "checkpoint_path": str(checkpoint),
                "spec_overrides": {
                    "name": "OnnxRegistryModel",
                    "architecture": DeepLearningArchitecture.AUTOENCODER,
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            },
        )
        result = processor.execute(sample_context)

        expected = np.full_like(original, _SingleArtifactOnnxSession._ARTIFACT_VALUE)
        np.testing.assert_allclose(result.get_estimated_noise(), expected, atol=1e-6)
        np.testing.assert_allclose(result.get_raw()._data, original - expected, atol=1e-6)
        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["model"] == "OnnxRegistryModel"
        assert run["runtime"] == DeepLearningRuntime.ONNX.value
        assert run["prediction_metadata"]["runtime_backend"] == "onnxruntime"


# ---------------------------------------------------------------------------
# NumpyInferenceAdapter tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNumpyInferenceAdapter:
    """Tests for NumpyInferenceAdapter — pure-NumPy checkpoint-backed models."""

    # ------------------------------------------------------------------
    # Construction and spec validation
    # ------------------------------------------------------------------

    def test_numpy_adapter_default_spec_has_numpy_runtime(self, tmp_path):
        weights = np.ones((4, 4))
        ckpt = tmp_path / "weights.npy"
        np.save(str(ckpt), weights)
        adapter = NumpyInferenceAdapter(
            checkpoint_path=str(ckpt),
            predict_fn=lambda data, w: data * 0.0,
        )
        assert adapter.spec.runtime == DeepLearningRuntime.NUMPY

    def test_numpy_adapter_rejects_non_numpy_runtime_override(self, tmp_path):
        ckpt = tmp_path / "weights.npy"
        np.save(str(ckpt), np.ones(4))
        with pytest.raises(ValueError, match="runtime='numpy'"):
            NumpyInferenceAdapter(
                checkpoint_path=str(ckpt),
                predict_fn=lambda data, w: data,
                spec_overrides={"runtime": DeepLearningRuntime.ONNX},
            )

    def test_numpy_adapter_rejects_non_callable_predict_fn(self, tmp_path):
        ckpt = tmp_path / "weights.npy"
        np.save(str(ckpt), np.ones(4))
        with pytest.raises(TypeError, match="callable"):
            NumpyInferenceAdapter(checkpoint_path=str(ckpt), predict_fn="not_a_fn")

    def test_numpy_adapter_registered_in_registry(self):
        registry = DeepLearningModelRegistry.get_instance()
        cls = registry.get("numpy_inference")
        assert cls is NumpyInferenceAdapter

    # ------------------------------------------------------------------
    # Artifact output via .npy checkpoint
    # ------------------------------------------------------------------

    def test_numpy_npy_artifact_output(self, sample_context, tmp_path):
        artifact_value = 0.05e-6
        weights = np.full((1,), artifact_value)
        ckpt = tmp_path / "artifact_scale.npy"
        np.save(str(ckpt), weights)

        def predict_fn(data, w):
            return np.full_like(data, w[0])

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            NumpyInferenceAdapter(
                checkpoint_path=str(ckpt),
                predict_fn=predict_fn,
                spec_overrides={
                    "name": "NumpyArtifact",
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            )
        )
        result = processor.execute(sample_context)

        np.testing.assert_allclose(
            result.get_estimated_noise(),
            np.full_like(original, artifact_value),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.get_raw()._data,
            original - artifact_value,
            atol=1e-12,
        )

    # ------------------------------------------------------------------
    # Clean output via .npz checkpoint
    # ------------------------------------------------------------------

    def test_numpy_npz_clean_output(self, sample_context, tmp_path):
        ckpt = tmp_path / "model.npz"
        np.savez(str(ckpt), scale=np.array([0.5]))

        def predict_fn(data, w):
            return data * float(w["scale"][0])

        original = sample_context.get_raw()._data.copy()
        processor = DeepLearningCorrection(
            NumpyInferenceAdapter(
                checkpoint_path=str(ckpt),
                predict_fn=predict_fn,
                spec_overrides={
                    "name": "NumpyClean",
                    "output_type": DeepLearningOutputType.CLEAN,
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            )
        )
        result = processor.execute(sample_context)

        # clean = original * 0.5  →  corrected = original * 0.5
        np.testing.assert_allclose(result.get_raw()._data, original * 0.5, atol=1e-12)

    # ------------------------------------------------------------------
    # Lazy-load: weights loaded once
    # ------------------------------------------------------------------

    def test_numpy_weights_loaded_once(self, sample_context, tmp_path):
        load_calls = [0]
        weights = np.zeros((2, 2))
        ckpt = tmp_path / "w.npy"
        np.save(str(ckpt), weights)

        original_np_load = np.load

        def counting_load(path, **kwargs):
            load_calls[0] += 1
            return original_np_load(path, **kwargs)

        adapter = NumpyInferenceAdapter(
            checkpoint_path=str(ckpt),
            predict_fn=lambda data, w: np.zeros_like(data),
            spec_overrides={
                "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                "supports_multichannel": True,
            },
        )

        import unittest.mock as mock
        with mock.patch("numpy.load", side_effect=counting_load):
            adapter._weights = None  # reset cache
            adapter.predict(sample_context)
            adapter.predict(sample_context)
            adapter.predict(sample_context)

        assert load_calls[0] == 1

    # ------------------------------------------------------------------
    # predict_fn shape mismatch raises
    # ------------------------------------------------------------------

    def test_numpy_wrong_output_shape_raises(self, sample_context, tmp_path):
        from facet.core import ProcessorValidationError

        ckpt = tmp_path / "bad.npy"
        np.save(str(ckpt), np.ones(1))

        adapter = NumpyInferenceAdapter(
            checkpoint_path=str(ckpt),
            predict_fn=lambda data, w: np.zeros((1, 10)),  # wrong shape
            spec_overrides={
                "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                "supports_multichannel": True,
            },
        )
        with pytest.raises(ProcessorValidationError, match="shape"):
            adapter.predict(sample_context)

    # ------------------------------------------------------------------
    # predict_fn exception is wrapped
    # ------------------------------------------------------------------

    def test_numpy_predict_fn_exception_wrapped(self, sample_context, tmp_path):
        from facet.core import ProcessorValidationError

        ckpt = tmp_path / "ok.npy"
        np.save(str(ckpt), np.ones(1))

        def bad_fn(data, w):
            raise RuntimeError("model exploded")

        adapter = NumpyInferenceAdapter(
            checkpoint_path=str(ckpt),
            predict_fn=bad_fn,
            spec_overrides={
                "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                "supports_multichannel": True,
            },
        )
        with pytest.raises(ProcessorValidationError, match="model exploded"):
            adapter.predict(sample_context)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def test_numpy_metadata_contains_runtime_backend(self, sample_context, tmp_path):
        ckpt = tmp_path / "w.npy"
        np.save(str(ckpt), np.ones((3,)))

        processor = DeepLearningCorrection(
            NumpyInferenceAdapter(
                checkpoint_path=str(ckpt),
                predict_fn=lambda data, w: np.zeros_like(data),
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            ),
            store_run_metadata=True,
        )
        result = processor.execute(sample_context)
        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["prediction_metadata"]["runtime_backend"] == "numpy"
        assert run["runtime"] == DeepLearningRuntime.NUMPY.value


# ---------------------------------------------------------------------------
# Blueprint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResearchBlueprints:
    """Tests for research blueprint completeness."""

    def test_eegdfus_blueprint_present(self):
        blueprints = list_deep_learning_blueprints()
        assert "eegdfus" in blueprints

    def test_eegdfus_blueprint_is_diffusion_architecture(self):
        spec = list_deep_learning_blueprints()["eegdfus"]
        assert spec.architecture == DeepLearningArchitecture.DIFFUSION

    def test_eegdfus_blueprint_is_offline(self):
        spec = list_deep_learning_blueprints()["eegdfus"]
        from facet.correction.deep_learning import DeepLearningLatencyProfile
        assert spec.latency_profile == DeepLearningLatencyProfile.OFFLINE

    def test_eegm2_blueprint_present(self):
        blueprints = list_deep_learning_blueprints()
        assert "eegm2" in blueprints

    def test_eegm2_blueprint_is_state_space_architecture(self):
        spec = list_deep_learning_blueprints()["eegm2"]
        assert spec.architecture == DeepLearningArchitecture.STATE_SPACE

    def test_eegm2_blueprint_supports_chunking(self):
        spec = list_deep_learning_blueprints()["eegm2"]
        assert spec.supports_chunking is True
        assert spec.chunk_size_samples == 8192

    def test_all_expected_blueprints_present(self):
        expected = {
            "dar", "dpae", "ic_u_net", "dhct_gan", "nested_gan",
            "d4pm", "denoise_mamba", "conv_tasnet", "demucs",
            "sepformer", "vit_spectrogram", "st_gnn",
            "eegdfus", "eegm2",
        }
        blueprints = list_deep_learning_blueprints()
        assert expected == set(blueprints.keys())

    def test_blueprints_returns_copy(self):
        a = list_deep_learning_blueprints()
        b = list_deep_learning_blueprints()
        assert a is not b


# ---------------------------------------------------------------------------
# Spec serialisation tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpecSerialization:
    """Tests for spec_to_dict / spec_from_dict round-trip."""

    def _base_spec(self, **kwargs) -> DeepLearningModelSpec:
        defaults = dict(
            name="TestSpec",
            architecture=DeepLearningArchitecture.AUTOENCODER,
            runtime=DeepLearningRuntime.NUMPY,
        )
        defaults.update(kwargs)
        return DeepLearningModelSpec(**defaults)

    # ------------------------------------------------------------------
    # spec_to_dict
    # ------------------------------------------------------------------

    def test_spec_to_dict_returns_dict(self):
        d = spec_to_dict(self._base_spec())
        assert isinstance(d, dict)

    def test_spec_to_dict_enum_fields_are_strings(self):
        d = spec_to_dict(self._base_spec())
        assert isinstance(d["architecture"], str)
        assert isinstance(d["runtime"], str)
        assert isinstance(d["domain"], str)
        assert isinstance(d["output_type"], str)

    def test_spec_to_dict_tags_are_list(self):
        spec = self._base_spec(tags=("cnn", "denoising"))
        d = spec_to_dict(spec)
        assert isinstance(d["tags"], list)
        assert d["tags"] == ["cnn", "denoising"]

    def test_spec_to_dict_is_json_serialisable(self):
        import json
        spec = self._base_spec(
            checkpoint_path="/tmp/model.npy",
            tags=("test",),
            min_sfreq=250.0,
        )
        d = spec_to_dict(spec)
        json.dumps(d)  # must not raise

    def test_spec_to_dict_preserves_none_optional_fields(self):
        d = spec_to_dict(self._base_spec())
        assert d["checkpoint_path"] is None
        assert d["min_sfreq"] is None

    # ------------------------------------------------------------------
    # spec_from_dict
    # ------------------------------------------------------------------

    def test_spec_from_dict_reconstructs_spec(self):
        original = self._base_spec(
            description="test model",
            tags=("a", "b"),
        )
        d = spec_to_dict(original)
        restored = spec_from_dict(d)
        assert restored == original

    def test_spec_from_dict_coerces_enum_strings(self):
        d = spec_to_dict(self._base_spec())
        d["architecture"] = "autoencoder"
        d["runtime"] = "numpy"
        spec = spec_from_dict(d)
        assert spec.architecture == DeepLearningArchitecture.AUTOENCODER
        assert spec.runtime == DeepLearningRuntime.NUMPY

    def test_spec_from_dict_restores_tags_as_tuple(self):
        d = spec_to_dict(self._base_spec(tags=("x", "y")))
        assert isinstance(d["tags"], list)
        spec = spec_from_dict(d)
        assert isinstance(spec.tags, tuple)
        assert spec.tags == ("x", "y")

    def test_spec_from_dict_ignores_unknown_keys(self):
        d = spec_to_dict(self._base_spec())
        d["future_field_unknown"] = "some_value"
        spec = spec_from_dict(d)  # must not raise
        assert spec.name == "TestSpec"

    def test_spec_from_dict_raises_on_invalid_enum_value(self):
        d = spec_to_dict(self._base_spec())
        d["architecture"] = "nonexistent_architecture"
        with pytest.raises(ValueError):
            spec_from_dict(d)

    # ------------------------------------------------------------------
    # Round-trip for all blueprints
    # ------------------------------------------------------------------

    def test_all_blueprints_round_trip(self):
        for key, spec in list_deep_learning_blueprints().items():
            d = spec_to_dict(spec)
            restored = spec_from_dict(d)
            assert restored == spec, f"Round-trip failed for blueprint '{key}'"


# ---------------------------------------------------------------------------
# Trigger-aligned Chunking tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTriggerAlignedChunking:
    """Tests for DeepLearningCorrection with trigger_aligned_chunking=True."""

    # ------------------------------------------------------------------
    # _trigger_chunk_ranges unit tests
    # ------------------------------------------------------------------

    def test_trigger_chunk_ranges_single_trigger_per_chunk(self):
        from facet.correction.deep_learning import DeepLearningCorrection as DLC
        triggers = np.array([100, 200, 300, 400])
        ranges = DLC._trigger_chunk_ranges(triggers, 500, triggers_per_chunk=1)
        assert ranges == [(100, 200), (200, 300), (300, 400), (400, 500)]

    def test_trigger_chunk_ranges_two_triggers_per_chunk(self):
        from facet.correction.deep_learning import DeepLearningCorrection as DLC
        triggers = np.array([100, 200, 300, 400])
        ranges = DLC._trigger_chunk_ranges(triggers, 500, triggers_per_chunk=2)
        assert ranges == [(100, 300), (300, 500)]

    def test_trigger_chunk_ranges_three_triggers_per_chunk_uneven(self):
        from facet.correction.deep_learning import DeepLearningCorrection as DLC
        # 4 triggers, 3 per chunk → one full group + one partial
        triggers = np.array([50, 150, 250, 350])
        ranges = DLC._trigger_chunk_ranges(triggers, 450, triggers_per_chunk=3)
        assert ranges == [(50, 350), (350, 450)]

    def test_trigger_chunk_ranges_unsorted_triggers_are_sorted(self):
        from facet.correction.deep_learning import DeepLearningCorrection as DLC
        triggers = np.array([400, 100, 300, 200])
        ranges = DLC._trigger_chunk_ranges(triggers, 500, triggers_per_chunk=1)
        assert ranges == [(100, 200), (200, 300), (300, 400), (400, 500)]

    def test_trigger_chunk_ranges_empty_triggers_returns_full_range(self):
        from facet.correction.deep_learning import DeepLearningCorrection as DLC
        ranges = DLC._trigger_chunk_ranges(np.array([]), 500, triggers_per_chunk=1)
        assert ranges == [(0, 500)]

    def test_trigger_chunk_ranges_pretrigger_data_excluded(self):
        from facet.correction.deep_learning import DeepLearningCorrection as DLC
        # First trigger at sample 50 → samples [0,50) are excluded
        triggers = np.array([50, 150])
        ranges = DLC._trigger_chunk_ranges(triggers, 200, triggers_per_chunk=1)
        assert ranges[0][0] == 50

    def test_trigger_chunk_ranges_last_chunk_extends_to_total(self):
        from facet.correction.deep_learning import DeepLearningCorrection as DLC
        triggers = np.array([0, 100, 200])
        total = 350
        ranges = DLC._trigger_chunk_ranges(triggers, total, triggers_per_chunk=1)
        assert ranges[-1][1] == total

    # ------------------------------------------------------------------
    # Validation: requires triggers in context
    # ------------------------------------------------------------------

    def test_trigger_aligned_chunking_raises_without_triggers(self, sample_context, tmp_path):
        from facet.core import ProcessorValidationError
        ckpt = tmp_path / "w.npy"
        np.save(str(ckpt), np.ones(1))
        processor = DeepLearningCorrection(
            NumpyInferenceAdapter(
                checkpoint_path=str(ckpt),
                predict_fn=lambda data, w: np.zeros_like(data),
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            ),
            trigger_aligned_chunking=True,
        )
        # sample_context has triggers, so strip them
        import dataclasses
        from facet.core.context import ProcessingMetadata
        no_trigger_ctx = sample_context.with_raw(sample_context.get_raw().copy())
        no_trigger_ctx.metadata.triggers = None

        with pytest.raises(ProcessorValidationError, match="trigger"):
            processor.validate(no_trigger_ctx)

    def test_triggers_per_chunk_zero_raises(self, tmp_path):
        ckpt = tmp_path / "w.npy"
        np.save(str(ckpt), np.ones(1))
        with pytest.raises(ValueError, match="triggers_per_chunk"):
            DeepLearningCorrection(
                NumpyInferenceAdapter(
                    checkpoint_path=str(ckpt),
                    predict_fn=lambda d, w: np.zeros_like(d),
                ),
                triggers_per_chunk=0,
            )

    # ------------------------------------------------------------------
    # End-to-end: trigger-aligned correction applies to correct windows
    # ------------------------------------------------------------------

    def test_trigger_aligned_chunking_corrects_only_trigger_windows(
        self, sample_context, tmp_path
    ):
        """Artifact subtraction happens only within [first_trigger, total_samples)."""
        ckpt = tmp_path / "artifact.npy"
        artifact_value = 0.1e-6
        np.save(str(ckpt), np.array([artifact_value]))

        def predict_fn(data, w):
            return np.full_like(data, w[0])

        original = sample_context.get_raw()._data.copy()
        triggers = sample_context.metadata.triggers  # e.g. [0, 250, 500, ...]
        first_trigger = int(np.min(triggers))

        processor = DeepLearningCorrection(
            NumpyInferenceAdapter(
                checkpoint_path=str(ckpt),
                predict_fn=predict_fn,
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            ),
            trigger_aligned_chunking=True,
            triggers_per_chunk=1,
        )
        result = processor.execute(sample_context)

        # Samples from first_trigger onward should have artifact subtracted
        np.testing.assert_allclose(
            result.get_raw()._data[:, first_trigger:],
            original[:, first_trigger:] - artifact_value,
            atol=1e-12,
        )
        # Samples before first trigger are untouched
        if first_trigger > 0:
            np.testing.assert_allclose(
                result.get_raw()._data[:, :first_trigger],
                original[:, :first_trigger],
                atol=1e-12,
            )

    def test_trigger_aligned_execution_mode_in_metadata(self, sample_context, tmp_path):
        ckpt = tmp_path / "w.npy"
        np.save(str(ckpt), np.ones(1))
        processor = DeepLearningCorrection(
            NumpyInferenceAdapter(
                checkpoint_path=str(ckpt),
                predict_fn=lambda data, w: np.zeros_like(data),
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            ),
            trigger_aligned_chunking=True,
            triggers_per_chunk=2,
            store_run_metadata=True,
        )
        result = processor.execute(sample_context)
        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["execution_mode"] == "trigger_aligned"
        assert run["triggers_per_chunk"] == 2

    def test_trigger_aligned_chunk_count_matches_trigger_groups(self, sample_context, tmp_path):
        ckpt = tmp_path / "w.npy"
        np.save(str(ckpt), np.ones(1))
        triggers = sample_context.metadata.triggers
        n_triggers = len(triggers)
        triggers_per_chunk = 2
        expected_chunks = (n_triggers + triggers_per_chunk - 1) // triggers_per_chunk

        processor = DeepLearningCorrection(
            NumpyInferenceAdapter(
                checkpoint_path=str(ckpt),
                predict_fn=lambda data, w: np.zeros_like(data),
                spec_overrides={
                    "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                    "supports_multichannel": True,
                },
            ),
            trigger_aligned_chunking=True,
            triggers_per_chunk=triggers_per_chunk,
            store_run_metadata=True,
        )
        result = processor.execute(sample_context)
        run = result.metadata.custom["deep_learning_runs"][0]
        assert run["chunk_count"] == expected_chunks


# ---------------------------------------------------------------------------
# Config-Loader tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeepLearningConfig:
    """Tests for to_config_dict / from_config_dict / save / load helpers."""

    def _make_onnx_processor(self, tmp_path, monkeypatch, **correction_kwargs):
        """Helper: build a patched OnnxInferenceAdapter-backed processor."""
        checkpoint = tmp_path / "model.onnx"
        checkpoint.write_bytes(b"placeholder")

        def session_factory(path, *, providers):
            return _SingleArtifactOnnxSession(path, providers=providers)

        _make_onnx_monkeypatch(monkeypatch, session_factory)

        adapter = OnnxInferenceAdapter(
            checkpoint_path=str(checkpoint),
            spec_overrides={
                "name": "ConfigTestModel",
                "architecture": DeepLearningArchitecture.AUTOENCODER,
                "execution_granularity": DeepLearningExecutionGranularity.MULTICHANNEL,
                "supports_multichannel": True,
            },
        )
        return DeepLearningCorrection(adapter, **correction_kwargs)

    # ------------------------------------------------------------------
    # to_config_dict
    # ------------------------------------------------------------------

    def test_to_config_dict_returns_json_serialisable_dict(self, tmp_path, monkeypatch):
        import json
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        d = processor.to_config_dict()
        json.dumps(d)  # must not raise

    def test_to_config_dict_contains_required_keys(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        d = processor.to_config_dict()
        for key in ("version", "processor", "adapter", "spec",
                    "store_run_metadata", "trigger_aligned_chunking", "triggers_per_chunk"):
            assert key in d, f"Missing key: {key}"

    def test_to_config_dict_adapter_name_is_onnx_inference(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        assert processor.to_config_dict()["adapter"] == "onnx_inference"

    def test_to_config_dict_spec_round_trips(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        d = processor.to_config_dict()
        restored_spec = spec_from_dict(d["spec"])
        assert restored_spec == processor.model.spec

    def test_to_config_dict_preserves_trigger_aligned_params(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(
            tmp_path, monkeypatch,
            trigger_aligned_chunking=True,
            triggers_per_chunk=3,
        )
        d = processor.to_config_dict()
        assert d["trigger_aligned_chunking"] is True
        assert d["triggers_per_chunk"] == 3

    def test_to_config_dict_raises_for_unregistered_adapter(self, tmp_path):
        class _UnregisteredAdapter(DeepLearningModelAdapter):
            spec = DeepLearningModelSpec(
                name="Unregistered",
                architecture=DeepLearningArchitecture.CUSTOM,
                runtime=DeepLearningRuntime.NUMPY,
            )
            def predict(self, context):
                return DeepLearningPrediction(artifact_data=np.zeros((1, 1)))

        processor = DeepLearningCorrection(_UnregisteredAdapter())
        with pytest.raises(ValueError, match="not registered"):
            processor.to_config_dict()

    # ------------------------------------------------------------------
    # from_config_dict
    # ------------------------------------------------------------------

    def test_from_config_dict_round_trips_onnx_processor(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        d = processor.to_config_dict()
        restored = DeepLearningCorrection.from_config_dict(d)
        assert restored.model.spec == processor.model.spec
        assert restored.store_run_metadata == processor.store_run_metadata
        assert restored.trigger_aligned_chunking == processor.trigger_aligned_chunking
        assert restored.triggers_per_chunk == processor.triggers_per_chunk

    def test_from_config_dict_raises_when_checkpoint_path_missing(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        d = processor.to_config_dict()
        d["spec"]["checkpoint_path"] = None
        with pytest.raises(ValueError, match="checkpoint_path"):
            DeepLearningCorrection.from_config_dict(d)

    def test_from_config_dict_raises_for_unknown_adapter(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        d = processor.to_config_dict()
        d["adapter"] = "nonexistent_adapter_xyz"
        with pytest.raises(KeyError):
            DeepLearningCorrection.from_config_dict(d)

    def test_from_config_dict_defaults_trigger_params_when_absent(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        d = processor.to_config_dict()
        d.pop("trigger_aligned_chunking", None)
        d.pop("triggers_per_chunk", None)
        restored = DeepLearningCorrection.from_config_dict(d)
        assert restored.trigger_aligned_chunking is False
        assert restored.triggers_per_chunk == 1

    # ------------------------------------------------------------------
    # save / load JSON round-trip
    # ------------------------------------------------------------------

    def test_save_and_load_config_json(self, tmp_path, monkeypatch):
        import json
        processor = self._make_onnx_processor(
            tmp_path, monkeypatch,
            trigger_aligned_chunking=True,
            triggers_per_chunk=2,
        )
        config_path = tmp_path / "config.json"
        save_deep_learning_config(processor, config_path)

        assert config_path.exists()
        with config_path.open() as f:
            raw = json.load(f)
        assert raw["adapter"] == "onnx_inference"

        restored = load_deep_learning_config(config_path)
        assert restored.model.spec == processor.model.spec
        assert restored.trigger_aligned_chunking is True
        assert restored.triggers_per_chunk == 2

    def test_save_config_creates_parent_dirs(self, tmp_path, monkeypatch):
        processor = self._make_onnx_processor(tmp_path, monkeypatch)
        nested = tmp_path / "deep" / "nested" / "config.json"
        save_deep_learning_config(processor, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# SpectrogramMixin
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpectrogramMixin:
    """Tests for SpectrogramMixin STFT/iSTFT wrapping."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_adapter(output_type=None, nperseg=64, noverlap=48, identity=True):
        """Return a concrete SpectrogramMixin + NumpyInferenceAdapter subclass."""
        from dataclasses import replace as _replace

        from facet.correction import (
            DeepLearningOutputType,
            NumpyInferenceAdapter,
            SpectrogramMixin,
        )

        _otype = output_type or DeepLearningOutputType.CLEAN

        class _SpectroAdapter(SpectrogramMixin, NumpyInferenceAdapter):
            # Override STFT params via class-level attributes
            pass

        _SpectroAdapter.nperseg = nperseg
        _SpectroAdapter.noverlap = noverlap

        # track whether _run_spectrogram_model was called
        _SpectroAdapter._call_count = 0

        def _run(self, magnitude, phase, context):
            type(self)._call_count += 1
            return magnitude  # identity

        if identity:
            _SpectroAdapter._run_spectrogram_model = _run

        adapter = _SpectroAdapter(
            checkpoint_path="/nonexistent/path.npz",
            predict_fn=lambda data, w: data,
            spec_overrides={"output_type": _otype},
        )
        return adapter

    @staticmethod
    def _make_context(n_channels=4, n_samples=512, sfreq=256.0):
        import mne

        data = np.random.default_rng(42).standard_normal((n_channels, n_samples)) * 1e-6
        ch_names = [f"EEG{i:02d}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
        raw = mne.io.RawArray(data, info, verbose=False)
        from facet.core import ProcessingContext
        return ProcessingContext(raw=raw)

    # ------------------------------------------------------------------
    # stft_forward
    # ------------------------------------------------------------------

    def test_stft_forward_output_shapes(self):
        """stft_forward returns (n_channels, n_freqs, n_frames) for magnitude and phase."""
        adapter = self._make_adapter()
        data = np.random.randn(4, 512).astype(np.float64)
        mag, phase, freqs, times = adapter.stft_forward(data, sfreq=256.0)
        assert mag.ndim == 3
        assert mag.shape[0] == 4
        assert phase.shape == mag.shape
        assert freqs.ndim == 1
        assert times.ndim == 1
        assert mag.shape[1] == freqs.shape[0]
        assert mag.shape[2] == times.shape[0]

    def test_stft_forward_phase_range(self):
        """Phase values must lie in [-pi, pi]."""
        adapter = self._make_adapter()
        data = np.random.randn(3, 512)
        _, phase, _, _ = adapter.stft_forward(data, sfreq=256.0)
        assert float(np.min(phase)) >= -np.pi - 1e-6
        assert float(np.max(phase)) <= np.pi + 1e-6

    def test_stft_forward_magnitude_non_negative(self):
        """Magnitude must be non-negative."""
        adapter = self._make_adapter()
        data = np.random.randn(2, 512)
        mag, _, _, _ = adapter.stft_forward(data, sfreq=256.0)
        assert float(np.min(mag)) >= 0.0

    def test_stft_forward_invalid_noverlap_raises(self):
        """noverlap >= nperseg must raise ValueError."""
        adapter = self._make_adapter(nperseg=64, noverlap=64)
        data = np.random.randn(2, 256)
        with pytest.raises(ValueError, match="noverlap"):
            adapter.stft_forward(data, sfreq=256.0)

    # ------------------------------------------------------------------
    # istft_backward
    # ------------------------------------------------------------------

    def test_istft_backward_output_shape(self):
        """istft_backward returns (n_channels, n_samples)."""
        adapter = self._make_adapter()
        data = np.random.randn(4, 512)
        mag, phase, _, _ = adapter.stft_forward(data, sfreq=256.0)
        out = adapter.istft_backward(mag, phase, sfreq=256.0, n_samples=512)
        assert out.shape == (4, 512)

    def test_stft_istft_roundtrip(self):
        """STFT → iSTFT roundtrip reconstructs signal up to numerical noise."""
        adapter = self._make_adapter(nperseg=128, noverlap=96)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 1024)).astype(np.float64)
        mag, phase, _, _ = adapter.stft_forward(data, sfreq=512.0)
        reconstructed = adapter.istft_backward(mag, phase, sfreq=512.0, n_samples=1024)
        np.testing.assert_allclose(reconstructed, data, atol=1e-5)

    def test_istft_backward_trims_to_n_samples(self):
        """Output is trimmed to the requested n_samples."""
        adapter = self._make_adapter()
        data = np.random.randn(2, 256)
        mag, phase, _, _ = adapter.stft_forward(data, sfreq=256.0)
        for target_len in (200, 256):
            out = adapter.istft_backward(mag, phase, sfreq=256.0, n_samples=target_len)
            assert out.shape == (2, target_len)

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def test_predict_returns_clean_prediction(self):
        """Identity adapter with CLEAN output_type should return clean_data."""
        from facet.correction import DeepLearningOutputType

        adapter = self._make_adapter(output_type=DeepLearningOutputType.CLEAN)
        ctx = self._make_context()
        pred = adapter.predict(ctx)
        assert pred.clean_data is not None
        assert pred.artifact_data is None
        assert pred.clean_data.shape == ctx.get_raw()._data.shape

    def test_predict_returns_artifact_prediction(self):
        """Identity adapter with ARTIFACT output_type should return artifact_data."""
        from facet.correction import DeepLearningOutputType

        adapter = self._make_adapter(output_type=DeepLearningOutputType.ARTIFACT)
        ctx = self._make_context()
        pred = adapter.predict(ctx)
        assert pred.artifact_data is not None
        assert pred.clean_data is None

    def test_predict_includes_stft_metadata(self):
        """Prediction metadata must carry stft_n_freqs, stft_n_frames, nperseg, noverlap."""
        adapter = self._make_adapter()
        ctx = self._make_context()
        pred = adapter.predict(ctx)
        for key in ("stft_n_freqs", "stft_n_frames", "nperseg", "noverlap"):
            assert key in pred.metadata, f"Missing key '{key}' in prediction metadata"
        assert pred.metadata["nperseg"] == 64
        assert pred.metadata["noverlap"] == 48

    def test_predict_calls_run_spectrogram_model(self):
        """predict() must delegate to _run_spectrogram_model."""
        adapter = self._make_adapter()
        type(adapter)._call_count = 0
        ctx = self._make_context()
        adapter.predict(ctx)
        assert type(adapter)._call_count == 1

    def test_predict_shape_mismatch_raises(self):
        """_run_spectrogram_model returning wrong shape must raise ProcessorValidationError."""
        from facet.correction import NumpyInferenceAdapter, SpectrogramMixin

        class _WrongShapeAdapter(SpectrogramMixin, NumpyInferenceAdapter):
            nperseg = 64
            noverlap = 48

            def _run_spectrogram_model(self, magnitude, phase, context):
                return magnitude[:, :, :1]  # wrong n_frames

        adapter = _WrongShapeAdapter(
            checkpoint_path="/nonexistent/path.npz",
            predict_fn=lambda d, w: d,
        )
        ctx = self._make_context()
        with pytest.raises(ProcessorValidationError):
            adapter.predict(ctx)

    def test_run_spectrogram_model_not_implemented_raises(self):
        """Concrete subclass that forgets _run_spectrogram_model raises NotImplementedError."""
        from facet.correction import NumpyInferenceAdapter, SpectrogramMixin

        class _NoImplAdapter(SpectrogramMixin, NumpyInferenceAdapter):
            nperseg = 64
            noverlap = 48

        adapter = _NoImplAdapter(
            checkpoint_path="/nonexistent/path.npz",
            predict_fn=lambda d, w: d,
        )
        ctx = self._make_context()
        with pytest.raises(NotImplementedError):
            adapter.predict(ctx)
