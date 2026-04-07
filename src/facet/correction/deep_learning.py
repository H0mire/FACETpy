"""Deep-learning integration primitives for artifact correction pipelines.

This module provides preparatory abstractions for integrating learned artifact
removal models into FACETpy without coupling the pipeline to a specific
framework or architecture family.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import mne
import numpy as np
from loguru import logger

from ..core import ProcessingContext, Processor, ProcessorValidationError, register_processor


class DeepLearningArchitecture(StrEnum):
    """High-level architecture families relevant for EEG-fMRI denoising."""

    AUTOENCODER = "autoencoder"
    UNET = "u_net"
    GAN = "gan"
    DIFFUSION = "diffusion"
    STATE_SPACE = "state_space"
    AUDIO_SOURCE_SEPARATION = "audio_source_separation"
    TRANSFORMER = "transformer"
    VISION_TRANSFORMER = "vision_transformer"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class DeepLearningRuntime(StrEnum):
    """Supported inference runtimes."""

    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    NUMPY = "numpy"
    CUSTOM = "custom"


class DeepLearningDomain(StrEnum):
    """Input representation families."""

    TIME_SERIES = "time_series"
    TIME_FREQUENCY = "time_frequency"
    GRAPH = "graph"


class DeepLearningOutputType(StrEnum):
    """What the model predicts."""

    ARTIFACT = "artifact"
    CLEAN = "clean"
    BOTH = "both"


class DeepLearningLatencyProfile(StrEnum):
    """Coarse deployment profile."""

    REALTIME = "realtime"
    NEAR_REALTIME = "near_realtime"
    OFFLINE = "offline"


class DeepLearningExecutionGranularity(StrEnum):
    """How a model consumes channel data during inference."""

    CHANNEL = "channel"
    CHANNEL_GROUP = "channel_group"
    MULTICHANNEL = "multichannel"


class DeepLearningChannelGroupingStrategy(StrEnum):
    """How neighboring channels are selected for channel-group execution."""

    INDEX = "index"
    POSITION = "position"


class DeepLearningDualOutputPolicy(StrEnum):
    """How to resolve simultaneous clean/artifact predictions."""

    STRICT = "strict"
    PREFER_CLEAN = "prefer_clean"
    PREFER_ARTIFACT = "prefer_artifact"


class TensorFlowTensorLayout(StrEnum):
    """Tensor layout convention for TensorFlow inputs and outputs."""

    BATCH_CHANNELS_TIME = "batch_channels_time"
    BATCH_TIME_CHANNELS = "batch_time_channels"


class PyTorchTensorLayout(StrEnum):
    """Tensor layout convention for PyTorch inputs and outputs."""

    BATCH_CHANNELS_TIME = "batch_channels_time"
    BATCH_TIME_CHANNELS = "batch_time_channels"


class OnnxTensorLayout(StrEnum):
    """Tensor layout convention for ONNX Runtime inputs and outputs."""

    BATCH_CHANNELS_TIME = "batch_channels_time"
    BATCH_TIME_CHANNELS = "batch_time_channels"


@dataclass(frozen=True)
class DeepLearningModelSpec:
    """Declarative metadata for a learned artifact correction model."""

    name: str
    architecture: DeepLearningArchitecture
    runtime: DeepLearningRuntime = DeepLearningRuntime.CUSTOM
    domain: DeepLearningDomain = DeepLearningDomain.TIME_SERIES
    output_type: DeepLearningOutputType = DeepLearningOutputType.ARTIFACT
    latency_profile: DeepLearningLatencyProfile = DeepLearningLatencyProfile.OFFLINE
    execution_granularity: DeepLearningExecutionGranularity = DeepLearningExecutionGranularity.MULTICHANNEL
    supports_multichannel: bool = True
    uses_triggers: bool = False
    requires_artifact_length: bool = False
    requires_estimated_noise: bool = False
    requires_channel_positions: bool = False
    min_sfreq: float | None = None
    recommended_sfreq: float | None = None
    checkpoint_path: str | None = None
    checkpoint_format: str | None = None
    device_preference: str = "auto"
    channel_group_size: int | None = None
    channel_grouping_strategy: DeepLearningChannelGroupingStrategy = DeepLearningChannelGroupingStrategy.INDEX
    supports_chunking: bool = False
    chunk_size_samples: int | None = None
    chunk_overlap_samples: int = 0
    dual_output_policy: DeepLearningDualOutputPolicy = DeepLearningDualOutputPolicy.STRICT
    dual_output_rtol: float = 1e-5
    dual_output_atol: float = 1e-12
    description: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.checkpoint_format is not None and self.checkpoint_path is None:
            raise ValueError("checkpoint_format requires checkpoint_path to be set")
        if self.checkpoint_path is not None and not str(self.checkpoint_path).strip():
            raise ValueError("checkpoint_path must be a non-empty string when provided")
        if self.checkpoint_format is not None and not str(self.checkpoint_format).strip():
            raise ValueError("checkpoint_format must be a non-empty string when provided")
        if not str(self.device_preference).strip():
            raise ValueError("device_preference must be a non-empty string")
        if self.chunk_size_samples is not None and self.chunk_size_samples <= 0:
            raise ValueError("chunk_size_samples must be > 0 when provided")
        if self.chunk_overlap_samples < 0:
            raise ValueError("chunk_overlap_samples must be >= 0")
        if self.chunk_size_samples is None and self.chunk_overlap_samples != 0:
            raise ValueError("chunk_overlap_samples requires chunk_size_samples to be set")
        if not self.supports_chunking and self.chunk_size_samples is not None:
            raise ValueError("chunk_size_samples can only be set when supports_chunking=True")
        if self.chunk_size_samples is not None and self.chunk_overlap_samples >= self.chunk_size_samples:
            raise ValueError("chunk_overlap_samples must be smaller than chunk_size_samples")
        if self.dual_output_rtol < 0 or self.dual_output_atol < 0:
            raise ValueError("dual_output_rtol and dual_output_atol must be >= 0")
        if self.execution_granularity == DeepLearningExecutionGranularity.CHANNEL_GROUP and not self.supports_multichannel:
            raise ValueError("channel_group execution requires supports_multichannel=True")
        if self.execution_granularity == DeepLearningExecutionGranularity.MULTICHANNEL and not self.supports_multichannel:
            raise ValueError("multichannel execution requires supports_multichannel=True")
        if (
            self.execution_granularity == DeepLearningExecutionGranularity.CHANNEL_GROUP
            and (self.channel_group_size is None or self.channel_group_size < 2)
        ):
            raise ValueError("channel_group execution requires channel_group_size >= 2")


@dataclass
class DeepLearningPrediction:
    """Output container returned by a deep-learning adapter."""

    clean_data: np.ndarray | None = None
    artifact_data: np.ndarray | None = None
    start_sample: int = 0
    stop_sample: int | None = None
    channel_indices: np.ndarray | list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.clean_data is None and self.artifact_data is None:
            raise ValueError("Provide at least one of clean_data or artifact_data")


class DeepLearningModelAdapter(ABC):
    """Base class for learned artifact removal adapters."""

    spec: DeepLearningModelSpec

    def __init__(self) -> None:
        spec = getattr(self, "spec", None)
        if not isinstance(spec, DeepLearningModelSpec):
            raise TypeError("DeepLearningModelAdapter subclasses must define a DeepLearningModelSpec as 'spec'")

    def validate_context(self, context: ProcessingContext) -> None:
        """Validate runtime availability and data prerequisites."""
        self._ensure_runtime_available()
        self._validate_checkpoint()

        if self.spec.uses_triggers and not context.has_triggers():
            raise ProcessorValidationError(f"Model '{self.spec.name}' requires trigger metadata")
        if self.spec.requires_artifact_length and context.get_artifact_length() is None:
            raise ProcessorValidationError(f"Model '{self.spec.name}' requires artifact_length metadata")
        if self.spec.requires_estimated_noise and not context.has_estimated_noise():
            raise ProcessorValidationError(f"Model '{self.spec.name}' requires an estimated noise reference")
        if self.spec.min_sfreq is not None and context.get_sfreq() < self.spec.min_sfreq:
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' requires sfreq >= {self.spec.min_sfreq:g} Hz, "
                f"got {context.get_sfreq():g} Hz"
            )
        if self.spec.requires_channel_positions and not self._has_channel_positions(context):
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' requires electrode positions in raw.info['chs'][i]['loc']"
            )

    @staticmethod
    def _runtime_module_name(runtime: DeepLearningRuntime) -> str | None:
        mapping = {
            DeepLearningRuntime.TENSORFLOW: "tensorflow",
            DeepLearningRuntime.PYTORCH: "torch",
            DeepLearningRuntime.ONNX: "onnxruntime",
            DeepLearningRuntime.NUMPY: None,
            DeepLearningRuntime.CUSTOM: None,
        }
        return mapping[runtime]

    def _ensure_runtime_available(self) -> None:
        module_name = self._runtime_module_name(self.spec.runtime)
        if module_name is None:
            return
        if importlib.util.find_spec(module_name) is None:
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' requires the optional runtime '{module_name}'. "
                "Install the corresponding dependency before running inference."
            )

    @staticmethod
    def _infer_checkpoint_format(checkpoint_path: Path) -> str | None:
        if checkpoint_path.is_dir():
            return "saved_model"

        suffix_map = {
            ".ckpt": "ckpt",
            ".h5": "h5",
            ".hdf5": "h5",
            ".keras": "keras",
            ".onnx": "onnx",
            ".pb": "pb",
            ".pt": "pt",
            ".pth": "pth",
            ".ts": "torchscript",
            ".torchscript": "torchscript",
            ".npy": "npy",
            ".npz": "npz",
        }
        return suffix_map.get(checkpoint_path.suffix.lower())

    @staticmethod
    def _allowed_checkpoint_formats(runtime: DeepLearningRuntime) -> set[str] | None:
        mapping = {
            DeepLearningRuntime.TENSORFLOW: {"saved_model", "keras", "h5", "ckpt", "pb"},
            DeepLearningRuntime.PYTORCH: {"pt", "pth", "ckpt", "torchscript"},
            DeepLearningRuntime.ONNX: {"onnx"},
            DeepLearningRuntime.NUMPY: {"npy", "npz"},
            DeepLearningRuntime.CUSTOM: None,
        }
        return mapping[runtime]

    def _validate_checkpoint(self) -> None:
        checkpoint_path = self.spec.checkpoint_path
        if checkpoint_path is None:
            return

        path = Path(checkpoint_path).expanduser()
        if not path.exists():
            raise ProcessorValidationError(f"Model '{self.spec.name}' checkpoint does not exist: {path}")

        resolved_format = self.spec.checkpoint_format or self._infer_checkpoint_format(path)
        allowed_formats = self._allowed_checkpoint_formats(self.spec.runtime)

        if allowed_formats is None:
            return
        if resolved_format is None:
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' checkpoint format could not be inferred from {path}. "
                "Set checkpoint_format explicitly."
            )
        if resolved_format not in allowed_formats:
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' checkpoint format '{resolved_format}' is incompatible with runtime "
                f"'{self.spec.runtime.value}'. Allowed: {sorted(allowed_formats)}"
            )

    @staticmethod
    def _has_channel_positions(context: ProcessingContext) -> bool:
        eeg_picks = mne.pick_types(context.get_raw().info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        if len(eeg_picks) == 0:
            return False

        for ch_idx in eeg_picks:
            loc = np.asarray(context.get_raw().info["chs"][ch_idx].get("loc", []), dtype=float)
            if loc.size < 3:
                return False
            xyz = loc[:3]
            if not np.all(np.isfinite(xyz)) or np.allclose(xyz, 0.0):
                return False
        return True

    @abstractmethod
    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        """Run model inference and return an artifact or clean-signal estimate."""


class TensorFlowInferenceAdapter(DeepLearningModelAdapter):
    """Generic TensorFlow inference adapter for checkpoint-backed artifact models."""

    spec = DeepLearningModelSpec(
        name="TensorFlowInferenceAdapter",
        architecture=DeepLearningArchitecture.CUSTOM,
        runtime=DeepLearningRuntime.TENSORFLOW,
        output_type=DeepLearningOutputType.ARTIFACT,
    )

    def __init__(
        self,
        *,
        checkpoint_path: str,
        spec_overrides: dict[str, Any] | None = None,
        input_layout: TensorFlowTensorLayout = TensorFlowTensorLayout.BATCH_CHANNELS_TIME,
        output_layout: TensorFlowTensorLayout | None = None,
        output_key: str | None = None,
        artifact_output_key: str = "artifact",
        clean_output_key: str = "clean",
        serving_signature_name: str | None = None,
        input_key: str | None = None,
        compile_model: bool = False,
    ) -> None:
        resolved_spec = replace(
            type(self).spec,
            checkpoint_path=checkpoint_path,
            **(spec_overrides or {}),
        )
        if resolved_spec.runtime != DeepLearningRuntime.TENSORFLOW:
            raise ValueError("TensorFlowInferenceAdapter requires a spec with runtime='tensorflow'")
        self.spec = resolved_spec
        self.input_layout = TensorFlowTensorLayout(input_layout)
        self.output_layout = TensorFlowTensorLayout(output_layout or input_layout)
        self.output_key = output_key
        self.artifact_output_key = artifact_output_key
        self.clean_output_key = clean_output_key
        self.serving_signature_name = serving_signature_name
        self.input_key = input_key
        self.compile_model = compile_model
        self._tf: Any | None = None
        self._model: Any | None = None
        super().__init__()

    def _load_tensorflow(self) -> Any:
        if self._tf is None:
            self._tf = importlib.import_module("tensorflow")
        return self._tf

    def _resolve_device(self, tf: Any) -> str | None:
        preference = self.spec.device_preference.strip().lower()
        if preference in {"", "auto"}:
            list_devices = getattr(getattr(tf, "config", None), "list_logical_devices", None)
            if callable(list_devices) and list_devices("GPU"):
                return "/GPU:0"
            return "/CPU:0"
        if preference in {"cpu", "cpu:0", "/cpu:0"}:
            return "/CPU:0"
        if preference in {"gpu", "gpu:0", "/gpu:0", "cuda", "cuda:0"}:
            list_devices = getattr(getattr(tf, "config", None), "list_logical_devices", None)
            if callable(list_devices) and not list_devices("GPU"):
                raise ProcessorValidationError(
                    f"Model '{self.spec.name}' requested device '{self.spec.device_preference}' but no GPU is available"
                )
            return "/GPU:0"
        return self.spec.device_preference

    def _resolve_checkpoint_format(self) -> str:
        checkpoint_path = self.spec.checkpoint_path
        if checkpoint_path is None:
            raise ProcessorValidationError(f"Model '{self.spec.name}' requires checkpoint_path")
        resolved_format = self.spec.checkpoint_format or self._infer_checkpoint_format(Path(checkpoint_path).expanduser())
        if resolved_format is None:
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' checkpoint format could not be inferred from {checkpoint_path}. "
                "Set checkpoint_format explicitly."
            )
        return resolved_format

    def _load_model(self) -> tuple[Any, str]:
        if self._model is not None:
            return self._model, self._resolve_checkpoint_format()

        tf = self._load_tensorflow()
        checkpoint_path = Path(self.spec.checkpoint_path).expanduser()
        checkpoint_format = self._resolve_checkpoint_format()

        if checkpoint_format in {"keras", "h5"}:
            self._model = tf.keras.models.load_model(str(checkpoint_path), compile=self.compile_model)
        elif checkpoint_format in {"saved_model", "pb"}:
            self._model = tf.saved_model.load(str(checkpoint_path))
        else:
            raise ProcessorValidationError(
                f"TensorFlowInferenceAdapter does not yet support checkpoint_format='{checkpoint_format}'. "
                "Use 'keras', 'h5', or SavedModel directories."
            )
        return self._model, checkpoint_format

    def _prepare_input(self, data: np.ndarray) -> np.ndarray:
        if self.input_layout == TensorFlowTensorLayout.BATCH_CHANNELS_TIME:
            return data[np.newaxis, ...].astype(np.float32, copy=False)
        if self.input_layout == TensorFlowTensorLayout.BATCH_TIME_CHANNELS:
            return np.moveaxis(data, 0, -1)[np.newaxis, ...].astype(np.float32, copy=False)
        raise ProcessorValidationError(f"Unsupported TensorFlow input layout: {self.input_layout.value}")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float)

    def _restore_output_layout(self, value: Any) -> np.ndarray:
        arr = self._to_numpy(value)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ProcessorValidationError(
                f"TensorFlow adapter expects 2D model outputs after removing batch dim, got shape {tuple(arr.shape)}"
            )
        if self.output_layout == TensorFlowTensorLayout.BATCH_CHANNELS_TIME:
            return arr
        if self.output_layout == TensorFlowTensorLayout.BATCH_TIME_CHANNELS:
            return np.moveaxis(arr, -1, 0)
        raise ProcessorValidationError(f"Unsupported TensorFlow output layout: {self.output_layout.value}")

    def _select_saved_model_signature(self, model: Any) -> tuple[str, Any]:
        signatures = getattr(model, "signatures", None)
        if not signatures:
            raise ProcessorValidationError(
                f"TensorFlow SavedModel for '{self.spec.name}' exposes no callable signatures"
            )

        if self.serving_signature_name is not None:
            if self.serving_signature_name not in signatures:
                raise ProcessorValidationError(
                    f"TensorFlow SavedModel for '{self.spec.name}' does not expose signature "
                    f"'{self.serving_signature_name}'. Available: {sorted(signatures)}"
                )
            return self.serving_signature_name, signatures[self.serving_signature_name]

        if "serving_default" in signatures:
            return "serving_default", signatures["serving_default"]

        if len(signatures) == 1:
            signature_name = next(iter(signatures))
            return signature_name, signatures[signature_name]

        raise ProcessorValidationError(
            f"TensorFlow SavedModel for '{self.spec.name}' exposes multiple signatures {sorted(signatures)}. "
            "Set serving_signature_name explicitly."
        )

    def _call_saved_model_signature(self, model: Any, input_tensor: Any) -> tuple[Any, str]:
        signature_name, signature = self._select_saved_model_signature(model)
        try:
            return signature(input_tensor), signature_name
        except TypeError:
            structured_input_signature = getattr(signature, "structured_input_signature", None)
            if structured_input_signature is None or len(structured_input_signature) != 2:
                raise ProcessorValidationError(
                    f"TensorFlow SavedModel signature '{signature_name}' for '{self.spec.name}' requires "
                    "structured_input_signature metadata for keyword invocation"
                ) from None

            _, keyword_spec = structured_input_signature
            if self.input_key is not None:
                if self.input_key not in keyword_spec:
                    raise ProcessorValidationError(
                        f"TensorFlow SavedModel signature '{signature_name}' for '{self.spec.name}' does not expose "
                        f"input key '{self.input_key}'. Available: {sorted(keyword_spec)}"
                    ) from None
                return signature(**{self.input_key: input_tensor}), signature_name

            if len(keyword_spec) == 1:
                input_name = next(iter(keyword_spec))
                return signature(**{input_name: input_tensor}), signature_name

            raise ProcessorValidationError(
                f"TensorFlow SavedModel signature '{signature_name}' for '{self.spec.name}' requires an explicit "
                "input_key because it exposes multiple keyword inputs"
            ) from None

    def _run_model(self, input_array: np.ndarray) -> tuple[Any, str, str | None]:
        tf = self._load_tensorflow()
        model, checkpoint_format = self._load_model()
        input_tensor = tf.convert_to_tensor(input_array)
        device_name = self._resolve_device(tf)
        signature_name: str | None = None

        with tf.device(device_name):
            if callable(model):
                try:
                    outputs = model(input_tensor, training=False)
                except TypeError:
                    outputs = model(input_tensor)
            else:
                outputs, signature_name = self._call_saved_model_signature(model, input_tensor)
        return outputs, device_name or "default", signature_name

    def _extract_single_output(self, outputs: Any) -> np.ndarray:
        if isinstance(outputs, dict):
            if self.output_key is not None:
                if self.output_key not in outputs:
                    raise ProcessorValidationError(
                        f"TensorFlow model '{self.spec.name}' did not return output key '{self.output_key}'"
                    )
                return self._restore_output_layout(outputs[self.output_key])
            if len(outputs) != 1:
                raise ProcessorValidationError(
                    f"TensorFlow model '{self.spec.name}' returned multiple outputs; set output_key explicitly"
                )
            return self._restore_output_layout(next(iter(outputs.values())))
        if isinstance(outputs, (list, tuple)):
            if len(outputs) != 1:
                raise ProcessorValidationError(
                    f"TensorFlow model '{self.spec.name}' returned {len(outputs)} outputs; expected exactly one"
                )
            return self._restore_output_layout(outputs[0])
        return self._restore_output_layout(outputs)

    def _extract_dual_outputs(self, outputs: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(outputs, dict):
            if self.artifact_output_key not in outputs or self.clean_output_key not in outputs:
                raise ProcessorValidationError(
                    f"TensorFlow model '{self.spec.name}' must return '{self.artifact_output_key}' and "
                    f"'{self.clean_output_key}' for output_type='both'"
                )
            artifact_data = self._restore_output_layout(outputs[self.artifact_output_key])
            clean_data = self._restore_output_layout(outputs[self.clean_output_key])
            return artifact_data, clean_data
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            artifact_data = self._restore_output_layout(outputs[0])
            clean_data = self._restore_output_layout(outputs[1])
            return artifact_data, clean_data
        raise ProcessorValidationError(
            f"TensorFlow model '{self.spec.name}' must return a dict or 2-tuple for output_type='both'"
        )

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        data = context.get_data(copy=False)
        input_array = self._prepare_input(data)
        outputs, device_name, signature_name = self._run_model(input_array)
        checkpoint_format = self._resolve_checkpoint_format()

        metadata = {
            "runtime_backend": "tensorflow",
            "device": device_name,
            "checkpoint_format": checkpoint_format,
            "input_layout": self.input_layout.value,
            "output_layout": self.output_layout.value,
        }
        if signature_name is not None:
            metadata["serving_signature_name"] = signature_name

        if self.spec.output_type == DeepLearningOutputType.ARTIFACT:
            return DeepLearningPrediction(
                artifact_data=self._extract_single_output(outputs),
                metadata=metadata,
            )
        if self.spec.output_type == DeepLearningOutputType.CLEAN:
            return DeepLearningPrediction(
                clean_data=self._extract_single_output(outputs),
                metadata=metadata,
            )

        artifact_data, clean_data = self._extract_dual_outputs(outputs)
        return DeepLearningPrediction(
            artifact_data=artifact_data,
            clean_data=clean_data,
            metadata=metadata,
        )


class PyTorchInferenceAdapter(DeepLearningModelAdapter):
    """Generic PyTorch inference adapter for checkpoint-backed artifact models."""

    spec = DeepLearningModelSpec(
        name="PyTorchInferenceAdapter",
        architecture=DeepLearningArchitecture.CUSTOM,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
    )

    def __init__(
        self,
        *,
        checkpoint_path: str,
        spec_overrides: dict[str, Any] | None = None,
        input_layout: PyTorchTensorLayout = PyTorchTensorLayout.BATCH_CHANNELS_TIME,
        output_layout: PyTorchTensorLayout | None = None,
        output_key: str | None = None,
        artifact_output_key: str = "artifact",
        clean_output_key: str = "clean",
        model_factory: Any | None = None,
        strict_load: bool = True,
    ) -> None:
        resolved_spec = replace(
            type(self).spec,
            checkpoint_path=checkpoint_path,
            **(spec_overrides or {}),
        )
        if resolved_spec.runtime != DeepLearningRuntime.PYTORCH:
            raise ValueError("PyTorchInferenceAdapter requires a spec with runtime='pytorch'")
        self.spec = resolved_spec
        self.input_layout = PyTorchTensorLayout(input_layout)
        self.output_layout = PyTorchTensorLayout(output_layout or input_layout)
        self.output_key = output_key
        self.artifact_output_key = artifact_output_key
        self.clean_output_key = clean_output_key
        self.model_factory = model_factory
        self.strict_load = strict_load
        self._torch: Any | None = None
        self._model: Any | None = None
        self._loaded_checkpoint_format: str | None = None
        self._model_device: str | None = None
        self._checkpoint_load_mode: str | None = None
        super().__init__()

    def _load_torch(self) -> Any:
        if self._torch is None:
            self._torch = importlib.import_module("torch")
        return self._torch

    def _resolve_device(self, torch: Any) -> str:
        preference = self.spec.device_preference.strip().lower()
        cuda_module = getattr(torch, "cuda", None)
        cuda_available = bool(getattr(cuda_module, "is_available", lambda: False)())

        if preference in {"", "auto"}:
            return "cuda:0" if cuda_available else "cpu"
        if preference in {"cpu", "cpu:0"}:
            return "cpu"
        if preference in {"gpu", "gpu:0", "cuda", "cuda:0"}:
            if not cuda_available:
                raise ProcessorValidationError(
                    f"Model '{self.spec.name}' requested device '{self.spec.device_preference}' but no GPU is available"
                )
            return "cuda:0"
        return self.spec.device_preference

    def _resolve_checkpoint_format(self) -> str:
        checkpoint_path = self.spec.checkpoint_path
        if checkpoint_path is None:
            raise ProcessorValidationError(f"Model '{self.spec.name}' requires checkpoint_path")
        resolved_format = self.spec.checkpoint_format or self._infer_checkpoint_format(Path(checkpoint_path).expanduser())
        if resolved_format is None:
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' checkpoint format could not be inferred from {checkpoint_path}. "
                "Set checkpoint_format explicitly."
            )
        return resolved_format

    @staticmethod
    def _coerce_state_dict(checkpoint_obj: Any) -> dict[str, Any]:
        if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
            state_dict = checkpoint_obj["state_dict"]
            if isinstance(state_dict, dict):
                return state_dict
        if isinstance(checkpoint_obj, dict):
            return checkpoint_obj
        raise ProcessorValidationError("PyTorch checkpoint does not contain a compatible state_dict payload")

    def _load_model(self) -> tuple[Any, str, str, str]:
        if self._model is not None:
            return (
                self._model,
                self._loaded_checkpoint_format,
                self._model_device,
                self._checkpoint_load_mode,
            )

        torch = self._load_torch()
        checkpoint_path = Path(self.spec.checkpoint_path).expanduser()
        checkpoint_format = self._resolve_checkpoint_format()
        device_name = self._resolve_device(torch)
        checkpoint_load_mode = "module"

        if checkpoint_format == "torchscript":
            jit_module = getattr(torch, "jit", None)
            if jit_module is None or not hasattr(jit_module, "load"):
                raise ProcessorValidationError("PyTorch runtime does not expose torch.jit.load for TorchScript checkpoints")
            model = jit_module.load(str(checkpoint_path), map_location=device_name)
            checkpoint_load_mode = "torchscript"
        elif checkpoint_format in {"pt", "pth", "ckpt"}:
            checkpoint_obj = torch.load(str(checkpoint_path), map_location=device_name)
            if self.model_factory is not None:
                model = self.model_factory()
                if not hasattr(model, "load_state_dict"):
                    raise ProcessorValidationError("model_factory must return an object with load_state_dict()")
                state_dict = self._coerce_state_dict(checkpoint_obj)
                model.load_state_dict(state_dict, strict=self.strict_load)
                checkpoint_load_mode = "state_dict"
            elif callable(checkpoint_obj):
                model = checkpoint_obj
                checkpoint_load_mode = "module"
            else:
                raise ProcessorValidationError(
                    f"PyTorch checkpoint format '{checkpoint_format}' was loaded as weights only. "
                    "Provide model_factory to reconstruct the module before inference."
                )
        else:
            raise ProcessorValidationError(
                f"PyTorchInferenceAdapter does not yet support checkpoint_format='{checkpoint_format}'. "
                "Use 'pt', 'pth', 'ckpt', or 'torchscript'."
            )

        if hasattr(model, "to"):
            model = model.to(device_name)
        if hasattr(model, "eval"):
            model.eval()

        self._model = model
        self._loaded_checkpoint_format = checkpoint_format
        self._model_device = device_name
        self._checkpoint_load_mode = checkpoint_load_mode
        return model, checkpoint_format, device_name, checkpoint_load_mode

    def _prepare_input(self, data: np.ndarray) -> np.ndarray:
        if self.input_layout == PyTorchTensorLayout.BATCH_CHANNELS_TIME:
            return data[np.newaxis, ...].astype(np.float32, copy=False)
        if self.input_layout == PyTorchTensorLayout.BATCH_TIME_CHANNELS:
            return np.moveaxis(data, 0, -1)[np.newaxis, ...].astype(np.float32, copy=False)
        raise ProcessorValidationError(f"Unsupported PyTorch input layout: {self.input_layout.value}")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=float)

    def _restore_output_layout(self, value: Any) -> np.ndarray:
        arr = self._to_numpy(value)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ProcessorValidationError(
                f"PyTorch adapter expects 2D model outputs after removing batch dim, got shape {tuple(arr.shape)}"
            )
        if self.output_layout == PyTorchTensorLayout.BATCH_CHANNELS_TIME:
            return arr
        if self.output_layout == PyTorchTensorLayout.BATCH_TIME_CHANNELS:
            return np.moveaxis(arr, -1, 0)
        raise ProcessorValidationError(f"Unsupported PyTorch output layout: {self.output_layout.value}")

    def _run_model(self, input_array: np.ndarray) -> tuple[Any, str, str, str]:
        torch = self._load_torch()
        model, checkpoint_format, device_name, checkpoint_load_mode = self._load_model()
        input_tensor = torch.as_tensor(input_array, device=device_name)
        no_grad = getattr(torch, "no_grad", None)

        if no_grad is None:
            outputs = model(input_tensor)
        else:
            with no_grad():
                outputs = model(input_tensor)
        return outputs, checkpoint_format, device_name, checkpoint_load_mode

    def _extract_single_output(self, outputs: Any) -> np.ndarray:
        if isinstance(outputs, dict):
            if self.output_key is not None:
                if self.output_key not in outputs:
                    raise ProcessorValidationError(
                        f"PyTorch model '{self.spec.name}' did not return output key '{self.output_key}'"
                    )
                return self._restore_output_layout(outputs[self.output_key])
            if len(outputs) != 1:
                raise ProcessorValidationError(
                    f"PyTorch model '{self.spec.name}' returned multiple outputs; set output_key explicitly"
                )
            return self._restore_output_layout(next(iter(outputs.values())))
        if isinstance(outputs, (list, tuple)):
            if len(outputs) != 1:
                raise ProcessorValidationError(
                    f"PyTorch model '{self.spec.name}' returned {len(outputs)} outputs; expected exactly one"
                )
            return self._restore_output_layout(outputs[0])
        return self._restore_output_layout(outputs)

    def _extract_dual_outputs(self, outputs: Any) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(outputs, dict):
            if self.artifact_output_key not in outputs or self.clean_output_key not in outputs:
                raise ProcessorValidationError(
                    f"PyTorch model '{self.spec.name}' must return '{self.artifact_output_key}' and "
                    f"'{self.clean_output_key}' for output_type='both'"
                )
            artifact_data = self._restore_output_layout(outputs[self.artifact_output_key])
            clean_data = self._restore_output_layout(outputs[self.clean_output_key])
            return artifact_data, clean_data
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            artifact_data = self._restore_output_layout(outputs[0])
            clean_data = self._restore_output_layout(outputs[1])
            return artifact_data, clean_data
        raise ProcessorValidationError(
            f"PyTorch model '{self.spec.name}' must return a dict or 2-tuple for output_type='both'"
        )

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        data = context.get_data(copy=False)
        input_array = self._prepare_input(data)
        outputs, checkpoint_format, device_name, checkpoint_load_mode = self._run_model(input_array)

        metadata = {
            "runtime_backend": "pytorch",
            "device": device_name,
            "checkpoint_format": checkpoint_format,
            "checkpoint_load_mode": checkpoint_load_mode,
            "input_layout": self.input_layout.value,
            "output_layout": self.output_layout.value,
        }

        if self.spec.output_type == DeepLearningOutputType.ARTIFACT:
            return DeepLearningPrediction(
                artifact_data=self._extract_single_output(outputs),
                metadata=metadata,
            )
        if self.spec.output_type == DeepLearningOutputType.CLEAN:
            return DeepLearningPrediction(
                clean_data=self._extract_single_output(outputs),
                metadata=metadata,
            )

        artifact_data, clean_data = self._extract_dual_outputs(outputs)
        return DeepLearningPrediction(
            artifact_data=artifact_data,
            clean_data=clean_data,
            metadata=metadata,
        )


class OnnxInferenceAdapter(DeepLearningModelAdapter):
    """Generic ONNX Runtime inference adapter for checkpoint-backed artifact models.

    Loads a ``.onnx`` model file via ``onnxruntime.InferenceSession`` and maps
    NumPy arrays to the session's named inputs/outputs.  No ML framework
    (TensorFlow, PyTorch) is required at inference time — only the lightweight
    ``onnxruntime`` package.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.onnx`` model file.
    spec_overrides:
        Optional keyword overrides merged into the default :class:`DeepLearningModelSpec`.
    input_layout:
        Memory layout expected by the model's first input node.
    output_layout:
        Memory layout produced by the model's output node(s).
        Defaults to *input_layout*.
    input_name:
        Name of the ONNX input node.  Auto-detected from the session when
        ``None`` (uses the first input).
    output_key:
        Name of the ONNX output node to use when the model returns a single
        prediction.  Auto-detected when ``None`` (requires exactly one output).
    artifact_output_key:
        Name of the artifact output node for ``output_type='both'`` models.
    clean_output_key:
        Name of the clean-signal output node for ``output_type='both'`` models.
    """

    spec = DeepLearningModelSpec(
        name="OnnxInferenceAdapter",
        architecture=DeepLearningArchitecture.CUSTOM,
        runtime=DeepLearningRuntime.ONNX,
        output_type=DeepLearningOutputType.ARTIFACT,
    )

    def __init__(
        self,
        *,
        checkpoint_path: str,
        spec_overrides: dict[str, Any] | None = None,
        input_layout: OnnxTensorLayout = OnnxTensorLayout.BATCH_CHANNELS_TIME,
        output_layout: OnnxTensorLayout | None = None,
        input_name: str | None = None,
        output_key: str | None = None,
        artifact_output_key: str = "artifact",
        clean_output_key: str = "clean",
    ) -> None:
        resolved_spec = replace(
            type(self).spec,
            checkpoint_path=checkpoint_path,
            **(spec_overrides or {}),
        )
        if resolved_spec.runtime != DeepLearningRuntime.ONNX:
            raise ValueError("OnnxInferenceAdapter requires a spec with runtime='onnx'")
        self.spec = resolved_spec
        self.input_layout = OnnxTensorLayout(input_layout)
        self.output_layout = OnnxTensorLayout(output_layout or input_layout)
        self.input_name = input_name
        self.output_key = output_key
        self.artifact_output_key = artifact_output_key
        self.clean_output_key = clean_output_key
        self._ort: Any | None = None
        self._session: Any | None = None
        super().__init__()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ort(self) -> Any:
        if self._ort is None:
            self._ort = importlib.import_module("onnxruntime")
        return self._ort

    def _resolve_providers(self) -> list[str]:
        """Map *device_preference* to an ordered ONNX execution-provider list."""
        preference = self.spec.device_preference.strip().lower()
        ort = self._load_ort()
        available: list[str] = list(getattr(ort, "get_available_providers", lambda: [])())
        has_cuda = "CUDAExecutionProvider" in available

        if preference in {"", "auto"}:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cuda else ["CPUExecutionProvider"]
        if preference in {"cpu", "cpu:0"}:
            return ["CPUExecutionProvider"]
        if preference in {"gpu", "gpu:0", "cuda", "cuda:0"}:
            if not has_cuda:
                raise ProcessorValidationError(
                    f"Model '{self.spec.name}' requested device '{self.spec.device_preference}' but "
                    "CUDAExecutionProvider is not available in this onnxruntime installation"
                )
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        # Pass through any custom provider string as-is.
        return [self.spec.device_preference]

    def _load_session(self) -> Any:
        """Lazily create and cache the ``onnxruntime.InferenceSession``."""
        if self._session is not None:
            return self._session

        ort = self._load_ort()
        checkpoint_path = Path(self.spec.checkpoint_path).expanduser()
        providers = self._resolve_providers()

        session_options_cls = getattr(ort, "SessionOptions", None)
        opts = session_options_cls() if session_options_cls is not None else None
        session_cls = getattr(ort, "InferenceSession")

        if opts is not None:
            self._session = session_cls(str(checkpoint_path), sess_options=opts, providers=providers)
        else:
            self._session = session_cls(str(checkpoint_path), providers=providers)

        return self._session

    def _resolve_input_name(self, session: Any) -> str:
        """Return the input-node name to feed, auto-detecting when unset."""
        if self.input_name is not None:
            return self.input_name
        inputs = session.get_inputs()
        if not inputs:
            raise ProcessorValidationError(f"Model '{self.spec.name}' ONNX session has no inputs")
        return inputs[0].name

    def _prepare_input(self, data: np.ndarray) -> np.ndarray:
        """Reshape raw EEG data to the layout expected by the model."""
        if self.input_layout == OnnxTensorLayout.BATCH_CHANNELS_TIME:
            return data[np.newaxis, ...].astype(np.float32, copy=False)
        if self.input_layout == OnnxTensorLayout.BATCH_TIME_CHANNELS:
            return np.moveaxis(data, 0, -1)[np.newaxis, ...].astype(np.float32, copy=False)
        raise ProcessorValidationError(f"Unsupported ONNX input layout: {self.input_layout.value}")

    def _restore_output_layout(self, value: np.ndarray) -> np.ndarray:
        """Strip the batch dimension and transpose back to (channels, samples)."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ProcessorValidationError(
                f"ONNX adapter expects 2D model outputs after removing batch dim, got shape {tuple(arr.shape)}"
            )
        if self.output_layout == OnnxTensorLayout.BATCH_CHANNELS_TIME:
            return arr
        if self.output_layout == OnnxTensorLayout.BATCH_TIME_CHANNELS:
            return np.moveaxis(arr, -1, 0)
        raise ProcessorValidationError(f"Unsupported ONNX output layout: {self.output_layout.value}")

    def _run_session(self, session: Any, input_array: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
        """Execute the ONNX session and return (outputs, output_names)."""
        input_name = self._resolve_input_name(session)
        output_names: list[str] = [o.name for o in session.get_outputs()]
        results: list[np.ndarray] = session.run(output_names, {input_name: input_array})
        return results, output_names

    def _extract_single_output(self, outputs: list[np.ndarray], output_names: list[str]) -> np.ndarray:
        if self.output_key is not None:
            if self.output_key not in output_names:
                raise ProcessorValidationError(
                    f"ONNX model '{self.spec.name}' did not return output key '{self.output_key}'. "
                    f"Available: {output_names}"
                )
            return self._restore_output_layout(outputs[output_names.index(self.output_key)])
        if len(outputs) != 1:
            raise ProcessorValidationError(
                f"ONNX model '{self.spec.name}' returned {len(outputs)} outputs; "
                "set output_key explicitly"
            )
        return self._restore_output_layout(outputs[0])

    def _extract_dual_outputs(
        self,
        outputs: list[np.ndarray],
        output_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        missing = [
            k for k in (self.artifact_output_key, self.clean_output_key) if k not in output_names
        ]
        if missing:
            raise ProcessorValidationError(
                f"ONNX model '{self.spec.name}' is missing output(s) {missing} for output_type='both'. "
                f"Available: {output_names}"
            )
        artifact_data = self._restore_output_layout(
            outputs[output_names.index(self.artifact_output_key)]
        )
        clean_data = self._restore_output_layout(
            outputs[output_names.index(self.clean_output_key)]
        )
        return artifact_data, clean_data

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        data = context.get_data(copy=False)
        input_array = self._prepare_input(data)

        session = self._load_session()
        outputs, output_names = self._run_session(session, input_array)

        metadata: dict[str, Any] = {
            "runtime_backend": "onnxruntime",
            "providers": list(session.get_providers()),
            "input_layout": self.input_layout.value,
            "output_layout": self.output_layout.value,
        }

        if self.spec.output_type == DeepLearningOutputType.ARTIFACT:
            return DeepLearningPrediction(
                artifact_data=self._extract_single_output(outputs, output_names),
                metadata=metadata,
            )
        if self.spec.output_type == DeepLearningOutputType.CLEAN:
            return DeepLearningPrediction(
                clean_data=self._extract_single_output(outputs, output_names),
                metadata=metadata,
            )

        artifact_data, clean_data = self._extract_dual_outputs(outputs, output_names)
        return DeepLearningPrediction(
            artifact_data=artifact_data,
            clean_data=clean_data,
            metadata=metadata,
        )


class DeepLearningModelRegistry:
    """Registry for deep-learning model adapters."""

    _instance: DeepLearningModelRegistry | None = None
    _registry: dict[str, type[DeepLearningModelAdapter]] = {}

    def __init__(self) -> None:
        if DeepLearningModelRegistry._instance is not None:
            raise RuntimeError("Use DeepLearningModelRegistry.get_instance() instead")

    @classmethod
    def get_instance(cls) -> DeepLearningModelRegistry:
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._registry = {}
        return cls._instance

    def register(self, name: str, model_class: type[DeepLearningModelAdapter], force: bool = False) -> None:
        if name in self._registry and not force:
            raise ValueError(f"Deep-learning model '{name}' is already registered. Use force=True to override.")
        if not issubclass(model_class, DeepLearningModelAdapter):
            raise TypeError(f"Model class must inherit from DeepLearningModelAdapter, got {model_class}")
        self._registry[name] = model_class
        logger.debug("Registered deep-learning model: {} -> {}", name, model_class.__name__)

    def unregister(self, name: str) -> None:
        if name not in self._registry:
            raise KeyError(f"Deep-learning model '{name}' is not registered")
        del self._registry[name]

    def get(self, name: str) -> type[DeepLearningModelAdapter]:
        if name not in self._registry:
            raise KeyError(f"Deep-learning model '{name}' is not registered. Available: {self.list_names()}")
        return self._registry[name]

    def list_names(self) -> list[str]:
        return list(self._registry.keys())

    def list_all(self) -> dict[str, type[DeepLearningModelAdapter]]:
        return self._registry.copy()


def register_deep_learning_model(
    model_class: type[DeepLearningModelAdapter] | None = None,
    *,
    name: str | None = None,
    force: bool = False,
):
    """Decorator to register a deep-learning model adapter."""

    def decorator(cls: type[DeepLearningModelAdapter]) -> type[DeepLearningModelAdapter]:
        model_name = name if name is not None else cls.spec.name
        registry = DeepLearningModelRegistry.get_instance()
        registry.register(model_name, cls, force=force)
        return cls

    if model_class is not None:
        return decorator(model_class)
    return decorator


def get_deep_learning_model(name: str) -> type[DeepLearningModelAdapter]:
    """Resolve a registered deep-learning model adapter by name."""
    return DeepLearningModelRegistry.get_instance().get(name)


def list_deep_learning_models() -> dict[str, type[DeepLearningModelAdapter]]:
    """Return registered deep-learning model adapters."""
    return DeepLearningModelRegistry.get_instance().list_all()


class NumpyInferenceAdapter(DeepLearningModelAdapter):
    """Inference adapter for pure-NumPy models stored as ``.npy`` or ``.npz`` files.

    This adapter is useful for lightweight models (e.g. Wiener filters, learned
    projection matrices, simple regression weights) that do not require a
    deep-learning framework at inference time.

    The adapter loads the checkpoint once (lazy) and passes the resulting
    array(s) together with the EEG data to a user-supplied *predict_fn*.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.npy`` (single array) or ``.npz`` (named arrays) file.
    predict_fn:
        Callable with signature ``(data, weights) -> np.ndarray`` where

        * *data* is the raw EEG array ``(n_channels, n_samples)`` as
          ``float64``.
        * *weights* is either the loaded ``np.ndarray`` (for ``.npy``) or the
          ``np.lib.npyio.NpzFile`` mapping (for ``.npz``).

        The function must return a 2-D ``float64`` array of shape
        ``(n_channels, n_samples)`` matching the declared *output_type*
        (artifact estimate **or** clean signal).
    spec_overrides:
        Optional keyword overrides merged into the default
        :class:`DeepLearningModelSpec`.
    """

    spec = DeepLearningModelSpec(
        name="NumpyInferenceAdapter",
        architecture=DeepLearningArchitecture.CUSTOM,
        runtime=DeepLearningRuntime.NUMPY,
        output_type=DeepLearningOutputType.ARTIFACT,
    )

    def __init__(
        self,
        *,
        checkpoint_path: str,
        predict_fn: Any,
        spec_overrides: dict[str, Any] | None = None,
    ) -> None:
        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable")
        resolved_spec = replace(
            type(self).spec,
            checkpoint_path=checkpoint_path,
            **(spec_overrides or {}),
        )
        if resolved_spec.runtime != DeepLearningRuntime.NUMPY:
            raise ValueError("NumpyInferenceAdapter requires a spec with runtime='numpy'")
        self.spec = resolved_spec
        self.predict_fn = predict_fn
        self._weights: Any | None = None
        super().__init__()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_weights(self) -> Any:
        """Lazily load and cache the checkpoint array(s)."""
        if self._weights is not None:
            return self._weights

        checkpoint_path = Path(self.spec.checkpoint_path).expanduser()
        suffix = checkpoint_path.suffix.lower()

        if suffix == ".npy":
            self._weights = np.load(str(checkpoint_path), allow_pickle=False)
        elif suffix in {".npz"}:
            self._weights = np.load(str(checkpoint_path), allow_pickle=False)
        else:
            # Fallback: try npy first, then npz
            try:
                self._weights = np.load(str(checkpoint_path), allow_pickle=False)
            except Exception as exc:
                raise ProcessorValidationError(
                    f"Model '{self.spec.name}' could not load checkpoint '{checkpoint_path}': {exc}"
                ) from exc

        return self._weights

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        data = context.get_data(copy=False).astype(np.float64, copy=False)
        weights = self._load_weights()

        try:
            result = self.predict_fn(data, weights)
        except Exception as exc:
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' predict_fn raised an error: {exc}"
            ) from exc

        result = np.asarray(result, dtype=np.float64)
        if result.ndim != 2 or result.shape != data.shape:
            raise ProcessorValidationError(
                f"Model '{self.spec.name}' predict_fn must return shape {data.shape}, "
                f"got {tuple(result.shape)}"
            )

        metadata: dict[str, Any] = {
            "runtime_backend": "numpy",
            "checkpoint_shape": tuple(weights.shape) if hasattr(weights, "shape") else None,
            "checkpoint_keys": list(weights.keys()) if hasattr(weights, "keys") else None,
        }

        if self.spec.output_type == DeepLearningOutputType.ARTIFACT:
            return DeepLearningPrediction(artifact_data=result, metadata=metadata)
        if self.spec.output_type == DeepLearningOutputType.CLEAN:
            return DeepLearningPrediction(clean_data=result, metadata=metadata)

        raise ProcessorValidationError(
            f"NumpyInferenceAdapter does not support output_type='both'. "
            "Return either artifact or clean signal from predict_fn and set output_type accordingly."
        )


register_deep_learning_model(TensorFlowInferenceAdapter, name="tensorflow_inference", force=True)
register_deep_learning_model(PyTorchInferenceAdapter, name="pytorch_inference", force=True)
register_deep_learning_model(OnnxInferenceAdapter, name="onnx_inference", force=True)
register_deep_learning_model(NumpyInferenceAdapter, name="numpy_inference", force=True)


_RESEARCH_BLUEPRINTS: dict[str, DeepLearningModelSpec] = {
    "dar": DeepLearningModelSpec(
        name="DAR",
        architecture=DeepLearningArchitecture.AUTOENCODER,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        device_preference="cpu",
        supports_chunking=True,
        chunk_size_samples=2048,
        chunk_overlap_samples=128,
        description="1D-CNN denoising autoencoder for direct clean-signal reconstruction.",
        tags=("cnn", "denoising", "supervised"),
    ),
    "dpae": DeepLearningModelSpec(
        name="DPAE",
        architecture=DeepLearningArchitecture.AUTOENCODER,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        description="Dual-pathway autoencoder with local/global temporal branches.",
        tags=("cnn", "multi_scale", "bcg_aware"),
    ),
    "ic_u_net": DeepLearningModelSpec(
        name="IC-U-Net",
        architecture=DeepLearningArchitecture.UNET,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.NEAR_REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        supports_chunking=True,
        chunk_size_samples=4096,
        chunk_overlap_samples=256,
        description="U-Net with ICA-guided component mixtures.",
        tags=("u_net", "ica", "hybrid"),
    ),
    "dhct_gan": DeepLearningModelSpec(
        name="DHCT-GAN",
        architecture=DeepLearningArchitecture.GAN,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.NEAR_REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=4096,
        chunk_overlap_samples=256,
        description="Dual-branch hybrid CNN/Transformer GAN.",
        tags=("gan", "transformer", "spectral_fidelity"),
    ),
    "nested_gan": DeepLearningModelSpec(
        name="Nested-GAN",
        architecture=DeepLearningArchitecture.GAN,
        runtime=DeepLearningRuntime.CUSTOM,
        domain=DeepLearningDomain.TIME_FREQUENCY,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.OFFLINE,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=4096,
        chunk_overlap_samples=512,
        description="Nested time-frequency and waveform GAN refinement.",
        tags=("gan", "spectrogram", "two_stage"),
    ),
    "d4pm": DeepLearningModelSpec(
        name="D4PM",
        architecture=DeepLearningArchitecture.DIFFUSION,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.BOTH,
        latency_profile=DeepLearningLatencyProfile.OFFLINE,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=4096,
        chunk_overlap_samples=512,
        description="Dual-branch diffusion with joint posterior sampling of signal and artifact.",
        tags=("diffusion", "source_separation", "offline"),
    ),
    "denoise_mamba": DeepLearningModelSpec(
        name="DenoiseMamba",
        architecture=DeepLearningArchitecture.STATE_SPACE,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=8192,
        chunk_overlap_samples=256,
        description="ConvSSD state-space denoiser for long high-rate sequences.",
        tags=("mamba", "state_space", "long_context"),
    ),
    "conv_tasnet": DeepLearningModelSpec(
        name="Conv-TasNet",
        architecture=DeepLearningArchitecture.AUDIO_SOURCE_SEPARATION,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.BOTH,
        latency_profile=DeepLearningLatencyProfile.REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=8192,
        chunk_overlap_samples=512,
        description="Waveform-domain source separation adapted from speech denoising.",
        tags=("audio", "source_separation", "time_domain"),
    ),
    "demucs": DeepLearningModelSpec(
        name="Demucs",
        architecture=DeepLearningArchitecture.AUDIO_SOURCE_SEPARATION,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.BOTH,
        latency_profile=DeepLearningLatencyProfile.NEAR_REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=8192,
        chunk_overlap_samples=512,
        description="U-Net plus recurrent source separation for rhythmic artifacts.",
        tags=("audio", "u_net", "sequence"),
    ),
    "sepformer": DeepLearningModelSpec(
        name="SepFormer",
        architecture=DeepLearningArchitecture.TRANSFORMER,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.BOTH,
        latency_profile=DeepLearningLatencyProfile.OFFLINE,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=4096,
        chunk_overlap_samples=512,
        description="Dual-path transformer source separator for local and inter-window structure.",
        tags=("transformer", "source_separation", "dual_path"),
    ),
    "vit_spectrogram": DeepLearningModelSpec(
        name="ViT Spectrogram Inpainting",
        architecture=DeepLearningArchitecture.VISION_TRANSFORMER,
        runtime=DeepLearningRuntime.CUSTOM,
        domain=DeepLearningDomain.TIME_FREQUENCY,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.OFFLINE,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=1024,
        chunk_overlap_samples=128,
        description="Vision-transformer or MAE-based spectrogram inpainting.",
        tags=("vit", "mae", "spectrogram"),
    ),
    "st_gnn": DeepLearningModelSpec(
        name="ST-GNN",
        architecture=DeepLearningArchitecture.GRAPH_NEURAL_NETWORK,
        runtime=DeepLearningRuntime.CUSTOM,
        domain=DeepLearningDomain.GRAPH,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.NEAR_REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        requires_channel_positions=True,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=2048,
        chunk_overlap_samples=128,
        description="Spatiotemporal graph model using electrode topology.",
        tags=("gnn", "topography", "spatial_constraints"),
    ),
    "eegdfus": DeepLearningModelSpec(
        name="EEGDfus",
        architecture=DeepLearningArchitecture.DIFFUSION,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.OFFLINE,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=4096,
        chunk_overlap_samples=512,
        description=(
            "Diffusion-based EEG artifact removal with guided denoising score matching. "
            "Models the clean-EEG posterior conditioned on the noisy observation."
        ),
        tags=("diffusion", "score_matching", "offline", "posterior_sampling"),
    ),
    "eegm2": DeepLearningModelSpec(
        name="EEGM2",
        architecture=DeepLearningArchitecture.STATE_SPACE,
        runtime=DeepLearningRuntime.CUSTOM,
        output_type=DeepLearningOutputType.CLEAN,
        latency_profile=DeepLearningLatencyProfile.NEAR_REALTIME,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        device_preference="cuda",
        supports_chunking=True,
        chunk_size_samples=8192,
        chunk_overlap_samples=256,
        description=(
            "Multi-scale Mamba state-space model for EEG artifact removal. "
            "Combines local convolutional blocks with hierarchical SSM layers "
            "to handle both the sharp GA edges and long-range TR periodicity."
        ),
        tags=("mamba", "state_space", "multi_scale", "long_context"),
    ),
}


def list_deep_learning_blueprints() -> dict[str, DeepLearningModelSpec]:
    """Return research blueprints for planned architecture families."""
    return _RESEARCH_BLUEPRINTS.copy()


def spec_to_dict(spec: DeepLearningModelSpec) -> dict[str, Any]:
    """Serialise a :class:`DeepLearningModelSpec` to a JSON-compatible dict.

    All :class:`~enum.StrEnum` values are stored as plain strings so the result
    can be round-tripped via :func:`spec_from_dict` or written directly to a
    JSON / YAML configuration file.

    Parameters
    ----------
    spec:
        The spec to serialise.

    Returns
    -------
    dict
        A shallow dict of primitive Python values.

    Examples
    --------
    >>> import json
    >>> d = spec_to_dict(my_spec)
    >>> json.dumps(d)  # safe — no enum instances
    """
    import dataclasses

    raw = dataclasses.asdict(spec)
    # StrEnum values serialise as str automatically via asdict, but convert
    # explicitly for clarity and forward-compatibility.
    str_fields = {
        "architecture", "runtime", "domain", "output_type",
        "latency_profile", "execution_granularity",
        "channel_grouping_strategy", "dual_output_policy",
    }
    for field in str_fields:
        if raw.get(field) is not None:
            raw[field] = str(raw[field])
    # tags is a tuple → list for JSON compatibility
    if isinstance(raw.get("tags"), tuple):
        raw["tags"] = list(raw["tags"])
    return raw


def spec_from_dict(data: dict[str, Any]) -> DeepLearningModelSpec:
    """Reconstruct a :class:`DeepLearningModelSpec` from a plain dict.

    This is the inverse of :func:`spec_to_dict` and accepts the JSON
    representations produced by that function.  Unknown keys are silently
    ignored so that serialised configs remain forward-compatible when new
    optional fields are added to the spec.

    Parameters
    ----------
    data:
        Dict as returned by :func:`spec_to_dict` or loaded from a JSON /
        YAML file.

    Returns
    -------
    DeepLearningModelSpec
        Fully reconstructed, validated spec instance.

    Raises
    ------
    KeyError
        If the required *name* or *architecture* keys are missing.
    ValueError
        If any enum value cannot be resolved.
    """
    import dataclasses

    known = {f.name for f in dataclasses.fields(DeepLearningModelSpec)}
    filtered = {k: v for k, v in data.items() if k in known}

    enum_map: dict[str, type] = {
        "architecture": DeepLearningArchitecture,
        "runtime": DeepLearningRuntime,
        "domain": DeepLearningDomain,
        "output_type": DeepLearningOutputType,
        "latency_profile": DeepLearningLatencyProfile,
        "execution_granularity": DeepLearningExecutionGranularity,
        "channel_grouping_strategy": DeepLearningChannelGroupingStrategy,
        "dual_output_policy": DeepLearningDualOutputPolicy,
    }
    for field, enum_cls in enum_map.items():
        if field in filtered and filtered[field] is not None:
            filtered[field] = enum_cls(filtered[field])

    # tags must be a tuple
    if "tags" in filtered and isinstance(filtered["tags"], list):
        filtered["tags"] = tuple(filtered["tags"])

    return DeepLearningModelSpec(**filtered)


@register_processor
class DeepLearningCorrection(Processor):
    """Apply a learned artifact correction model inside a FACETpy pipeline."""

    name = "deep_learning_correction"
    description = "Deep-learning-based artifact correction"
    version = "1.0.0"

    requires_raw = True
    requires_triggers = False
    modifies_raw = True
    parallel_safe = False
    channel_wise = False

    def __init__(
        self,
        model: str | type[DeepLearningModelAdapter] | DeepLearningModelAdapter,
        model_kwargs: dict[str, Any] | None = None,
        store_run_metadata: bool = True,
        trigger_aligned_chunking: bool = False,
        triggers_per_chunk: int = 1,
    ) -> None:
        if triggers_per_chunk < 1:
            raise ValueError("triggers_per_chunk must be >= 1")
        self.model = self._coerce_model(model, model_kwargs=model_kwargs)
        self.channel_wise = self.model.spec.execution_granularity == DeepLearningExecutionGranularity.CHANNEL
        self.store_run_metadata = store_run_metadata
        self.trigger_aligned_chunking = trigger_aligned_chunking
        self.triggers_per_chunk = triggers_per_chunk
        super().__init__()

    def _coerce_model(
        self,
        model: str | type[DeepLearningModelAdapter] | DeepLearningModelAdapter,
        *,
        model_kwargs: dict[str, Any] | None,
    ) -> DeepLearningModelAdapter:
        kwargs = model_kwargs or {}
        if isinstance(model, DeepLearningModelAdapter):
            if kwargs:
                raise ValueError("model_kwargs are only supported when model is a registry name or adapter class")
            return model
        if isinstance(model, str):
            model_class = get_deep_learning_model(model)
            return model_class(**kwargs)
        if isinstance(model, type) and issubclass(model, DeepLearningModelAdapter):
            return model(**kwargs)
        raise TypeError("model must be a registered name, adapter class, or DeepLearningModelAdapter instance")

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        self.model.validate_context(context)
        if self.trigger_aligned_chunking and not context.has_triggers():
            raise ProcessorValidationError(
                f"Model '{self.model.spec.name}' has trigger_aligned_chunking=True "
                "but the context contains no trigger positions. "
                "Run TriggerDetector before this processor."
            )

    def validate_execution_mode(self, *, parallel: bool, channel_sequential: bool) -> None:
        spec = self.model.spec
        if channel_sequential and spec.execution_granularity != DeepLearningExecutionGranularity.CHANNEL:
            raise ProcessorValidationError(
                f"Model '{spec.name}' uses execution_granularity='{spec.execution_granularity.value}' and cannot "
                "run with channel_sequential=True. Use serial execution or a channel-granular model."
            )

    def _get_parameters(self) -> dict[str, Any]:
        spec = self.model.spec
        return {
            "model": spec.name,
            "architecture": spec.architecture.value,
            "runtime": spec.runtime.value,
            "domain": spec.domain.value,
            "output_type": spec.output_type.value,
            "execution_granularity": spec.execution_granularity.value,
            "channel_group_size": spec.channel_group_size,
            "channel_grouping_strategy": spec.channel_grouping_strategy.value,
            "checkpoint_path": spec.checkpoint_path,
            "checkpoint_format": spec.checkpoint_format,
            "device_preference": spec.device_preference,
            "supports_chunking": spec.supports_chunking,
            "chunk_size_samples": spec.chunk_size_samples,
            "chunk_overlap_samples": spec.chunk_overlap_samples,
            "dual_output_policy": spec.dual_output_policy.value,
            "store_run_metadata": self.store_run_metadata,
            "trigger_aligned_chunking": self.trigger_aligned_chunking,
            "triggers_per_chunk": self.triggers_per_chunk,
        }

    def _validate_prediction_contract(self, prediction: DeepLearningPrediction) -> None:
        spec = self.model.spec
        if spec.output_type == DeepLearningOutputType.ARTIFACT and prediction.artifact_data is None:
            raise ProcessorValidationError(f"Model '{spec.name}' declared output_type='artifact' but returned none")
        if spec.output_type == DeepLearningOutputType.CLEAN and prediction.clean_data is None:
            raise ProcessorValidationError(f"Model '{spec.name}' declared output_type='clean' but returned none")
        if (
            spec.output_type == DeepLearningOutputType.BOTH
            and (prediction.clean_data is None or prediction.artifact_data is None)
        ):
            raise ProcessorValidationError(
                f"Model '{spec.name}' declared output_type='both' and must return both clean_data and artifact_data"
            )

    @staticmethod
    def _trigger_chunk_ranges(
        triggers: np.ndarray,
        total_samples: int,
        triggers_per_chunk: int = 1,
    ) -> list[tuple[int, int]]:
        """Build chunk ranges aligned to TR (trigger) boundaries.

        Each chunk starts at the first sample of a trigger group and ends at
        the first sample of the next group.  The final chunk extends to
        *total_samples*.  Samples before the first trigger are excluded
        (gradient artifact has not started yet).

        Parameters
        ----------
        triggers:
            Sorted or unsorted array of trigger sample positions.
        total_samples:
            Total number of samples in the recording.
        triggers_per_chunk:
            Number of consecutive TRs (triggers) to include in each chunk.
            ``1`` (default) means one TR per chunk.

        Returns
        -------
        list[tuple[int, int]]
            List of ``(start, stop)`` sample pairs, non-overlapping,
            covering ``[triggers[0], total_samples)``.
        """
        if len(triggers) == 0:
            return [(0, total_samples)]

        sorted_triggers = np.sort(triggers)
        # Select every triggers_per_chunk-th trigger as a chunk boundary.
        group_starts = sorted_triggers[::triggers_per_chunk].tolist()
        ranges: list[tuple[int, int]] = []
        for i, start in enumerate(group_starts):
            stop = group_starts[i + 1] if i + 1 < len(group_starts) else total_samples
            start_i, stop_i = int(start), int(stop)
            if start_i < stop_i:
                ranges.append((start_i, stop_i))
        return ranges

    @staticmethod
    def _chunk_ranges(total_samples: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
        if chunk_size >= total_samples:
            return [(0, total_samples)]

        ranges: list[tuple[int, int]] = []
        step = chunk_size - overlap
        start = 0
        while start < total_samples:
            stop = min(start + chunk_size, total_samples)
            ranges.append((start, stop))
            if stop == total_samples:
                break
            start += step
        return ranges

    @staticmethod
    def _data_channel_indices(raw: mne.io.BaseRaw) -> list[int]:
        eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
        if len(eeg_picks) == 0:
            return list(range(len(raw.ch_names)))
        return eeg_picks.tolist()

    @staticmethod
    def _channel_positions(raw: mne.io.BaseRaw, channel_indices: list[int]) -> np.ndarray | None:
        positions: list[np.ndarray] = []
        for ch_idx in channel_indices:
            loc = np.asarray(raw.info["chs"][ch_idx].get("loc", []), dtype=float)
            if loc.size < 3:
                return None
            xyz = loc[:3]
            if not np.all(np.isfinite(xyz)) or np.allclose(xyz, 0.0):
                return None
            positions.append(xyz)
        return np.vstack(positions)

    def _precompute_position_groups(
        self,
        raw: mne.io.BaseRaw,
        data_indices: list[int],
    ) -> dict[int, list[int]]:
        spec = self.model.spec
        max_group_size = min(spec.channel_group_size, len(data_indices))
        if max_group_size == len(data_indices):
            return {
                target_abs_idx: [target_abs_idx] + [idx for idx in data_indices if idx != target_abs_idx]
                for target_abs_idx in data_indices
            }

        positions = self._channel_positions(raw, data_indices)
        if positions is None:
            raise ProcessorValidationError(
                f"Model '{spec.name}' requires valid channel positions for position-based channel grouping"
            )

        distance_matrix = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
        precomputed_groups: dict[int, list[int]] = {}
        for target_row, target_abs_idx in enumerate(data_indices):
            ordered = [data_indices[i] for i in np.argsort(distance_matrix[target_row], kind="stable")]
            selected = ordered[:max_group_size]
            precomputed_groups[target_abs_idx] = [target_abs_idx] + [idx for idx in selected if idx != target_abs_idx]
        return precomputed_groups

    def _select_channel_group(
        self,
        raw: mne.io.BaseRaw,
        target_abs_idx: int,
        *,
        data_indices: list[int] | None = None,
        precomputed_position_groups: dict[int, list[int]] | None = None,
    ) -> list[int]:
        spec = self.model.spec
        if precomputed_position_groups is not None:
            return precomputed_position_groups[target_abs_idx].copy()

        data_indices = self._data_channel_indices(raw) if data_indices is None else data_indices
        max_group_size = min(spec.channel_group_size, len(data_indices))
        if max_group_size == len(data_indices):
            return [target_abs_idx] + [idx for idx in data_indices if idx != target_abs_idx]

        distances = np.asarray([abs(idx - target_abs_idx) for idx in data_indices], dtype=float)
        ordered = [data_indices[i] for i in np.argsort(distances, kind="stable")]
        selected = ordered[:max_group_size]
        return [target_abs_idx] + [idx for idx in selected if idx != target_abs_idx]

    def _build_channel_group_context(
        self,
        context: ProcessingContext,
        channel_indices: list[int],
    ) -> ProcessingContext:
        raw = context.get_raw()
        ch_names = [raw.ch_names[idx] for idx in channel_indices]
        group_data = raw.get_data(picks=ch_names)
        info = mne.pick_info(raw.info, channel_indices)
        group_raw = mne.io.RawArray(group_data, info, verbose=False)
        group_context = ProcessingContext(
            raw=group_raw,
            raw_original=context.get_raw_original(),
            metadata=context.metadata.copy(),
        )
        group_context._history = context.get_history()
        group_context.metadata.custom.update(
            {
                "channel_group_indices": channel_indices.copy(),
                "channel_group_target_index_local": 0,
                "channel_group_target_name": group_raw.ch_names[0],
            }
        )

        if context.has_estimated_noise():
            noise = context.get_estimated_noise()
            if noise is not None and noise.ndim == 2:
                group_context.set_estimated_noise(noise[channel_indices].copy())

        return group_context

    def _build_chunk_context(self, context: ProcessingContext, start: int, stop: int) -> ProcessingContext:
        raw = context.get_raw()
        raw_chunk = mne.io.RawArray(raw._data[:, start:stop].copy(), raw.info.copy(), verbose=False)
        metadata = context.metadata.copy()

        if metadata.triggers is not None:
            local_triggers = metadata.triggers - start
            valid = (local_triggers >= 0) & (local_triggers < (stop - start))
            metadata.triggers = local_triggers[valid].astype(np.int32, copy=False)

        if metadata.acq_start_sample is not None:
            metadata.acq_start_sample = max(metadata.acq_start_sample - start, 0)
        if metadata.acq_end_sample is not None:
            metadata.acq_end_sample = min(metadata.acq_end_sample - start, stop - start)

        metadata.custom.update(
            {
                "chunk_start_sample": start,
                "chunk_stop_sample": stop,
                "chunk_length_samples": stop - start,
            }
        )

        chunk_context = ProcessingContext(raw=raw_chunk, raw_original=raw_chunk.copy(), metadata=metadata)
        if context.has_estimated_noise():
            chunk_context.set_estimated_noise(context.get_estimated_noise()[:, start:stop].copy())
        return chunk_context

    @staticmethod
    def _resolve_channel_indices(channel_indices: np.ndarray | list[int] | None, n_channels: int) -> np.ndarray:
        if channel_indices is None:
            return np.arange(n_channels, dtype=int)

        indices = np.asarray(channel_indices, dtype=int)
        if indices.ndim != 1:
            raise ProcessorValidationError("channel_indices must be a one-dimensional array of channel indices")
        if len(indices) == 0:
            raise ProcessorValidationError("channel_indices cannot be empty")
        if np.any(indices < 0) or np.any(indices >= n_channels):
            raise ProcessorValidationError(f"channel_indices must fall within [0, {n_channels - 1}]")
        return indices

    @staticmethod
    def _resolve_sample_bounds(prediction: DeepLearningPrediction, n_samples: int) -> tuple[int, int]:
        start = int(prediction.start_sample)
        stop = n_samples if prediction.stop_sample is None else int(prediction.stop_sample)
        if start < 0 or stop > n_samples or start >= stop:
            raise ProcessorValidationError(
                f"Prediction sample range must satisfy 0 <= start < stop <= {n_samples}, got [{start}, {stop})"
            )
        return start, stop

    @staticmethod
    def _coerce_segment(
        segment: np.ndarray | None,
        *,
        n_channels: int,
        n_samples: int,
        label: str,
    ) -> np.ndarray | None:
        if segment is None:
            return None

        arr = np.asarray(segment, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ProcessorValidationError(f"{label} must be a 2D array with shape (n_channels, n_samples)")
        if arr.shape != (n_channels, n_samples):
            raise ProcessorValidationError(
                f"{label} must have shape {(n_channels, n_samples)}, got {tuple(arr.shape)}"
            )
        return arr

    def _resolve_artifact_prediction(
        self,
        data: np.ndarray,
        prediction: DeepLearningPrediction,
    ) -> tuple[np.ndarray, int, int, np.ndarray]:
        self._validate_prediction_contract(prediction)

        channel_indices = self._resolve_channel_indices(prediction.channel_indices, data.shape[0])
        start, stop = self._resolve_sample_bounds(prediction, data.shape[1])
        segment_length = stop - start
        original_segment = data[channel_indices, start:stop].copy()

        artifact_segment = self._coerce_segment(
            prediction.artifact_data,
            n_channels=len(channel_indices),
            n_samples=segment_length,
            label="artifact_data",
        )
        clean_segment = self._coerce_segment(
            prediction.clean_data,
            n_channels=len(channel_indices),
            n_samples=segment_length,
            label="clean_data",
        )

        if clean_segment is None:
            clean_segment = original_segment - artifact_segment
        if artifact_segment is None:
            artifact_segment = original_segment - clean_segment

        reconstructed_clean = original_segment - artifact_segment
        spec = self.model.spec
        if not np.allclose(
            reconstructed_clean,
            clean_segment,
            rtol=spec.dual_output_rtol,
            atol=spec.dual_output_atol,
        ):
            if spec.dual_output_policy == DeepLearningDualOutputPolicy.STRICT:
                raise ProcessorValidationError(
                    f"Model '{spec.name}' returned inconsistent clean_data and artifact_data predictions"
                )
            if spec.dual_output_policy == DeepLearningDualOutputPolicy.PREFER_CLEAN:
                artifact_segment = original_segment - clean_segment
            else:
                clean_segment = original_segment - artifact_segment

        return channel_indices, start, stop, artifact_segment

    def _process_single_pass(self, context: ProcessingContext) -> tuple[mne.io.BaseRaw, np.ndarray, dict[str, Any]]:
        raw = context.get_raw().copy()
        prediction = self.model.predict(context)
        channel_indices, start, stop, artifact_segment = self._resolve_artifact_prediction(raw._data, prediction)

        raw._data[channel_indices, start:stop] -= artifact_segment
        estimated_artifacts = np.zeros_like(raw._data)
        estimated_artifacts[channel_indices, start:stop] = artifact_segment

        execution_metadata = {
            "start_sample": start,
            "stop_sample": stop,
            "channel_indices": channel_indices.tolist(),
            "prediction_metadata": deepcopy(prediction.metadata),
            "execution_mode": "single_pass",
        }
        return raw, estimated_artifacts, execution_metadata

    def _infer_local_context(self, context: ProcessingContext) -> tuple[mne.io.BaseRaw, np.ndarray, dict[str, Any]]:
        spec = self.model.spec
        if self.trigger_aligned_chunking and context.has_triggers():
            return self._process_trigger_aligned(context)
        use_chunking = spec.supports_chunking and spec.chunk_size_samples is not None
        if use_chunking:
            return self._process_chunked(context)
        return self._process_single_pass(context)

    def _process_trigger_aligned(
        self, context: ProcessingContext
    ) -> tuple[mne.io.BaseRaw, np.ndarray, dict[str, Any]]:
        """Run inference with chunks aligned to TR (trigger) boundaries."""
        raw = context.get_raw().copy()
        total_samples = raw._data.shape[1]
        triggers = context.metadata.triggers
        chunk_ranges = self._trigger_chunk_ranges(triggers, total_samples, self.triggers_per_chunk)
        estimated_artifacts = np.zeros_like(raw._data)
        chunk_summaries: list[dict[str, Any]] = []

        for chunk_start, chunk_stop in chunk_ranges:
            chunk_context = self._build_chunk_context(context, chunk_start, chunk_stop)
            prediction = self.model.predict(chunk_context)
            channel_indices, local_start, local_stop, artifact_segment = self._resolve_artifact_prediction(
                chunk_context.get_raw()._data,
                prediction,
            )
            global_start = chunk_start + local_start
            global_stop = chunk_start + local_stop
            estimated_artifacts[channel_indices, global_start:global_stop] = artifact_segment
            chunk_summaries.append(
                {
                    "chunk_start_sample": chunk_start,
                    "chunk_stop_sample": chunk_stop,
                    "prediction_start_sample": global_start,
                    "prediction_stop_sample": global_stop,
                    "channel_indices": channel_indices.tolist(),
                    "prediction_metadata": deepcopy(prediction.metadata),
                }
            )

        raw._data -= estimated_artifacts
        execution_metadata = {
            "start_sample": 0,
            "stop_sample": total_samples,
            "channel_indices": list(range(raw._data.shape[0])),
            "prediction_metadata": {"chunks": chunk_summaries},
            "execution_mode": "trigger_aligned",
            "chunk_count": len(chunk_ranges),
            "triggers_per_chunk": self.triggers_per_chunk,
        }
        return raw, estimated_artifacts, execution_metadata

    def _process_chunked(self, context: ProcessingContext) -> tuple[mne.io.BaseRaw, np.ndarray, dict[str, Any]]:
        spec = self.model.spec
        raw = context.get_raw().copy()
        total_samples = raw._data.shape[1]
        chunk_ranges = self._chunk_ranges(total_samples, spec.chunk_size_samples, spec.chunk_overlap_samples)
        estimated_artifacts = np.zeros_like(raw._data)
        chunk_summaries: list[dict[str, Any]] = []

        if spec.chunk_overlap_samples == 0:
            for chunk_start, chunk_stop in chunk_ranges:
                chunk_context = self._build_chunk_context(context, chunk_start, chunk_stop)
                prediction = self.model.predict(chunk_context)
                channel_indices, local_start, local_stop, artifact_segment = self._resolve_artifact_prediction(
                    chunk_context.get_raw()._data,
                    prediction,
                )
                global_start = chunk_start + local_start
                global_stop = chunk_start + local_stop

                estimated_artifacts[channel_indices, global_start:global_stop] = artifact_segment
                chunk_summaries.append(
                    {
                        "chunk_start_sample": chunk_start,
                        "chunk_stop_sample": chunk_stop,
                        "prediction_start_sample": global_start,
                        "prediction_stop_sample": global_stop,
                        "channel_indices": channel_indices.tolist(),
                        "prediction_metadata": deepcopy(prediction.metadata),
                    }
                )

            raw._data -= estimated_artifacts
            execution_metadata = {
                "start_sample": 0,
                "stop_sample": total_samples,
                "channel_indices": list(range(raw._data.shape[0])),
                "prediction_metadata": {"chunks": chunk_summaries},
                "execution_mode": "chunked",
                "chunk_count": len(chunk_ranges),
            }
            return raw, estimated_artifacts, execution_metadata

        artifact_counts = np.zeros(raw._data.shape, dtype=np.uint16)

        for chunk_start, chunk_stop in chunk_ranges:
            chunk_context = self._build_chunk_context(context, chunk_start, chunk_stop)
            prediction = self.model.predict(chunk_context)
            channel_indices, local_start, local_stop, artifact_segment = self._resolve_artifact_prediction(
                chunk_context.get_raw()._data,
                prediction,
            )
            global_start = chunk_start + local_start
            global_stop = chunk_start + local_stop

            estimated_artifacts[channel_indices, global_start:global_stop] += artifact_segment
            artifact_counts[channel_indices, global_start:global_stop] += 1
            chunk_summaries.append(
                {
                    "chunk_start_sample": chunk_start,
                    "chunk_stop_sample": chunk_stop,
                    "prediction_start_sample": global_start,
                    "prediction_stop_sample": global_stop,
                    "channel_indices": channel_indices.tolist(),
                    "prediction_metadata": deepcopy(prediction.metadata),
                }
            )

        covered_mask = artifact_counts > 0
        estimated_artifacts[covered_mask] = estimated_artifacts[covered_mask] / artifact_counts[covered_mask]
        raw._data[covered_mask] -= estimated_artifacts[covered_mask]

        execution_metadata = {
            "start_sample": 0,
            "stop_sample": total_samples,
            "channel_indices": list(range(raw._data.shape[0])),
            "prediction_metadata": {"chunks": chunk_summaries},
            "execution_mode": "chunked",
            "chunk_count": len(chunk_ranges),
        }
        return raw, estimated_artifacts, execution_metadata

    def _process_channel_groups(self, context: ProcessingContext) -> tuple[mne.io.BaseRaw, np.ndarray, dict[str, Any]]:
        raw = context.get_raw().copy()
        estimated_artifacts = np.zeros_like(raw._data)
        group_summaries: list[dict[str, Any]] = []
        data_indices = self._data_channel_indices(raw)
        precomputed_position_groups: dict[int, list[int]] | None = None

        if self.model.spec.channel_grouping_strategy == DeepLearningChannelGroupingStrategy.POSITION:
            precomputed_position_groups = self._precompute_position_groups(raw, data_indices)

        for target_abs_idx in data_indices:
            group_indices = self._select_channel_group(
                raw,
                target_abs_idx,
                data_indices=data_indices,
                precomputed_position_groups=precomputed_position_groups,
            )
            group_context = self._build_channel_group_context(context, group_indices)
            _, local_estimated_artifacts, local_metadata = self._infer_local_context(group_context)

            estimated_artifacts[target_abs_idx] = local_estimated_artifacts[0]
            raw._data[target_abs_idx] -= local_estimated_artifacts[0]
            group_summaries.append(
                {
                    "target_channel_index": target_abs_idx,
                    "target_channel_name": raw.ch_names[target_abs_idx],
                    "group_channel_indices": group_indices.copy(),
                    "group_channel_names": [raw.ch_names[idx] for idx in group_indices],
                    "local_execution_mode": local_metadata["execution_mode"],
                    "prediction_metadata": deepcopy(local_metadata["prediction_metadata"]),
                }
            )

        execution_metadata = {
            "start_sample": 0,
            "stop_sample": raw._data.shape[1],
            "channel_indices": data_indices,
            "prediction_metadata": {"groups": group_summaries},
            "execution_mode": "channel_group",
            "group_count": len(group_summaries),
        }
        return raw, estimated_artifacts, execution_metadata

    def process(self, context: ProcessingContext) -> ProcessingContext:
        spec = self.model.spec
        if spec.execution_granularity == DeepLearningExecutionGranularity.CHANNEL_GROUP:
            raw, estimated_artifacts, execution_metadata = self._process_channel_groups(context)
        else:
            raw, estimated_artifacts, execution_metadata = self._infer_local_context(context)

        new_ctx = context.with_raw(raw)
        new_ctx.accumulate_noise(estimated_artifacts)

        if self.store_run_metadata:
            run_metadata = {
                "model": self.model.spec.name,
                "architecture": self.model.spec.architecture.value,
                "runtime": self.model.spec.runtime.value,
                "domain": self.model.spec.domain.value,
                "latency_profile": self.model.spec.latency_profile.value,
                "execution_granularity": self.model.spec.execution_granularity.value,
                "channel_group_size": self.model.spec.channel_group_size,
                "channel_grouping_strategy": self.model.spec.channel_grouping_strategy.value,
                "checkpoint_path": self.model.spec.checkpoint_path,
                "checkpoint_format": self.model.spec.checkpoint_format,
                "device_preference": self.model.spec.device_preference,
                "supports_chunking": self.model.spec.supports_chunking,
                "chunk_size_samples": self.model.spec.chunk_size_samples,
                "chunk_overlap_samples": self.model.spec.chunk_overlap_samples,
                "dual_output_policy": self.model.spec.dual_output_policy.value,
                **execution_metadata,
            }
            new_ctx.metadata.custom.setdefault("deep_learning_runs", []).append(run_metadata)

        logger.info(
            "Applied deep-learning correction '{}' in {} mode",
            self.model.spec.name,
            execution_metadata["execution_mode"],
        )
        return new_ctx

    # ------------------------------------------------------------------
    # Config serialisation
    # ------------------------------------------------------------------

    def to_config_dict(self) -> dict[str, Any]:
        """Serialise this processor to a JSON-compatible configuration dict.

        The returned dict can be saved as a JSON file and later restored via
        :meth:`from_config_dict`.  Only adapters registered in
        :class:`DeepLearningModelRegistry` with a serialisable
        ``checkpoint_path`` are supported (i.e. TensorFlow, PyTorch, ONNX).
        :class:`NumpyInferenceAdapter` instances that use a runtime callable
        cannot be round-tripped.

        Returns
        -------
        dict
            JSON-safe dict with keys ``version``, ``processor``,
            ``adapter``, ``spec``, ``store_run_metadata``,
            ``trigger_aligned_chunking``, ``triggers_per_chunk``.

        Raises
        ------
        ValueError
            If the adapter is not registered in the model registry.
        """
        registry = DeepLearningModelRegistry.get_instance()
        adapter_name: str | None = next(
            (name for name, cls in registry.list_all().items() if type(self.model) is cls),
            None,
        )
        if adapter_name is None:
            raise ValueError(
                f"Adapter type '{type(self.model).__name__}' is not registered in "
                "DeepLearningModelRegistry. Register it first via register_deep_learning_model()."
            )
        return {
            "version": "1",
            "processor": self.name,
            "adapter": adapter_name,
            "spec": spec_to_dict(self.model.spec),
            "store_run_metadata": self.store_run_metadata,
            "trigger_aligned_chunking": self.trigger_aligned_chunking,
            "triggers_per_chunk": self.triggers_per_chunk,
        }

    @classmethod
    def from_config_dict(cls, data: dict[str, Any]) -> "DeepLearningCorrection":
        """Reconstruct a :class:`DeepLearningCorrection` from a config dict.

        This is the inverse of :meth:`to_config_dict`.  The adapter is
        looked up by name in :class:`DeepLearningModelRegistry` and
        instantiated with the deserialised :class:`DeepLearningModelSpec`.

        Parameters
        ----------
        data:
            Dict as returned by :meth:`to_config_dict` or loaded from a
            JSON file via :func:`load_deep_learning_config`.

        Returns
        -------
        DeepLearningCorrection
            Fully reconstructed processor ready to be added to a pipeline.

        Raises
        ------
        KeyError
            If the adapter name is not registered.
        ValueError
            If ``spec.checkpoint_path`` is ``None``.
        """
        import dataclasses

        adapter_name: str = data["adapter"]
        spec = spec_from_dict(data["spec"])

        if spec.checkpoint_path is None:
            raise ValueError(
                "Cannot reconstruct adapter from config: spec.checkpoint_path is None. "
                "Set checkpoint_path in the config spec."
            )

        adapter_cls = get_deep_learning_model(adapter_name)

        # Build spec_overrides: all spec fields except checkpoint_path
        # (which is passed as a positional kwarg to stay compatible with all adapters).
        spec_fields = {f.name for f in dataclasses.fields(DeepLearningModelSpec)}
        spec_overrides = {
            k: getattr(spec, k)
            for k in spec_fields
            if k != "checkpoint_path"
        }

        adapter = adapter_cls(
            checkpoint_path=spec.checkpoint_path,
            spec_overrides=spec_overrides,
        )

        return cls(
            model=adapter,
            store_run_metadata=data.get("store_run_metadata", True),
            trigger_aligned_chunking=data.get("trigger_aligned_chunking", False),
            triggers_per_chunk=data.get("triggers_per_chunk", 1),
        )


# ---------------------------------------------------------------------------
# Convenience I/O helpers
# ---------------------------------------------------------------------------

def save_deep_learning_config(processor: "DeepLearningCorrection", path: "str | Path") -> None:
    """Save a :class:`DeepLearningCorrection` configuration to a JSON file.

    Parameters
    ----------
    processor:
        The processor to serialise.
    path:
        Destination file path.  The ``.json`` suffix is recommended.

    Example
    -------
    >>> save_deep_learning_config(my_processor, "onnx_correction.json")
    """
    import json

    config = processor.to_config_dict()
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    logger.info("Saved DeepLearningCorrection config to {}", output_path)


def load_deep_learning_config(path: "str | Path") -> "DeepLearningCorrection":
    """Load a :class:`DeepLearningCorrection` from a JSON configuration file.

    Parameters
    ----------
    path:
        Path to a JSON file previously created by
        :func:`save_deep_learning_config`.

    Returns
    -------
    DeepLearningCorrection
        Reconstructed processor ready to be added to a pipeline.

    Example
    -------
    >>> processor = load_deep_learning_config("onnx_correction.json")
    >>> pipeline = Pipeline([processor])
    """
    import json

    input_path = Path(path).expanduser()
    with input_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    logger.info("Loaded DeepLearningCorrection config from {}", input_path)
    return DeepLearningCorrection.from_config_dict(config)
