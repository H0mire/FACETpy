"""Deep-learning integration primitives for artifact correction pipelines.

This module provides preparatory abstractions for integrating learned artifact
removal models into FACETpy without coupling the pipeline to a specific
framework or architecture family.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
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
}


def list_deep_learning_blueprints() -> dict[str, DeepLearningModelSpec]:
    """Return research blueprints for planned architecture families."""
    return _RESEARCH_BLUEPRINTS.copy()


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
    ) -> None:
        self.model = self._coerce_model(model, model_kwargs=model_kwargs)
        self.channel_wise = self.model.spec.execution_granularity == DeepLearningExecutionGranularity.CHANNEL
        self.store_run_metadata = store_run_metadata
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

    def _select_channel_group(self, raw: mne.io.BaseRaw, target_abs_idx: int) -> list[int]:
        spec = self.model.spec
        data_indices = self._data_channel_indices(raw)
        max_group_size = min(spec.channel_group_size, len(data_indices))
        if max_group_size == len(data_indices):
            return [target_abs_idx] + [idx for idx in data_indices if idx != target_abs_idx]

        if spec.channel_grouping_strategy == DeepLearningChannelGroupingStrategy.POSITION:
            positions = self._channel_positions(raw, data_indices)
            if positions is None:
                raise ProcessorValidationError(
                    f"Model '{spec.name}' requires valid channel positions for position-based channel grouping"
                )
            target_pos = positions[data_indices.index(target_abs_idx)]
            distances = np.linalg.norm(positions - target_pos, axis=1)
        else:
            distances = np.asarray([abs(idx - target_abs_idx) for idx in data_indices], dtype=float)

        ordered = [data_indices[i] for i in np.argsort(distances, kind="stable")]
        selected = ordered[:max_group_size]
        selected = [target_abs_idx] + [idx for idx in selected if idx != target_abs_idx]
        return selected

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
        group_context = context.with_raw(group_raw)
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
        else:
            group_context._estimated_noise = None

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
        use_chunking = spec.supports_chunking and spec.chunk_size_samples is not None
        if use_chunking:
            return self._process_chunked(context)
        return self._process_single_pass(context)

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

        for target_abs_idx in self._data_channel_indices(raw):
            group_indices = self._select_channel_group(raw, target_abs_idx)
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
            "channel_indices": self._data_channel_indices(raw),
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
