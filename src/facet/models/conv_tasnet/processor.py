"""Inference integration for the Conv-TasNet artifact source separator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np

from ...core import ProcessingContext, ProcessorValidationError, register_processor
from ...correction.deep_learning import (
    DeepLearningArchitecture,
    DeepLearningCorrection,
    DeepLearningExecutionGranularity,
    DeepLearningModelAdapter,
    DeepLearningModelSpec,
    DeepLearningOutputType,
    DeepLearningPrediction,
    DeepLearningRuntime,
)


class ConvTasNetAdapter(DeepLearningModelAdapter):
    """TorchScript adapter that runs Conv-TasNet on overlapping channel windows."""

    spec = DeepLearningModelSpec(
        name="ConvTasNetAdapter",
        architecture=DeepLearningArchitecture.CUSTOM,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        supports_chunking=True,
        chunk_size_samples=512,
        description="Channel-wise Conv-TasNet artifact source separator.",
        tags=("conv_tasnet", "source_separation", "tcn", "torchscript", "artifact_prediction"),
    )

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        chunk_size_samples: int = 512,
        chunk_overlap_samples: int = 0,
        device: str = "cpu",
        eeg_only: bool = True,
        channel_indices: list[int] | None = None,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.chunk_size_samples = int(chunk_size_samples)
        self.chunk_overlap_samples = int(chunk_overlap_samples)
        self.device = device
        self.eeg_only = bool(eeg_only)
        self.channel_indices = None if channel_indices is None else [int(idx) for idx in channel_indices]
        self.demean_input = bool(demean_input)
        self.remove_prediction_mean = bool(remove_prediction_mean)
        self._torch: Any | None = None
        self._model: Any | None = None
        self.spec = DeepLearningModelSpec(
            name="ConvTasNetAdapter",
            architecture=DeepLearningArchitecture.CUSTOM,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=DeepLearningOutputType.ARTIFACT,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=False,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="torchscript",
            device_preference=device,
            supports_chunking=True,
            chunk_size_samples=self.chunk_size_samples,
            chunk_overlap_samples=self.chunk_overlap_samples,
            description="Channel-wise Conv-TasNet artifact source separator.",
            tags=("conv_tasnet", "source_separation", "tcn", "torchscript", "artifact_prediction"),
        )
        super().__init__()

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.chunk_size_samples <= 0:
            raise ProcessorValidationError("chunk_size_samples must be positive")
        if self.chunk_overlap_samples < 0 or self.chunk_overlap_samples >= self.chunk_size_samples:
            raise ProcessorValidationError(
                "chunk_overlap_samples must be >= 0 and smaller than chunk_size_samples"
            )
        raw = context.get_raw()
        if raw.n_times < self.chunk_size_samples:
            raise ProcessorValidationError(
                f"Conv-TasNet requires at least {self.chunk_size_samples} samples, got {raw.n_times}"
            )
        self._resolve_channels(raw)

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        raw = context.get_raw()
        data = raw._data
        channels = self._resolve_channels(raw)
        model, torch = self._load_model()

        artifact = np.zeros_like(data)
        counts = np.zeros_like(data, dtype=np.uint16)
        chunk_ranges = self._chunk_ranges(data.shape[1])

        with torch.no_grad():
            for start, stop in chunk_ranges:
                if stop - start != self.chunk_size_samples:
                    continue
                for ch_idx in channels:
                    segment = data[ch_idx : ch_idx + 1, start:stop].astype(np.float32, copy=True)
                    if self.demean_input:
                        segment = segment - segment.mean(axis=-1, keepdims=True)
                    prediction = self._predict_segment(model, torch, segment)
                    if self.remove_prediction_mean:
                        prediction = prediction - prediction.mean(axis=-1, keepdims=True)
                    artifact[ch_idx, start:stop] += prediction.astype(data.dtype, copy=False)
                    counts[ch_idx, start:stop] += 1

        covered = counts > 0
        artifact[covered] = artifact[covered] / counts[covered]
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "chunk_size_samples": self.chunk_size_samples,
            "chunk_overlap_samples": self.chunk_overlap_samples,
            "chunk_count": len(chunk_ranges),
            "channels": [raw.ch_names[idx] for idx in channels],
            "device": self.device,
            "demean_input": self.demean_input,
            "remove_prediction_mean": self.remove_prediction_mean,
            "coverage_fraction": float(np.count_nonzero(covered) / covered.size),
        }
        return DeepLearningPrediction(artifact_data=artifact, metadata=metadata)

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ProcessorValidationError(
                "Conv-TasNet requires PyTorch. Install the pytorch extra first."
            ) from exc
        model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        model.eval()
        self._model = model
        self._torch = torch
        return model, torch

    def _predict_segment(self, model: Any, torch: Any, segment: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(segment[np.newaxis, ...], dtype=torch.float32, device=self.device)
        output = model(tensor)
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        output = np.asarray(output, dtype=np.float32)

        if output.ndim != 3 or output.shape[0] != 1:
            raise ProcessorValidationError(
                "Conv-TasNet TorchScript model must return shape (1, n_sources, samples), "
                f"got {tuple(output.shape)}"
            )
        if output.shape[1] < 2:
            raise ProcessorValidationError(
                f"Conv-TasNet expects at least 2 sources (clean, artifact), got {output.shape[1]}"
            )
        if output.shape[2] != segment.shape[-1]:
            raise ProcessorValidationError(
                f"Conv-TasNet output sample count {output.shape[2]} does not match input {segment.shape[-1]}"
            )
        return output[0, 1]

    def _resolve_channels(self, raw: mne.io.BaseRaw) -> list[int]:
        if self.channel_indices is not None:
            if len(self.channel_indices) == 0:
                raise ProcessorValidationError("channel_indices cannot be empty")
            invalid = [idx for idx in self.channel_indices if idx < 0 or idx >= len(raw.ch_names)]
            if invalid:
                raise ProcessorValidationError(f"channel_indices out of range: {invalid}")
            return list(self.channel_indices)
        if self.eeg_only:
            return [int(idx) for idx in mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)]
        return list(range(len(raw.ch_names)))

    def _chunk_ranges(self, n_samples: int) -> list[tuple[int, int]]:
        step = self.chunk_size_samples - self.chunk_overlap_samples
        ranges: list[tuple[int, int]] = []
        start = 0
        while start + self.chunk_size_samples <= n_samples:
            ranges.append((start, start + self.chunk_size_samples))
            start += step
        return ranges


@register_processor
class ConvTasNetCorrection(DeepLearningCorrection):
    """Pipeline processor for Conv-TasNet single-channel artifact separation."""

    name = "conv_tasnet_correction"
    description = "Channel-wise Conv-TasNet source-separation artifact correction"
    version = "0.1.0"

    requires_raw = True
    requires_triggers = False
    modifies_raw = True
    parallel_safe = False
    channel_wise = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        chunk_size_samples: int = 512,
        chunk_overlap_samples: int = 0,
        device: str = "cpu",
        eeg_only: bool = True,
        channel_indices: list[int] | None = None,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
        store_run_metadata: bool = True,
    ) -> None:
        adapter = ConvTasNetAdapter(
            checkpoint_path=checkpoint_path,
            chunk_size_samples=chunk_size_samples,
            chunk_overlap_samples=chunk_overlap_samples,
            device=device,
            eeg_only=eeg_only,
            channel_indices=channel_indices,
            demean_input=demean_input,
            remove_prediction_mean=remove_prediction_mean,
        )
        super().__init__(model=adapter, store_run_metadata=store_run_metadata)

    def validate_execution_mode(self, *, parallel: bool, channel_sequential: bool) -> None:
        if parallel:
            raise ProcessorValidationError(
                f"{self.name} loads a stateful TorchScript model and must not run in parallel mode"
            )
        super().validate_execution_mode(parallel=parallel, channel_sequential=channel_sequential)
