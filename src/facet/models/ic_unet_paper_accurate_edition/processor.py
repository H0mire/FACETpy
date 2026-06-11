"""Inference integration for the paper-accurate IC-U-Net edition.

Mirrors the original ``facet.models.ic_unet.processor`` structure but:

- the network is a **sensor-level** U-Net (no in-graph ICA in the default
  trained model), and
- the output head is driven by the trained ``output_type``: ``clean`` (the
  paper-faithful default; the network reconstructs the clean center epoch and
  FACET subtracts clean from noisy) or ``artifact`` (FACETpy-compatible; the
  network predicts the artifact directly).

The ``@register_processor`` name is the GLOBALLY-UNIQUE
``"ic_unet_paper_accurate_correction"`` (NOT the original
``"ic_unet_correction"``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...core import ProcessingContext, ProcessorValidationError, register_processor
from ...correction.deep_learning import (
    DeepLearningArchitecture,
    DeepLearningCorrection,
    DeepLearningExecutionGranularity,
    DeepLearningModelSpec,
    DeepLearningOutputType,
    DeepLearningPrediction,
    DeepLearningRuntime,
    EpochContextArtifactAdapter,
    _resample_1d,
)


def _resample_2d(values: np.ndarray, target_samples: int) -> np.ndarray:
    """Resample each row of a ``(n_channels, n_samples)`` array."""
    if values.ndim != 2:
        raise ValueError(f"Expected 2D values, got shape {values.shape}")
    if values.shape[1] == target_samples:
        return values.astype(np.float32, copy=False)
    return np.stack([_resample_1d(row, target_samples) for row in values], axis=0)


class IcUNetPaperAccurateAdapter(EpochContextArtifactAdapter):
    """TorchScript adapter for the sensor-level paper-accurate IC-U-Net.

    Runs the trained U-Net on a multichannel epoch context. The trained model
    outputs either the clean center epoch (``output_type='clean'``) or the
    artifact center epoch (``output_type='artifact'``). For the clean head we
    recover the artifact as ``noisy_center - predicted_clean`` so the rest of the
    FACET correction pipeline (which subtracts an artifact estimate) works
    unchanged, while still returning ``clean_data`` in the prediction for
    downstream consumers that prefer it.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        context_epochs: int = 7,
        epoch_samples: int = 512,
        output_type: str = "clean",
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        normalize: str = "zscore",
        remove_prediction_mean: bool = True,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.context_epochs = int(context_epochs)
        self.epoch_samples = int(epoch_samples)
        normalized_output = str(output_type).strip().lower()
        if normalized_output not in {"clean", "artifact"}:
            raise ValueError(f"output_type must be 'clean' or 'artifact', got '{output_type}'")
        self.output_type = normalized_output
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.device = device
        self.channel_indices = channel_indices
        self.eeg_only = bool(eeg_only)
        normalized_mode = str(normalize).strip().lower()
        if normalized_mode not in {"zscore", "demean", "none"}:
            raise ValueError(f"normalize must be 'zscore', 'demean' or 'none', got '{normalize}'")
        self.normalize = normalized_mode
        self.remove_prediction_mean = bool(remove_prediction_mean)
        self._model: Any | None = None
        self._torch: Any | None = None

        spec_output = (
            DeepLearningOutputType.CLEAN if self.output_type == "clean" else DeepLearningOutputType.ARTIFACT
        )
        self.spec = DeepLearningModelSpec(
            name="IcUNetPaperAccurateAdapter",
            architecture=DeepLearningArchitecture.UNET,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=spec_output,
            execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
            supports_multichannel=True,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="torchscript",
            device_preference=device,
            description="Paper-accurate sensor-level IC-U-Net (Chuang et al. 2022) for the multichannel epoch context.",
            tags=("ic_unet", "paper_accurate", "u_net", "sensor_level", "torchscript", "multichannel"),
        )
        super().__init__()

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ProcessorValidationError("context_epochs must be a positive odd integer")
        if self.epoch_samples <= 0:
            raise ProcessorValidationError("epoch_samples must be positive")
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < self.context_epochs + 1:
            raise ProcessorValidationError(
                f"Need at least {self.context_epochs + 1} triggers for {self.context_epochs}-epoch context, "
                f"got {len(triggers)}"
            )

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        raw = context.get_raw()
        data = raw._data
        triggers = np.asarray(context.get_triggers(), dtype=int)
        starts, stops = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)
        model, torch = self._load_model()
        estimated_artifacts = np.zeros_like(data)
        estimated_clean: np.ndarray | None = np.zeros_like(data) if self.output_type == "clean" else None
        radius = self.context_epochs // 2
        corrected_epochs = 0

        with torch.no_grad():
            for center_idx in range(radius, len(starts) - radius):
                center_start = starts[center_idx]
                center_stop = stops[center_idx]
                center_len = int(center_stop - center_start)
                if center_len <= 0:
                    continue

                context_indices = list(range(center_idx - radius, center_idx + radius + 1))
                noisy_full = self._build_context_input(data, channels, starts, stops, context_indices)
                artifact_center, clean_center = self._predict_center(model, torch, noisy_full)

                for local_idx, ch_idx in enumerate(channels):
                    artifact_native = _resample_1d(artifact_center[local_idx], center_len).astype(
                        data.dtype, copy=False
                    )
                    estimated_artifacts[ch_idx, center_start:center_stop] += artifact_native
                    if estimated_clean is not None:
                        clean_native = _resample_1d(clean_center[local_idx], center_len).astype(data.dtype, copy=False)
                        estimated_clean[ch_idx, center_start:center_stop] += clean_native

                corrected_epochs += 1

        lengths = stops - starts
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "context_epochs": self.context_epochs,
            "epoch_samples": self.epoch_samples,
            "output_type": self.output_type,
            "normalize": self.normalize,
            "corrected_epochs": corrected_epochs,
            "skipped_edge_epochs": min(len(starts), self.context_epochs - 1),
            "channels": [raw.ch_names[idx] for idx in channels],
            "epoch_length_min": int(lengths.min()) if len(lengths) else 0,
            "epoch_length_median": float(np.median(lengths)) if len(lengths) else 0.0,
            "epoch_length_max": int(lengths.max()) if len(lengths) else 0,
            "device": self.device,
            "remove_prediction_mean": self.remove_prediction_mean,
        }
        if estimated_clean is not None:
            return DeepLearningPrediction(
                artifact_data=estimated_artifacts,
                clean_data=estimated_clean,
                metadata=metadata,
            )
        return DeepLearningPrediction(artifact_data=estimated_artifacts, metadata=metadata)

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ProcessorValidationError(
                "IcUNetPaperAccurate requires PyTorch. Install the pytorch extra first."
            ) from exc
        model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        model.eval()
        self._model = model
        self._torch = torch
        return model, torch

    def _build_epoch_boundaries(
        self,
        context: ProcessingContext,
        triggers: np.ndarray,
        n_times: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sfreq = context.get_sfreq()
        artifact_offset = (
            context.metadata.artifact_to_trigger_offset
            if self.artifact_to_trigger_offset is None
            else self.artifact_to_trigger_offset
        )
        offset_samples = int(round(artifact_offset * sfreq))
        starts = triggers[:-1] + offset_samples
        stops = triggers[1:] + offset_samples
        valid = (starts >= 0) & (stops > starts) & (stops <= n_times)
        starts = starts[valid].astype(int)
        stops = stops[valid].astype(int)
        if len(starts) < self.context_epochs:
            raise ProcessorValidationError(
                f"Only {len(starts)} valid trigger epochs remain after clipping; need {self.context_epochs}"
            )
        return starts, stops

    def _build_context_input(
        self,
        data: np.ndarray,
        channels: list[int],
        starts: np.ndarray,
        stops: np.ndarray,
        context_indices: list[int],
    ) -> np.ndarray:
        epochs: list[np.ndarray] = []
        for epoch_idx in context_indices:
            segment = data[np.asarray(channels), starts[epoch_idx] : stops[epoch_idx]]
            epochs.append(_resample_2d(np.asarray(segment, dtype=np.float32), self.epoch_samples))
        stack = np.stack(epochs, axis=0)
        return stack.transpose(1, 0, 2).reshape(len(channels), self.context_epochs * self.epoch_samples)

    def _normalize_input(self, noisy_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the training-time normalisation; return (x, mean, std)."""
        mean = noisy_full.mean(axis=-1, keepdims=True, dtype=np.float32)
        std = noisy_full.std(axis=-1, keepdims=True, dtype=np.float32)
        if self.normalize == "none":
            return noisy_full, np.zeros_like(mean), np.ones_like(std)
        x = noisy_full - mean
        if self.normalize == "zscore":
            denom = std + np.float32(1e-8)
            return x / denom, mean, denom
        # demean
        return x, mean, np.ones_like(std)

    def _predict_center(
        self,
        model: Any,
        torch: Any,
        noisy_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (artifact_center, clean_center), both in native units."""
        center_index = self.context_epochs // 2
        center_start = center_index * self.epoch_samples
        center_stop = center_start + self.epoch_samples
        noisy_center = noisy_full[:, center_start:center_stop].astype(np.float32, copy=True)

        x, mean, denom = self._normalize_input(noisy_full)
        tensor = torch.as_tensor(x[None, ...], dtype=torch.float32, device=self.device)
        output = model(tensor)
        prediction = output.detach().cpu().numpy()
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 2:
            raise ProcessorValidationError(
                f"TorchScript model must return (batch, channels, samples), got {tuple(output.shape)}"
            )
        prediction = prediction.astype(np.float32, copy=False)

        if self.output_type == "clean":
            # Network output is the normalised clean center; denormalise.
            clean_center = prediction * denom + mean
            artifact_center = noisy_center - clean_center
        else:
            # Network output is the (normalised) artifact center; rescale by std
            # only (the artifact has no per-channel mean offset under demean/zscore).
            artifact_center = prediction * denom
            clean_center = noisy_center - artifact_center

        if self.remove_prediction_mean:
            artifact_center = artifact_center - artifact_center.mean(axis=-1, keepdims=True, dtype=np.float32)
        return artifact_center, clean_center


@register_processor
class IcUNetPaperAccurateCorrection(DeepLearningCorrection):
    """Pipeline processor for the paper-accurate sensor-level IC-U-Net."""

    name = "ic_unet_paper_accurate_correction"
    description = "Paper-accurate sensor-level IC-U-Net artifact correction (Chuang et al. 2022)"
    version = "0.1.0"

    requires_raw = True
    requires_triggers = True
    modifies_raw = True
    parallel_safe = False
    channel_wise = False

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        context_epochs: int = 7,
        epoch_samples: int = 512,
        output_type: str = "clean",
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        normalize: str = "zscore",
        remove_prediction_mean: bool = True,
        store_run_metadata: bool = True,
    ) -> None:
        adapter = IcUNetPaperAccurateAdapter(
            checkpoint_path=checkpoint_path,
            context_epochs=context_epochs,
            epoch_samples=epoch_samples,
            output_type=output_type,
            artifact_to_trigger_offset=artifact_to_trigger_offset,
            device=device,
            channel_indices=channel_indices,
            eeg_only=eeg_only,
            normalize=normalize,
            remove_prediction_mean=remove_prediction_mean,
        )
        super().__init__(model=adapter, store_run_metadata=store_run_metadata)

    def validate_execution_mode(self, *, parallel: bool, channel_sequential: bool) -> None:
        if parallel:
            raise ProcessorValidationError(
                f"{self.name} loads a stateful TorchScript model and must not run in parallel mode"
            )
        if channel_sequential:
            raise ProcessorValidationError(f"{self.name} is multichannel; channel-sequential execution is unsupported")
        super().validate_execution_mode(parallel=parallel, channel_sequential=channel_sequential)
