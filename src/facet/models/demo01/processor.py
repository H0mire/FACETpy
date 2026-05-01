"""
Trigger-context deep learning correction processors.

This module contains inference utilities for models that consume several
consecutive trigger-defined artifact epochs and predict only the center epoch.
The implementation keeps the model input length fixed while allowing variable
artifact epoch lengths in the original recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
from loguru import logger

from ...core import ProcessingContext, Processor, ProcessorValidationError, register_processor


def _resample_1d(values: np.ndarray, target_samples: int) -> np.ndarray:
    """Linearly resample one channel epoch to a fixed number of samples."""
    if values.ndim != 1:
        raise ValueError(f"Expected 1D values, got shape {values.shape}")
    if target_samples <= 0:
        raise ValueError(f"target_samples must be positive, got {target_samples}")
    if len(values) == target_samples:
        return values.astype(np.float32, copy=False)
    if len(values) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if len(values) == 1:
        return np.full(target_samples, float(values[0]), dtype=np.float32)

    source_x = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    return np.interp(target_x, source_x, values).astype(np.float32)


@register_processor
class EpochContextDeepLearningCorrection(Processor):
    """Apply a TorchScript model to consecutive trigger-defined epochs.

    The model input is shaped as ``(batch, context_epochs, 1, epoch_samples)``.
    For every valid center epoch, the processor builds a multi-epoch context,
    predicts the center artifact epoch, resamples the prediction back to the
    original center-epoch length, subtracts it from the raw data, and stores the
    estimated artifact in ``context.estimated_noise``.

    This processor is intentionally channel-wise compatible. In normal pipeline
    execution it loops over all selected channels; in ``channel_sequential`` mode
    the pipeline may pass a single-channel context, keeping peak memory bounded
    by one channel plus the TorchScript model.
    """

    name = "epoch_context_deep_learning_correction"
    description = "Deep learning correction using multi-epoch trigger context"
    version = "0.1.0"

    requires_raw = True
    requires_triggers = True
    modifies_raw = True
    parallel_safe = False
    channel_wise = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        context_epochs: int = 7,
        epoch_samples: int | None = None,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.context_epochs = context_epochs
        self.epoch_samples = epoch_samples
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.device = device
        self.channel_indices = channel_indices
        self.eeg_only = eeg_only
        self.demean_input = demean_input
        self.remove_prediction_mean = remove_prediction_mean
        self._model: Any | None = None
        self._torch: Any | None = None
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ProcessorValidationError("context_epochs must be a positive odd integer")
        if self.epoch_samples is not None and self.epoch_samples <= 0:
            raise ProcessorValidationError("epoch_samples must be positive when provided")
        if not Path(self.checkpoint_path).exists():
            raise ProcessorValidationError(f"TorchScript checkpoint does not exist: {self.checkpoint_path}")

        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < self.context_epochs + 1:
            raise ProcessorValidationError(
                f"Need at least {self.context_epochs + 1} triggers for {self.context_epochs}-epoch context, "
                f"got {len(triggers)}"
            )

    def validate_execution_mode(self, *, parallel: bool, channel_sequential: bool) -> None:
        if parallel:
            raise ProcessorValidationError(
                f"{self.name} is stateful during TorchScript inference and must not run in parallel mode"
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        data = raw._data
        triggers = np.asarray(context.get_triggers(), dtype=int)
        starts, stops, target_samples = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)

        if len(channels) == 0:
            raise ProcessorValidationError("No channels selected for epoch-context deep learning correction")

        model, torch = self._load_model()
        estimated_artifacts = np.zeros_like(data)
        radius = self.context_epochs // 2
        corrected_epochs = 0

        with torch.no_grad():
            for center_idx in range(radius, len(starts) - radius):
                context_indices = range(center_idx - radius, center_idx + radius + 1)
                center_start = starts[center_idx]
                center_stop = stops[center_idx]
                center_len = center_stop - center_start
                if center_len <= 0:
                    continue

                for ch_idx in channels:
                    epoch_stack = np.stack(
                        [
                            _resample_1d(data[ch_idx, starts[epoch_idx] : stops[epoch_idx]], target_samples)
                            for epoch_idx in context_indices
                        ],
                        axis=0,
                    )
                    prediction = self._predict_center_artifact(model, torch, epoch_stack)
                    artifact_native = _resample_1d(prediction, center_len).astype(data.dtype, copy=False)
                    data[ch_idx, center_start:center_stop] -= artifact_native
                    estimated_artifacts[ch_idx, center_start:center_stop] += artifact_native

                corrected_epochs += 1

        new_ctx = context.with_raw(raw)
        new_ctx.accumulate_noise(estimated_artifacts)
        self._store_run_metadata(new_ctx, channels, starts, stops, target_samples, corrected_epochs)
        logger.info(
            "Epoch-context DL correction complete: {} channels, {} center epochs, input epoch samples={}",
            len(channels),
            corrected_epochs,
            target_samples,
        )
        return new_ctx

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProcessorValidationError(
                "EpochContextDeepLearningCorrection requires PyTorch. Install the pytorch extra first."
            ) from exc

        model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        model.eval()
        self._model = model
        self._torch = torch
        return model, torch

    def _build_epoch_boundaries(
        self, context: ProcessingContext, triggers: np.ndarray, n_times: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
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

        lengths = stops - starts
        target_samples = self.epoch_samples or int(round(float(np.median(lengths))))
        if target_samples <= 0:
            raise ProcessorValidationError("Resolved epoch_samples must be positive")

        return starts, stops, target_samples

    def _resolve_channels(self, raw: mne.io.BaseRaw) -> list[int]:
        if self.channel_indices is not None:
            return [int(idx) for idx in self.channel_indices]
        if self.eeg_only:
            return [int(idx) for idx in mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)]
        return list(range(len(raw.ch_names)))

    def _predict_center_artifact(self, model: Any, torch: Any, epoch_stack: np.ndarray) -> np.ndarray:
        if self.demean_input:
            epoch_stack = epoch_stack - np.mean(epoch_stack, axis=-1, keepdims=True, dtype=np.float32)
        tensor = torch.as_tensor(epoch_stack[None, :, None, :], dtype=torch.float32, device=self.device)
        output = model(tensor)
        prediction = output.detach().cpu().numpy()

        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim == 2 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 1:
            raise ProcessorValidationError(
                "TorchScript model must return a single center epoch with shape (batch, 1, samples), "
                f"got {tuple(output.shape)}"
            )
        prediction = prediction.astype(np.float32, copy=False)
        if self.remove_prediction_mean:
            prediction = prediction - np.mean(prediction, dtype=np.float32)
        return prediction

    def _store_run_metadata(
        self,
        context: ProcessingContext,
        channels: list[int],
        starts: np.ndarray,
        stops: np.ndarray,
        target_samples: int,
        corrected_epochs: int,
    ) -> None:
        lengths = stops - starts
        run = {
            "checkpoint_path": self.checkpoint_path,
            "context_epochs": self.context_epochs,
            "epoch_samples": target_samples,
            "corrected_epochs": corrected_epochs,
            "skipped_edge_epochs": min(len(starts), self.context_epochs - 1),
            "channels": [context.get_raw().ch_names[idx] for idx in channels],
            "epoch_length_min": int(lengths.min()),
            "epoch_length_median": float(np.median(lengths)),
            "epoch_length_max": int(lengths.max()),
            "device": self.device,
            "demean_input": self.demean_input,
            "remove_prediction_mean": self.remove_prediction_mean,
        }
        context.metadata.custom.setdefault("epoch_context_deep_learning_runs", []).append(run)
