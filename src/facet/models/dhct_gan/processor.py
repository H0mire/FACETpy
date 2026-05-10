"""Inference integration for the DHCT-GAN generator."""

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


def _resample_1d(values: np.ndarray, target_samples: int) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {values.shape}")
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if len(values) == target_samples:
        return values.astype(np.float32, copy=False)
    if len(values) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    return np.interp(target_x, source_x, values).astype(np.float32)


class DHCTGanAdapter(DeepLearningModelAdapter):
    """Per-channel TorchScript adapter for the DHCT-GAN artifact generator.

    The generator was trained on demeaned, single-channel epoch windows of
    fixed length (default 512 samples). At inference time we extract each
    trigger-to-trigger artifact epoch from the raw signal, resample it to the
    model-domain length, predict the artifact, and resample the prediction
    back to the native epoch length before subtraction by
    :class:`DeepLearningCorrection`.
    """

    spec = DeepLearningModelSpec(
        name="DHCTGanAdapter",
        architecture=DeepLearningArchitecture.GAN,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        uses_triggers=True,
        description="Dual-branch hybrid CNN-Transformer GAN artifact predictor (per channel).",
        tags=("dhct_gan", "gan", "transformer", "cnn", "torchscript", "artifact_prediction"),
    )

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        epoch_samples: int | None = 512,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.epoch_samples = None if epoch_samples is None else int(epoch_samples)
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.device = device
        self.channel_indices = channel_indices
        self.eeg_only = bool(eeg_only)
        self.demean_input = bool(demean_input)
        self.remove_prediction_mean = bool(remove_prediction_mean)
        self._model: Any | None = None
        self._torch: Any | None = None
        self.spec = DeepLearningModelSpec(
            name="DHCTGanAdapter",
            architecture=DeepLearningArchitecture.GAN,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=DeepLearningOutputType.ARTIFACT,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="torchscript",
            device_preference=device,
            description="Dual-branch hybrid CNN-Transformer GAN artifact predictor (per channel).",
            tags=("dhct_gan", "gan", "transformer", "cnn", "torchscript", "artifact_prediction"),
        )
        super().__init__()

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.epoch_samples is not None and self.epoch_samples <= 0:
            raise ProcessorValidationError("epoch_samples must be positive when provided")
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < 2:
            raise ProcessorValidationError(
                f"DHCT-GAN inference requires at least two triggers, got {len(triggers)}"
            )

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        raw = context.get_raw()
        data = raw._data
        triggers = np.asarray(context.get_triggers(), dtype=int)
        starts, stops, target_samples = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)
        model, torch = self._load_model()
        artifact_estimate = np.zeros_like(data)
        corrected_epochs = 0

        with torch.no_grad():
            for epoch_idx in range(len(starts)):
                start = int(starts[epoch_idx])
                stop = int(stops[epoch_idx])
                native_length = stop - start
                if native_length <= 0:
                    continue
                for ch_idx in channels:
                    segment = data[ch_idx, start:stop]
                    resampled = _resample_1d(segment, target_samples)
                    artifact_native = self._predict_segment(model, torch, resampled, native_length)
                    artifact_estimate[ch_idx, start:stop] += artifact_native.astype(data.dtype, copy=False)
                corrected_epochs += 1

        lengths = stops - starts
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "epoch_samples": target_samples,
            "corrected_epochs": corrected_epochs,
            "channels": [raw.ch_names[idx] for idx in channels],
            "epoch_length_min": int(lengths.min()),
            "epoch_length_median": float(np.median(lengths)),
            "epoch_length_max": int(lengths.max()),
            "device": self.device,
            "demean_input": self.demean_input,
            "remove_prediction_mean": self.remove_prediction_mean,
        }
        return DeepLearningPrediction(artifact_data=artifact_estimate, metadata=metadata)

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ProcessorValidationError(
                "DHCTGanAdapter requires PyTorch. Install the pytorch extra first."
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
        if len(starts) == 0:
            raise ProcessorValidationError("No valid trigger-to-trigger epochs remain after clipping")
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

    def _predict_segment(
        self, model: Any, torch: Any, segment: np.ndarray, native_length: int
    ) -> np.ndarray:
        segment = segment.astype(np.float32, copy=True)
        if self.demean_input:
            segment = segment - segment.mean()
        tensor = torch.as_tensor(segment[None, None, :], dtype=torch.float32, device=self.device)
        output = model(tensor)
        prediction = output.detach().cpu().numpy()
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim == 2 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 1:
            raise ProcessorValidationError(
                f"DHCT-GAN TorchScript model must return shape (batch, 1, samples), got {tuple(output.shape)}"
            )
        prediction = prediction.astype(np.float32, copy=False)
        if self.remove_prediction_mean:
            prediction = prediction - prediction.mean()
        return _resample_1d(prediction, native_length)


@register_processor
class DHCTGanCorrection(DeepLearningCorrection):
    """Pipeline processor for DHCT-GAN per-channel artifact correction."""

    name = "dhct_gan_correction"
    description = "Dual-branch hybrid CNN-Transformer GAN artifact correction (per channel)"
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
        epoch_samples: int | None = 512,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
        store_run_metadata: bool = True,
    ) -> None:
        adapter = DHCTGanAdapter(
            checkpoint_path=checkpoint_path,
            epoch_samples=epoch_samples,
            artifact_to_trigger_offset=artifact_to_trigger_offset,
            device=device,
            channel_indices=channel_indices,
            eeg_only=eeg_only,
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
