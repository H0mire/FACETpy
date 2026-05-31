"""Inference integration for the 7-epoch cascaded context DAE."""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.signal import resample_poly

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


def _resample_axis(arr: np.ndarray, target_samples: int, axis: int = -1) -> np.ndarray:
    """Bandlimited polyphase resampling of an N-D array along ``axis``.

    Uses ``scipy.signal.resample_poly`` (FIR polyphase filter) instead of
    linear interpolation. Two reasons this matters for the cascaded
    context DAE:

    * **Input direction (native → 512)**: at small resampling ratios the
      difference vs. ``np.interp`` is in the stopband behaviour only;
      passband content is preserved.
    * **Output direction (512 → native)**: linear interpolation has a
      triangle impulse response (``sinc^2`` in frequency) — a very steep
      lowpass that destroys phase and HF content in the predicted
      artifact. Polyphase resampling preserves both, which is critical
      because fMRI gradient artifacts have substantial energy in the
      100-1000 Hz EPI-readout band.

    Edge handling: ``resample_poly`` does *not* assume periodicity
    (unlike FFT-based ``resample``), so it does not introduce ringing at
    the epoch boundaries. The Kaiser-window FIR filter handles the edges
    by symmetric reflection internally.

    Vectorisation note: passing a 2-D or 3-D array and resampling along
    ``axis=-1`` amortises the Kaiser FIR design over all rows. For a
    single Niazy-style inference (~833 epochs × 30 channels × 8 calls)
    this is ~30× faster than calling per row.
    """
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    n = int(arr.shape[axis])
    if n == target_samples:
        return arr.astype(np.float32, copy=False)
    if n == 0:
        shape = list(arr.shape)
        shape[axis] = target_samples
        return np.zeros(shape, dtype=np.float32)

    g = gcd(target_samples, n)
    up, down = target_samples // g, n // g
    out = resample_poly(arr.astype(np.float64, copy=False), up, down, axis=axis)

    # Crop/pad so the output length is exactly target_samples regardless
    # of integer-rounding effects in resample_poly.
    current = out.shape[axis]
    if current > target_samples:
        slicer = [slice(None)] * out.ndim
        slicer[axis] = slice(0, target_samples)
        out = out[tuple(slicer)]
    elif current < target_samples:
        pad_width = [(0, 0)] * out.ndim
        pad_width[axis] = (0, target_samples - current)
        out = np.pad(out, pad_width, mode="edge")
    return out.astype(np.float32, copy=False)


def _resample_1d(values: np.ndarray, target_samples: int) -> np.ndarray:
    """1-D convenience wrapper around :func:`_resample_axis`.

    Kept for backward compatibility with any external code that imports
    ``_resample_1d`` from this module.
    """
    if values.ndim != 1:
        raise ValueError(f"Expected 1D values, got shape {values.shape}")
    return _resample_axis(values, target_samples, axis=-1)


class CascadedContextDenoisingAutoencoderAdapter(DeepLearningModelAdapter):
    """TorchScript adapter that builds 7-epoch channel-wise contexts."""

    spec = DeepLearningModelSpec(
        name="CascadedContextDenoisingAutoencoderAdapter",
        architecture=DeepLearningArchitecture.AUTOENCODER,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        uses_triggers=True,
        description="Seven-epoch channel-wise cascaded denoising autoencoder artifact predictor.",
        tags=("cascaded_context_dae", "denoising_autoencoder", "context", "torchscript", "artifact_prediction"),
    )

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        context_epochs: int = 7,
        epoch_samples: int | None = 512,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.context_epochs = int(context_epochs)
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
            name="CascadedContextDenoisingAutoencoderAdapter",
            architecture=DeepLearningArchitecture.AUTOENCODER,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=DeepLearningOutputType.ARTIFACT,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="torchscript",
            device_preference=device,
            description="Seven-epoch channel-wise cascaded denoising autoencoder artifact predictor.",
            tags=("cascaded_context_dae", "denoising_autoencoder", "context", "torchscript", "artifact_prediction"),
        )
        super().__init__()

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ProcessorValidationError("context_epochs must be a positive odd integer")
        if self.epoch_samples is not None and self.epoch_samples <= 0:
            raise ProcessorValidationError("epoch_samples must be positive when provided")
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
        starts, stops, target_samples = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)
        channels_arr = np.asarray(channels, dtype=int)
        model, torch = self._load_model()
        estimated_artifacts = np.zeros_like(data)
        radius = self.context_epochs // 2
        corrected_epochs = 0

        with torch.no_grad():
            for center_idx in range(radius, len(starts) - radius):
                center_start = int(starts[center_idx])
                center_stop = int(stops[center_idx])
                center_len = center_stop - center_start
                if center_len <= 0:
                    continue

                # 1) Gather all 7 context epochs across all channels, resample
                #    each context position once per call (batched over channels).
                context_blocks: list[np.ndarray] = []
                for epoch_idx in range(center_idx - radius, center_idx + radius + 1):
                    ep_start = int(starts[epoch_idx])
                    ep_stop = int(stops[epoch_idx])
                    block = data[channels_arr, ep_start:ep_stop]  # (n_ch, native_len)
                    context_blocks.append(_resample_axis(block, target_samples, axis=-1))

                # (n_channels, 7, target_samples)
                context_stack = np.stack(context_blocks, axis=1)

                if self.demean_input:
                    context_stack = context_stack - context_stack.mean(
                        axis=-1, keepdims=True, dtype=np.float32
                    )

                # 2) One model forward pass batched across channels.
                #    Input layout: (n_channels, 7, 1, target_samples)
                tensor = torch.as_tensor(
                    context_stack[:, :, None, :],
                    dtype=torch.float32,
                    device=self.device,
                )
                output = model(tensor)
                predictions = output.detach().cpu().numpy().astype(np.float32, copy=False)

                # Squeeze model output to (n_channels, target_samples).
                if predictions.ndim == 3 and predictions.shape[1] == 1:
                    predictions = predictions[:, 0, :]
                elif predictions.ndim == 2:
                    pass  # already (n_channels, target_samples)
                else:
                    raise ProcessorValidationError(
                        "TorchScript model output has unexpected rank for batched "
                        f"inference: got shape {tuple(output.shape)}"
                    )

                if self.remove_prediction_mean:
                    predictions = predictions - predictions.mean(
                        axis=-1, keepdims=True, dtype=np.float32
                    )

                # 3) Batched output resample (n_channels, target_samples) ->
                #    (n_channels, center_len).
                artifact_native = _resample_axis(
                    predictions, center_len, axis=-1
                ).astype(data.dtype, copy=False)

                # 4) Scatter into the output buffer.
                estimated_artifacts[channels_arr, center_start:center_stop] += artifact_native

                corrected_epochs += 1

        lengths = stops - starts
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "context_epochs": self.context_epochs,
            "epoch_samples": target_samples,
            "corrected_epochs": corrected_epochs,
            "skipped_edge_epochs": min(len(starts), self.context_epochs - 1),
            "channels": [raw.ch_names[idx] for idx in channels],
            "epoch_length_min": int(lengths.min()),
            "epoch_length_median": float(np.median(lengths)),
            "epoch_length_max": int(lengths.max()),
            "device": self.device,
            "demean_input": self.demean_input,
            "remove_prediction_mean": self.remove_prediction_mean,
        }
        return DeepLearningPrediction(artifact_data=estimated_artifacts, metadata=metadata)

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ProcessorValidationError(
                "CascadedContextDenoisingAutoencoder requires PyTorch. Install the pytorch extra first."
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


@register_processor
class CascadedContextDenoisingAutoencoderCorrection(DeepLearningCorrection):
    """Pipeline processor for 7-epoch cascaded context DAE inference."""

    name = "cascaded_context_dae_correction"
    description = "Seven-epoch channel-wise cascaded denoising-autoencoder artifact correction"
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
        epoch_samples: int | None = 512,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
        store_run_metadata: bool = True,
    ) -> None:
        adapter = CascadedContextDenoisingAutoencoderAdapter(
            checkpoint_path=checkpoint_path,
            context_epochs=context_epochs,
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
