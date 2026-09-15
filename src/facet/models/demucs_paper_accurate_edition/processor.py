"""Inference integration for the paper-accurate time-domain Demucs model.

Mirrors ``facet.models.demucs.processor`` but adds the paper's only
inference-time trick — the *shift trick* (arXiv:1911.13254, Sec 4.4): predict
over several circular time-shifts of the input, inverse-shift, and average. The
processor registers under the GLOBALLY UNIQUE name
``demucs_paper_accurate_correction`` so it can coexist with the original
``demucs_correction``.
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


class DemucsPaperAccurateAdapter(EpochContextArtifactAdapter):
    """TorchScript adapter that flattens N trigger-defined epochs per channel.

    The model consumes a single 1D waveform of length ``context_epochs *
    epoch_samples`` per channel and returns an artifact prediction of the same
    length. The adapter slices the center epoch from the prediction and
    resamples it back to the native trigger-to-trigger length before subtraction.

    Adds the paper's test-time shift trick (Sec 4.4) via ``n_shifts``: with
    ``n_shifts > 1`` the concatenated context waveform is circularly rolled by a
    set of offsets, the model is run on each, predictions are inverse-rolled and
    averaged before the center epoch is sliced. ``n_shifts=1`` reproduces the
    original single-pass behaviour.
    """

    spec = DeepLearningModelSpec(
        name="DemucsPaperAccurateAdapter",
        architecture=DeepLearningArchitecture.AUDIO_SOURCE_SEPARATION,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        uses_triggers=True,
        description="Paper-accurate time-domain Demucs (U-Net + BiLSTM + shift trick) channel-wise artifact predictor.",
        tags=(
            "demucs",
            "paper_accurate",
            "audio_source_separation",
            "u_net",
            "lstm",
            "torchscript",
            "artifact_prediction",
        ),
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
        n_shifts: int = 1,
        shift_max_fraction: float = 0.5,
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
        self.n_shifts = max(1, int(n_shifts))
        self.shift_max_fraction = float(shift_max_fraction)
        self._model: Any | None = None
        self._torch: Any | None = None
        self.spec = DeepLearningModelSpec(
            name="DemucsPaperAccurateAdapter",
            architecture=DeepLearningArchitecture.AUDIO_SOURCE_SEPARATION,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=DeepLearningOutputType.ARTIFACT,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="torchscript",
            device_preference=device,
            description=(
                "Paper-accurate time-domain Demucs (U-Net + BiLSTM + shift trick) "
                "channel-wise gradient artifact predictor."
            ),
            tags=(
                "demucs",
                "paper_accurate",
                "audio_source_separation",
                "u_net",
                "lstm",
                "torchscript",
                "artifact_prediction",
            ),
        )
        super().__init__()

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ProcessorValidationError("context_epochs must be a positive odd integer")
        if self.epoch_samples is not None and self.epoch_samples <= 0:
            raise ProcessorValidationError("epoch_samples must be positive when provided")
        if not 0.0 < self.shift_max_fraction <= 1.0:
            raise ProcessorValidationError("shift_max_fraction must be in (0, 1]")
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
        model, torch = self._load_model()
        estimated_artifacts = np.zeros_like(data)
        radius = self.context_epochs // 2
        corrected_epochs = 0

        with torch.no_grad():
            for center_idx in range(radius, len(starts) - radius):
                center_start = starts[center_idx]
                center_stop = stops[center_idx]
                center_len = center_stop - center_start
                if center_len <= 0:
                    continue
                context_indices = range(center_idx - radius, center_idx + radius + 1)
                for ch_idx in channels:
                    epoch_stack = np.concatenate(
                        [
                            _resample_1d(data[ch_idx, starts[epoch_idx] : stops[epoch_idx]], target_samples)
                            for epoch_idx in context_indices
                        ],
                        axis=0,
                    )
                    prediction_center = self._predict_center_artifact(model, torch, epoch_stack, target_samples, radius)
                    artifact_native = _resample_1d(prediction_center, center_len).astype(data.dtype, copy=False)
                    estimated_artifacts[ch_idx, center_start:center_stop] += artifact_native
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
            "n_shifts": self.n_shifts,
        }
        return DeepLearningPrediction(artifact_data=estimated_artifacts, metadata=metadata)

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ProcessorValidationError("Demucs requires PyTorch. Install the deeplearning extra first.") from exc
        model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        model.eval()
        self._model = model
        self._torch = torch
        return model, torch

    def _shift_offsets(self, length: int) -> list[int]:
        """Deterministic, evenly-spaced circular shift offsets for the shift trick.

        The paper samples S random shifts; we use a fixed ``linspace`` so the
        correction is reproducible across runs. Offsets span up to
        ``shift_max_fraction`` of the waveform length.
        """
        if self.n_shifts <= 1 or length <= 1:
            return [0]
        max_shift = max(1, int(round(length * self.shift_max_fraction)))
        offsets = np.linspace(0, max_shift, self.n_shifts, endpoint=False)
        return [int(round(o)) % length for o in offsets]

    def _run_model(self, model: Any, torch: Any, waveform: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(waveform[None, None, :], dtype=torch.float32, device=self.device)
        output = model(tensor)
        prediction = output.detach().cpu().numpy()
        if prediction.ndim == 3 and prediction.shape[0] == 1 and prediction.shape[1] == 1:
            prediction = prediction[0, 0]
        elif prediction.ndim == 2 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 1:
            raise ProcessorValidationError(
                f"Demucs TorchScript model must return shape (batch, 1, samples), got {tuple(output.shape)}"
            )
        return prediction.astype(np.float32, copy=False)

    def _predict_center_artifact(
        self,
        model: Any,
        torch: Any,
        epoch_stack: np.ndarray,
        target_samples: int,
        radius: int,
    ) -> np.ndarray:
        if self.demean_input:
            epoch_stack = epoch_stack - np.mean(epoch_stack, dtype=np.float32)

        length = epoch_stack.shape[-1]
        offsets = self._shift_offsets(length)
        accumulated = np.zeros(length, dtype=np.float32)
        for offset in offsets:
            shifted = np.roll(epoch_stack, offset, axis=-1) if offset else epoch_stack
            prediction = self._run_model(model, torch, shifted)
            # Inverse-roll so all predictions align before averaging (Sec 4.4).
            aligned = np.roll(prediction, -offset, axis=-1) if offset else prediction
            accumulated += aligned
        prediction = accumulated / float(len(offsets))

        center_slice = prediction[radius * target_samples : (radius + 1) * target_samples]
        if self.remove_prediction_mean:
            center_slice = center_slice - np.mean(center_slice, dtype=np.float32)
        return center_slice


@register_processor
class DemucsPaperAccurateCorrection(DeepLearningCorrection):
    """Pipeline processor for paper-accurate Demucs gradient-artifact inference."""

    name = "demucs_paper_accurate_correction"
    description = "Paper-accurate time-domain Demucs (U-Net + BiLSTM + shift trick) channel-wise gradient-artifact correction"
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
        n_shifts: int = 1,
        shift_max_fraction: float = 0.5,
        store_run_metadata: bool = True,
    ) -> None:
        adapter = DemucsPaperAccurateAdapter(
            checkpoint_path=checkpoint_path,
            context_epochs=context_epochs,
            epoch_samples=epoch_samples,
            artifact_to_trigger_offset=artifact_to_trigger_offset,
            device=device,
            channel_indices=channel_indices,
            eeg_only=eeg_only,
            demean_input=demean_input,
            remove_prediction_mean=remove_prediction_mean,
            n_shifts=n_shifts,
            shift_max_fraction=shift_max_fraction,
        )
        super().__init__(model=adapter, store_run_metadata=store_run_metadata)

    def validate_execution_mode(self, *, parallel: bool, channel_sequential: bool) -> None:
        if parallel:
            raise ProcessorValidationError(
                f"{self.name} loads a stateful TorchScript model and must not run in parallel mode"
            )
        super().validate_execution_mode(parallel=parallel, channel_sequential=channel_sequential)
