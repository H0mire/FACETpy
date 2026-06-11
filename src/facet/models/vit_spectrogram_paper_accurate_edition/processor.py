"""Inference integration for the paper-accurate ViT/MAE spectrogram inpainter.

Mirrors ``facet.models.vit_spectrogram.processor`` exactly in its inference
contract (per-channel, 7-epoch trigger context, ``artifact = noisy - clean``)
but loads a TorchScript export of the MAE-faithful asymmetric autoencoder from
this edition and registers under a GLOBALLY UNIQUE processor name
``vit_spectrogram_paper_accurate_correction``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...core import ProcessingContext, ProcessorValidationError, register_processor
from ...correction.deep_learning import (
    DeepLearningArchitecture,
    DeepLearningCorrection,
    DeepLearningDomain,
    DeepLearningExecutionGranularity,
    DeepLearningModelSpec,
    DeepLearningOutputType,
    DeepLearningPrediction,
    DeepLearningRuntime,
    EpochContextArtifactAdapter,
    _resample_1d,
)

_TAGS = (
    "vit_spectrogram",
    "vit",
    "mae",
    "spectrogram",
    "inpainting",
    "asymmetric_autoencoder",
    "paper_accurate",
    "torchscript",
    "artifact_prediction",
)
_DESCRIPTION = (
    "Paper-accurate MAE-style asymmetric ViT spectrogram inpainter for fMRI "
    "gradient artifact removal."
)


class ViTSpectrogramMAEAdapter(EpochContextArtifactAdapter):
    """TorchScript adapter running the MAE asymmetric ViT inpainter per channel.

    Predicts the *clean* center epoch from a 7-epoch trigger-defined context
    and converts that to an artifact estimate via ``noisy - clean``. The
    inference path of the TorchScript graph is the model's ``eval()`` forward,
    which returns the time-domain center epoch (shape ``(batch, 1, samples)``);
    the MAE training-mode dict output is never exported.
    """

    spec = DeepLearningModelSpec(
        name="ViTSpectrogramMAEAdapter",
        architecture=DeepLearningArchitecture.VISION_TRANSFORMER,
        runtime=DeepLearningRuntime.PYTORCH,
        domain=DeepLearningDomain.TIME_FREQUENCY,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        uses_triggers=True,
        description=_DESCRIPTION,
        tags=_TAGS,
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
            name="ViTSpectrogramMAEAdapter",
            architecture=DeepLearningArchitecture.VISION_TRANSFORMER,
            runtime=DeepLearningRuntime.PYTORCH,
            domain=DeepLearningDomain.TIME_FREQUENCY,
            output_type=DeepLearningOutputType.ARTIFACT,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="torchscript",
            device_preference=device,
            description=_DESCRIPTION,
            tags=_TAGS,
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
                    epoch_stack = np.stack(
                        [
                            _resample_1d(data[ch_idx, starts[epoch_idx] : stops[epoch_idx]], target_samples)
                            for epoch_idx in context_indices
                        ],
                        axis=0,
                    )
                    predicted_clean_center = self._predict_center_clean(model, torch, epoch_stack)
                    noisy_center_native = data[ch_idx, center_start:center_stop].astype(np.float32, copy=True)
                    if self.demean_input:
                        noisy_center_native = noisy_center_native - noisy_center_native.mean()
                    predicted_clean_native = _resample_1d(predicted_clean_center, center_len)
                    artifact_native = (noisy_center_native - predicted_clean_native).astype(data.dtype, copy=False)
                    if self.remove_prediction_mean:
                        artifact_native = artifact_native - np.mean(artifact_native, dtype=np.float32)
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
        }
        return DeepLearningPrediction(artifact_data=estimated_artifacts, metadata=metadata)

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ProcessorValidationError(
                "ViTSpectrogramMAEAdapter requires PyTorch. Install the pytorch extra first."
            ) from exc
        model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        model.eval()
        self._model = model
        self._torch = torch
        return model, torch

    def _predict_center_clean(self, model: Any, torch: Any, epoch_stack: np.ndarray) -> np.ndarray:
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
                "TorchScript model must return shape (batch, 1, samples) for the clean center epoch; "
                f"got {tuple(output.shape)}"
            )
        return prediction.astype(np.float32, copy=False)


@register_processor
class ViTSpectrogramMAECorrection(DeepLearningCorrection):
    """Pipeline processor for paper-accurate MAE-ViT spectrogram-inpainting correction."""

    name = "vit_spectrogram_paper_accurate_correction"
    description = "Paper-accurate MAE-ViT spectrogram-inpainting artifact correction"
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
        adapter = ViTSpectrogramMAEAdapter(
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
