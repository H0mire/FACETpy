"""Demo 01 epoch-context deep-learning correction.

Demo 01 is intentionally model-specific: it builds trigger-defined multi-epoch
contexts, runs a TorchScript artifact predictor, and returns a full-length
artifact estimate. The actual correction application is delegated to the generic
:class:`facet.correction.DeepLearningCorrection` machinery.
"""

from __future__ import annotations

from copy import deepcopy
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


class Demo01EpochContextTorchScriptAdapter(EpochContextArtifactAdapter):
    """Adapter that builds Demo-01 epoch contexts from a ProcessingContext."""

    spec = DeepLearningModelSpec(
        name="Demo01EpochContextTorchScriptAdapter",
        architecture=DeepLearningArchitecture.CUSTOM,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        uses_triggers=True,
        description="Demo 01 seven-epoch context TorchScript artifact predictor.",
        tags=("demo01", "epoch_context", "torchscript", "artifact_prediction"),
    )

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

        self.spec = DeepLearningModelSpec(
            name="Demo01EpochContextTorchScriptAdapter",
            architecture=DeepLearningArchitecture.CUSTOM,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=DeepLearningOutputType.ARTIFACT,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="torchscript",
            device_preference=device,
            description="Demo 01 seven-epoch context TorchScript artifact predictor.",
            tags=("demo01", "epoch_context", "torchscript", "artifact_prediction"),
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

        if len(channels) == 0:
            raise ProcessorValidationError("No channels selected for Demo 01 epoch-context correction")

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
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProcessorValidationError(
                "Demo01EpochContextTorchScriptAdapter requires PyTorch. Install the pytorch extra first."
            ) from exc

        model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        model.eval()
        self._model = model
        self._torch = torch
        return model, torch

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
class EpochContextDeepLearningCorrection(DeepLearningCorrection):
    """Demo-01 convenience processor backed by generic DeepLearningCorrection.

    The class remains available under its previous name for closed-beta
    pipelines. Internally, model-specific context construction lives in
    ``Demo01EpochContextTorchScriptAdapter`` and correction application is
    handled by ``DeepLearningCorrection``.
    """

    name = "epoch_context_deep_learning_correction"
    description = "Demo 01 deep learning correction using multi-epoch trigger context"
    version = "0.2.0"

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
        store_run_metadata: bool = True,
        store_legacy_metadata: bool = True,
    ) -> None:
        self.store_legacy_metadata = store_legacy_metadata
        adapter = Demo01EpochContextTorchScriptAdapter(
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
                f"{self.name} is stateful during TorchScript inference and must not run in parallel mode"
            )
        super().validate_execution_mode(parallel=parallel, channel_sequential=channel_sequential)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        new_context = super().process(context)
        if self.store_legacy_metadata:
            runs = new_context.metadata.custom.get("deep_learning_runs", [])
            if runs:
                prediction_metadata = deepcopy(runs[-1].get("prediction_metadata", {}))
                new_context.metadata.custom.setdefault("epoch_context_deep_learning_runs", []).append(
                    prediction_metadata
                )
        return new_context
