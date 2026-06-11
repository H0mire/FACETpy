"""Inference integration for the paper-accurate Dual-Pathway Autoencoder (DPAE).

Mirrors the original ``facet.models.dpae.processor`` structure but:

*   defaults to the CLEAN-EEG reconstruction target (paper Sec. 2.3), returning
    ``DeepLearningPrediction(clean_data=...)``; ``target_type='artifact'`` keeps
    the original subtract-the-artifact path;
*   applies the paper's per-segment normalisation (subtract std, divide by
    max-abs) before the network and rescales the output back to native amplitude
    (Sec. 3.1), mirroring ``build_dataset`` exactly;
*   uses a NEW, globally-unique ``@register_processor`` name
    ``dpae_paper_accurate_correction`` (the original is ``dpae_correction``).
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


def _output_type_for(target_type: str) -> DeepLearningOutputType:
    return (
        DeepLearningOutputType.ARTIFACT
        if str(target_type).strip().lower() == "artifact"
        else DeepLearningOutputType.CLEAN
    )


class DPAEPaperAccurateAdapter(EpochContextArtifactAdapter):
    """TorchScript adapter running paper-accurate DPAE per channel and epoch.

    Each native trigger-to-trigger epoch is resampled to ``epoch_samples``,
    per-segment normalised (subtract std, divide by max-abs), passed through the
    network, then rescaled back and resampled to the native epoch length. With
    ``target_type='clean'`` the network output is the clean estimate; with
    ``'artifact'`` it is the artifact. The adapter assembles a full-length
    output array of the requested kind and lets ``DeepLearningCorrection`` apply
    it (clean output: the framework derives ``artifact = original - clean``).
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        epoch_samples: int = 512,
        target_type: str = "clean",
        normalize: bool = True,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.epoch_samples = int(epoch_samples)
        self.target_type = str(target_type).strip().lower()
        if self.target_type not in {"clean", "artifact"}:
            raise ValueError(f"target_type must be 'clean' or 'artifact', got '{target_type}'")
        self.normalize = bool(normalize)
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.device = device
        self.channel_indices = channel_indices
        self.eeg_only = bool(eeg_only)
        self._model: Any | None = None
        self._torch: Any | None = None

        output_type = _output_type_for(self.target_type)
        self.spec = DeepLearningModelSpec(
            name="DPAEPaperAccurateAdapter",
            architecture=DeepLearningArchitecture.AUTOENCODER,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=output_type,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="torchscript",
            device_preference=device,
            description="Paper-accurate dual-pathway autoencoder (symmetric fusion, clean target).",
            tags=("dpae", "dual_pathway_autoencoder", "paper_accurate", "torchscript"),
        )
        super().__init__()

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.epoch_samples <= 0:
            raise ProcessorValidationError("epoch_samples must be positive")
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < 2:
            raise ProcessorValidationError(
                f"DPAE requires at least 2 triggers to define one epoch, got {len(triggers)}"
            )

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        raw = context.get_raw()
        data = raw._data
        triggers = np.asarray(context.get_triggers(), dtype=int)
        starts, stops = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)
        model, torch = self._load_model()
        corrected_epochs = 0

        if self.target_type == "clean":
            # Start from a copy of the original; replace corrected epochs with
            # the predicted clean estimate. Untouched samples => artifact 0.
            clean_out = data.astype(np.float32, copy=True)
        else:
            estimated_artifacts = np.zeros_like(data)

        with torch.no_grad():
            for epoch_start, epoch_stop in zip(starts, stops, strict=False):
                epoch_len = epoch_stop - epoch_start
                if epoch_len <= 0:
                    continue
                for ch_idx in channels:
                    native_segment = data[ch_idx, epoch_start:epoch_stop]
                    resampled = _resample_1d(native_segment, self.epoch_samples)
                    prediction, std, scale = self._predict_segment(model, torch, resampled)
                    if self.target_type == "clean":
                        # Network output is normalised clean; invert the norm.
                        clean_resampled = prediction * scale + std if self.normalize else prediction
                        clean_native = _resample_1d(clean_resampled, epoch_len).astype(data.dtype, copy=False)
                        clean_out[ch_idx, epoch_start:epoch_stop] = clean_native
                    else:
                        # Network output is the normalised artifact (a diff: only
                        # the scale applies, the std shift cancels).
                        artifact_resampled = prediction * scale if self.normalize else prediction
                        artifact_native = _resample_1d(artifact_resampled, epoch_len).astype(
                            data.dtype, copy=False
                        )
                        estimated_artifacts[ch_idx, epoch_start:epoch_stop] += artifact_native
                corrected_epochs += 1

        lengths = stops - starts
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "epoch_samples": self.epoch_samples,
            "target_type": self.target_type,
            "normalize": self.normalize,
            "corrected_epochs": corrected_epochs,
            "channels": [raw.ch_names[idx] for idx in channels],
            "epoch_length_min": int(lengths.min()) if lengths.size else 0,
            "epoch_length_median": float(np.median(lengths)) if lengths.size else 0.0,
            "epoch_length_max": int(lengths.max()) if lengths.size else 0,
            "device": self.device,
        }
        if self.target_type == "clean":
            return DeepLearningPrediction(clean_data=clean_out, metadata=metadata)
        return DeepLearningPrediction(artifact_data=estimated_artifacts, metadata=metadata)

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ProcessorValidationError(
                "DualPathwayAutoencoder requires PyTorch. Install the pytorch extra first."
            ) from exc
        model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        model.eval()
        self._model = model
        self._torch = torch
        return model, torch

    def _build_epoch_boundaries(
        self, context: ProcessingContext, triggers: np.ndarray, n_times: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sfreq = context.get_sfreq()
        artifact_offset = (
            context.metadata.artifact_to_trigger_offset
            if self.artifact_to_trigger_offset is None
            else self.artifact_to_trigger_offset
        )
        offset_samples = int(round((artifact_offset or 0.0) * sfreq))
        starts = triggers[:-1] + offset_samples
        stops = triggers[1:] + offset_samples
        valid = (starts >= 0) & (stops > starts) & (stops <= n_times)
        starts = starts[valid].astype(int)
        stops = stops[valid].astype(int)
        if len(starts) == 0:
            raise ProcessorValidationError("No valid trigger epochs remain after clipping")
        return starts, stops

    def _predict_segment(
        self, model: Any, torch: Any, segment: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        std = 0.0
        scale = 1.0
        if self.normalize:
            # Paper Sec. 3.1: subtract std, divide by max-abs.
            seg = np.asarray(segment, dtype=np.float32)
            std = float(np.std(seg))
            shifted = seg - std
            scale = float(np.max(np.abs(shifted)))
            if scale < 1e-8:
                scale = 1.0
            segment = (shifted / scale).astype(np.float32, copy=False)
        else:
            segment = np.asarray(segment, dtype=np.float32)

        tensor = torch.as_tensor(segment[None, None, :], dtype=torch.float32, device=self.device)
        output = model(tensor)
        prediction = output.detach().cpu().numpy()
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim == 2 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 1:
            raise ProcessorValidationError(
                f"DPAE TorchScript model must return shape (batch, 1, samples), got {tuple(output.shape)}"
            )
        return prediction.astype(np.float32, copy=False), std, scale


@register_processor
class DPAEPaperAccurateCorrection(DeepLearningCorrection):
    """Pipeline processor for paper-accurate DPAE inference (UNIQUE name)."""

    name = "dpae_paper_accurate_correction"
    description = "Paper-accurate dual-pathway autoencoder artifact correction"
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
        epoch_samples: int = 512,
        target_type: str = "clean",
        normalize: bool = True,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        store_run_metadata: bool = True,
    ) -> None:
        adapter = DPAEPaperAccurateAdapter(
            checkpoint_path=checkpoint_path,
            epoch_samples=epoch_samples,
            target_type=target_type,
            normalize=normalize,
            artifact_to_trigger_offset=artifact_to_trigger_offset,
            device=device,
            channel_indices=channel_indices,
            eeg_only=eeg_only,
        )
        super().__init__(model=adapter, store_run_metadata=store_run_metadata)

    def validate_execution_mode(self, *, parallel: bool, channel_sequential: bool) -> None:
        if parallel:
            raise ProcessorValidationError(
                f"{self.name} loads a stateful TorchScript model and must not run in parallel mode"
            )
        super().validate_execution_mode(parallel=parallel, channel_sequential=channel_sequential)
