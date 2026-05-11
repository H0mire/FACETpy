"""Inference integration for the D4PM single-branch conditional diffusion model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np

from facet.core import ProcessingContext, ProcessorValidationError, register_processor
from facet.correction.deep_learning import (
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
        raise ValueError(f"Expected 1D values, got shape {values.shape}")
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    if len(values) == target_samples:
        return values.astype(np.float32, copy=False)
    if len(values) == 0:
        return np.zeros(target_samples, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_samples, dtype=np.float64)
    return np.interp(target_x, source_x, values).astype(np.float32)


class D4PMArtifactDiffusionAdapter(DeepLearningModelAdapter):
    """Conditional-diffusion adapter that samples a per-channel artifact estimate.

    The adapter loads a checkpoint produced by ``facet-train`` (a state-dict
    ``.pt`` file written by :class:`PyTorchModelWrapper.save_checkpoint`).
    A ``D4PMTrainingModule`` is reinstantiated with the architecture
    hyperparameters declared on the adapter, the predictor weights are
    loaded, and inference proceeds via DDPM reverse sampling with optional
    data-consistency reinforcement.
    """

    spec = DeepLearningModelSpec(
        name="D4PMArtifactDiffusionAdapter",
        architecture=DeepLearningArchitecture.DIFFUSION,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        uses_triggers=True,
        description="Single-branch conditional DDPM gradient-artifact predictor.",
        tags=("d4pm", "diffusion", "ddpm", "artifact_prediction"),
    )

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        epoch_samples: int = 512,
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        feats: int = 64,
        d_model: int = 128,
        d_ff: int = 512,
        n_heads: int = 2,
        n_layers: int = 2,
        embed_dim: int = 128,
        sample_steps: int = 50,
        data_consistency_weight: float = 0.5,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
        seed: int = 0,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.epoch_samples = int(epoch_samples)
        self.num_steps = int(num_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.feats = int(feats)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.embed_dim = int(embed_dim)
        self.sample_steps = int(sample_steps)
        self.data_consistency_weight = float(data_consistency_weight)
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.device = device
        self.channel_indices = channel_indices
        self.eeg_only = bool(eeg_only)
        self.demean_input = bool(demean_input)
        self.remove_prediction_mean = bool(remove_prediction_mean)
        self.seed = int(seed)

        self._torch: Any | None = None
        self._module: Any | None = None
        self.spec = DeepLearningModelSpec(
            name="D4PMArtifactDiffusionAdapter",
            architecture=DeepLearningArchitecture.DIFFUSION,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=DeepLearningOutputType.ARTIFACT,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="pt",
            device_preference=device,
            description="Single-branch conditional DDPM gradient-artifact predictor.",
            tags=("d4pm", "diffusion", "ddpm", "artifact_prediction"),
        )
        super().__init__()

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.epoch_samples <= 0:
            raise ProcessorValidationError("epoch_samples must be positive")
        if self.sample_steps < 1 or self.sample_steps > self.num_steps:
            raise ProcessorValidationError(
                f"sample_steps must be in [1, {self.num_steps}], got {self.sample_steps}"
            )
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < 2:
            raise ProcessorValidationError(
                f"D4PM requires at least 2 triggers to define epochs, got {len(triggers)}"
            )

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        raw = context.get_raw()
        data = raw._data
        triggers = np.asarray(context.get_triggers(), dtype=int)
        starts, stops = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)
        module, torch = self._load_module()

        estimated_artifacts = np.zeros_like(data)
        corrected_epochs = 0

        with torch.no_grad():
            for epoch_idx in range(len(starts)):
                start = starts[epoch_idx]
                stop = stops[epoch_idx]
                native_len = stop - start
                if native_len <= 0:
                    continue
                for ch_idx in channels:
                    native = data[ch_idx, start:stop]
                    resampled = _resample_1d(native, self.epoch_samples)
                    artifact_512 = self._sample_artifact(module, torch, resampled)
                    artifact_native = _resample_1d(artifact_512, native_len).astype(
                        data.dtype, copy=False
                    )
                    estimated_artifacts[ch_idx, start:stop] += artifact_native
                corrected_epochs += 1

        lengths = stops - starts
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "epoch_samples": self.epoch_samples,
            "num_steps": self.num_steps,
            "sample_steps": self.sample_steps,
            "data_consistency_weight": self.data_consistency_weight,
            "corrected_epochs": corrected_epochs,
            "channels": [raw.ch_names[idx] for idx in channels],
            "epoch_length_min": int(lengths.min()) if len(lengths) else 0,
            "epoch_length_median": float(np.median(lengths)) if len(lengths) else 0.0,
            "epoch_length_max": int(lengths.max()) if len(lengths) else 0,
            "device": self.device,
            "demean_input": self.demean_input,
            "remove_prediction_mean": self.remove_prediction_mean,
        }
        return DeepLearningPrediction(artifact_data=estimated_artifacts, metadata=metadata)

    def _load_module(self) -> tuple[Any, Any]:
        if self._module is not None and self._torch is not None:
            return self._module, self._torch
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ProcessorValidationError(
                "D4PM requires PyTorch. Install the pytorch extra first."
            ) from exc

        from .training import D4PMTrainingModule

        module = D4PMTrainingModule(
            epoch_samples=self.epoch_samples,
            num_steps=self.num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            feats=self.feats,
            d_model=self.d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embed_dim=self.embed_dim,
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        module.load_state_dict(state_dict, strict=True)
        module.to(self.device)
        module.eval()
        self._module = module
        self._torch = torch
        return module, torch

    def _build_epoch_boundaries(
        self, context: ProcessingContext, triggers: np.ndarray, n_times: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sfreq = context.get_sfreq()
        artifact_offset = (
            context.metadata.artifact_to_trigger_offset
            if self.artifact_to_trigger_offset is None
            else self.artifact_to_trigger_offset
        )
        if artifact_offset is None:
            artifact_offset = 0.0
        offset_samples = int(round(artifact_offset * sfreq))
        starts = triggers[:-1] + offset_samples
        stops = triggers[1:] + offset_samples
        valid = (starts >= 0) & (stops > starts) & (stops <= n_times)
        starts = starts[valid].astype(int)
        stops = stops[valid].astype(int)
        if len(starts) == 0:
            raise ProcessorValidationError("No valid trigger epochs after clipping")
        return starts, stops

    def _resolve_channels(self, raw: mne.io.BaseRaw) -> list[int]:
        if self.channel_indices is not None:
            return [int(idx) for idx in self.channel_indices]
        if self.eeg_only:
            return [
                int(idx)
                for idx in mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False)
            ]
        return list(range(len(raw.ch_names)))

    def _sample_artifact(
        self, module: Any, torch: Any, noisy_y: np.ndarray
    ) -> np.ndarray:
        device = self.device
        y_mean = float(noisy_y.mean()) if self.demean_input else 0.0
        y = torch.as_tensor(
            noisy_y - y_mean, dtype=torch.float32, device=device
        ).view(1, 1, -1)

        h_t = torch.randn_like(y)

        step_indices = torch.linspace(
            self.num_steps - 1, 0, self.sample_steps, device=device
        ).long()

        for step_idx, t_int in enumerate(step_indices.tolist()):
            t_tensor = torch.tensor([t_int], dtype=torch.long, device=device)
            noise_level = module.sqrt_alphas_cumprod[t_tensor]
            pred_noise = module.predictor(h_t, y, noise_level)

            sqrt_alpha = module.sqrt_alphas_cumprod[t_int]
            sqrt_one_minus = module.sqrt_one_minus_alphas_cumprod[t_int]
            h0_pred = (h_t - sqrt_one_minus * pred_noise) / sqrt_alpha

            if self.data_consistency_weight > 0.0:
                residual = y - h0_pred
                h0_pred = h0_pred + self.data_consistency_weight * residual

            if step_idx == len(step_indices) - 1:
                h_t = h0_pred
                break

            t_prev = step_indices[step_idx + 1].item()
            sqrt_alpha_prev = module.sqrt_alphas_cumprod[t_prev]
            sqrt_one_minus_prev = module.sqrt_one_minus_alphas_cumprod[t_prev]
            h_t = sqrt_alpha_prev * h0_pred + sqrt_one_minus_prev * torch.randn_like(h_t)

        artifact = h_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if self.remove_prediction_mean:
            artifact = artifact - artifact.mean()
        return artifact


@register_processor
class D4PMArtifactCorrection(DeepLearningCorrection):
    """Pipeline processor for the D4PM conditional diffusion artifact predictor."""

    name = "d4pm_correction"
    description = "Single-branch conditional DDPM gradient-artifact correction"
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
        num_steps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        feats: int = 64,
        d_model: int = 128,
        d_ff: int = 512,
        n_heads: int = 2,
        n_layers: int = 2,
        embed_dim: int = 128,
        sample_steps: int = 50,
        data_consistency_weight: float = 0.5,
        artifact_to_trigger_offset: float | None = None,
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        demean_input: bool = True,
        remove_prediction_mean: bool = True,
        seed: int = 0,
        store_run_metadata: bool = True,
    ) -> None:
        adapter = D4PMArtifactDiffusionAdapter(
            checkpoint_path=checkpoint_path,
            epoch_samples=epoch_samples,
            num_steps=num_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            feats=feats,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            embed_dim=embed_dim,
            sample_steps=sample_steps,
            data_consistency_weight=data_consistency_weight,
            artifact_to_trigger_offset=artifact_to_trigger_offset,
            device=device,
            channel_indices=channel_indices,
            eeg_only=eeg_only,
            demean_input=demean_input,
            remove_prediction_mean=remove_prediction_mean,
            seed=seed,
        )
        super().__init__(model=adapter, store_run_metadata=store_run_metadata)

    def validate_execution_mode(self, *, parallel: bool, channel_sequential: bool) -> None:
        if parallel:
            raise ProcessorValidationError(
                f"{self.name} loads a stateful PyTorch module and must not run in parallel mode"
            )
        super().validate_execution_mode(parallel=parallel, channel_sequential=channel_sequential)
