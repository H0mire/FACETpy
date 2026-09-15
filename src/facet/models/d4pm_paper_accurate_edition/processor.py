"""Inference integration for the paper-accurate D4PM diffusion model.

Implements the paper's Joint Posterior Sampling (Algorithm 1) when the
checkpoint is dual-branch, and falls back to the documented single-branch
data-consistency reduction otherwise. The reverse sampler uses the TRUE DDPM
ancestral update (posterior_mean_coef1/coef2 + posterior_variance), unlike the
original d4pm which used an ad-hoc re-noising step.

Imports use THREE dots: this package sits at the same depth as the original
``facet.models.d4pm`` package.
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


class D4PMPaperAccurateAdapter(EpochContextArtifactAdapter):
    """Paper-accurate D4PM diffusion adapter with joint posterior sampling.

    Loads a ``facet-train`` state-dict ``.pt`` checkpoint, reinstantiates a
    :class:`D4PMTrainingModule` with the declared architecture hyperparameters,
    and runs DDPM reverse sampling per channel-epoch:

    - ``dual_branch=True`` (paper-faithful): Joint Posterior Sampling
      (Algorithm 1). At each reverse step predict ``x0`` (clean) and ``x'0``
      (artifact), compute residual ``r = y - (x0 + x'0*lambda_snr)``, apply
      consistency ``x_hat0 = x0 + lambda_dc*r`` and
      ``x'_hat0 = x'0 + (1-lambda_dc)*r``, then run both DDPM ancestral
      reverse updates. Returns the artifact estimate.
    - ``dual_branch=False`` (cheap reduction): single artifact branch with the
      documented ``h0 += lambda_dc*(y - h0)`` data consistency.

    The ancestral update uses the exact posterior coefficients
    (Algorithm 1 lines 18-21), with no added noise on the final step.
    """

    spec = DeepLearningModelSpec(
        name="D4PMPaperAccurateAdapter",
        architecture=DeepLearningArchitecture.DIFFUSION,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
        supports_multichannel=False,
        uses_triggers=True,
        description="Paper-accurate dual-branch DDPM gradient-artifact predictor with joint posterior sampling.",
        tags=("d4pm", "diffusion", "ddpm", "dual_branch", "joint_posterior", "artifact_prediction"),
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
        n_layers: int = 3,
        embed_dim: int = 128,
        num_classes: int = 1,
        norm_first: bool = False,
        dual_branch: bool = True,
        lambda_snr: float = 1.0,
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
        self.num_classes = int(num_classes)
        self.norm_first = bool(norm_first)
        self.dual_branch = bool(dual_branch)
        self.lambda_snr = float(lambda_snr)
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
            name="D4PMPaperAccurateAdapter",
            architecture=DeepLearningArchitecture.DIFFUSION,
            runtime=DeepLearningRuntime.PYTORCH,
            output_type=DeepLearningOutputType.ARTIFACT,
            execution_granularity=DeepLearningExecutionGranularity.CHANNEL,
            supports_multichannel=False,
            uses_triggers=True,
            checkpoint_path=self.checkpoint_path,
            checkpoint_format="pt",
            device_preference=device,
            description="Paper-accurate dual-branch DDPM gradient-artifact predictor with joint posterior sampling.",
            tags=("d4pm", "diffusion", "ddpm", "dual_branch", "joint_posterior", "artifact_prediction"),
        )
        super().__init__()

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.epoch_samples <= 0:
            raise ProcessorValidationError("epoch_samples must be positive")
        if self.sample_steps < 1 or self.sample_steps > self.num_steps:
            raise ProcessorValidationError(f"sample_steps must be in [1, {self.num_steps}], got {self.sample_steps}")
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < 2:
            raise ProcessorValidationError(f"D4PM requires at least 2 triggers to define epochs, got {len(triggers)}")

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
                    artifact_model = self._sample_artifact(module, torch, resampled)
                    artifact_native = _resample_1d(artifact_model, native_len).astype(data.dtype, copy=False)
                    estimated_artifacts[ch_idx, start:stop] += artifact_native
                corrected_epochs += 1

        lengths = stops - starts
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "epoch_samples": self.epoch_samples,
            "num_steps": self.num_steps,
            "sample_steps": self.sample_steps,
            "dual_branch": self.dual_branch,
            "lambda_snr": self.lambda_snr,
            "data_consistency_weight": self.data_consistency_weight,
            "sampler": "ddpm_ancestral_joint_posterior" if self.dual_branch else "ddpm_ancestral_single_branch",
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
            raise ProcessorValidationError("D4PM requires PyTorch. Install the pytorch extra first.") from exc

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
            num_classes=self.num_classes,
            norm_first=self.norm_first,
            dual_branch=self.dual_branch,
            lambda_snr=self.lambda_snr,
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

    def _predict_x0(
        self, module: Any, predictor: Any, x_t: Any, y: Any, t_int: int, torch: Any
    ) -> Any:
        """One epsilon prediction -> x0 estimate via the closed form."""
        t_tensor = torch.tensor([t_int], dtype=torch.long, device=self.device)
        sqrt_abar = module.sqrt_alphas_cumprod[t_int]
        sqrt_one_minus = module.sqrt_one_minus_alphas_cumprod[t_int]
        noise_level = module.sqrt_alphas_cumprod[t_tensor]
        class_idx = torch.zeros(1, dtype=torch.long, device=self.device)
        pred_eps = predictor(x_t, y, noise_level, class_idx)
        x0 = (x_t - sqrt_one_minus * pred_eps) / sqrt_abar
        return x0

    def _ancestral_step(self, module: Any, x_t: Any, x0_hat: Any, t_int: int, t_prev: int, torch: Any) -> Any:
        """True DDPM ancestral update (Algorithm 1 lines 18-21).

        mu_t = coef1 * x0_hat + coef2 * x_t; add posterior noise unless this is
        the final step (t_prev <= 0).
        """
        coef1 = module.posterior_mean_coef1[t_int]
        coef2 = module.posterior_mean_coef2[t_int]
        mean = coef1 * x0_hat + coef2 * x_t
        if t_prev <= 0:
            return mean
        var = module.posterior_variance[t_int]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    def _sample_artifact(self, module: Any, torch: Any, noisy_y: np.ndarray) -> np.ndarray:
        device = self.device
        if self.seed is not None:
            torch.manual_seed(self.seed)
        y_mean = float(noisy_y.mean()) if self.demean_input else 0.0
        y = torch.as_tensor(noisy_y - y_mean, dtype=torch.float32, device=device).view(1, 1, -1)

        # Strided ancestral schedule (subsample for speed); each consecutive
        # pair (t -> t_prev) is one ancestral reverse step.
        step_indices = torch.linspace(self.num_steps - 1, 0, self.sample_steps, device=device).long().tolist()

        x_t_art = torch.randn_like(y)
        x_t_clean = torch.randn_like(y) if self.dual_branch else None
        lambda_dc = self.data_consistency_weight

        for step_idx, t_int in enumerate(step_indices):
            t_prev = step_indices[step_idx + 1] if step_idx + 1 < len(step_indices) else -1

            if self.dual_branch:
                x0_clean = self._predict_x0(module, module.predictor_clean, x_t_clean, y, t_int, torch)
                x0_art = self._predict_x0(module, module.predictor, x_t_art, y, t_int, torch)
                # Joint posterior consistency (Eq 4 / Algorithm 1).
                residual = y - (x0_clean + x0_art * self.lambda_snr)
                x0_clean_hat = x0_clean + lambda_dc * residual
                x0_art_hat = x0_art + (1.0 - lambda_dc) * residual
                x_t_clean = self._ancestral_step(module, x_t_clean, x0_clean_hat, t_int, t_prev, torch)
                x_t_art = self._ancestral_step(module, x_t_art, x0_art_hat, t_int, t_prev, torch)
                if t_prev <= 0:
                    final = x0_art_hat
                    break
            else:
                x0_art = self._predict_x0(module, module.predictor, x_t_art, y, t_int, torch)
                if lambda_dc > 0.0:
                    x0_art = x0_art + lambda_dc * (y - x0_art)
                x_t_art = self._ancestral_step(module, x_t_art, x0_art, t_int, t_prev, torch)
                if t_prev <= 0:
                    final = x0_art
                    break
        else:  # pragma: no cover - loop always breaks on final step
            final = x0_art if not self.dual_branch else x0_art_hat

        artifact = final.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if self.remove_prediction_mean:
            artifact = artifact - artifact.mean()
        return artifact


@register_processor
class D4PMPaperAccurateCorrection(DeepLearningCorrection):
    """Pipeline processor for the paper-accurate D4PM diffusion artifact predictor.

    Registered under the GLOBALLY-UNIQUE name ``d4pm_paper_accurate_correction``
    (the original is ``d4pm_correction``).
    """

    name = "d4pm_paper_accurate_correction"
    description = "Paper-accurate dual-branch DDPM gradient-artifact correction (joint posterior sampling)"
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
        n_layers: int = 3,
        embed_dim: int = 128,
        num_classes: int = 1,
        norm_first: bool = False,
        dual_branch: bool = True,
        lambda_snr: float = 1.0,
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
        adapter = D4PMPaperAccurateAdapter(
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
            num_classes=num_classes,
            norm_first=norm_first,
            dual_branch=dual_branch,
            lambda_snr=lambda_snr,
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
