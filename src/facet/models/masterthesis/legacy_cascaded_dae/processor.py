"""The FACETpy 0.1.0 deep-learning corrector, inside a v2 pipeline.

The legacy corrector lives on ``feature/deeplearning`` as
``facet.frameworks.deeplearning_torch.CascadedDenoisingEstimator``: two tiny
fully-connected denoising autoencoders (8-4-8, so a **four-unit bottleneck**)
over the whole epoch flattened across all channels, applied in cascade — stage 2
is trained on stage 1's residual.

Running it here rather than through its own example is the point: the same
recording, the same pre- and post-processing, the same window, the same
diagnostics as every other arm. Otherwise "the legacy one corrected better" and
"these ones don't" are two numbers from two different measurements.

**What it was trained on, and why that decides how to read it.** The legacy
example runs AAS first and then builds the estimator from the resulting object,
so ``prepare_epochs`` pairs ``noisy`` (the original) with ``clean`` (the
*AAS-corrected* signal) and the training target is ``noisy − AAS_corrected`` —
AAS's own artifact estimate, on the very same 162 seconds it is then applied to.
The model is a distillation of AAS, not an independent corrector. Reproducing
AAS closely is what it optimises, so doing so is not evidence that it corrects
better; it is evidence that the fit converged. The Weg-A models are trained
against an independent clean signal and cannot copy the template, which is a
strictly harder problem. The two numbers do not belong in the same ranking
column without that sentence next to them.

Two deliberate differences from the legacy application path, both so this arm is
comparable to the others rather than to itself:

* Epochs are cut trigger-to-trigger with the chain's artifact offset, like every
  other adapter here, not with ``mne.Epochs(tmin, tmax)``.
* The prediction is made at the model's native epoch length and resampled back,
  because the correction stage sits on the up-sampled signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from facet.core import ProcessorValidationError
from facet.correction.deep_learning import (
    DeepLearningArchitecture,
    DeepLearningExecutionGranularity,
    DeepLearningModelSpec,
    DeepLearningOutputType,
    DeepLearningPrediction,
    DeepLearningRuntime,
    EpochContextArtifactAdapter,
    _resample_1d,
)
from facet.models.masterthesis.adapters import require_artifact


def _build_dae(input_size: int, hidden_units: list[int], dropout: float):
    """The legacy ``DenoisingAutoencoder``, rebuilt here.

    Copied rather than imported because importing it would drag the whole 0.1.0
    package onto ``sys.path``, where its top-level name ``facet`` collides with
    the installed 2.0 package. The layer list is the contract; the state dict
    loads with ``strict=True``, which is what checks the copy.
    """
    import torch.nn as nn

    return nn.ModuleDict(
        {
            "encoder": nn.Sequential(
                nn.Linear(input_size, hidden_units[0]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(hidden_units[0], hidden_units[1]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ),
            "decoder": nn.Sequential(
                nn.Linear(hidden_units[1], hidden_units[2]),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(hidden_units[2], input_size),
            ),
        }
    )


class LegacyDLAdapter(EpochContextArtifactAdapter):
    """FACETpy 0.1.0's cascaded denoising autoencoder as a v2 corrector."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 64,
        eeg_only: bool = True,
        n_channels: int = 30,
        epoch_samples: int = 294,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint).expanduser())
        self.device = device
        self.batch_size = int(batch_size)
        self.eeg_only = eeg_only
        self.context_epochs = 1
        # The base class asks for these before predict() runs; the model's own
        # epoch length comes from the checkpoint, which is not loaded yet.
        self.epoch_samples = int(epoch_samples)
        self.n_channels = int(n_channels)
        self.channel_indices = None
        self.edge_mode = "mirror"
        self.artifact_to_trigger_offset = None
        self._bundle = None
        self._models = None
        super().__init__()

    spec = DeepLearningModelSpec(
        name="LegacyDLAdapter",
        architecture=DeepLearningArchitecture.AUTOENCODER,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        supports_multichannel=True,
        uses_triggers=True,
        description="FACETpy 0.1.0 cascaded FC denoising autoencoder (feature/deeplearning).",
        tags=("legacy", "artifact_prediction"),
    )

    # ------------------------------------------------------------------ model

    def _load(self):
        import torch

        if self._models is None:
            b = torch.load(str(require_artifact(self.checkpoint_path)), map_location="cpu", weights_only=False)
            mods = []
            for key in ("stage1", "stage2"):
                m = _build_dae(b["input_size"], b["hidden_units"], b["dropout_rate"])
                sd = {k: v for k, v in b[key].items()}
                m.load_state_dict(sd, strict=True)
                mods.append(m.float().to(self.device).eval())
            self._bundle, self._models = b, mods
        return self._bundle, self._models

    def validate_context(self, context) -> None:
        super().validate_context(context)
        if not Path(self.checkpoint_path).exists():
            raise ProcessorValidationError(
                f"Legacy checkpoint is missing: {self.checkpoint_path}. Supply the recorded Phase-0 checkpoint."
            )

    # ------------------------------------------------------------- prediction

    def predict(self, context):
        import torch

        bundle, (stage1, stage2) = self._load()
        raw = context.get_raw()
        data = raw._data
        triggers = np.asarray(context.get_triggers(), dtype=int)
        starts, stops, _ = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)

        # The separately trained delivery checkpoint omits epochs_info. Its
        # recorded native shape is supplied explicitly by the adapter settings.
        dimensions = bundle.get(
            "epochs_info",
            {
                "n_channels": self.n_channels,
                "n_times": self.epoch_samples,
            },
        )
        n_ch = int(dimensions["n_channels"])
        n_t = int(dimensions["n_times"])
        if n_ch * n_t != int(bundle["input_size"]):
            raise ProcessorValidationError("Native epoch dimensions do not match the legacy checkpoint.")
        self.epoch_samples, self.n_channels = n_t, n_ch
        if len(channels) != n_ch:
            raise ProcessorValidationError(
                f"The legacy model expects {n_ch} channels; the pipeline supplies "
                f"{len(channels)}. The original channel order and input size are required."
            )

        # (n_epochs, n_channels, n_times) at the model's own epoch length.
        ep = np.stack(
            [
                np.stack([_resample_1d(data[ch, a:b], n_t) for ch in channels])
                for a, b in zip(starts, stops, strict=False)
            ]
        )
        x = (ep - bundle["input_mean"]) / (bundle["input_std"] + 1e-8)

        preds = []
        with torch.no_grad():
            for i in range(0, x.shape[0], self.batch_size):
                t = torch.as_tensor(
                    np.ascontiguousarray(x[i : i + self.batch_size], dtype=np.float32), device=self.device
                )
                flat = t.reshape(t.shape[0], -1)
                # Cascade: both stages see the same input, their outputs add.
                out = stage1["decoder"](stage1["encoder"](flat)) + stage2["decoder"](stage2["encoder"](flat))
                preds.append(out.reshape(t.shape).cpu().float().numpy())
        pred = np.concatenate(preds, axis=0)
        pred = pred * (bundle["artifact_std"] + 1e-8) + bundle["artifact_mean"]

        estimated = np.zeros_like(data)
        for e, (lo, hi) in enumerate(zip(starts, stops, strict=False)):
            for k, ch in enumerate(channels):
                estimated[ch, lo:hi] += _resample_1d(pred[e, k], hi - lo).astype(data.dtype, copy=False)

        return DeepLearningPrediction(
            artifact_data=estimated,
            metadata={
                "model_id": "legacy_dl",
                "source": "feature/deeplearning, facet.frameworks.deeplearning_torch",
                "architecture": f"Two-stage FC-DAE {bundle['hidden_units']}, flattened {n_ch}x{n_t}",
                "normalisation": "global mean and standard deviation over the dataset",
                "training_target": "noisy - AAS_corrected from the same recording",
                "caveat": "AAS distillation: trained and applied to the same 162 seconds.",
                "n_epochs": len(starts),
                "epoch_samples": n_t,
                "channels": [raw.ch_names[i] for i in channels],
            },
        )
