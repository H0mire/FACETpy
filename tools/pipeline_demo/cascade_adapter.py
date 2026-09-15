"""Pipeline adapter for the FARM-residual cascade (Demucs-MC).

Why this exists. The cascade was trained on ``noisy - template``: FARM removes
the epoch-repeatable component, the network predicts only what FARM leaves. In a
FACETpy pipeline that formulation needs no special plumbing — it needs the
correct *order*. Placed after ``FARMCorrection``, this adapter sees exactly the
signal it was trained on, because FARM has already subtracted its estimate from
the raw. Placed before FARM it would see the raw signal and produce nonsense, so
the ordering is checked rather than assumed.

Two properties of the training contract are reproduced here:

* **Trigger-aligned epochs, resampled to a fixed length.** The model consumes
  ``(1, context_epochs, n_channels, epoch_samples)``; the dataset builder wrote
  trigger-to-trigger epochs resampled to 512 samples, so inference resamples the
  same way and maps the prediction back to the native epoch length.
* **Target electrode first, then its nearest neighbours.** The builder guarantees
  index 0 is the electrode being corrected. Getting this wrong silently corrects
  the wrong channel, so the ordering is constructed explicitly from the montage
  and falls back to index neighbours only when no positions exist.

Registered as ``cascade_demucs_correction`` — a name of its own, so it can never
shadow an existing processor.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import mne
import numpy as np

from facet.core import ProcessingContext, ProcessorValidationError
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

DEFAULT_KWARGS = {
    "depth": 4, "initial_channels": 32, "kernel_size": 8, "stride": 4,
    "lstm_layers": 2, "n_heads": 4, "attention_levels": 2, "rescale": 0.1,
}


class CascadeDemucsAdapter(EpochContextArtifactAdapter):
    """Predicts the residual artifact FARM leaves behind, per target electrode."""

    spec = DeepLearningModelSpec(
        name="CascadeDemucsAdapter",
        architecture=DeepLearningArchitecture.AUDIO_SOURCE_SEPARATION,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        supports_multichannel=True,
        uses_triggers=True,
        description="FARM-residual cascade: multichannel Demucs predicting the post-FARM residual artifact.",
        tags=("demucs", "cascade", "farm_residual", "artifact_prediction"),
    )

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        model_factory: str = "facet.models.demucs_mc.training:build_model",
        model_kwargs: dict[str, Any] | None = None,
        context_epochs: int = 7,
        n_channels: int = 3,
        epoch_samples: int | None = 512,
        artifact_to_trigger_offset: float | None = None,
        edge_mode: str = "mirror",
        device: str = "cpu",
        channel_indices: list[int] | None = None,
        eeg_only: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.model_factory = model_factory
        self.model_kwargs = dict(model_kwargs or DEFAULT_KWARGS)
        self.context_epochs = int(context_epochs)
        if edge_mode not in ("mirror", "skip"):
            raise ValueError(f"edge_mode must be 'mirror' or 'skip', got {edge_mode!r}")
        self.edge_mode = edge_mode
        self.n_channels = int(n_channels)
        self.epoch_samples = None if epoch_samples is None else int(epoch_samples)
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.device = device
        self.channel_indices = channel_indices
        self.eeg_only = bool(eeg_only)
        self.batch_size = max(1, int(batch_size))
        self._model: Any | None = None
        self._torch: Any | None = None
        self.spec = DeepLearningModelSpec(
            **{**{k: getattr(type(self).spec, k) for k in (
                "name", "architecture", "runtime", "output_type", "execution_granularity",
                "supports_multichannel", "uses_triggers", "description", "tags")},
               "checkpoint_path": self.checkpoint_path,
               # The runtime validator accepts torch's own extensions; a training
               # checkpoint is a .pt holding a state dict, not a TorchScript archive.
               "checkpoint_format": "pt",
               "device_preference": device},
        )
        super().__init__()

    # ------------------------------------------------------------- validation

    def validate_context(self, context: ProcessingContext) -> None:
        super().validate_context(context)
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ProcessorValidationError("context_epochs must be a positive odd integer")
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < self.context_epochs + 1:
            raise ProcessorValidationError(
                f"Need at least {self.context_epochs + 1} triggers for a "
                f"{self.context_epochs}-epoch context, got {len(triggers)}"
            )
        # The cascade only makes sense downstream of FARM. Running it on the raw
        # signal is not a degraded result, it is a different task, so it is
        # refused rather than silently producing a number.
        history = [entry.get("processor", "") for entry in getattr(context, "history", []) or []]
        if history and not any("farm" in h.lower() or "aas" in h.lower() for h in history):
            raise ProcessorValidationError(
                "CascadeDemucsAdapter expects the FARM/AAS correction to have run first — its input "
                "is 'signal - template'. Place it after FARMCorrection in the pipeline. "
                f"History so far: {history}"
            )

    @staticmethod
    def _context_indices(centre: int, n_epochs: int, radius: int) -> list[int]:
        """Epoch indices of one context window, mirrored at the recording edges.

        Near the start there is no epoch at ``centre - 3``. Skipping those centres
        leaves the first and last few epochs **uncorrected**, which on a real
        recording is not a small edge effect: it is the full gradient artifact
        standing next to a corrected stretch, and it dominates any plot of the
        scan onset.

        Mirroring is the natural fill here because the window is symmetric about
        the centre by construction: the missing epoch at offset ``-k`` is replaced
        by the existing one at ``+k``. The artifact is quasi-periodic, so a
        neighbour on the other side is a far better stand-in than zero padding or
        a repeated edge epoch — and the centre epoch, the one actually being
        corrected, always keeps its true position in the window.

        If both sides are out of range (a recording shorter than the context),
        the index is clamped, which degrades gracefully to repeating the edge.
        """
        out: list[int] = []
        for offset in range(-radius, radius + 1):
            j = centre + offset
            if 0 <= j < n_epochs:
                out.append(j)
                continue
            mirrored = centre - offset
            out.append(mirrored if 0 <= mirrored < n_epochs
                       else int(min(max(j, 0), n_epochs - 1)))
        return out

    # ------------------------------------------------------------ neighbours

    def _neighbour_table(self, raw: mne.io.BaseRaw, channels: list[int]) -> dict[int, list[int]]:
        """Target electrode first, then its nearest neighbours by montage position.

        Falls back to index adjacency when the montage carries no positions; that
        fallback is reported in the metadata, because a neighbour set that does
        not reflect geometry changes what the model sees.
        """
        positions = None
        montage_used = "index"
        try:
            loc = np.array([raw.info["chs"][i]["loc"][:3] for i in channels], dtype=float)
            if np.isfinite(loc).all() and np.abs(loc).sum() > 0:
                positions = loc
                montage_used = "position"
        except (KeyError, IndexError, TypeError):
            positions = None

        table: dict[int, list[int]] = {}
        extra = self.n_channels - 1
        for pos, ch in enumerate(channels):
            if positions is not None:
                d = np.linalg.norm(positions - positions[pos], axis=1)
                order = [channels[j] for j in np.argsort(d) if channels[j] != ch]
            else:
                order = [channels[j] for j in range(len(channels)) if j != pos]
                order.sort(key=lambda other: abs(channels.index(other) - pos))
            neighbours = order[:extra]
            while len(neighbours) < extra:                 # tiny montages: repeat
                neighbours.append(ch)
            table[ch] = [ch, *neighbours]
        self._montage_used = montage_used
        return table

    # -------------------------------------------------------------- inference

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._torch
        try:
            import torch
        except ImportError as exc:                          # pragma: no cover
            raise ProcessorValidationError(
                "The cascade adapter requires PyTorch. Install the deeplearning extra first."
            ) from exc
        module_name, _, attr = self.model_factory.partition(":")
        factory = getattr(import_module(module_name), attr)
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        samples = self.epoch_samples or 512
        model = factory(input_shape=(self.context_epochs, self.n_channels, samples), **self.model_kwargs)
        model.load_state_dict(state)
        model = model.to(self.device).eval()
        export = getattr(model, "export_module", None)
        if callable(export):
            model = export().to(self.device).eval()
        self._model, self._torch = model, torch
        return model, torch

    def predict(self, context: ProcessingContext) -> DeepLearningPrediction:
        raw = context.get_raw()
        data = raw._data
        triggers = np.asarray(context.get_triggers(), dtype=int)
        starts, stops, target_samples = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)
        neighbours = self._neighbour_table(raw, channels)
        model, torch = self._load_model()

        estimated = np.zeros_like(data)
        radius = self.context_epochs // 2
        n_epochs = len(starts)
        centres = (range(radius, n_epochs - radius) if self.edge_mode == "skip"
                   else range(n_epochs))
        index_table = {c: self._context_indices(c, n_epochs, radius) for c in centres}
        # Pre-resample every epoch once per channel: the same epoch is read by up
        # to context_epochs different centres, so resampling per centre would
        # repeat the polyphase filter seven times for nothing.
        resampled = {
            ch: np.stack([_resample_1d(data[ch, a:b], target_samples) for a, b in zip(starts, stops)])
            for ch in channels
        }

        jobs: list[tuple[int, int]] = [(ch, centre) for centre in centres for ch in channels]
        corrected_epochs = set()
        with torch.no_grad():
            for begin in range(0, len(jobs), self.batch_size):
                batch = jobs[begin:begin + self.batch_size]
                stack = np.stack([
                    np.stack([
                        resampled[member][index_table[centre]]
                        for member in neighbours[ch]
                    ], axis=1)                              # (epochs, members, samples)
                    for ch, centre in batch
                ])
                out = model(torch.from_numpy(stack.astype(np.float32)).to(self.device))
                pred = out.detach().cpu().numpy()
                if pred.ndim == 3:
                    pred = pred[:, 0, :]
                for (ch, centre), row in zip(batch, pred):
                    lo, hi = starts[centre], stops[centre]
                    estimated[ch, lo:hi] += _resample_1d(row, hi - lo).astype(data.dtype, copy=False)
                    corrected_epochs.add(centre)

        lengths = stops - starts
        metadata = {
            "checkpoint_path": self.checkpoint_path,
            "model_factory": self.model_factory,
            "model_kwargs": self.model_kwargs,
            "context_epochs": self.context_epochs,
            "n_channels": self.n_channels,
            "epoch_samples": target_samples,
            "corrected_epochs": len(corrected_epochs),
            "n_epochs": n_epochs,
            "edge_mode": self.edge_mode,
            "skipped_edge_epochs": (min(n_epochs, self.context_epochs - 1)
                                    if self.edge_mode == "skip" else 0),
            "edge_epochs_mirrored": (0 if self.edge_mode == "skip"
                                     else sum(1 for c in centres
                                              if c < radius or c >= n_epochs - radius)),
            "channels": [raw.ch_names[i] for i in channels],
            "neighbour_selection": getattr(self, "_montage_used", "index"),
            "epoch_length_min": int(lengths.min()),
            "epoch_length_median": float(np.median(lengths)),
            "epoch_length_max": int(lengths.max()),
            "device": self.device,
            "input_contract": "signal - FARM template (run this processor after FARMCorrection)",
        }
        return DeepLearningPrediction(artifact_data=estimated, metadata=metadata)
