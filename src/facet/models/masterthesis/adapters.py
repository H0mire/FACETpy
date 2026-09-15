"""Recorded input packing and reconstruction for thesis model families.

Models and artifacts are explicit inputs. This module does not read the thesis
catalog or search for a repository checkout. The canonical input array has shape
(batch, context epochs, channels, samples), in volts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class PackingSpec:
    """How one model wants its input, and what its output means."""

    model_id: str
    family: str
    context: str  # "single" | "stack"
    packing: str  # see predict_from_context
    demean: str  # "per_segment" | "per_epoch" | "single_mean" | "per_channel" | "none"
    output: str  # "artifact" | "clean" | "sources"
    multichannel: bool = False
    slice_centre: bool = False
    remove_prediction_dc: bool = False
    loader: str = "torchscript"  # "torchscript" | "python_source" | "d4pm"
    sampler: str = ""  # "" = plain forward; "d4pm" = DDPM reverse loop
    batch_size: int = 32
    note: str = ""


FAMILY_SPECS: dict[str, PackingSpec] = {
    "dpae": PackingSpec(
        "dpae",
        "Discriminative",
        "single",
        "b1s",
        "per_segment",
        "artifact",
        remove_prediction_dc=True,
        batch_size=128,
        note="(B,1,S) in, (B,1,S) artifact out.",
    ),
    "cascaded_dae": PackingSpec(
        "cascaded_dae",
        "Autoencoder (cascaded MLP)",
        "single",
        "b1s",
        "per_segment",
        "artifact",
        remove_prediction_dc=True,
        batch_size=256,
        note="Identical contract to DPAE.",
    ),
    "dhct_gan": PackingSpec(
        "dhct_gan",
        "GAN (single-epoch)",
        "single",
        "b1s",
        "per_segment",
        "artifact",
        remove_prediction_dc=True,
        batch_size=128,
        note="Single epoch, no context — kept for completeness.",
    ),
    "denoise_mamba": PackingSpec(
        "denoise_mamba",
        "SSM",
        "single",
        "b1s",
        "per_segment",
        "artifact",
        remove_prediction_dc=True,
        loader="python_source",
        batch_size=128,
        note="TorchScript has a CUDA device baked into the SSM scan; rebuilt from source.",
    ),
    "conv_tasnet": PackingSpec(
        "conv_tasnet",
        "Audio (TCN)",
        "single",
        "b1s",
        "per_segment",
        "sources",
        batch_size=64,
        note="(B,n_sources,S) out; source 1 is the artifact.",
    ),
    "d4pm": PackingSpec(
        "d4pm",
        "Diffusion",
        "single",
        "b1s",
        "per_segment",
        "artifact",
        loader="d4pm",
        sampler="d4pm",
        batch_size=64,
        note="DDPM reverse loop; the shipped .ts is a stub. Slow.",
    ),
    "sepformer": PackingSpec(
        "sepformer", "Audio (Transformer)", "stack", "bt1s", "per_epoch", "artifact", batch_size=32
    ),
    "nested_gan": PackingSpec("nested_gan", "GAN (TF+Time)", "stack", "bt1s", "per_epoch", "artifact", batch_size=32),
    "cascaded_context_dae": PackingSpec(
        "cascaded_context_dae", "Autoencoder (context MLP)", "stack", "bt1s", "per_epoch", "artifact", batch_size=128
    ),
    "vit_spectrogram": PackingSpec(
        "vit_spectrogram",
        "Vision (MAE)",
        "stack",
        "bt1s",
        "per_epoch",
        "clean",
        batch_size=32,
        note="Predicts the CLEAN centre epoch, not the artifact.",
    ),
    "dhct_gan_v2": PackingSpec(
        "dhct_gan_v2",
        "GAN (hybrid CNN+Transformer)",
        "stack",
        "bts",
        "per_epoch",
        "artifact",
        batch_size=64,
        note="Flat (B,T,S) packing, not (B,T,1,S).",
    ),
    "demucs": PackingSpec(
        "demucs",
        "Audio (U-Net+LSTM)",
        "stack",
        "b1ts",
        "single_mean",
        "artifact",
        slice_centre=True,
        batch_size=16,
        note="Seven epochs concatenated; one mean over the whole stack; centre sliced.",
    ),
    "ic_unet": PackingSpec(
        "ic_unet",
        "Discriminative + ICA",
        "stack",
        "bcts",
        "per_channel",
        "artifact",
        multichannel=True,
        slice_centre=True,
        batch_size=8,
        note="All channels at once, epochs concatenated along time.",
    ),
    "st_gnn": PackingSpec(
        "st_gnn",
        "Graph (GNN)",
        "stack",
        "btcs",
        "per_epoch",
        "artifact",
        multichannel=True,
        batch_size=8,
        note="Full multichannel context; channel order is load-bearing.",
    ),
}


#: The deployment editions, derived from the fourteen above rather than written
#: out again. Two fields change and both changes are structural, not cosmetic:
#:
#: * ``demean="none"`` — the deployment model z-scores and demeans its *own*
#:   input, so demeaning again here would remove a mean that is already zero and,
#:   for the ``single_mean`` families, remove the wrong one.
#: * ``remove_prediction_dc=False`` — every predicted epoch already leaves the
#:   model with zero mean. Subtracting it a second time is a no-op, but leaving
#:   the flag set would hide a regression in the model if the guarantee ever broke.
DEPLOYMENT_SPECS: dict[str, PackingSpec] = {
    f"{model_id}_deployment": PackingSpec(
        model_id=f"{model_id}_deployment",
        family=f"{spec.family} (Deployment)",
        context=spec.context,
        packing=spec.packing,
        demean="none",
        output="artifact",  # every edition returns the artifact
        multichannel=spec.multichannel,
        slice_centre=False,  # the model slices its own centre
        remove_prediction_dc=False,
        loader="torchscript",
        sampler="",
        batch_size=spec.batch_size,
        note=f"Deployment edition of {model_id}: same network, recovered-clean objective.",
    )
    for model_id, spec in FAMILY_SPECS.items()
    if model_id != "d4pm"  # the diffusion sampler is not wired here
}


def _demean(arr: np.ndarray, mode: str) -> np.ndarray:
    """Remove the mean over the axis the model was trained with.

    Separated and named because every mode below is a silent failure if wrong:
    the prediction stays plausible and simply sits at the wrong offset.
    """
    if mode == "none":
        return arr
    if mode in ("per_segment", "per_epoch", "per_channel"):
        return arr - arr.mean(axis=-1, keepdims=True)
    if mode == "single_mean":
        return arr - arr.mean(axis=(-2, -1), keepdims=True)
    raise ValueError(f"unknown demean mode {mode!r}")


def _forward(model, batch: np.ndarray, device: str) -> np.ndarray:
    """One forward pass, always in float32.

    MPS has no float64 at all, and the raw MNE data is float64, so the cast is
    not an optimisation — without it the whole adapter is CPU-only.
    """
    import torch

    x = torch.as_tensor(np.ascontiguousarray(batch, dtype=np.float32), device=device)
    with torch.no_grad():
        out = model(x)
    return out.detach().cpu().float().numpy()


def _batched(model, arr: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    return np.concatenate(
        [_forward(model, arr[i : i + batch_size], device) for i in range(0, arr.shape[0], batch_size)], axis=0
    )


#: D4PM sampler settings, taken from ``tools/evaluation/eval_unified_holdout.py:infer_d4pm``
#: so the pipeline and the holdout evaluation run the *same* reverse process.
D4PM_SAMPLE_STEPS = 50
D4PM_DATA_CONSISTENCY_WEIGHT = 0.5
D4PM_SEED = 0


def _d4pm_sample(module, arr: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    """DDPM reverse loop for D4PM.

    Every other family is a single forward pass, so :func:`_batched` suffices.
    D4PM is not: its ``forward`` is the *training* objective (it returns
    ``[pred_noise, noise]`` for a noised input), and calling it the way the other
    families are called does not produce an artifact estimate at all. The reverse
    loop below is therefore not an optimisation but the model's actual inference.

    It is a line-for-line copy of the holdout evaluator's sampler, including the
    seed, so the two paths are comparable rather than merely similar.
    """
    import torch

    torch.manual_seed(D4PM_SEED)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, arr.shape[0], batch_size):
            noisy_y = torch.as_tensor(
                np.ascontiguousarray(arr[start : start + batch_size], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            h_t = torch.randn_like(noisy_y)
            step_indices = torch.linspace(module.num_steps - 1, 0, D4PM_SAMPLE_STEPS, device=device).long()
            for step_idx, t_int in enumerate(step_indices.tolist()):
                t_tensor = torch.full((noisy_y.shape[0],), t_int, dtype=torch.long, device=device)
                noise_level = module.sqrt_alphas_cumprod[t_tensor]
                pred_noise = module.predictor(h_t, noisy_y, noise_level)
                sqrt_alpha = module.sqrt_alphas_cumprod[t_int]
                sqrt_one_minus = module.sqrt_one_minus_alphas_cumprod[t_int]
                h0_pred = (h_t - sqrt_one_minus * pred_noise) / sqrt_alpha
                if D4PM_DATA_CONSISTENCY_WEIGHT > 0.0:
                    h0_pred = h0_pred + D4PM_DATA_CONSISTENCY_WEIGHT * (noisy_y - h0_pred)
                if step_idx == len(step_indices) - 1:
                    h_t = h0_pred
                    break
                t_prev = step_indices[step_idx + 1].item()
                h_t = module.sqrt_alphas_cumprod[t_prev] * h0_pred + module.sqrt_one_minus_alphas_cumprod[
                    t_prev
                ] * torch.randn_like(h_t)
            out.append(h_t.detach().cpu().float().numpy())
    return np.concatenate(out, axis=0)


def predict_from_context(
    spec: PackingSpec,
    model: Any,
    ctx: np.ndarray,
    *,
    device: str,
    batch_size: int | None = None,
) -> np.ndarray:
    """Artifact estimate of the centre epoch for every channel.

    ``ctx`` is ``(N, T, C, S)`` — N windows, T context epochs, C channels,
    S samples per epoch, in the model's native units. Returns ``(N, C, S)``.

    This function is the contract. Both the pipeline adapter and the
    verification script call it, so the two can never disagree.
    """
    if spec.sampler and spec.multichannel:
        raise NotImplementedError(f"{spec.model_id}: samplers are only wired for the single-channel path")
    n, t, c, s = ctx.shape
    centre = t // 2
    bs = batch_size or spec.batch_size
    ctx = ctx.astype(np.float32, copy=True)

    if spec.multichannel:
        if spec.packing == "bcts":  # (N, C, T*S)
            x = _demean(ctx.transpose(0, 2, 1, 3).reshape(n, c, t * s), spec.demean)
        elif spec.packing == "btcs":  # (N, T, C, S)
            x = _demean(ctx, spec.demean)
        elif spec.packing == "bcs":  # (N, C, S), centre epoch only
            # Every electrode, one epoch. The contract the FACETpy 0.1.0 DAE had,
            # and the one the single-epoch families were missing entirely.
            x = _demean(ctx[:, centre], spec.demean)
        else:
            raise ValueError(f"{spec.model_id}: packing {spec.packing!r} is not multichannel")
        out = _batched(model, x, bs, device)
        if out.ndim != 3 or out.shape[1] != c:
            raise RuntimeError(f"{spec.model_id}: unexpected output {out.shape}")
        if out.shape[2] == t * s:
            out = out[:, :, centre * s : (centre + 1) * s]
        elif out.shape[2] != s:
            raise RuntimeError(f"{spec.model_id}: time dim {out.shape[2]} is neither {s} nor {t * s}")
        if spec.remove_prediction_dc:
            # The multichannel branch used to skip this, so dc_mode had no effect
            # at all on ic_unet and st_gnn — the two models whose output is 96 %
            # constant offset. Silent, because nothing asserted the flag was read.
            out = out - out.mean(axis=-1, keepdims=True)
        pred = out
    else:
        if spec.context == "single":
            flat = ctx[:, centre].transpose(0, 1, 2).reshape(n * c, 1, s)
            x = _demean(flat, spec.demean)
        else:
            stack = ctx.transpose(0, 2, 1, 3).reshape(n * c, t, s)  # (N*C, T, S)
            if spec.packing == "bt1s":
                x = _demean(stack[:, :, None, :], spec.demean)  # (N*C, T, 1, S)
            elif spec.packing == "bts":
                x = _demean(stack, spec.demean)
            elif spec.packing == "b1ts":
                x = _demean(stack, spec.demean).reshape(n * c, 1, t * s)
            else:
                raise ValueError(f"{spec.model_id}: unknown packing {spec.packing!r}")
        out = _d4pm_sample(model, x, bs, device) if spec.sampler == "d4pm" else _batched(model, x, bs, device)
        if spec.output == "sources":
            if out.ndim != 3 or out.shape[1] < 2:
                raise RuntimeError(f"{spec.model_id}: expected sources, got {out.shape}")
            out = out[:, 1, :]
        if out.ndim == 4:
            out = out.squeeze(2)
        if out.ndim == 3 and out.shape[1] == 1:
            out = out.squeeze(1)
        if out.ndim != 2:
            raise RuntimeError(f"{spec.model_id}: unexpected output {out.shape}")
        if spec.slice_centre and out.shape[-1] == t * s:
            out = out[:, centre * s : (centre + 1) * s]
        out = out[:, -s:]
        if spec.remove_prediction_dc:
            out = out - out.mean(axis=-1, keepdims=True)
        pred = out.reshape(n, c, s)

    if spec.output == "clean":
        # The model predicts the clean centre epoch in demeaned space, so the
        # artifact is what is left of the demeaned noisy centre.
        noisy_centre = ctx[:, centre].transpose(0, 1, 2)  # (N, C, S)
        noisy_centre = noisy_centre - noisy_centre.mean(axis=-1, keepdims=True)
        pred = noisy_centre - pred
    return pred.astype(np.float32, copy=False)


# ------------------------------------------------------------------ loading


def require_artifact(path: str | Path) -> Path:
    """Reject missing weights and Git LFS pointers before backend loading."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {path}")
    with path.open("rb") as stream:
        header = stream.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise FileNotFoundError(
            f"{path} is a Git LFS pointer. Materialize this artifact with Git LFS "
            "or supply its local binary file before loading the model."
        )
    if not header:
        raise ValueError(f"Model artifact is empty: {path}")
    return path


def load_model(
    model_id: str,
    device: str = "cpu",
    *,
    checkpoint: str | Path,
    model_factory=None,
    model_kwargs: dict[str, Any] | None = None,
):
    """Load an explicit export or a strict state dictionary.

    TorchScript exports load on CPU before conversion to the requested device.
    State dictionaries require their recorded factory and arguments. The two
    Phase-1 source loaders retain their original, fixed architecture settings.
    Callers select CPU-compatible exports explicitly; no alternative is searched.
    """
    import importlib

    import torch

    path = require_artifact(checkpoint)
    if path.suffix == ".ts":
        model = torch.jit.load(str(path), map_location="cpu")
        return model.float().to(device).eval()
    kwargs = dict(model_kwargs or {})
    if model_factory is None:
        if model_id == "denoise_mamba":
            from facet.models.masterthesis.denoise_mamba.training import build_model

            model_factory = build_model
            kwargs = (
                dict(
                    epoch_samples=512,
                    d_model=64,
                    d_state=16,
                    expand=2,
                    d_conv=4,
                    n_blocks=4,
                    dropout=0.1,
                    input_kernel_size=7,
                )
                | kwargs
            )
        elif model_id == "d4pm":
            from facet.models.masterthesis.d4pm.training import D4PMTrainingModule

            model_factory = D4PMTrainingModule
            kwargs = (
                dict(
                    epoch_samples=512,
                    num_steps=200,
                    beta_start=1e-4,
                    beta_end=0.02,
                    feats=64,
                    d_model=128,
                    d_ff=512,
                    n_heads=2,
                    n_layers=2,
                    embed_dim=128,
                )
                | kwargs
            )
        else:
            raise ValueError("State dictionaries require model_factory and the recorded model_kwargs.")
    if isinstance(model_factory, str):
        module, separator, name = model_factory.partition(":")
        if not separator:
            module, _, name = model_factory.rpartition(".")
        model_factory = getattr(importlib.import_module(module), name)
    model = model_factory(**kwargs)
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    model.load_state_dict(state, strict=True)
    export_module = getattr(model, "export_module", None)
    if callable(export_module) and model_id != "d4pm":
        model = export_module()
    return model.float().to(device).eval()


# --------------------------------------------------------------- pipeline use

#: Channel order of the training dataset. ST-GNN and IC-U-Net consume all
#: channels at once, and ST-GNN's graph is built from this order — feeding a
#: differently ordered montage produces a plausible-looking but wrong result, so
#: the order is asserted rather than assumed.
TRAINING_CHANNEL_ORDER = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
    "AF4",
    "AF3",
    "FC2",
    "FC1",
    "CP1",
    "CP2",
    "PO3",
    "PO4",
    "FC6",
    "FC5",
    "CP5",
]


class FamilyAdapter(EpochContextArtifactAdapter):
    """Runs any registered model family inside a FACETpy pipeline.

    The epoch handling is the same for all of them — trigger-to-trigger epochs
    resampled to the training length, prediction mapped back — so only the
    packing differs, and that comes from :data:`FAMILY_SPECS`.

    Two things are checked rather than assumed:

    * **Channel order.** Multichannel models were trained on one montage order.
      A mismatch is refused, because the failure is silent.
    * **Edge epochs.** Missing context epochs are mirrored about the centre, the
      same rule the cascade adapter uses, so no epoch is left uncorrected.
    """

    def __init__(
        self,
        model_id: str,
        *,
        checkpoint: str | Path | None = None,
        model_factory=None,
        model_kwargs: dict[str, Any] | None = None,
        packing_spec: PackingSpec | None = None,
        device: str = "cpu",
        batch_size: int | None = None,
        context_epochs: int = 7,
        epoch_samples: int = 512,
        edge_mode: str = "mirror",
        eeg_only: bool = True,
        dc_mode: str = "as_evaluated",
    ) -> None:
        """``dc_mode`` decides what happens to the prediction's constant offset.

        ``as_evaluated`` (default)
            Exactly what the model's own inference function in
            ``tools/evaluation/eval_unified_holdout.py`` does — four of the fourteen remove
            the prediction's mean, ten do not. This is the setting the adapters
            are verified bit-identical under, and it must stay the default:
            without it the family comparison stops being about the models.
        ``reconcile``
            Remove the prediction's mean for every model. Each model is trained
            on demeaned segments, so the constant offset of its prediction
            carries no information; subtracting it anyway shifts each epoch's
            baseline and the reassembled recording steps at every seam. Whether
            that offset is removed is an *evaluation* choice inherited from the
            original per-model scripts, not a property of the architecture — so
            it belongs in a named mode rather than in the packing table.

            Measured: it fixes the offset (nested_gan 253 -> 50 µV residual) but
            makes the *epoch-periodic* residual slightly worse, because throwing
            the offset away also stops the correction from removing the segment's
            real mean. Which is what the third mode is for.
        ``segment_mean``
            The one that follows from the training contract. The models are fed
            demeaned segments and fitted against demeaned targets, so a
            prediction estimates ``artifact - mean(artifact)``. The EEG is
            high-passed, so ``mean(clean) ~ 0`` and therefore
            ``mean(artifact) ~ mean(noisy)``. The artifact estimate that belongs
            in the recording is then ``(pred - mean(pred)) + mean(noisy_segment)``:
            the model's spurious offset goes, the segment's real offset is still
            removed, and no seam is created.
        """
        registry = {**FAMILY_SPECS, **DEPLOYMENT_SPECS}
        if model_id not in registry:
            raise ValueError(f"unknown model family {model_id!r}; known: {sorted(registry)}")
        if dc_mode not in ("as_evaluated", "reconcile", "segment_mean"):
            raise ValueError(f"unknown dc_mode {dc_mode!r}")
        self.dc_mode = dc_mode
        self.packing = packing_spec or registry[model_id]
        if dc_mode in ("reconcile", "segment_mean"):
            import dataclasses

            self.packing = dataclasses.replace(self.packing, remove_prediction_dc=True)
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.model_factory = model_factory
        self.model_kwargs = dict(model_kwargs or {})
        self.device = device
        self.batch_size = batch_size or self.packing.batch_size
        self.context_epochs = int(context_epochs)
        self.epoch_samples = int(epoch_samples)
        self.edge_mode = edge_mode
        self.eeg_only = eeg_only
        # None means "take the offset the Loader recorded in the context", which
        # is what every model was trained with (-0.005 s). Hard-coding 0.0 here
        # would silently undo the offset the chain sets.
        self.artifact_to_trigger_offset = None
        self.n_channels = len(TRAINING_CHANNEL_ORDER)
        self.channel_indices = None
        self._model = None
        super().__init__()

    spec = DeepLearningModelSpec(
        name="FamilyAdapter",
        architecture=DeepLearningArchitecture.AUDIO_SOURCE_SEPARATION,
        runtime=DeepLearningRuntime.PYTORCH,
        output_type=DeepLearningOutputType.ARTIFACT,
        execution_granularity=DeepLearningExecutionGranularity.MULTICHANNEL,
        supports_multichannel=True,
        uses_triggers=True,
        description="Any evaluated model family, run inside a FACETpy pipeline.",
        tags=("family", "artifact_prediction"),
    )

    # ------------------------------------------------------------- validation

    def validate_context(self, context) -> None:
        super().validate_context(context)
        triggers = np.asarray(context.get_triggers(), dtype=int)
        if len(triggers) < self.context_epochs + 1:
            raise ProcessorValidationError(
                f"{self.model_id}: need at least {self.context_epochs + 1} triggers, got {len(triggers)}"
            )
        if self.packing.multichannel:
            raw = context.get_raw()
            names = [raw.ch_names[i] for i in self._resolve_channels(raw)]
            if names != TRAINING_CHANNEL_ORDER:
                raise ProcessorValidationError(
                    f"{self.model_id} consumes all channels at once and was trained on a "
                    f"fixed montage order. A different order silently corrects the wrong "
                    f"channels.\nExpected: {TRAINING_CHANNEL_ORDER}\nGot:      {names}"
                )

    # ------------------------------------------------------------------ model

    def _load_model(self):
        import torch

        if self._model is None:
            if self.checkpoint is None:
                raise ValueError("Provide an explicit checkpoint path to FamilyAdapter.")
            self._model = load_model(
                self.model_id,
                self.device,
                checkpoint=self.checkpoint,
                model_factory=self.model_factory,
                model_kwargs=self.model_kwargs,
            )
        return self._model, torch

    # ------------------------------------------------------------ prediction

    @staticmethod
    def _context_indices(centre: int, n_epochs: int, radius: int) -> list[int]:
        """Context epoch indices, mirrored about the centre at the edges."""
        out: list[int] = []
        for offset in range(-radius, radius + 1):
            j = centre + offset
            if 0 <= j < n_epochs:
                out.append(j)
                continue
            mirrored = centre - offset
            out.append(mirrored if 0 <= mirrored < n_epochs else int(min(max(j, 0), n_epochs - 1)))
        return out

    def predict(self, context):
        raw = context.get_raw()
        data = raw._data
        triggers = np.asarray(context.get_triggers(), dtype=int)
        starts, stops, target_samples = self._build_epoch_boundaries(context, triggers, raw.n_times)
        channels = self._resolve_channels(raw)
        model, _ = self._load_model()

        radius = self.context_epochs // 2
        n_epochs = len(starts)
        # Resample every epoch once per channel — the same epoch is read by up to
        # context_epochs different centres.
        resampled = np.stack(
            [
                np.stack([_resample_1d(data[ch, a:b], target_samples) for a, b in zip(starts, stops, strict=False)])
                for ch in channels
            ],
            axis=1,
        )  # (n_epochs, n_channels, samples)

        estimated = np.zeros_like(data)
        index_table = [self._context_indices(c, n_epochs, radius) for c in range(n_epochs)]
        # Batch over centres: the model already batches internally over channels.
        stride = max(1, self.batch_size // max(1, len(channels))) if not self.packing.multichannel else self.batch_size
        for begin in range(0, n_epochs, stride):
            block = list(range(begin, min(begin + stride, n_epochs)))
            ctx = np.stack([resampled[index_table[c]] for c in block])  # (B, T, C, S)
            pred = predict_from_context(self.packing, model, ctx, device=self.device, batch_size=self.batch_size)
            for row, centre in enumerate(block):
                lo, hi = starts[centre], stops[centre]
                for k, ch in enumerate(channels):
                    piece = _resample_1d(pred[row, k], hi - lo).astype(data.dtype, copy=False)
                    if self.dc_mode == "segment_mean":
                        piece = piece + data[ch, lo:hi].mean()
                    estimated[ch, lo:hi] += piece

        metadata = {
            "model_id": self.model_id,
            "family": self.packing.family,
            "packing": self.packing.packing,
            "demean": self.packing.demean,
            "output_convention": self.packing.output,
            "multichannel": self.packing.multichannel,
            "context_epochs": self.context_epochs,
            "epoch_samples": target_samples,
            "n_epochs": n_epochs,
            "edge_mode": self.edge_mode,
            "dc_mode": self.dc_mode,
            "channels": [raw.ch_names[i] for i in channels],
            "contract_verified_against": "tools/evaluation/eval_unified_holdout.py:INFERENCE_FUNCS (bit-identical, "
            "output/model_evaluations/family_adapters/adapter_verification.json)",
        }
        return DeepLearningPrediction(artifact_data=estimated, metadata=metadata)
