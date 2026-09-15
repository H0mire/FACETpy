"""D4PM deployment edition — the paper's ε-objective, scored on the waveform too.

**What the base edition does on a real recording.** 94.57 µV residual gradient
artifact against FARM's 4.62 and 105.9x FARM's power in the EEG band. The
uncorrected arm leaves 259 µV, so D4PM removed some of the artifact and added far
more distortion than it removed — worse than doing nothing on the band that
matters.

**Why this edition could not simply swap the loss.** Every other family maps
input to signal, so replacing MSE-on-the-artifact with the recovered-clean
objective is a one-line change. D4PM does not: its ``forward`` *is* the diffusion
training step. It samples a timestep ``t`` and noise ``ε``, forms the noised
artifact ``h_t = √ᾱ_t·h₀ + √(1-ᾱ_t)·ε``, and returns ``(ε̂, ε)`` for an
ε-prediction loss. There is no waveform in it to score.

**What it does instead.** The ε-prediction is kept exactly as the paper has it,
and the artifact is recovered from it analytically::

    ĥ₀ = (h_t − √(1-ᾱ_t)·ε̂) / √ᾱ_t

This is an identity, not an approximation, and it is differentiable — so the
recovered-clean objective can score ``ĥ₀`` while the network still learns to
predict noise. Nothing about the architecture, the schedule or the
parameterisation changes.

**And a defect the measurement exposed.** DDPM assumes data at roughly unit
variance: ``h_t = √ᾱ_t·h₀ + √(1-ᾱ_t)·ε`` mixes the signal with ``ε ~ N(0, 1)``.
Our artifact is stored in **volts**, RMS 1.88e-03. At the very first timestep
``√(1-ᾱ₀) = 0.01``, so the signal-to-noise ratio is already 1.88e-03 / 1e-02 =
**0.19** — the entire schedule runs in a regime where the artifact is far below
the noise it is supposed to be separated from, and recovering ``ĥ₀`` is a
difference of two O(1) quantities to extract an O(1e-3) one. An untrained model
returned a recovered artifact 82 to 761 times the clean signal's RMS depending on
where the schedule was cut.

Every other deployment edition normalises its input inside the module; D4PM had
no normalisation at all. This one z-scores the pair (conditioning signal and
artifact) by the conditioning signal's own standard deviation before the
diffusion and restores the scale afterwards, so the schedule operates where it
was designed to. That is the same invariant, not a new idea — but here it is not
a nicety, it is the difference between a diffusion process and a random walk.

**The one place judgement was needed.** ``1/√ᾱ_t`` diverges as ``t`` approaches
the end of the schedule: at ``t = 199`` of 200 the factor is ~14, so an ε-error
of 1 % becomes a 14 % waveform error and the gradient of the waveform term is
dominated by the timesteps where the artifact is least recoverable. The waveform
term is therefore applied only below :data:`WAVEFORM_T_FRACTION` of the
schedule, where the reconstruction carries information, and the ε-term covers the
whole schedule as the paper intends. The split is a parameter, not a constant, so
the choice is visible in the config rather than buried here.

Reported per epoch alongside the terms: ``eps`` (the paper's own objective) and
the recovered-clean terms, so the two can be read against each other instead of
one being inferred from the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from facet.models.masterthesis.d4pm.training import D4PMTrainingModule
from facet.training.dataset import NPZContextArtifactDataset
from facet.training.deployment_losses import LOSS_KEYS, RecoveredCleanObjective

PACKING = "b1s"
CORE_OUTPUT = "artifact"
CONTEXT_EPOCHS = 7
EPOCH_SAMPLES = 512

#: Only timesteps below this fraction of the schedule contribute the waveform
#: term. Above it ``1/√ᾱ_t`` amplifies the ε-error faster than the term carries
#: information about the artifact.
#:
#: 0.25 is measured, not chosen for neatness. Untrained model, same batch, with
#: the scale normalisation in place:
#:
#: ==========  ===========  =============  ===========================
#: cut         amplitude    energy_ratio   share of the batch scored
#: ==========  ===========  =============  ===========================
#: t < 0.50          2.456          1.869                        55 %
#: t < 0.25          0.510          1.235                        30 %
#: t < 0.10          0.097          1.059                        14 %
#: ==========  ===========  =============  ===========================
#:
#: At half the schedule the ``1/√ᾱ_t`` amplification is already what the term
#: measures; at a tenth the recovery is near-exact by construction and the term
#: has little left to say. A quarter keeps both the information and a useful
#: share of each batch.
WAVEFORM_T_FRACTION = 0.25


class D4PMWaveformModule(D4PMTrainingModule):
    """:class:`D4PMTrainingModule` that also hands back the recovered artifact.

    Returns ``(batch, 3, samples)``: row 0 the recovered artifact ``ĥ₀``, row 1
    the predicted noise, row 2 the true noise. The loss splits them; the module
    stays a diffusion training step.

    Parameters
    ----------
    waveform_t_fraction : float
        Timesteps at or above this fraction of the schedule get their waveform
        row zeroed *and* are marked in the returned mask, so the loss can average
        over the contributing examples only rather than over a batch that is
        half zeros.
    demean_output : bool
        Remove the recovered artifact's own mean, the same guarantee every other
        deployment edition makes structurally.
    """

    def __init__(
        self, *args: Any, waveform_t_fraction: float = WAVEFORM_T_FRACTION, demean_output: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.waveform_t_fraction = float(waveform_t_fraction)
        self.demean_output = bool(demean_output)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.shape[1] != 2:
            raise ValueError(f"expected 2 input channels, got {packed.shape[1]}")
        if torch.jit.is_tracing():
            zero = torch.zeros_like(packed[:, 0:1, :])
            return torch.cat([zero, zero, zero], dim=1)

        cond_y = packed[:, 0:1, :]
        h0 = packed[:, 1:2, :]
        batch, device = packed.shape[0], packed.device

        # Into the scale the schedule assumes. The conditioning signal is the
        # reference rather than the artifact, because at inference only the
        # conditioning signal exists.
        scale = cond_y.std(dim=-1, keepdim=True).clamp_min(1e-30)
        cond_y, h0 = cond_y / scale, h0 / scale

        if self.training:
            t = torch.randint(0, self.num_steps, (batch,), device=device)
            noise = torch.randn_like(h0)
        else:
            # Deterministic evaluation, as in the base edition: fixed timesteps
            # spread over the schedule and no noise, so val_loss is comparable
            # between epochs rather than a fresh random draw each time.
            arange = torch.arange(batch, device=device, dtype=torch.long)
            t = (arange * (self.num_steps // max(batch, 1))) % self.num_steps
            noise = torch.zeros_like(h0)

        h_t = self.q_sample(h0, t, noise)
        pred_noise = self.predictor(h_t, cond_y, self.sqrt_alphas_cumprod[t])

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        h0_hat = (h_t - sqrt_one_minus * pred_noise) / sqrt_alpha
        h0_hat = h0_hat * scale  # back into volts
        if self.demean_output:
            h0_hat = h0_hat - h0_hat.mean(dim=-1, keepdim=True)

        usable = (t < int(self.waveform_t_fraction * self.num_steps)).view(-1, 1, 1)
        h0_hat = torch.where(usable, h0_hat, torch.zeros_like(h0_hat))
        # The mask rides along as a constant channel so the loss can normalise by
        # the number of contributing examples instead of by the batch size.
        mask = usable.to(h0_hat.dtype).expand_as(h0_hat)
        return torch.cat([h0_hat, pred_noise, noise, mask], dim=1)


class D4PMDeploymentLoss(nn.Module):
    """The paper's ε-loss plus the recovered-clean objective on ``ĥ₀``.

    Parameters
    ----------
    eps_weight : float
        Weight of ``MSE(ε̂, ε)``, the objective D4PM is defined by. Keeping it is
        deliberate: the sampler at inference walks the reverse process, and a
        model whose ε-prediction has been traded away for waveform accuracy at
        low noise levels has nothing to walk with.
    waveform_weight : float
        Weight of the recovered-clean objective.
    """

    def __init__(self, eps_weight: float = 1.0, waveform_weight: float = 1.0, **objective_kwargs: Any) -> None:
        super().__init__()
        self.eps_weight = float(eps_weight)
        self.waveform_weight = float(waveform_weight)
        objective_kwargs = {k: v for k, v in objective_kwargs.items() if k in LOSS_KEYS}
        objective_kwargs.setdefault("prediction_is", "artifact")
        objective_kwargs["rows"] = tuple(objective_kwargs.get("rows", ("artifact", "clean", "noisy")))
        self.objective = RecoveredCleanObjective(**objective_kwargs)
        self.last_terms: dict[str, float] = {}

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape[1] != 4:
            raise ValueError(
                f"expected [h0_hat, pred_noise, noise, mask] on axis 1, got {prediction.shape[1]} channels"
            )
        # The epsilon terms keep the model's full length: scoring them on the
        # centre epoch alone would train the diffusion on a seventh of its own
        # input and stop being D4PM. Only the waveform row is reduced.
        eps_loss = (prediction[:, 1:2] - prediction[:, 2:3]).pow(2).mean()
        mask = prediction[:, 3:4]
        contributing = mask[:, 0, 0] > 0

        h0_hat = prediction[:, 0:1]
        target_rows, target_len = target.shape[-2], target.shape[-1]
        if h0_hat.shape[-1] != target_len:
            rows = h0_hat.shape[-1] // target_len
            if rows * target_len != h0_hat.shape[-1]:
                raise ValueError(
                    f"prediction length {h0_hat.shape[-1]} is not a whole number of target epochs of {target_len}"
                )
            reshaped = h0_hat.reshape(h0_hat.shape[0], rows, target_len)
            # One target row means the context axis was epochs and only the centre
            # one is scored; many rows means it was channels and every electrode
            # has its own target.
            h0_hat = reshaped[:, rows // 2 : rows // 2 + 1] if target_rows == 1 else reshaped

        total = self.eps_weight * eps_loss
        terms = {"eps": float(eps_loss.detach())}
        if self.waveform_weight and bool(contributing.any()):
            waveform = self.objective(h0_hat[contributing], target[contributing])
            total = total + self.waveform_weight * waveform
            terms.update(self.objective.last_terms)
            terms["waveform_share"] = float(contributing.float().mean())
        self.last_terms = terms
        return total


class D4PMDeploymentDataset:
    """Packs ``[noisy, artifact]`` as input and ``[artifact, clean, noisy]`` as target.

    The input packing is the base edition's — diffusion training needs the clean
    target *as an input* to form the noised state. The target packing is the
    deployment contract's, so the recovered-clean objective sees what it needs.
    All rows are demeaned per epoch, in the same space the module's output demean
    produces.
    """

    def __init__(self, base_dataset: Any, max_examples: int | None = None) -> None:
        self.base_dataset = base_dataset
        first_noisy, first_target = base_dataset[0]
        self.context_epochs = int(first_noisy.shape[0])
        self.n_channels = int(first_noisy.shape[1])
        self.epoch_samples = int(first_noisy.shape[2])
        self.centre = self.context_epochs // 2
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))
        self.target_rows = ("artifact", "clean", "noisy")
        total = len(base_dataset) * self.n_channels
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        base_idx, channel = divmod(int(idx), self.n_channels)
        ctx, target = self.base_dataset[base_idx]
        target = target[:, channel]  # (3, S)
        noisy = ctx[self.centre, channel]  # (S,)
        packed = np.stack([noisy, target[0]], axis=0)  # [noisy, artifact]
        return packed.astype(np.float32), target[:, np.newaxis, :].astype(np.float32)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (2, self.epoch_samples)

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (3, 1, self.epoch_samples)

    @property
    def n_chunks(self) -> int:
        return len(self)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        n_base = len(self.base_dataset)
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_base)
        val_base = set(order[: max(1, int(n_base * val_ratio))].tolist())
        train = [i for i in range(len(self)) if (i // self.n_channels) not in val_base]
        val = [i for i in range(len(self)) if (i // self.n_channels) in val_base]
        return _Subset(self, train), _Subset(self, val)


class _Subset:
    def __init__(self, parent: D4PMDeploymentDataset, indices: list[int]) -> None:
        self._parent, self._indices = parent, indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


# --------------------------------------------------------------- facet-train

_MODULE_KEYS = frozenset(
    {
        "epoch_samples",
        "num_steps",
        "beta_start",
        "beta_end",
        "feats",
        "d_model",
        "d_ff",
        "n_heads",
        "n_layers",
        "embed_dim",
        "waveform_t_fraction",
        "demean_output",
    }
)


def build_model(**kwargs: Any) -> D4PMWaveformModule:
    kwargs = {k: v for k, v in kwargs.items() if k in _MODULE_KEYS}
    kwargs.setdefault("epoch_samples", EPOCH_SAMPLES)
    return D4PMWaveformModule(**kwargs)


def build_loss(name: str = "d4pm_deployment", **kwargs: Any) -> nn.Module:
    if name.strip().lower() not in {"d4pm_deployment", "recovered_clean", "deployment"}:
        raise ValueError(f"Unknown loss {name!r}")
    eps_weight = float(kwargs.pop("eps_weight", 1.0))
    waveform_weight = float(kwargs.pop("waveform_weight", 1.0))
    if kwargs.get("freq_band") is not None:
        kwargs["freq_band"] = tuple(kwargs["freq_band"])
    return D4PMDeploymentLoss(eps_weight=eps_weight, waveform_weight=waveform_weight, **kwargs)


def build_dataset(path: str | Path, max_examples: int | None = None, **_: Any) -> D4PMDeploymentDataset:
    base = NPZContextArtifactDataset(
        path, target_key="artifact_center", demean_input=True, demean_target=True, target_extras=("clean", "noisy")
    )
    return D4PMDeploymentDataset(base, max_examples=max_examples)


__all__ = [
    "CORE_OUTPUT",
    "PACKING",
    "D4PMDeploymentDataset",
    "D4PMDeploymentLoss",
    "D4PMWaveformModule",
    "build_dataset",
    "build_loss",
    "build_model",
]

# ---------------------------------------------------------------------------
# Context variants: never one epoch and one channel
# ---------------------------------------------------------------------------
#
# D4PM cannot use facet.training.context_variants: that helper flattens a
# single-channel input into one long 1-D signal, and D4PM's input already has two
# channels -- the conditioning signal and the artifact the forward diffusion
# needs. So the axis is built here instead, on the same principle: concatenate
# the rows in time, leave the network's shape alone.
#
# One decision worth naming. The returned tensor keeps *all* rows for the noise
# terms and reduces only the waveform row, so the epsilon objective still covers
# the whole noised tensor -- scoring it on the centre epoch alone would train the
# diffusion on a seventh of its own input and quietly stop being D4PM.


class D4PMContextDataset(D4PMDeploymentDataset):
    """``[noisy, artifact]`` over a whole context axis rather than one epoch.

    Parameters
    ----------
    axis : {"epochs", "channels"}
        ``"epochs"`` gives ``(2, 7*S)`` per electrode; ``"channels"`` gives
        ``(2, 30*S)`` per window.
    """

    def __init__(
        self,
        base_dataset: Any,
        axis: str = "channels",
        max_examples: int | None = None,
        artifact_context: np.ndarray | None = None,
    ) -> None:
        if axis not in ("epochs", "channels"):
            raise ValueError(f"axis must be epochs|channels, got {axis!r}")
        super().__init__(base_dataset, max_examples=None)
        self.axis = axis
        # Read straight from the bundle rather than reconstructing noisy - clean:
        # the forward diffusion noises every row, so it needs the artifact on every
        # row, and artifact_context is exactly that.
        if axis == "epochs" and artifact_context is None:
            raise ValueError("the epochs axis needs artifact_context from the bundle")
        self.artifact_context = artifact_context
        self.rows = self.context_epochs if axis == "epochs" else self.n_channels
        total = len(base_dataset) * (self.n_channels if axis == "epochs" else 1)
        self._length = total if max_examples is None else max(0, min(int(max_examples), total))

    def __getitem__(self, idx: int):
        idx = int(idx)
        if self.axis == "epochs":
            base_idx, channel = divmod(idx, self.n_channels)
            ctx, target = self.base_dataset[base_idx]
            noisy = ctx[:, channel]  # (T, S)
            art_rows = self.artifact_context[base_idx][:, channel]  # (T, S)
            art_rows = art_rows - art_rows.mean(axis=-1, keepdims=True)
            packed = np.stack([noisy.reshape(-1), art_rows.reshape(-1)], axis=0)
            tgt = target[:, channel][:, np.newaxis, :]  # (3, 1, S)
        else:
            ctx, target = self.base_dataset[idx]
            centre = ctx[self.centre]  # (C, S)
            packed = np.stack([centre.reshape(-1), target[0].reshape(-1)], axis=0)
            tgt = target  # (3, C, S)
        return packed.astype(np.float32), np.ascontiguousarray(tgt, dtype=np.float32)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (2, self.rows * self.epoch_samples)

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (3, 1 if self.axis == "epochs" else self.n_channels, self.epoch_samples)


def build_context_model(
    axis: str = "channels",
    n_channels: int = 30,
    context_epochs: int = CONTEXT_EPOCHS,
    epoch_samples: int = EPOCH_SAMPLES,
    **kwargs: Any,
):
    """D4PM over a context axis: the same module, a longer input."""
    rows = context_epochs if axis == "epochs" else n_channels
    kwargs = {k: v for k, v in kwargs.items() if k in _MODULE_KEYS}
    kwargs["epoch_samples"] = rows * epoch_samples
    module = D4PMWaveformModule(**kwargs)
    module.context_rows = rows
    module.context_axis = axis
    module.centre_row = context_epochs // 2 if axis == "epochs" else None
    module.epoch_length = epoch_samples
    return module


def build_context_dataset(path: str | Path, axis: str = "channels", max_examples: int | None = None, **_: Any):
    base = NPZContextArtifactDataset(
        path, target_key="artifact_center", demean_input=True, demean_target=True, target_extras=("clean", "noisy")
    )
    art_ctx = None
    if axis == "epochs":
        with np.load(Path(path).expanduser(), allow_pickle=False) as bundle:
            art_ctx = bundle["artifact_context"].astype(np.float32, copy=False)
    return D4PMContextDataset(base, axis=axis, max_examples=max_examples, artifact_context=art_ctx)
