"""Cascaded DAE deployment edition.

**What the base edition does on a real recording.** 42.9 µV residual and 7.8x FARM's power in the EEG band: it *added* energy rather than removing it.

**Why.** Its objective is MSE between the predicted artifact and
``artifact_center``. On this dataset the artifact is 1881 µV RMS against 495 µV
of clean EEG, so handing back the input as "the artifact" — deleting the EEG —
scores well. Nothing in the objective, and nothing in the training log, said
otherwise.

**What changes here.** Only the objective and the four invariants in
:class:`facet.training.deployment_model.PackedDeploymentModel`; the network is
imported unchanged from :mod:`facet.models.cascaded_dae.training`.

1. :class:`facet.training.deployment_losses.RecoveredCleanObjective` scores
   ``clean_hat = noisy - prediction`` against the clean EEG, with every term
   divided by the clean signal's own energy. Deleting the signal scores exactly
   1.0 per term; a perfect reconstruction scores 0.0. The four terms are
   IC-U-Net's published ensemble (amplitude, velocity, acceleration, frequency),
   plus SI-SDR.
2. The model z-scores its own input and restores the scale on the way out, so
   the external contract is unchanged: raw volts in, raw volts out.
3. Every predicted epoch leaves with zero mean, which removes the seam step
   structurally instead of leaving it to the inference adapter.
4. Early stopping uses a *relative* ``min_delta``. An absolute one is either a
   no-op or a hard stop depending on the loss scale, and nothing warns which.

Packing ``b1s`` and core output ``artifact`` are the base edition's,
verified against the inference adapter by ``tests/test_deployment_data.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from facet.models.cascaded_dae.training import build_model as _build_core
from facet.training.deployment_data import (
    build_packed_dataset, model_input_shape, single_row_target_shape)
from facet.training.deployment_losses import build_deployment_loss
from facet.training.deployment_model import PackedDeploymentModel

#: The base edition's tensor layout and output meaning. Changing either here
#: without changing ``tools/pipeline_demo/family_adapters.py`` breaks inference
#: silently, which is why the test compares the two.
PACKING = "b1s"
CORE_OUTPUT = "artifact"
CONTEXT_EPOCHS = 7
EPOCH_SAMPLES = 512


def build_model(
    context_epochs: int = CONTEXT_EPOCHS,
    epoch_samples: int = EPOCH_SAMPLES,
    normalise: bool = True,
    identity_init: bool = True,
    demean_output: bool = True,
    **core_kwargs: Any,
) -> PackedDeploymentModel:
    """The base network, wrapped so it starts at "change nothing"."""
    core_kwargs.setdefault("epoch_samples", epoch_samples)
    core_kwargs.setdefault("context_epochs", context_epochs)
    # Defaults for everything facet-train would otherwise inject, so the edition
    # can be built without a dataset on disk — a trace check or a doctest should
    # not need one.
    n_channels = int(core_kwargs.setdefault("n_channels", 30))
    core_kwargs.setdefault("chunk_size", epoch_samples)
    core_kwargs.setdefault("input_shape", model_input_shape(
        PACKING, n_channels=n_channels, context_epochs=context_epochs,
        epoch_samples=epoch_samples))
    # facet-train injects the *stacked* target shape, whose leading axis is the
    # loss's extra rows. A core that sizes its output head from it builds a head
    # as many times too wide as there are rows.
    core_kwargs["target_shape"] = single_row_target_shape(
        PACKING, n_channels=n_channels, epoch_samples=epoch_samples)
    core = _build_core(**core_kwargs)
    return PackedDeploymentModel(
        core, packing=PACKING, context_epochs=context_epochs,
        epoch_samples=epoch_samples, core_output=CORE_OUTPUT,
        normalise=normalise, identity_init=identity_init, demean_output=demean_output)


def build_loss(name: str = "recovered_clean", **kwargs: Any) -> nn.Module:
    """Loss factory. ``sfreq`` is injected by facet-train and gates the band."""
    return build_deployment_loss(name, **kwargs)


def build_dataset(path: str | Path, max_examples: int | None = None,
                  **_: Any) -> Any:
    return build_packed_dataset(path, PACKING, max_examples=max_examples)

# ---------------------------------------------------------------------------
# All-channel variant: the contract the FACETpy 0.1.0 DAE actually had
# ---------------------------------------------------------------------------

#: Packing for the all-channel variant. ``bcs`` is one epoch across every
#: electrode, ``b1s`` is one epoch of one electrode.
ALL_CHANNEL_PACKING = "bcs"


class FlattenChannels(nn.Module):
    """Present ``(B, C, S)`` to a core that expects ``(B, 1, C*S)``.

    Why this exists. ``cascaded_dae`` sees **one electrode at a time**: no
    temporal and no spatial context, 512 samples and nothing else. The cascaded
    DAE that *did* reproduce AAS — FACETpy 0.1.0, 88.6 % agreement on the
    validation tail — was an 8-4-8 autoencoder over ``(channels x timepoints)``
    flattened, so it saw all 30 electrodes at once through a **four-unit**
    bottleneck.

    That difference is not incidental. The gradient artifact is produced by the
    same switching gradients at every electrode, so its waveform is nearly the
    same on all of them up to a per-channel gain — which is exactly the
    redundancy averaging methods exploit. A model that sees all channels can use
    it; a single-channel model has to infer the artifact from one noisy
    realisation. And a four-unit bottleneck across 8820 inputs can represent
    almost nothing *except* what the channels share, which is as strong an
    inductive bias towards the artifact as one could ask for.

    The wrapper changes the input contract and nothing else: the same cascade,
    the same two stages, the same residual structure.
    """

    def __init__(self, core: nn.Module, n_channels: int, epoch_samples: int) -> None:
        super().__init__()
        self.core = core
        self.n_channels = int(n_channels)
        self.epoch_samples = int(epoch_samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        flat = x.reshape(batch, 1, self.n_channels * self.epoch_samples)
        return self.core(flat).reshape(batch, self.n_channels, self.epoch_samples)


def build_all_channel_model(
    context_epochs: int = CONTEXT_EPOCHS,
    epoch_samples: int = EPOCH_SAMPLES,
    n_channels: int = 30,
    bottleneck: int = 4,
    normalise: bool = True,
    identity_init: bool = True,
    demean_output: bool = True,
    **core_kwargs: Any,
) -> PackedDeploymentModel:
    """The cascade over every electrode at once, with the legacy bottleneck.

    Parameters
    ----------
    bottleneck : int
        Latent width of the middle layer. 4 is the FACETpy 0.1.0 value; the
        per-channel edition uses 128, which on 30x512 inputs is wide enough to
        encode each channel separately and therefore removes the very pressure
        that makes the architecture work.
    """
    hidden = core_kwargs.pop("hidden_units", None)
    if hidden is None:
        outer = int(core_kwargs.pop("outer_units", 8))
        hidden = [outer, int(bottleneck), outer]
    core = _build_core(input_shape=(1, n_channels * epoch_samples),
                       chunk_size=n_channels * epoch_samples,
                       hidden_units=list(hidden),
                       dropout_rate=float(core_kwargs.pop("dropout_rate", 0.2)))
    wrapped = FlattenChannels(core, n_channels=n_channels, epoch_samples=epoch_samples)
    return PackedDeploymentModel(
        wrapped, packing=ALL_CHANNEL_PACKING, context_epochs=context_epochs,
        epoch_samples=epoch_samples, core_output=CORE_OUTPUT,
        normalise=normalise, identity_init=identity_init, demean_output=demean_output)


def build_all_channel_dataset(path: str | Path, max_examples: int | None = None,
                              **_: Any) -> Any:
    return build_packed_dataset(path, ALL_CHANNEL_PACKING, max_examples=max_examples)

# ---------------------------------------------------------------------------
# Context variants: the rule is never one epoch *and* one channel
# ---------------------------------------------------------------------------
#
# docs/research/run_7_paper_strict_rebuild.md states it: at least one comparison
# axis has to be present, or there is nothing in the input from which the
# artifact could be separated. This edition inherited a single-epoch,
# single-channel contract from its base edition; these two factories give it each
# axis in turn, so "which context does this family actually need" becomes a
# measurement instead of an assumption.
#
# See facet.training.context_variants for how the axes are built and for the one
# caveat that comes with the cheap construction (seams at the joins).


def build_axis_model(axis: str = "channels", **kwargs: Any) -> Any:
    """The base network with one comparison axis. ``axis`` is epochs|channels."""
    from facet.training.context_variants import build_context_variant
    kwargs.pop("input_shape", None)
    kwargs.pop("target_shape", None)
    kwargs.pop("chunk_size", None)
    kwargs.pop("sfreq", None)
    return build_context_variant(_build_core, axis, core_output=CORE_OUTPUT, **kwargs)


def build_axis_dataset(path: str | Path, axis: str = "channels",
                       max_examples: int | None = None, **_: Any) -> Any:
    packing = "b1ts" if axis == "epochs" else "bcs"
    return build_packed_dataset(path, packing, max_examples=max_examples)
