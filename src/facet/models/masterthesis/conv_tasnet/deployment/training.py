"""Conv-TasNet deployment edition.

**What the base edition does on a real recording.** 15.9 µV residual, 1.78x FARM in the EEG band: the artifact was largely left in place.

**Why.** Its objective is MSE between the predicted artifact and
``artifact_center``. On this dataset the artifact is 1881 µV RMS against 495 µV
of clean EEG, so handing back the input as "the artifact" — deleting the EEG —
scores well. Nothing in the objective, and nothing in the training log, said
otherwise.

**What changes here.** Only the objective and the four invariants in
:class:`facet.training.deployment_model.PackedDeploymentModel`; the network is
imported unchanged from :mod:`facet.models.masterthesis.conv_tasnet.training`.

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

The artifact is source index 1, as in the base edition; the separation head is unchanged.

Packing ``b1s`` and core output ``sources`` are the base edition's,
verified against the inference adapter by ``tests/test_deployment_data.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from facet.models.masterthesis.conv_tasnet.training import build_model as _build_core
from facet.training.deployment_data import build_packed_dataset, model_input_shape, single_row_target_shape
from facet.training.deployment_losses import build_deployment_loss
from facet.training.deployment_model import PackedDeploymentModel

#: The base edition's tensor layout and output meaning. Changing either here
#: without changing ``src/facet/models/masterthesis/adapters.py`` breaks inference
#: silently, which is why the test compares the two.
PACKING = "b1s"
CORE_OUTPUT = "sources"
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
    core_kwargs.setdefault(
        "input_shape",
        model_input_shape(PACKING, n_channels=n_channels, context_epochs=context_epochs, epoch_samples=epoch_samples),
    )
    # facet-train injects the *stacked* target shape, whose leading axis is the
    # loss's extra rows. A core that sizes its output head from it builds a head
    # as many times too wide as there are rows.
    core_kwargs["target_shape"] = single_row_target_shape(PACKING, n_channels=n_channels, epoch_samples=epoch_samples)
    core = _build_core(**core_kwargs)
    return PackedDeploymentModel(
        core,
        packing=PACKING,
        context_epochs=context_epochs,
        epoch_samples=epoch_samples,
        core_output=CORE_OUTPUT,
        normalise=normalise,
        identity_init=identity_init,
        demean_output=demean_output,
    )


def build_loss(name: str = "recovered_clean", **kwargs: Any) -> nn.Module:
    """Loss factory. ``sfreq`` is injected by facet-train and gates the band."""
    return build_deployment_loss(name, **kwargs)


def build_dataset(path: str | Path, max_examples: int | None = None, **_: Any) -> Any:
    return build_packed_dataset(path, PACKING, max_examples=max_examples)


# ---------------------------------------------------------------------------
# Context variants: the rule is never one epoch *and* one channel
# ---------------------------------------------------------------------------
#
# docs/source/thesis_reference/selected_variants.rst states it: at least one comparison
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


def build_axis_dataset(path: str | Path, axis: str = "channels", max_examples: int | None = None, **_: Any) -> Any:
    packing = "b1ts" if axis == "epochs" else "bcs"
    return build_packed_dataset(path, packing, max_examples=max_examples)
