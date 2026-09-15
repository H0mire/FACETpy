"""Cascaded context DAE deployment edition.

**What the base edition does on a real recording.** 48.4 µV residual and 10.8x FARM's EEG-band power — energy added, not removed.

**Why.** Its objective is MSE between the predicted artifact and
``artifact_center``. On this dataset the artifact is 1881 µV RMS against 495 µV
of clean EEG, so handing back the input as "the artifact" — deleting the EEG —
scores well. Nothing in the objective, and nothing in the training log, said
otherwise.

**What changes here.** Only the objective and the four invariants in
:class:`facet.training.deployment_model.PackedDeploymentModel`; the network is
imported unchanged from :mod:`facet.models.masterthesis.cascaded_context_dae.training`.

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

Packing ``bt1s`` and core output ``artifact`` are the base edition's,
verified against the inference adapter by ``tests/test_deployment_data.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from facet.models.masterthesis.cascaded_context_dae.training import build_model as _build_core
from facet.training.deployment_data import build_packed_dataset, model_input_shape, single_row_target_shape
from facet.training.deployment_losses import build_deployment_loss
from facet.training.deployment_model import PackedDeploymentModel

#: The base edition's tensor layout and output meaning. Changing either here
#: without changing ``src/facet/models/masterthesis/adapters.py`` breaks inference
#: silently, which is why the test compares the two.
PACKING = "bt1s"
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
