"""Give a single-epoch, single-channel model a comparison axis.

``docs/research/run_7_paper_strict_rebuild.md`` states the rule: **never one
epoch and one channel**. At least one comparison axis has to be there, because
without one there is nothing in the input from which the artifact could be
separated — a model can learn the artifact's stereotyped *shape*, and an
amplitude, and nothing else.

Six editions inherited exactly that contract from their base editions:
``conv_tasnet``, ``d4pm``, ``denoise_mamba``, ``dhct_gan``, ``dpae`` and
``cascaded_dae``. Keeping the base contract was deliberate — only the objective
was meant to vary — but holding an invalid contract fixed does not make a
comparison clean, it preserves the fault. The pipeline plot shows the
consequence: ``dpae`` and ``cascaded_dae`` return an EEG-like baseline with a
periodic spike train through it. They learned the shape, not the instance.

This module supplies the two axes, and supplies them the same way for every
family so the comparison is about context rather than about capacity:

``epochs``
    the seven context epochs concatenated in time, ``(B, 1, 7*S)``. The core is
    built for the longer input and the centre epoch is sliced from its output.
``channels``
    all electrodes concatenated in time, ``(B, 1, C*S)``, reshaped back to
    ``(B, C, S)``. This is the FACETpy 0.1.0 contract, which reached 88.6 %
    agreement with AAS on its validation tail.

**Both flatten rather than widening the network.** No layer changes shape, no
parameter count jumps, nothing about the architecture moves — so a difference
between the arms is a difference in what the model could see. The cost is seams:
six at the epoch joins, which are real boundaries in the signal anyway, and
twenty-nine at the channel joins, which are not. A convolution whose kernel
straddles a channel seam sees two unrelated electrodes; that is a genuine
limitation of the cheap construction and belongs next to any result from it.

Why the channel axis should matter at all: the gradient artifact is produced by
the same switching gradients at every electrode, so its waveform is nearly
identical across them up to a per-channel gain. That redundancy is precisely what
averaging methods exploit, and a single-channel model cannot.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

from facet.training.deployment_model import PackedDeploymentModel

#: Axis name -> (packing, samples the core must accept, output channels)
AXES = {
    "epochs": ("b1ts", "context_epochs * epoch_samples", 1),
    "channels": ("bcs", "n_channels * epoch_samples", "n_channels"),
}


class FlattenToOneDimension(nn.Module):
    """Present a multi-row input to a core that expects ``(B, 1, L)``.

    Parameters
    ----------
    core : nn.Module
        Built for an input of length ``rows * samples``.
    rows, samples : int
        The shape being flattened, ``(B, rows, samples)``.
    keep_rows : bool
        Reshape the core's output back to ``(B, rows, samples)``. True for the
        channel axis, where every electrode needs its own output; False for the
        epoch axis, where the wrapper slices one epoch out of a flat signal.
    """

    def __init__(self, core: nn.Module, rows: int, samples: int,
                 keep_rows: bool = True) -> None:
        super().__init__()
        self.core = core
        self.rows = int(rows)
        self.samples = int(samples)
        self.keep_rows = bool(keep_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        out = self.core(x.reshape(batch, 1, self.rows * self.samples))
        if out.dim() == 3 and out.shape[1] != 1:
            # A core that returns separated sources keeps its own axis; leave it
            # to PackedDeploymentModel, which knows which index is the artifact.
            return out
        if self.keep_rows:
            return out.reshape(batch, self.rows, self.samples)
        return out.reshape(batch, 1, self.rows * self.samples)


def build_context_variant(
    core_factory: Callable[..., nn.Module],
    axis: str,
    *,
    core_output: str = "artifact",
    core_output_index: int = 1,
    context_epochs: int = 7,
    epoch_samples: int = 512,
    n_channels: int = 30,
    length_kwargs: tuple[str, ...] = ("epoch_samples", "chunk_size"),
    normalise: bool = True,
    identity_init: bool = True,
    demean_output: bool = True,
    **core_kwargs: Any,
) -> PackedDeploymentModel:
    """Wrap ``core_factory`` so the model sees one comparison axis.

    Parameters
    ----------
    core_factory : callable
        The family's own ``build_model``.
    axis : {"epochs", "channels"}
    length_kwargs : tuple of str
        Names under which this core takes its input length. Every one present in
        its signature is set to the flattened length — families disagree about
        whether it is ``epoch_samples``, ``chunk_size`` or ``input_shape``, and
        setting only one silently leaves the core at 512 samples.

    Examples
    --------
    ::

        model = build_context_variant(dpae_build_model, "channels")
        model(torch.randn(2, 30, 512))        # -> (2, 30, 512)
    """
    if axis not in AXES:
        raise ValueError(f"axis must be one of {sorted(AXES)}, got {axis!r}")
    rows = context_epochs if axis == "epochs" else n_channels
    length = rows * epoch_samples
    packing = AXES[axis][0]

    for name in length_kwargs:
        core_kwargs[name] = length
    core_kwargs["input_shape"] = (1, length)
    core_kwargs["target_shape"] = (1, length)
    core_kwargs.pop("n_channels", None)
    core = core_factory(**core_kwargs)

    wrapped = FlattenToOneDimension(core, rows=rows, samples=epoch_samples,
                                    keep_rows=(axis == "channels"))
    return PackedDeploymentModel(
        wrapped, packing=packing, context_epochs=context_epochs,
        epoch_samples=epoch_samples, core_output=core_output,
        core_output_index=core_output_index, normalise=normalise,
        identity_init=identity_init, demean_output=demean_output)
