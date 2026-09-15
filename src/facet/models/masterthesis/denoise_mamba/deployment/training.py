"""DenoiseMamba deployment edition — the same recurrence, computed in parallel.

**What the base edition does on a real recording.** 2.55 µV residual gradient
artifact but only 0.28x FARM's power in the EEG band: it removed the artifact by
removing the signal.

**And it could not be trained anyway.** Measured on an idle RTX 4090:

============================================  ==========
                                                      ms
============================================  ==========
selective scan, forward only                        67.1
selective scan, forward **and backward**          5478.2
the (B, L, d_inner, d_state) tensors alone          26.5
the 512-step Python loop alone                    2611.1
============================================  ==========

Forward 67 ms, backward 5.5 seconds. The cost is not arithmetic — it is that
``for t in range(512)`` builds an autograd graph 512 nodes deep, four times per
pass. Throughput is 8 examples/s and **flat in batch size** (64, 128 and 256 all
give 8/s, 512 runs out of memory), so nothing about the configuration can fix it:
43 minutes per epoch, ~43 hours for the 60-epoch budget the other families get.
That is also why the *base* run in run 6 reached only 11 of its 60 epochs.

``torch.compile`` is not the way out: inductor unrolls the loop and runs out of
memory on an otherwise empty 24 GB card.

**The fix is a parallel scan, and it changes no mathematics.** The recurrence

    s_t = a_t · s_{t-1} + b_t

is a composition of affine maps, and affine maps compose associatively::

    (A_u, B_u) then (A_t, B_t)  ==  (A_t·A_u,  A_t·B_u + B_t)

so a Hillis-Steele scan computes the same states in ``log2(512) = 9`` sequential
steps instead of 512. Only the order of the floating-point additions differs;
``tests/test_denoise_mamba_parallel_scan.py`` asserts the two agree to float32
precision on the real parameter ranges, and the prototype measured a relative
difference of 1.7e-07 with a 14.8x speed-up.

Everything else is the base architecture and the shared deployment contract:
the same ``SelectiveSSM`` parameters, the same blocks, the recovered-clean
objective, self-normalisation, a demeaned output and a relative ``min_delta``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from facet.models.masterthesis.denoise_mamba.training import SelectiveSSM
from facet.models.masterthesis.denoise_mamba.training import build_model as _build_core
from facet.training.deployment_data import build_packed_dataset, model_input_shape, single_row_target_shape
from facet.training.deployment_losses import build_deployment_loss
from facet.training.deployment_model import PackedDeploymentModel

PACKING = "b1s"
CORE_OUTPUT = "artifact"
CONTEXT_EPOCHS = 7
EPOCH_SAMPLES = 512


def _shift(x: torch.Tensor, offset: int, fill: float) -> torch.Tensor:
    """Shift along time, filling the head with the operator's identity element."""
    head = torch.full_like(x[:, :offset], fill)
    return torch.cat([head, x[:, :-offset]], dim=1)


def parallel_affine_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """States of ``s_t = a_t * s_{t-1} + b_t`` with ``s_{-1} = 0``, in log2(L) steps.

    Parameters
    ----------
    a, b : torch.Tensor
        Shape ``(batch, length, ...)``. ``a`` is the per-step decay, ``b`` the
        per-step input.

    Returns
    -------
    torch.Tensor
        Same shape as ``b``; element ``t`` is the state after step ``t``.

    Notes
    -----
    Each iteration composes the affine map of every position with the map
    ``offset`` positions earlier, so after ``ceil(log2(L))`` iterations position
    ``t`` carries the composition of steps ``0..t``. The head of each shift is
    filled with the identity map ``(1, 0)``, which is why the result is exact
    rather than approximate at the boundaries.
    """
    A, B = a, b
    length = a.shape[1]
    offset = 1
    while offset < length:
        A_prev = _shift(A, offset, 1.0)
        B_prev = _shift(B, offset, 0.0)
        B = A * B_prev + B
        A = A * A_prev
        offset *= 2
    return B


class ParallelScanSSM(SelectiveSSM):
    """:class:`SelectiveSSM` with the sequential loop replaced by a parallel scan.

    The parameters, their shapes and their meaning are the base class's — a state
    dict from one loads into the other. Only :meth:`forward` differs, and only in
    how it evaluates the same recurrence.

    Parameters
    ----------
    checkpoint_scan : bool
        Recompute the scan during the backward pass instead of keeping it.

        Not an optimisation but a requirement. The scan holds ``log2(512) = 9``
        levels of ``(batch, 512, d_inner, d_state)`` tensors, which is 4.8 GB per
        block at batch 64 and 19 GB across the four blocks — it runs out of
        memory on a 24 GB card at *every* batch size. Recomputing costs one extra
        forward pass, and the forward pass was measured at 67 ms against 5.5 s
        for the backward, so the trade is heavily in its favour.
    """

    def __init__(self, *args: Any, checkpoint_scan: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint_scan = bool(checkpoint_scan)

    def _scan(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_scan and torch.is_grad_enabled() and self.training:
            return torch.utils.checkpoint.checkpoint(parallel_affine_scan, a, b, use_reentrant=False)
        return parallel_affine_scan(a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, d_inner = x.shape
        if d_inner != self.d_inner:
            raise ValueError(f"expected d_inner={self.d_inner}, got {d_inner}")

        x_dbl = self.x_proj(x)
        delta_unproj, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = nn.functional.softplus(self.dt_proj(delta_unproj))

        A = -torch.exp(self.A_log.float())
        delta_a_exp = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        delta_b_x = torch.einsum("bld,bln,bld->bldn", delta, B, x)

        states = self._scan(delta_a_exp, delta_b_x)  # (B, L, D, N)
        y = torch.einsum("bldn,bln->bld", states, C)
        return y + x * self.D


def _swap_in_parallel_scan(core: nn.Module, checkpoint_scan: bool = True) -> nn.Module:
    """Replace every :class:`SelectiveSSM` with the parallel-scan version.

    Done by re-parenting the existing parameters rather than by rebuilding, so
    the initialisation is bit-identical to what the base edition would have had
    for the same seed. A different initialisation would make "the parallel scan
    changes nothing" untestable.
    """
    replaced = 0
    for block in core.blocks:
        old = block.mamba.ssm
        new = ParallelScanSSM(d_inner=old.d_inner, d_state=old.d_state, checkpoint_scan=checkpoint_scan)
        new.load_state_dict(old.state_dict(), strict=True)
        block.mamba.ssm = new
        replaced += 1
    if replaced == 0:
        raise RuntimeError("no SelectiveSSM found to replace — has the core changed?")
    return core


def build_model(
    context_epochs: int = CONTEXT_EPOCHS,
    epoch_samples: int = EPOCH_SAMPLES,
    normalise: bool = True,
    identity_init: bool = True,
    demean_output: bool = True,
    parallel_scan: bool = True,
    checkpoint_scan: bool = True,
    **core_kwargs: Any,
) -> PackedDeploymentModel:
    """The base network with the parallel scan, wrapped in the deployment contract.

    ``parallel_scan=False`` keeps the original sequential loop, which is what the
    equivalence test compares against.
    """
    core_kwargs.setdefault("epoch_samples", epoch_samples)
    core_kwargs.setdefault("context_epochs", context_epochs)
    n_channels = int(core_kwargs.setdefault("n_channels", 30) or 30)
    core_kwargs.setdefault("chunk_size", epoch_samples)
    core_kwargs.setdefault(
        "input_shape",
        model_input_shape(PACKING, n_channels=n_channels, context_epochs=context_epochs, epoch_samples=epoch_samples),
    )
    core_kwargs["target_shape"] = single_row_target_shape(PACKING, n_channels=n_channels, epoch_samples=epoch_samples)

    core = _build_core(**core_kwargs)
    if parallel_scan:
        core = _swap_in_parallel_scan(core, checkpoint_scan=checkpoint_scan)
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


__all__ = [
    "CORE_OUTPUT",
    "PACKING",
    "ParallelScanSSM",
    "build_dataset",
    "build_loss",
    "build_model",
    "parallel_affine_scan",
]

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


def build_axis_model(
    axis: str = "channels", *, parallel_scan: bool = True, checkpoint_scan: bool = True, **kwargs: Any
) -> Any:
    """The base network with one comparison axis. ``axis`` is epochs|channels.

    The scan swap has to be repeated here. :func:`build_model` applies it to the
    core it builds itself, but the context variants go through
    :func:`build_context_variant`, which calls the *original* factory directly —
    so without this the axis arms would run the sequential loop with the whole
    graph retained. On the channel axis that is a 15,360-step scan rather than
    512, which is where the memory goes: the epoch axis merely gets slow, the
    channel axis does not fit on the card at all.

    ``parallel_scan`` and ``checkpoint_scan`` cannot simply be passed through
    either. The original ``build_model`` swallows unknown keywords via ``**_``,
    so they would be accepted and silently ignored.
    """
    from facet.training.context_variants import build_context_variant

    kwargs.pop("input_shape", None)
    kwargs.pop("target_shape", None)
    kwargs.pop("chunk_size", None)
    kwargs.pop("sfreq", None)

    def _core(**core_kwargs: Any) -> nn.Module:
        core = _build_core(**core_kwargs)
        if parallel_scan:
            core = _swap_in_parallel_scan(core, checkpoint_scan=checkpoint_scan)
        return core

    return build_context_variant(_core, axis, core_output=CORE_OUTPUT, **kwargs)


def build_axis_dataset(path: str | Path, axis: str = "channels", max_examples: int | None = None, **_: Any) -> Any:
    packing = "b1ts" if axis == "epochs" else "bcs"
    return build_packed_dataset(path, packing, max_examples=max_examples)
