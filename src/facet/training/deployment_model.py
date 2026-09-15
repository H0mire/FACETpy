"""A template that makes a model start at "change nothing" instead of "delete everything".

:mod:`facet.training.deployment_losses` fixes what the models were scored on.
This module fixes what they start from, which turned out to matter just as much.

Most of the families predict the artifact as ``noisy - core(noisy)``. An untrained
``core`` outputs approximately zero, so an untrained *model* outputs approximately
the whole input as the artifact — that is, it begins training already deleting the
EEG, at the exact point the old objective rewarded most. Measured on IC-U-Net at
initialisation: the recovered signal carried **4 %** of the clean EEG's amplitude
before a single useful gradient step. Adding a skip so the core predicts a
*correction* to its input puts the same initialisation at exactly 1.0 — the
trivial "do nothing" baseline, which is the honest starting point for a corrector.

Four invariants live here, in one place, because they are the same four for every
family and each of them is a silent failure when wrong:

1. **Self-normalisation.** IC-U-Net, ART, DAR and EEGdenoiseNet all z-score before
   the network; we were the only ones feeding raw volts into a BatchNorm stack.
   Doing it inside the module — rather than in the dataset — keeps the external
   contract at "raw in, raw out", so the pipeline adapter, the TorchScript export
   and the holdout evaluator need no scaling metadata and cannot disagree about it.
2. **Identity initialisation**, as above.
3. **A demeaned output.** Learned correctors work epoch by epoch, and reassembled
   into a continuous recording their segments no longer share a baseline: the
   signal steps at every trigger. Eight of fourteen families showed a seam-step
   ratio above 2.5 against a working corrector's 1.0. Removing the mean in the
   module makes it structural rather than something four inference paths each
   have to remember.
4. **Scale restoration**, so 1 and 3 do not change the units of the answer.

Subclasses supply the two genuinely family-specific pieces, :meth:`centre_of` and
:meth:`core_clean`, and inherit the rest. Everything above is exercised by
``tests/test_deployment_model.py`` against a stub core, so a new edition gets the
invariants tested for free.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn

#: Below this a channel's standard deviation is not a scale, it is a flat
#: channel, and dividing by it manufactures noise with a gain of 1/floor.
SCALE_FLOOR = 1e-30


class DeploymentArtifactModel(nn.Module):
    """Normalise, run the core with an identity skip, demean, restore the scale.

    Parameters
    ----------
    normalise : bool
        Z-score the input over ``normalise_dims`` before the core and multiply
        the result back afterwards.
    normalise_dims : tuple of int
        Axes the mean and standard deviation are taken over.
    identity_init : bool
        Treat the core's output as a correction to its input rather than as the
        clean signal, so an untrained model changes nothing.
    demean_output : bool
        Remove the predicted epoch's own mean before returning it.

    Notes
    -----
    The statistics span the whole context, never a single epoch: a per-epoch
    scale would make the seven context epochs incomparable and reintroduce the
    very seam that ``demean_output`` exists to remove. For packings that keep the
    context on its own axis, pass that axis in ``normalise_dims``.
    """

    def __init__(
        self,
        normalise: bool = True,
        identity_init: bool = True,
        demean_output: bool = True,
        normalise_dims: tuple[int, ...] = (-1,),
    ) -> None:
        super().__init__()
        self.normalise = bool(normalise)
        self.identity_init = bool(identity_init)
        self.demean_output = bool(demean_output)
        #: Axes the input statistics are taken over. For a packing that keeps the
        #: context epochs on their own axis this must include that axis: a scale
        #: computed per epoch makes the seven epochs incomparable and puts the
        #: seam back that ``demean_output`` removes.
        self.normalise_dims = tuple(int(d) for d in normalise_dims)

    # ----------------------------------------------------------- family hooks

    @abstractmethod
    def centre_of(self, x: torch.Tensor) -> torch.Tensor:
        """The noisy centre epoch, in the same space and shape as the output."""

    @abstractmethod
    def core_clean(self, x: torch.Tensor) -> torch.Tensor:
        """The core's estimate of the clean centre epoch, same shape as :meth:`centre_of`.

        Implementations that use the identity skip should add their input back
        inside this method when the core's natural output is a correction; the
        template only knows the *centre*, not the core's internal layout, so it
        cannot apply the skip on the family's behalf.
        """

    # --------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = list(self.normalise_dims)
        x = x - x.mean(dim=dims, keepdim=True)
        if self.normalise:
            scale = x.std(dim=dims, keepdim=True).clamp_min(SCALE_FLOOR)
            x = x / scale
        else:
            scale = None

        artifact = self.centre_of(x) - self.core_clean(x)
        if self.demean_output:
            artifact = artifact - artifact.mean(dim=-1, keepdim=True)
        if scale is not None:
            # ``scale`` may carry axes the centre does not (a context axis, say);
            # reduce it to a single factor per remaining axis rather than
            # broadcasting a mismatched shape into silence.
            artifact = artifact * self._align_scale(scale, artifact)
        return artifact

    @staticmethod
    def _align_scale(scale: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Reduce ``scale`` so it broadcasts against ``like`` without inventing axes."""
        while scale.dim() > like.dim():
            scale = scale.mean(dim=1)
        for axis in range(1, scale.dim() - 1):
            if scale.shape[axis] != like.shape[axis] and scale.shape[axis] != 1:
                scale = scale.mean(dim=axis, keepdim=True)
        return scale


#: Input layouts the fourteen families use, and where the centre epoch lives in
#: each. Copied from ``src/facet/models/masterthesis/adapters.py`` deliberately: the
#: training-time packing and the inference-time packing have to agree, and the
#: verification script asserts they do.
PACKINGS = {
    "b1s": "(B, 1, S) — one epoch, no context",
    "bcs": "(B, C, S) — one epoch, every electrode at once",
    "bt1s": "(B, T, 1, S) — context on axis 1",
    "bts": "(B, T, S) — context on axis 1, flat",
    "b1ts": "(B, 1, T*S) — context concatenated in time",
    "bcts": "(B, C, T*S) — multichannel, context concatenated in time",
    "btcs": "(B, T, C, S) — multichannel, context on axis 1",
}

#: Which axes carry the context for each packing, for ``normalise_dims``.
#:
#: Every axis a packing uses as context has to be *inside* the reduction, the
#: channel axis included. The gradient artifact is the same waveform at every
#: electrode up to a per-channel gain -- on this dataset the artifact runs from
#: 602 µV at F3 to 3501 µV at O2, a spread of 5.8x -- and that gain is the whole
#: reason a multichannel view can do better than a single-channel one. Reducing
#: over ``(-1,)`` alone on a ``(B, C, S)`` input takes the standard deviation
#: *per channel per example*, so every electrode reaches the core at unit
#: variance and the gain is gone before the network sees anything. It is
#: multiplied back onto the output afterwards, so the amplitude is right and the
#: loss never complains; only the input is impoverished.
#:
#: FACETpy 0.1.0 got this right in all three of its variants, by different
#: routes: the torch cascade normalised with a single global scalar over the
#: whole recording, and the two TensorFlow variants used dataset-level
#: per-channel constants -- fixed numbers a channel-attention block can learn
#: back, unlike a per-example rescaling, which leaves nothing to learn.
_CONTEXT_DIMS = {
    "b1s": (-1,),
    "bcs": (1, -1),
    "bt1s": (1, -1),
    "bts": (1, -1),
    "b1ts": (-1,),
    "bcts": (1, -1),
    "btcs": (1, 2, -1),
}


class PackedDeploymentModel(DeploymentArtifactModel):
    """:class:`DeploymentArtifactModel` for a core that speaks one of :data:`PACKINGS`.

    The core is called with the input in its native packing and must return
    something the centre epoch can be read out of — either the full context or
    the centre alone. Both are accepted, because the families disagree and the
    disagreement is not interesting.

    Parameters
    ----------
    core : nn.Module
        The family's network, unchanged.
    packing : str
        One of :data:`PACKINGS`.
    context_epochs, epoch_samples : int
        Geometry, needed to locate the centre.
    core_output : {"clean", "artifact", "sources"}
        What the core's output means, which decides whether the identity skip is
        needed at all.

        ``"clean"`` cores (IC-U-Net, ViT-Spectrogram) return the denoised signal,
        so the artifact is ``noisy - core(noisy)`` and an untrained core -- output
        near zero -- makes the model claim the whole input is artifact. That is
        the deletion failure, present before training starts, and it is what
        ``identity_init`` fixes.

        ``"artifact"`` cores return the artifact directly, so they are already
        residual: an untrained core predicts no artifact, which is the "do
        nothing" baseline. ``identity_init`` is a no-op for them and is ignored,
        rather than silently applied to the wrong quantity.

        ``"sources"`` cores (Conv-TasNet) return several separated signals;
        ``core_output_index`` selects the artifact and the rest follows
        ``"artifact"``.
    core_output_index : int
        Index into axis 1 of a ``"sources"`` output.

    Examples
    --------
    ::

        model = PackedDeploymentModel(my_core, packing="bt1s",
                                      context_epochs=7, epoch_samples=512)
        artifact = model(torch.randn(4, 7, 1, 512))
    """

    def __init__(
        self,
        core: nn.Module,
        packing: str,
        context_epochs: int,
        epoch_samples: int,
        core_output: str = "clean",
        core_output_index: int = 1,
        **kwargs,
    ) -> None:
        if packing not in PACKINGS:
            raise ValueError(f"unknown packing {packing!r}; known: {sorted(PACKINGS)}")
        if core_output not in {"clean", "artifact", "sources"}:
            raise ValueError(f"core_output must be clean/artifact/sources, got {core_output!r}")
        kwargs.setdefault("normalise_dims", _CONTEXT_DIMS[packing])
        super().__init__(**kwargs)
        self.core_output = core_output
        if core_output != "clean":
            # Residual by construction; nothing to skip. Say so rather than
            # leaving a flag set that does nothing.
            self.identity_init = False
        self.core = core
        self.packing = packing
        self.context_epochs = int(context_epochs)
        self.epoch_samples = int(epoch_samples)
        self.centre_index = self.context_epochs // 2
        self.centre_start = self.centre_index * self.epoch_samples
        self.centre_stop = self.centre_start + self.epoch_samples
        self.core_output_index = core_output_index

    def _read_centre(self, tensor: torch.Tensor) -> torch.Tensor:
        """Centre epoch of a tensor in this model's packing, or of the centre itself."""
        if self.packing in ("b1s", "bcs"):
            return tensor
        if self.packing in ("bt1s", "bts", "btcs"):
            return tensor[:, self.centre_index] if tensor.shape[1] == self.context_epochs else tensor
        # b1ts / bcts: the context is concatenated along time.
        if tensor.shape[-1] == self.context_epochs * self.epoch_samples:
            return tensor[..., self.centre_start : self.centre_stop]
        return tensor

    def centre_of(self, x: torch.Tensor) -> torch.Tensor:
        return self._read_centre(x)

    def core_clean(self, x: torch.Tensor) -> torch.Tensor:
        out = self.core(x)
        if self.core_output == "sources":
            if out.dim() < 3 or out.shape[1] <= self.core_output_index:
                raise RuntimeError(
                    f"core returned {tuple(out.shape)}; expected sources on axis 1 with index {self.core_output_index}"
                )
            out = out[:, self.core_output_index]
        centre_noisy = self.centre_of(x)
        centre = self._read_centre(out)
        if centre.numel() != centre_noisy.numel():
            raise RuntimeError(
                f"core output {tuple(out.shape)} does not reduce to the centre epoch {tuple(centre_noisy.shape)}"
            )
        centre = centre.reshape(centre_noisy.shape)
        if self.core_output in ("artifact", "sources"):
            # forward() computes centre - core_clean, so hand back what makes
            # that identity hold for a core that already predicts the artifact.
            return centre_noisy - centre
        return centre + centre_noisy if self.identity_init else centre
