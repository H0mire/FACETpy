"""ViT-Spectrogram deployment edition — a complex mask, because magnitude cannot work.

**What the base edition does on a real recording.** 0.38 µV residual gradient
artifact and 0.26x FARM's power in the EEG band: the flattest line of all
fourteen arms. It removed the artifact by removing the signal.

**Why, and why this edition needs more than a new objective.** Every other
deployment edition changes only what the network is scored against. This one
cannot, and the reason is measurable rather than argued.

The base model predicts the clean *magnitude* spectrogram and rebuilds the
waveform with ``torch.polar(pred_magnitude, phase)``, where ``phase`` comes from
the **noisy** input. So substitute the true clean magnitude, keep everything else
identical, and you have the architecture's ceiling — the best it could do with a
perfect magnitude predictor:

=====================================================  =======  ======  ======  ======
variant                                                   loss    ampl     vel   accel
=====================================================  =======  ======  ======  ======
ceiling: true clean magnitude + noisy phase               4.835   1.493   1.475   1.466
control: true magnitude *and* true phase                 -1.000   0.000   0.000   0.000
do nothing                                               43.055  14.141  10.093   9.543
trained base edition, epoch 60                           48.155  20.535  13.347  12.450
**deleting the signal** (the objective's floor)         **3.046**  1.000   1.000   1.000
=====================================================  =======  ======  ======  ======

**The ceiling is worse than deleting the signal.** A perfect magnitude predictor
still scores above 1.0 on every time-domain term, because the borrowed phase
carries the artifact. The control row shows the STFT round-trip itself is
lossless, so the crop is not the culprit — it is the phase and nothing else.

That also explains the training curve: over 60 epochs the base edition's
velocity term moved from 13.342 to 13.347 and acceleration from 12.451 to
12.450, both frozen, while only the frequency term fell (74.8 → 29.4). The model
was learning the one thing it could still change.

It is the same failure IC-U-Net's ablation reports from the other direction: a
frequency-magnitude-only objective reaches **-1.14 dB**, worse than no correction
at all, against 22.60 dB for amplitude (arXiv:2111.10026, Table 2).

**The fix, and its source.** Predict a **complex** ratio mask instead of a
magnitude, so the model can move the phase. Yang et al. give exactly this
argument for the complex-valued restormer in their nested GAN: operating on the
complex spectrum lets the model use amplitude *and* phase, "minimizing the loss
of useful EEG signal components during artifact removal"
(Biomed. Phys. Eng. Express 11:065054, 2025). Since ``Z_clean = (Z_clean /
Z_noisy) · Z_noisy``, a complex mask can in principle be exact — the ceiling
moves from 4.835 to the control row's -1.000.

Everything else is the base architecture: the same STFT, the same patching, the
same masked-patch ViT encoder, the same depth and width. Only the decoder head
is twice as wide, and what it means changed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from facet.models.masterthesis.vit_spectrogram.training import ViTSpectrogramInpainter
from facet.training.deployment_data import (
    build_packed_dataset, model_input_shape, single_row_target_shape)
from facet.training.deployment_losses import build_deployment_loss
from facet.training.deployment_model import PackedDeploymentModel

#: Arguments :class:`ViTSpectrogramInpainter` accepts, minus the two this
#: factory sets itself. facet-train also injects ``sfreq``, ``n_channels``,
#: ``chunk_size``, ``input_shape`` and ``target_shape``; forwarding those to the
#: core is a TypeError at construction, which is where the first run died.
_CORE_KEYS = frozenset({
    "n_fft", "hop_length", "freq_bins", "time_frames", "patch_freq", "patch_time",
    "embed_dim", "depth", "n_heads", "mlp_ratio", "dropout", "mask_margin_patches",
})

PACKING = "bt1s"
CORE_OUTPUT = "clean"
CONTEXT_EPOCHS = 7
EPOCH_SAMPLES = 512


class ComplexMaskViTSpectrogram(ViTSpectrogramInpainter):
    """The base inpainter with a complex ratio mask in place of a magnitude head.

    Parameters
    ----------
    mask_bound : float
        The mask is squashed to ``(-mask_bound, mask_bound)`` per component. An
        unbounded complex mask multiplies the spectrum by an arbitrary factor,
        which is how a spectral model turns 1881 µV of artifact into more; the
        bound makes over-amplification unreachable rather than merely penalised.
        2.0 leaves room to *add* where the artifact cancelled real signal.
    **kwargs
        Passed to :class:`ViTSpectrogramInpainter` unchanged.

    Notes
    -----
    The head is initialised so the mask is exactly ``1 + 0i`` — the model starts
    as the identity and its first prediction is "this signal is already clean".
    That is the "do nothing" baseline, and it is why the deployment wrapper's own
    ``identity_init`` must be **off** for this core: applying it as well would
    add the centre epoch a second time.
    """

    def __init__(self, mask_bound: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mask_bound = float(mask_bound)
        # Two components per pixel instead of one: real and imaginary.
        self.decoder_head = nn.Linear(self.embed_dim, 2 * self.patch_pixels)
        nn.init.zeros_(self.decoder_head.weight)
        with torch.no_grad():
            bias = torch.zeros(2 * self.patch_pixels)
            # Solve bound * tanh(b / bound) == 1 so the *squashed* real part is
            # exactly one, not merely close to it.
            bias[: self.patch_pixels] = self.mask_bound * math.atanh(1.0 / self.mask_bound)
            self.decoder_head.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        signal = x.reshape(batch, self.total_samples)

        Z = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length,
                       win_length=self.n_fft, window=self.stft_window,
                       center=True, return_complex=True)
        full_freq_bins, full_time_frames = Z.shape[-2], Z.shape[-1]
        Z_cropped = Z[:, : self.freq_bins, : self.time_frames]

        # The encoder still reads the log-magnitude image: the ViT's input is
        # unchanged, only its output is.
        log_mag = torch.log1p(Z_cropped.abs())
        tokens = self.patch_embed(self._patchify(log_mag))
        tokens = self._apply_mask(tokens)
        tokens = self.norm_in(tokens + self._positional_embedding())
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm_out(tokens)

        raw = self.decoder_head(tokens)
        real = self._unpatchify(raw[..., : self.patch_pixels])
        imag = self._unpatchify(raw[..., self.patch_pixels:])
        bound = self.mask_bound
        mask = torch.complex(bound * torch.tanh(real / bound),
                             bound * torch.tanh(imag / bound))

        complex_spec = nn.functional.pad(
            mask * Z_cropped,
            (0, full_time_frames - self.time_frames, 0, full_freq_bins - self.freq_bins))
        time_signal = torch.istft(
            complex_spec, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.stft_window, center=True,
            length=self.total_samples)
        centre = time_signal[:, self.center_start_sample:self.center_stop_sample]
        return centre.unsqueeze(1)


def build_model(
    context_epochs: int = CONTEXT_EPOCHS,
    epoch_samples: int = EPOCH_SAMPLES,
    normalise: bool = True,
    demean_output: bool = True,
    mask_bound: float = 2.0,
    **core_kwargs: Any,
) -> PackedDeploymentModel:
    """The complex-mask inpainter, wrapped so it starts at "change nothing"."""
    # facet-train injects whatever the signature will swallow; the core only
    # accepts its own named arguments, so filter rather than forward blindly.
    core_kwargs = {k: v for k, v in core_kwargs.items() if k in _CORE_KEYS}
    core = ComplexMaskViTSpectrogram(
        mask_bound=mask_bound, context_epochs=context_epochs,
        epoch_samples=epoch_samples, **core_kwargs)
    return PackedDeploymentModel(
        core, packing=PACKING, context_epochs=context_epochs,
        epoch_samples=epoch_samples, core_output=CORE_OUTPUT,
        normalise=normalise,
        # Off on purpose: the complex mask is initialised to 1 + 0i, so the core
        # is already the identity. Adding the wrapper's skip would add the centre
        # epoch twice and make the untrained model predict *minus* the signal.
        identity_init=False,
        demean_output=demean_output)


def build_loss(name: str = "recovered_clean", **kwargs: Any) -> nn.Module:
    """Loss factory. ``sfreq`` is injected by facet-train and gates the band."""
    kwargs.setdefault("prediction_is", "artifact")
    return build_deployment_loss(name, **kwargs)


def build_dataset(path: str | Path, max_examples: int | None = None, **_: Any) -> Any:
    return build_packed_dataset(path, PACKING, max_examples=max_examples)


__all__ = ["CORE_OUTPUT", "PACKING", "ComplexMaskViTSpectrogram",
           "build_dataset", "build_loss", "build_model",
           "model_input_shape", "single_row_target_shape"]
