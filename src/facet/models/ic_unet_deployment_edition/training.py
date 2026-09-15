"""IC-U-Net, retrained against an objective that cannot be won by deleting the EEG.

`ic_unet` corrects nothing on a real recording. Run through a FACETpy pipeline on
the Niazy EDF it leaves 11.88 µV of residual gradient artifact against FARM's
4.62 µV, its output is ~96 % constant offset, and the epoch-seam step ratio is
well above a working corrector's 1.0. On the prepared holdout tensors it scores
RRMSE_t ≈ 0.98 with a correlation near zero — the signature of a model that
returns almost nothing.

That is not a bug in the architecture. It is the objective. The base edition
minimises MSE between its predicted artifact and ``artifact_center``, and on this
dataset the artifact is 1881 µV RMS against 495 µV of clean EEG, so "hand back
the input as the artifact" — deleting the EEG entirely — is a *good* solution to
the problem as posed. Chuang et al. never posed it that way: IC-U-Net is trained
against the **clean** signal, with an ensemble of four terms whose own ablation
(arXiv:2111.10026, Table 2) is the argument for all four —

======================================  ==========
objective                               SNR (dB)
======================================  ==========
amplitude only                          22.60
velocity only                            7.76
acceleration only                        7.60
frequency magnitude only                **-1.14**
ensemble, weights [1, 1, 1, 1]          **24.98**
======================================  ==========

The -1.14 dB row is worth pausing on: a frequency-magnitude objective is worse
than doing nothing, and it is exactly what our ``vit_spectrogram`` edition
optimises.

Three changes, and only three, separate this edition from ``ic_unet``:

1. **The loss scores the recovered clean, normalised.**
   :class:`facet.training.deployment_losses.RecoveredCleanObjective` evaluates
   ``clean_hat = noisy - prediction`` against the clean EEG and divides every
   term by the clean signal's own energy. Deleting the signal now scores exactly
   1.0 per term instead of scoring well, and the number in the training log means
   "this fraction of the EEG is wrong" rather than "some µV²".
2. **The model normalises its own input.** IC-U-Net, ART, DAR and EEGdenoiseNet
   all z-score before the network and restore the scale afterwards; we were the
   only ones feeding raw µV into a BatchNorm stack. Doing it *inside* the module
   rather than in the dataset keeps the external contract identical — raw µV in,
   raw µV out — so the pipeline adapter, the TorchScript export and the holdout
   evaluator need no change at all.
3. **The prediction is demeaned per epoch before it leaves the model.** The
   reassembled recording stepped at every trigger because each epoch came back
   with its own baseline; the user could see it in the plot before any metric
   reported it. Removing the DC in the module makes the guarantee structural
   rather than something the inference adapter has to remember.

The U-Net itself, the channel ladder, the kernel sizes, the frozen ICA and the
7-epoch context are untouched. If this edition corrects and ``ic_unet`` does not,
the objective was the cause.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from facet.models.ic_unet.training import IcUnet1D, _fit_ica_matrix
from facet.training.dataset import NPZContextArtifactDataset
from facet.training.deployment_losses import build_deployment_loss
from facet.training.deployment_model import DeploymentArtifactModel


#: The base edition's tensor layout and output meaning, named here so the
#: deployment editions all answer the same two questions the same way.
PACKING = "bcts"
CORE_OUTPUT = "clean"
CONTEXT_EPOCHS = 7
EPOCH_SAMPLES = 512


def _invert(matrix: np.ndarray) -> np.ndarray:
    """Inverse of a square unmixing matrix, pseudo-inverse only if it is singular.

    ``np.linalg.pinv`` emits "divide by zero encountered in matmul" on this
    numpy/BLAS build for *any* input, the identity included, and returns the
    right answer anyway. Under ``-W error`` — which is how the test suite runs —
    that turns a benign warning into a failure to construct the model. An ICA
    unmixing matrix is square and invertible by construction, so the ordinary
    inverse is both the correct call and the quiet one; the pseudo-inverse stays
    as the fallback for the degenerate case rather than as the default.
    """
    try:
        return np.linalg.inv(matrix).astype(np.float32)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix).astype(np.float32)


class SelfNormalisingIcUnet(DeploymentArtifactModel):
    """IC-U-Net that z-scores its own input and restores the scale on the way out.

    The normalisation lives here rather than in the dataset on purpose. A
    dataset-side z-score has to be undone at inference by whoever calls the
    model, using statistics that then have to be shipped alongside the weights —
    and the four inference paths in this repo (pipeline adapter, holdout
    evaluator, TorchScript export, processor) would each have to remember. Inside
    the module the contract stays "raw µV in, raw µV out" and there is nothing to
    remember.

    Parameters
    ----------
    n_channels, context_epochs, epoch_samples : int
        Geometry of the input, ``(B, n_channels, context_epochs * epoch_samples)``.
    base_channels : int
        First U-Net level width; the ladder is ``64 -> 128 -> 256 -> 512``.
    ica_init : np.ndarray, optional
        Frozen unmixing matrix. Identity when omitted.
    demean_output : bool
        Remove each predicted epoch's own mean before returning it. Leave on
        unless you are deliberately reproducing the seam-step failure.
    identity_init : bool
        Let the U-Net predict a *correction* to its input rather than the clean
        signal outright, so an untrained network is the identity. Without it the
        module computes ``artifact = noisy - unet(noisy)`` and an untrained
        ``unet`` outputs approximately zero, which makes the artifact estimate
        approximately the whole input — the network starts at "delete the EEG"
        and has to climb out of it. Measured on the smoke run: ``energy_ratio``
        began at 0.037, meaning the recovered signal carried under 4 % of the
        clean EEG's amplitude before a single useful gradient step. With the
        skip, the same initialisation starts at "change nothing", which is the
        trivial baseline rather than a failure state.

    Examples
    --------
    ::

        model = SelfNormalisingIcUnet(30, 7, 512)
        artifact = model(torch.randn(2, 30, 7 * 512) * 1e3)   # µV in, µV out
        # untrained: near zero, i.e. "change nothing", not "delete everything"
    """

    def __init__(
        self,
        n_channels: int,
        context_epochs: int,
        epoch_samples: int,
        base_channels: int = 64,
        ica_init: np.ndarray | None = None,
        demean_output: bool = True,
        identity_init: bool = True,
        normalise: bool = True,
    ) -> None:
        super().__init__(normalise=normalise, identity_init=identity_init,
                         demean_output=demean_output)
        self.n_channels = int(n_channels)
        self.context_epochs = int(context_epochs)
        self.epoch_samples = int(epoch_samples)
        self.full_samples = self.context_epochs * self.epoch_samples
        self.center_start = (self.context_epochs // 2) * self.epoch_samples
        self.center_stop = self.center_start + self.epoch_samples

        self.unet = IcUnet1D(in_channels=self.n_channels, out_channels=self.n_channels,
                             base_channels=base_channels)

        if ica_init is None:
            ica_init = np.eye(self.n_channels, dtype=np.float32)
        else:
            ica_init = np.asarray(ica_init, dtype=np.float32)
            if ica_init.shape != (self.n_channels, self.n_channels):
                raise ValueError(
                    f"ica_init must be ({self.n_channels}, {self.n_channels}), got {ica_init.shape}")
        self.register_buffer("ica_W", torch.from_numpy(ica_init))
        self.register_buffer("ica_W_pinv", torch.from_numpy(_invert(ica_init)))

    def centre_of(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.center_start:self.center_stop]

    def core_clean(self, x: torch.Tensor) -> torch.Tensor:
        # The U-Net runs in IC space over the whole context; the skip is applied
        # there rather than after the inverse ICA, so "the core predicts a
        # correction" holds in the space the core actually sees.
        ic = torch.einsum("ij,bjt->bit", self.ica_W, x)
        ic_clean = ic + self.unet(ic) if self.identity_init else self.unet(ic)
        clean_full = torch.einsum("ij,bjt->bit", self.ica_W_pinv, ic_clean)
        return clean_full[..., self.center_start:self.center_stop]


class ContextIcUnetDeploymentDataset:
    """Flattens the 7-epoch context and keeps the clean/noisy rows for the loss.

    Identical to ``ic_unet``'s wrapper except that the target is the
    ``(3, channels, samples)`` stack the recovered-clean objective needs. All
    three rows are demeaned per epoch by the base dataset, in the same space the
    model's own output demean produces — if those two disagree the loss is off by
    a constant and nothing raises.
    """

    def __init__(self, base_dataset: Any, *, max_examples: int | None = None) -> None:
        self.base_dataset = base_dataset
        first_noisy, first_target = base_dataset[0]
        if first_noisy.ndim != 3:
            raise ValueError("base input must be (context_epochs, channels, samples)")
        if first_target.ndim != 3:
            raise ValueError(
                "base target must be (rows, channels, samples) — build the dataset with "
                "target_extras=('clean', 'noisy')")

        self.context_epochs = int(first_noisy.shape[0])
        self.n_channels = int(first_noisy.shape[1])
        self.epoch_samples = int(first_noisy.shape[2])
        self.full_samples = self.context_epochs * self.epoch_samples
        self.chunk_size = self.epoch_samples
        self.target_type = "artifact"
        self.trigger_aligned = True
        self.sfreq = float(getattr(base_dataset, "sfreq", float("nan")))
        self.target_rows = tuple(getattr(base_dataset, "target_rows", ("artifact", "clean", "noisy")))
        n = len(base_dataset)
        self._length = n if max_examples is None else max(0, min(int(max_examples), n))

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        noisy_context, target = self.base_dataset[idx]
        flat = noisy_context.transpose(1, 0, 2).reshape(
            self.n_channels, self.full_samples).astype(np.float32, copy=True)
        return flat, target.astype(np.float32, copy=True)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (self.n_channels, self.full_samples)

    @property
    def target_shape(self) -> tuple[int, int, int]:
        return (len(self.target_rows), self.n_channels, self.epoch_samples)

    @property
    def n_chunks(self) -> int:
        return len(self)

    def train_val_split(self, val_ratio: float = 0.2, seed: int = 42):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self)).tolist()
        n_val = max(1, int(len(self) * val_ratio))
        val = set(indices[:n_val])
        return (_Subset(self, [i for i in range(len(self)) if i not in val]),
                _Subset(self, [i for i in range(len(self)) if i in val]))


class _Subset:
    def __init__(self, parent: ContextIcUnetDeploymentDataset, indices: list[int]) -> None:
        self._parent, self._indices = parent, indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._parent[self._indices[idx]]


# --------------------------------------------------------------- facet-train


def build_model(
    n_channels: int = 30,
    context_epochs: int = 7,
    epoch_samples: int = 512,
    base_channels: int = 64,
    fit_ica: bool = True,
    ica_random_state: int = 0,
    dataset_path: str | Path | None = None,
    demean_output: bool = True,
    identity_init: bool = True,
    normalise: bool = True,
    **_: Any,
) -> SelfNormalisingIcUnet:
    ica = None
    if fit_ica and dataset_path is not None:
        ica = _fit_ica_matrix(dataset_path, n_channels, random_state=ica_random_state)
    return SelfNormalisingIcUnet(
        n_channels=n_channels, context_epochs=context_epochs, epoch_samples=epoch_samples,
        base_channels=base_channels, ica_init=ica, demean_output=demean_output,
        identity_init=identity_init, normalise=normalise)


def build_loss(name: str = "recovered_clean", **kwargs: Any) -> nn.Module:
    """Loss factory. ``sfreq`` is forwarded by facet-train and gates the band."""
    return build_deployment_loss(name, **kwargs)


def build_dataset(
    path: str | Path,
    max_examples: int | None = None,
    **_: Any,
) -> ContextIcUnetDeploymentDataset:
    base = NPZContextArtifactDataset(
        path, target_key="artifact_center", max_examples=max_examples,
        demean_input=True, demean_target=True, target_extras=("clean", "noisy"))
    return ContextIcUnetDeploymentDataset(base)
