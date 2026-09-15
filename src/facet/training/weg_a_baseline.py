"""``facet-train`` factories for the Run 3 / Weg A spatio-temporal dataset.

The Weg-A builder emits a contract no existing model consumed: one example per
*(target channel c, centre epoch e)*, input ``(context_epochs, 1 + k_neighbors,
core)`` — the target channel plus its geodesic montage neighbours over
consecutive epochs — and target ``(1, core)``, the artifact of ``(c, e)``. The
model factories under :mod:`facet.models` all expect the older
``NPZContextArtifactDataset`` layout (all channels in, all channels out), so this
module supplies the missing bridge plus a deliberately plain baseline.

It is a *baseline*, not a contender: a fully convolutional stack with dilated
receptive field over the flattened ``context_epochs * channels`` axis. Its job is
to produce the first trustworthy validation number on the leakage-free split
(run_3 §7a) that better architectures can then be measured against.

Usage::

    uv run facet-train fit --config configs/weg_a_baseline.yaml
"""

from __future__ import annotations

from pathlib import Path

import torch

from facet.training.dataset import NPZSpatioTemporalDataset

DEFAULT_DATASET = "./output/weg_a_real_v3_512/weg_a_spatiotemporal_dataset.npz"


class SpatioTemporalArtifactNet(torch.nn.Module):
    """Predict the centre-epoch artifact of one channel from its spatio-temporal context.

    Input ``(batch, context_epochs, channels, samples)`` is flattened along the
    (epoch, channel) axes — the ordering is fixed by the builder (self channel
    first, then neighbours, epochs in time order), so a plain convolution can pick
    up both the cross-epoch periodicity and the cross-channel signature. Output is
    ``(batch, 1, samples)``: the artifact of the target channel only.

    The dilated stack is what gives the model a receptive field wide enough to see
    a whole slice period rather than a local burst.
    """

    def __init__(
        self,
        context_epochs: int,
        n_channels: int,
        hidden_channels: int = 64,
        n_blocks: int = 6,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.context_epochs = context_epochs
        self.n_channels = n_channels
        in_channels = context_epochs * n_channels

        self.stem = torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = torch.nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2**i
            pad = dilation * (kernel_size - 1) // 2
            self.blocks.append(
                torch.nn.Sequential(
                    torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=pad, dilation=dilation),
                    torch.nn.GELU(),
                    torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                    torch.nn.GELU(),
                )
            )
        self.head = torch.nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (batch, context_epochs, channels, samples), got {tuple(x.shape)}")
        batch, context_epochs, n_channels, samples = x.shape
        h = self.stem(x.reshape(batch, context_epochs * n_channels, samples))
        for block in self.blocks:
            h = h + block(h)  # residual: the stack refines, it does not restart
        return self.head(h)


def build_dataset(
    path: str = DEFAULT_DATASET,
    max_examples: int | None = None,
    max_shift: int | None = None,
    fractional_shift: bool = False,
    background_mix_prob: float = 0.0,
    demean_input: bool = False,
    demean_target: bool = False,
    target_key: str = "artifact_center",
    target_extras: tuple[str, ...] = (),
    residual_mode: bool = False,
    seed: int = 0,
    **_: object,
) -> NPZSpatioTemporalDataset:
    """Load the Weg-A spatio-temporal dataset.

    Augmentations default to **off** so the first baseline number reflects the
    data, not the augmentation policy; switch them on for later runs.
    """
    return NPZSpatioTemporalDataset(
        path=Path(path).expanduser(),
        target_key=target_key,
        max_examples=max_examples,
        max_shift=max_shift,
        fractional_shift=fractional_shift,
        background_mix_prob=background_mix_prob,
        demean_input=demean_input,
        demean_target=demean_target,
        target_extras=tuple(target_extras),
        residual_mode=residual_mode,
        seed=seed,
    )


def build_model(
    input_shape: tuple[int, int, int],
    hidden_channels: int = 64,
    n_blocks: int = 6,
    kernel_size: int = 9,
    **_: object,
) -> SpatioTemporalArtifactNet:
    context_epochs, n_channels, _samples = input_shape
    return SpatioTemporalArtifactNet(
        context_epochs=context_epochs,
        n_channels=n_channels,
        hidden_channels=hidden_channels,
        n_blocks=n_blocks,
        kernel_size=kernel_size,
    )


class SpikeWeightedMSELoss(torch.nn.Module):
    """MSE that weights the artifact error where a known IED sits (run_6 Phase C).

    Plain MSE optimises the *average* sample, and IED samples are ~0.02 % of the
    data, so the objective is indifferent to them. That is not an abstract worry:
    with plain MSE this baseline reaches a 2.5x lower bulk RMSE than FARM yet
    scores a spike morphology correlation of ~0, because its residual (~44 µV) is
    larger than a typical injected spike (~23 µV) — the spike drowns in the
    model's own error. Weighting the error inside the labelled spike region makes
    accuracy *there* worth ``spike_weight`` times as much as elsewhere.

    Expects the dataset to be built with ``target_with_spike_mask=True``: the
    target arrives as ``(batch, 2, T)`` with row 0 the artifact and row 1 the
    (window-shifted) spike mask, because facet-train passes the loss only
    ``(prediction, target)``.
    """

    def __init__(self, spike_weight: float = 20.0) -> None:
        super().__init__()
        if spike_weight < 1.0:
            raise ValueError(f"spike_weight must be >= 1, got {spike_weight}")
        self.spike_weight = float(spike_weight)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-2] < 2:
            raise ValueError(
                "SpikeWeightedMSELoss needs the spike mask in the target; build the dataset "
                "with target_with_spike_mask=True"
            )
        artifact = target[..., :1, :]
        mask = (target[..., 1:2, :] > 0).to(prediction.dtype)
        weight = 1.0 + (self.spike_weight - 1.0) * mask
        return torch.sum(weight * (prediction - artifact) ** 2) / torch.sum(weight)


class RecoveredCleanLoss(torch.nn.Module):
    """Score the EEG the model hands back, not the artifact it predicts.

    Why the artifact target is degenerate here. The artifact is ~1961 µV RMS and
    the clean EEG ~19 µV, so a model that simply returns its input as "the
    artifact" — deleting the EEG entirely — already scores 18.8 µV MSE, which beat
    every model actually trained against that objective (26.8 and 43.6 µV). The
    optimum of MSE-on-artifact is a model that preserves nothing, and the trained
    models drifted towards it: a 101 µV IED came back as 4 µV
    (``docs/source/thesis_reference/phase_3_grid_search.rst`` §3b).

    The fix is to evaluate ``clean_hat = noisy - prediction`` and to include a
    **scale-invariant** term. SI-SDR of an all-zero estimate is -inf, so deletion
    is not merely penalised, it is unreachable — the loophole closes rather than
    narrowing. An MSE term is kept alongside because SI-SDR is blind to scale,
    while artifact subtraction depends on absolute µV; this mirrors the
    ``si_snr_mse`` pairing the SepFormer edition already uses.

    Expects ``target_extras=("clean", "spike")``: rows are
    ``[artifact, clean, spike_mask]``, from which ``noisy`` is reconstructed.
    ``spike_weight`` additionally emphasises the MSE inside the labelled IED.
    """

    def __init__(
        self,
        mse_weight: float = 10.0,
        spike_weight: float = 20.0,
        si_sdr_max: float = 30.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.spike_weight = float(spike_weight)
        self.si_sdr_max = float(si_sdr_max)
        self.eps = float(eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-2] < 3:
            raise ValueError(
                "RecoveredCleanLoss needs [artifact, clean, spike] rows; build the dataset "
                "with target_extras=('clean', 'spike')"
            )
        artifact = target[..., :1, :]
        clean = target[..., 1:2, :]
        mask = (target[..., 2:3, :] > 0).to(prediction.dtype)
        clean_hat = (artifact + clean) - prediction  # what the clinician sees

        est = clean_hat.reshape(clean_hat.shape[0], -1)
        ref = clean.reshape(clean.shape[0], -1)
        est = est - est.mean(dim=-1, keepdim=True)
        ref = ref - ref.mean(dim=-1, keepdim=True)
        proj = ((est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + self.eps)) * ref
        noise = est - proj
        si_sdr = 10.0 * torch.log10((proj.pow(2).sum(-1) + self.eps) / (noise.pow(2).sum(-1) + self.eps))
        loss = -torch.clamp(si_sdr, max=self.si_sdr_max).mean()

        if self.mse_weight > 0:
            # SI-SDR is scale-invariant, so on its own it lets the estimate be
            # correlated with the clean at an arbitrary amplitude. Anchor the scale
            # with a *normalised* MSE: 0 when perfect, 1 when the estimate carries
            # no signal, and quadratic in any scale error. A logarithmic anchor was
            # tried first and was far too weak — a 2500x amplitude blow-up cost
            # ~7 points against up to 30 points of SI-SDR reward, so the model took
            # the trade and produced 46,921 uV of error.
            weight = 1.0 + (self.spike_weight - 1.0) * mask
            mse = torch.sum(weight * (clean_hat - clean) ** 2) / torch.sum(weight)
            loss = loss + self.mse_weight * mse / (clean.pow(2).mean() + self.eps)
        return loss


def build_loss(name: str = "mse", spike_weight: float = 20.0, mse_weight: float = 10.0, **_: object):
    """Loss factory.

    MSE is the deliberate default: it anchors absolute amplitude in µV, which
    artifact subtraction depends on. Structure-aware objectives are only worth
    adding once a measured gap justifies them (run_3 §0).
    """
    normalized = name.strip().lower()
    if normalized in {"l1", "mae"}:
        return torch.nn.L1Loss()
    if normalized in {"smooth_l1", "huber"}:
        return torch.nn.SmoothL1Loss()
    if normalized == "mse":
        return torch.nn.MSELoss()
    if normalized in {"spike_mse", "spike_weighted_mse"}:
        return SpikeWeightedMSELoss(spike_weight=spike_weight)
    if normalized in {"recovered_clean", "si_sdr_clean"}:
        return RecoveredCleanLoss(mse_weight=mse_weight, spike_weight=spike_weight)
    raise ValueError(f"Unsupported loss '{name}'. Use one of: mse, l1, smooth_l1, spike_mse, recovered_clean.")
