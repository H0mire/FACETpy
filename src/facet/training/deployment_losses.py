"""Objectives that score the EEG a model hands back, not the artifact it predicts.

Fourteen trained families were run end to end on a real recording for the first
time in run 7, and four of them "won" their training objective by deleting the
signal: ``vit_spectrogram``, ``denoise_mamba``, ``ic_unet`` and the Weg-A direct
baseline all returned something close to a flat line, which scores *better* on
MSE-against-the-artifact than any honest correction does. That is not a training
accident, it is the optimum of the objective they were given — on this dataset
the artifact is 3.8x the RMS of the clean EEG, so "return the input as the
artifact" is a strong solution to "predict the artifact".

Four published papers solve this the same way, and this module implements all
four mechanisms against one shared contract:

* **Derivative terms.** IC-U-Net (Chuang et al., arXiv:2111.10026) trains against
  amplitude *plus* velocity *plus* acceleration. Their own ablation is the
  argument: amplitude-only reaches 22.60 dB, the four-term ensemble 24.98 dB, and
  a frequency-magnitude-only objective reaches **-1.14 dB** — worse than doing
  nothing at all, which is exactly what our ``vit_spectrogram`` optimises.
  A constant output has zero velocity and zero acceleration, so both terms are at
  their maximum for a deleted signal.
* **A spectral term.** IC-U-Net's fourth term and the ``Loss_Frequency`` of Li,
  Bortel, Ayad & Shmuel (EMBC 2025) are the same idea: match the magnitude
  spectrum of the reference. Useful *alongside* the time-domain terms and
  catastrophic alone, per the ablation above.
* **A scale-invariant term.** SI-SDR on the recovered clean, as the SepFormer
  edition and :class:`facet.training.weg_a_baseline.RecoveredCleanLoss` already
  use.
* **An identity anchor.** ``Loss_Identity = E[|X - G(X)|]`` from Li et al., which
  they introduce verbatim "to avoid signal distortion". See
  :class:`RecoveredCleanObjective` for why this one is *not* used literally here.

**Every term is normalised by the reference statistic it measures.** That is the
part that matters beyond any single model:

===============================  ==========================
recovered clean                  loss value
===============================  ==========================
exactly right                    0.0
deleted (flat line)              1.0 per time-domain term
===============================  ==========================

So the loss reads as a fraction of the clean signal's own energy. Two
consequences. An absolute ``min_delta`` becomes meaningful again — the ViT run
was configured with ``min_delta: 1.0e-06`` against a loss whose *entire* value
was 2.45e-07, so early stopping could never see an improvement and the run
stopped at epoch 5 of 120. And "the model deleted the signal" is now legible in
the training log as a loss sitting at 1.0, instead of as a suspiciously small
number.

The diagnostics in :attr:`RecoveredCleanObjective.last_terms` are the other half:
``energy_ratio`` is ``RMS(clean_hat) / RMS(clean)``, which is 0.0 for a deleted
signal and 1.0 for a correctly scaled one, and it is reported per epoch whatever
the loss weights are. The failure we could not see is now a column.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Row names understood by :class:`RecoveredCleanObjective`. The dataset stacks
#: these along a leading axis of the target, because facet-train hands the loss
#: only ``(prediction, target)`` — anything else a loss needs has to ride along.
KNOWN_ROWS = ("artifact", "clean", "noisy", "spike")


#: Relative floor for every denominator in this module. Guards are taken as a
#: fraction of the reference's own energy rather than as an absolute epsilon,
#: because an absolute one makes the loss depend on the unit the data is stored
#: in. MNE keeps EEG in volts, so the clean signal here has a mean square around
#: 2.5e-07 and a plain ``+ 1e-8`` in the denominator was 4 % of it: "delete the
#: whole signal" scored 0.976 instead of the 1.000 the whole design rests on, and
#: the same objective in µV would have scored 1.000. Nothing raised; the tests
#: used O(1) data and passed.
_REL_FLOOR = 1e-12


def _finite_difference(x: torch.Tensor, order: int) -> torch.Tensor:
    for _ in range(order):
        x = x[..., 1:] - x[..., :-1]
    return x


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """``numerator / denominator`` with a floor that scales with the denominator."""
    return numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)


class RecoveredCleanObjective(nn.Module):
    """Normalised multi-term loss on ``clean_hat``, whatever the model predicts.

    Parameters
    ----------
    prediction_is : {"artifact", "clean"}
        What the model's output means. ``"artifact"`` gives
        ``clean_hat = noisy - prediction``; ``"clean"`` gives
        ``clean_hat = prediction``. Getting this wrong does not raise — it
        silently trains the model to predict the complement — so it is a required
        named argument rather than something inferred.
    rows : tuple of str
        Names of the target rows along axis 1, e.g. ``("artifact", "clean",
        "noisy")``. Must contain ``"clean"``, and ``"noisy"`` when
        ``prediction_is="artifact"``.
    amplitude_weight, velocity_weight, acceleration_weight, frequency_weight : float
        IC-U-Net's four ensemble terms. Their paper uses ``[1, 1, 1, 1]``.
    si_sdr_weight : float
        Weight of the scale-invariant term. Normalised by ``si_sdr_max``, so it
        contributes -1.0 at a perfect reconstruction and 0.0 at a deleted one.
    si_sdr_max : float
        Cap in dB. Without it the term is unbounded above and dominates once the
        fit is good, which is how a bounded objective turns into an unstable one.
    identity_weight : float
        Weight of ``mean|prediction| / mean|artifact|``, the scale-free rendering
        of Li et al.'s ``Loss_Identity``. **Default 0.0, deliberately.** Their
        generator's input is the *post-AAS residual*, where the true artifact is
        small and shrinking the output towards zero is a sensible prior. Applied
        to a direct corrector, whose input carries an artifact 3.8x the size of
        the EEG, the literal term penalises correct removal exactly as hard as
        over-removal. Enable it for cascade-formulation models, where the
        assumption the paper makes actually holds; ``identity_hinge`` keeps only
        the over-removal half if you want it in the direct case.
    identity_hinge : bool
        Penalise only ``mean|prediction| > mean|artifact|`` (over-removal). This
        is a deviation from the published term and is flagged as such in
        :attr:`last_terms` under ``identity_hinge``.
    sfreq : float, optional
        Sampling rate, forwarded by facet-train. Without it the frequency term
        falls back to the whole positive-frequency axis instead of ``freq_band``.
    freq_band : tuple of float
        Band for the frequency term, in Hz.
    spike_weight : float
        Extra weight on samples marked by a ``"spike"`` row, if present. 1.0
        disables it.
    eps : float
        Retained for API compatibility. Every denominator in this module is
        floored *relative to the reference*, not by an absolute epsilon — see
        :data:`_REL_FLOOR` for the unit-dependence bug that forced the change.

    Examples
    --------
    ::

        loss = RecoveredCleanObjective(
            prediction_is="artifact", rows=("artifact", "clean", "noisy"), sfreq=1000.0)
        value = loss(prediction, target)          # target: (B, 3, C, S)
        print(loss.last_terms["energy_ratio"])    # 0.0 => the signal was deleted
    """

    def __init__(
        self,
        prediction_is: str = "artifact",
        rows: tuple[str, ...] = ("artifact", "clean", "noisy"),
        amplitude_weight: float = 1.0,
        velocity_weight: float = 1.0,
        acceleration_weight: float = 1.0,
        frequency_weight: float = 1.0,
        si_sdr_weight: float = 1.0,
        si_sdr_max: float = 30.0,
        identity_weight: float = 0.0,
        identity_hinge: bool = False,
        sfreq: float | None = None,
        freq_band: tuple[float, float] = (1.0, 70.0),
        spike_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if prediction_is not in {"artifact", "clean"}:
            raise ValueError(f"prediction_is must be 'artifact' or 'clean', got {prediction_is!r}")
        for name in rows:
            if name not in KNOWN_ROWS:
                raise ValueError(f"unknown target row {name!r}; known: {KNOWN_ROWS}")
        if "clean" not in rows:
            raise ValueError("rows must contain 'clean': every term is scored against it")
        if prediction_is == "artifact" and "noisy" not in rows:
            raise ValueError(
                "rows must contain 'noisy' when the model predicts the artifact — "
                "clean_hat = noisy - prediction cannot be formed otherwise"
            )
        self.prediction_is = prediction_is
        self.rows = tuple(rows)
        self._row_index = {name: i for i, name in enumerate(rows)}
        self.amplitude_weight = float(amplitude_weight)
        self.velocity_weight = float(velocity_weight)
        self.acceleration_weight = float(acceleration_weight)
        self.frequency_weight = float(frequency_weight)
        self.si_sdr_weight = float(si_sdr_weight)
        self.si_sdr_max = float(si_sdr_max)
        self.identity_weight = float(identity_weight)
        self.identity_hinge = bool(identity_hinge)
        self.sfreq = None if sfreq is None else float(sfreq)
        self.freq_band = (float(freq_band[0]), float(freq_band[1]))
        self.spike_weight = float(spike_weight)
        self.eps = float(eps)
        #: Per-term values of the last forward pass, for the training log.
        self.last_terms: dict[str, float] = {}

    # ------------------------------------------------------------------ terms

    def _normalised_mse(
        self, estimate: torch.Tensor, reference: torch.Tensor, weight: torch.Tensor | None
    ) -> torch.Tensor:
        """MSE divided by the reference's own mean square.

        0.0 when the estimate is exact and **exactly 1.0 when the estimate is
        zero**, which is the property the whole module is built around: a deleted
        signal is not a small number any more.
        """
        if weight is None:
            num = (estimate - reference).pow(2).mean()
        else:
            num = (weight * (estimate - reference).pow(2)).sum() / weight.sum()
        return _safe_ratio(num, reference.pow(2).mean())

    def _frequency_term(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        est = torch.fft.rfft(estimate, dim=-1).abs()
        ref = torch.fft.rfft(reference, dim=-1).abs()
        if self.sfreq is not None:
            freqs = torch.fft.rfftfreq(estimate.shape[-1], d=1.0 / self.sfreq, device=estimate.device)
            band = (freqs >= self.freq_band[0]) & (freqs <= self.freq_band[1])
            if bool(band.any()):
                est, ref = est[..., band], ref[..., band]
        return _safe_ratio((est - ref).pow(2).mean(), ref.pow(2).mean())

    def _si_sdr(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        est = estimate.reshape(estimate.shape[0], -1)
        ref = reference.reshape(reference.shape[0], -1)
        est = est - est.mean(dim=-1, keepdim=True)
        ref = ref - ref.mean(dim=-1, keepdim=True)
        ref_energy = ref.pow(2).sum(-1, keepdim=True)
        scale = _safe_ratio((est * ref).sum(-1, keepdim=True), ref_energy)
        proj = scale * ref
        noise = est - proj
        # Floor both sides at the same fraction of the reference energy, so the
        # ratio is unit-free. A fixed epsilon would cap SI-SDR at whatever the
        # data's absolute scale happens to make it.
        floor = _REL_FLOOR * ref_energy.squeeze(-1)
        sdr = 10.0 * torch.log10(
            torch.clamp(proj.pow(2).sum(-1), min=floor) / torch.clamp(noise.pow(2).sum(-1), min=floor)
        )
        return torch.clamp(sdr, max=self.si_sdr_max).mean()

    # ---------------------------------------------------------------- forward

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] != len(self.rows):
            raise ValueError(
                f"target has {target.shape[1]} rows on axis 1 but the loss was configured "
                f"for {len(self.rows)}: {self.rows}. Build the dataset with "
                f"target_extras matching this layout."
            )
        clean = target[:, self._row_index["clean"]]
        if self.prediction_is == "artifact":
            noisy = target[:, self._row_index["noisy"]]
            clean_hat = noisy - prediction
        else:
            noisy = None
            clean_hat = prediction
        if clean_hat.shape != clean.shape:
            raise ValueError(
                f"prediction shape {tuple(prediction.shape)} does not line up with the clean row {tuple(clean.shape)}"
            )

        weight = None
        if self.spike_weight != 1.0 and "spike" in self._row_index:
            mask = (target[:, self._row_index["spike"]] > 0).to(prediction.dtype)
            weight = 1.0 + (self.spike_weight - 1.0) * mask

        terms: dict[str, torch.Tensor] = {}
        if self.amplitude_weight:
            terms["amplitude"] = self._normalised_mse(clean_hat, clean, weight)
        if self.velocity_weight:
            terms["velocity"] = self._normalised_mse(
                _finite_difference(clean_hat, 1), _finite_difference(clean, 1), None
            )
        if self.acceleration_weight:
            terms["acceleration"] = self._normalised_mse(
                _finite_difference(clean_hat, 2), _finite_difference(clean, 2), None
            )
        if self.frequency_weight:
            terms["frequency"] = self._frequency_term(clean_hat, clean)
        if self.si_sdr_weight:
            terms["si_sdr"] = -self._si_sdr(clean_hat, clean) / self.si_sdr_max
        if self.identity_weight:
            artifact = target[:, self._row_index["artifact"]] if "artifact" in self._row_index else noisy - clean
            ratio = _safe_ratio(prediction.abs().mean(), artifact.abs().mean())
            terms["identity"] = torch.clamp(ratio - 1.0, min=0.0) if self.identity_hinge else ratio

        weights = {
            "amplitude": self.amplitude_weight,
            "velocity": self.velocity_weight,
            "acceleration": self.acceleration_weight,
            "frequency": self.frequency_weight,
            "si_sdr": self.si_sdr_weight,
            "identity": self.identity_weight,
        }
        total = sum(weights[k] * v for k, v in terms.items())

        with torch.no_grad():
            self.last_terms = {k: float(v.detach()) for k, v in terms.items()}
            # Reported whatever the weights are: this is the number that would
            # have made the four deletion failures visible during training.
            self.last_terms["energy_ratio"] = float(
                _safe_ratio(clean_hat.pow(2).mean().sqrt(), clean.pow(2).mean().sqrt()).detach()
            )
        return total


#: Arguments :class:`RecoveredCleanObjective` accepts. facet-train injects
#: whatever a factory's signature will swallow — ``n_channels``, ``chunk_size``,
#: ``target_type``, ``training_config`` — so a ``**kwargs`` passthrough hands the
#: loss arguments it has never heard of and the run dies at construction.
#: Filtering against a named set keeps ``sfreq``, the one injected value the
#: frequency term actually needs, and drops the rest.
LOSS_KEYS = frozenset(
    {
        "prediction_is",
        "rows",
        "amplitude_weight",
        "velocity_weight",
        "acceleration_weight",
        "frequency_weight",
        "si_sdr_weight",
        "si_sdr_max",
        "identity_weight",
        "identity_hinge",
        "sfreq",
        "freq_band",
        "spike_weight",
        "eps",
    }
)


def build_deployment_loss(name: str = "recovered_clean", **kwargs) -> nn.Module:
    """Factory for facet-train's ``loss_factory``, shared by every edition.

    ``ic_unet_ensemble`` is the published four-term ensemble with no
    scale-invariant term; ``recovered_clean`` adds SI-SDR on top, which is what
    the deployment editions use.
    """
    normalized = name.strip().lower()
    if normalized not in {"recovered_clean", "deployment", "ic_unet_ensemble", "ensemble"}:
        raise ValueError(f"Unknown deployment loss {name!r}")
    kwargs = {k: v for k, v in kwargs.items() if k in LOSS_KEYS}
    if normalized in {"ic_unet_ensemble", "ensemble"}:
        kwargs.setdefault("si_sdr_weight", 0.0)
    kwargs["rows"] = tuple(kwargs.get("rows", ("artifact", "clean", "noisy")))
    kwargs.setdefault("prediction_is", "artifact")
    if kwargs.get("freq_band") is not None:
        kwargs["freq_band"] = tuple(kwargs["freq_band"])
    return RecoveredCleanObjective(**kwargs)
