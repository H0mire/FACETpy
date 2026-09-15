"""Every deployment edition builds, runs, and traces — checked for all thirteen.

The editions are thin by design: a family's own network wrapped in
:class:`facet.training.deployment_model.PackedDeploymentModel`. Thin does not
mean safe. Each one still has to agree with its base edition about the tensor
layout, has to survive ``torch.jit.trace`` because the pipeline arm loads a
TorchScript export, and has to be constructible without a dataset on disk — the
last one because ``build_model`` leans on values facet-train injects, and a
default that is only ever supplied by the CLI is a default nobody tested.
"""

from __future__ import annotations

import importlib
import io
import warnings

import pytest
import yaml

torch = pytest.importorskip("torch")

from facet.training.deployment_data import model_input_shape  # noqa: E402

EDITIONS = [
    "ic_unet",
    "vit_spectrogram",
    "denoise_mamba",
    "st_gnn",
    "demucs",
    "conv_tasnet",
    "sepformer",
    "nested_gan",
    "cascaded_dae",
    "cascaded_context_dae",
    "dpae",
    "dhct_gan_v2",
    "dhct_gan",
]
S, T, C = 512, 7, 30


def _build(family: str):
    name = "dhct_gan.v2" if family == "dhct_gan_v2" else family
    module = importlib.import_module(f"facet.models.masterthesis.{name}.deployment.training")
    with open(f"tests/fixtures/deployment_configs/{family}_deployment_edition.yaml") as stream:
        cfg = yaml.safe_load(stream)
    kwargs = dict(cfg["model"].get("kwargs") or {})
    kwargs.pop("dataset_path", None)
    kwargs["fit_ica"] = False  # no ICA fit without the bundle
    return module, module.build_model(**kwargs).eval(), cfg


@pytest.mark.unit
@pytest.mark.parametrize("family", EDITIONS)
def test_edition_builds_without_a_dataset_and_returns_the_centre_epoch(family):
    module, model, _ = _build(family)
    shape = model_input_shape(module.PACKING, n_channels=C, context_epochs=T, epoch_samples=S)
    with torch.no_grad():
        out = model(torch.randn(2, *shape))
    assert out.shape[0] == 2
    assert out.shape[-1] == S, f"{family}: output is not one epoch long"
    assert torch.isfinite(out).all()


@pytest.mark.unit
@pytest.mark.parametrize("family", EDITIONS)
def test_edition_traces_to_torchscript(family):
    """The pipeline arm loads a TorchScript export, so a model that cannot be
    traced cannot be deployed however well it trains."""
    module, model, _ = _build(family)
    shape = model_input_shape(module.PACKING, n_channels=C, context_epochs=T, epoch_samples=S)
    x = torch.randn(1, *shape)
    with torch.no_grad():
        expected = model(x)
    with warnings.catch_warnings():
        # Tracing reports every Python-level branch on a tensor as "might cause
        # the trace to be incorrect". For a fixed input shape they are constant,
        # which is exactly what the reload comparison below checks.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(model, x, strict=False)
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    buf.seek(0)
    with torch.no_grad():
        assert torch.allclose(torch.jit.load(buf)(x), expected, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("family", EDITIONS)
def test_every_predicted_epoch_leaves_with_zero_mean(family):
    module, model, _ = _build(family)
    shape = model_input_shape(module.PACKING, n_channels=C, context_epochs=T, epoch_samples=S)
    with torch.no_grad():
        out = model(torch.randn(2, *shape) * 100.0 + 7.0)
    scale = float(out.abs().max())
    assert float(out.mean(dim=-1).abs().max()) <= max(1e-5, 1e-4 * scale), family


@pytest.mark.unit
@pytest.mark.parametrize("family", EDITIONS)
def test_config_points_at_the_edition_and_monitors_validation(family):
    """A config that still points at the base module trains the wrong model, and
    ``monitor: loss`` is what made two GAN runs pick epoch 1 of 16 and 1 of 34."""
    _, _, cfg = _build(family)
    name = "dhct_gan.v2" if family == "dhct_gan_v2" else family
    module = f"masterthesis.{name}.deployment"
    for key in ("factory", "loss_factory"):
        assert cfg["model"][key].startswith(f"facet.models.{module}.")
    assert cfg["data"]["dataset_factory"].startswith(f"facet.models.{module}.")
    assert cfg["checkpoint"]["monitor"] == "val_loss"
    assert cfg["early_stopping"]["monitor"] == "val_loss"
    assert cfg["early_stopping"]["min_delta_rel"] > 0.0
    assert cfg["export"]["path"] == f"exports/{family}_deployment.ts"


@pytest.mark.unit
def test_the_registry_and_the_editions_agree_on_which_families_exist():
    """A spec without an edition cannot be trained; an edition without a spec
    cannot be run through a pipeline. Either way the mismatch is silent."""

    from facet.models.masterthesis.adapters import DEPLOYMENT_SPECS

    assert {k.removesuffix("_deployment") for k in DEPLOYMENT_SPECS} == set(EDITIONS)
    for key, spec in DEPLOYMENT_SPECS.items():
        family = key.removesuffix("_deployment")
        name = "dhct_gan.v2" if family == "dhct_gan_v2" else family
        module = importlib.import_module(f"facet.models.masterthesis.{name}.deployment.training")
        assert spec.packing == module.PACKING, family
        assert spec.demean == "none", family
        assert not spec.remove_prediction_dc, family


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(512,), (1, 512), (30, 512), (3, 1, 512), (3, 30, 512)])
def test_prediction_snapshots_survive_a_row_stacked_target(shape):
    """The plotting helper must reach 1-D from any rank, not from rank 2 only.

    It peeled exactly one axis after squeezing, so a ``(3, 1, 512)`` target — the
    layout every deployment edition now uses — came back two-dimensional and
    matplotlib raised *after* training finished, killing the TorchScript export
    with it. A model that trains for an hour and then cannot be deployed is
    indistinguishable from one that failed.
    """
    import numpy as np

    from facet.training.callbacks import _to_1d

    out = _to_1d(np.zeros(shape, dtype=np.float32))
    assert out.ndim == 1
    assert out.shape == (512,)
