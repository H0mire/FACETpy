"""CPU smoke tests for the paper-accurate Nested-GAN edition.

Every architecture dimension is shrunk (channels=8, levels=2, 1 block per
level, n_fft=16, target_samples=64, batch=2-4, ~5 optimizer steps) so the
whole module runs in a couple of seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from facet.models.nested_gan_paper_accurate_edition.training import (
    ChannelWiseContextArtifactDataset,
    HierarchicalSpectrogramRestormer,
    NestedGANGenerator,
    NestedGANLoss,
    build_dataset,
    build_loss,
    build_model,
)

# Tiny hierarchical config reused across tests.
_SMOKE_KWARGS = dict(
    input_shape=(3, 1, 64),
    inner_channels=8,
    inner_levels=2,
    inner_block_counts=[1, 1, 1],
    inner_head_counts=[1, 2, 2],
    inner_refinement_blocks=1,
    inner_expansion=2.0,
    outer_base_channels=8,
    n_fft=16,
    hop_length=8,
    win_length=16,
)


@pytest.mark.unit
def test_build_model_returns_hierarchical_generator_with_correct_shape():
    torch = pytest.importorskip("torch")
    model = build_model(**_SMOKE_KWARGS)
    assert isinstance(model, NestedGANGenerator)
    # The inner branch must be the new hierarchical Restormer, not a flat stack.
    assert isinstance(model.inner, HierarchicalSpectrogramRestormer)
    assert model.inner.levels == 2
    # GDFN expansion default is the Restormer value when not overridden.
    assert build_model(input_shape=(3, 1, 64), n_fft=16, hop_length=8, win_length=16).inner is not None

    x = torch.randn(2, 3, 1, 64)
    y = model(x)
    assert tuple(y.shape) == (2, 1, 64)


@pytest.mark.unit
def test_forward_shape_with_default_expansion_is_2_66():
    # Confirms the paper-faithful GDFN gamma default propagates.
    from facet.models.nested_gan_paper_accurate_edition.training import _GDFN

    gdfn = _GDFN(8)  # default expansion
    # hidden = int(8 * 2.66) = 21 -> project_in produces 2*hidden channels.
    assert gdfn.project_in.out_channels == 21 * 2


@pytest.mark.unit
def test_generator_trains_loss_decreases():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = build_model(**_SMOKE_KWARGS)
    loss_fn = build_loss(lambda_time=1.0, lambda_mrstft=0.5, fft_sizes=[16, 32], hop_fraction=0.25)

    # A simple deterministic target the network can fit a little in a few steps.
    x = torch.randn(4, 3, 1, 64)
    target = torch.randn(4, 1, 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    initial_loss = None
    final_loss = None
    for step in range(5):
        optimizer.zero_grad()
        pred = model(x)
        assert tuple(pred.shape) == (4, 1, 64)
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)
        if step == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss


@pytest.mark.unit
def test_inner_branch_uses_global_residual():
    torch = pytest.importorskip("torch")
    # If the output_proj produced exact zeros (R=0), the inner artifact should
    # equal the iSTFT of the unmodified center spectrogram (global residual).
    inner = HierarchicalSpectrogramRestormer(
        n_fft=16,
        hop_length=8,
        win_length=16,
        in_channels=2,
        base_channels=8,
        levels=2,
        block_counts=(1, 1, 1),
        head_counts=(1, 2, 2),
        refinement_blocks=1,
        target_samples=64,
    )
    with torch.no_grad():
        inner.output_proj.weight.zero_()
        center = torch.randn(2, 64)
        out = inner(center.unsqueeze(1))
        # iSTFT(STFT(center)) reconstructs center (up to edge effects); with R=0
        # the global-residual branch returns exactly that reconstruction.
        recon = inner._istft(inner._stft(center))
        torch.testing.assert_close(out, recon, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_inner_neighbor_epochs_changes_input_channels():
    torch = pytest.importorskip("torch")
    model = build_model(
        input_shape=(5, 1, 64),
        inner_channels=8,
        inner_levels=1,
        inner_block_counts=[1, 1],
        inner_head_counts=[1, 2],
        inner_refinement_blocks=0,
        inner_neighbor_epochs=1,
        outer_base_channels=8,
        n_fft=16,
        hop_length=8,
        win_length=16,
    )
    # center + 2 neighbours = 3 time channels -> 6 STFT (real/imag) channels.
    assert model.inner.in_channels == 6
    x = torch.randn(2, 5, 1, 64)
    assert tuple(model(x).shape) == (2, 1, 64)


@pytest.mark.unit
def test_bias_free_layernorm_has_no_bias():
    from facet.models.nested_gan_paper_accurate_edition.training import _LayerNorm2d

    ln = _LayerNorm2d(8, bias_free=True)
    assert ln.bias is None
    ln_default = _LayerNorm2d(8)
    assert ln_default.bias is not None


@pytest.mark.unit
def test_nested_gan_loss_is_non_negative_scalar():
    torch = pytest.importorskip("torch")
    loss_fn = NestedGANLoss(lambda_time=1.0, lambda_mrstft=0.5, fft_sizes=(16, 32), hop_fraction=0.25)
    pred = torch.randn(3, 1, 64)
    target = pred + 0.01 * torch.randn_like(pred)
    value = loss_fn(pred, target)
    assert value.dim() == 0
    assert float(value) >= 0.0


@pytest.mark.unit
def test_build_dataset_loads_synthetic_npz(tmp_path):
    path = tmp_path / "context.npz"
    rng = np.random.default_rng(0)
    n_examples, context_epochs, n_channels, n_samples = 4, 3, 2, 64
    clean = rng.standard_normal((n_examples, context_epochs, n_channels, n_samples)).astype(np.float32)
    artifact = rng.standard_normal((n_examples, n_channels, n_samples)).astype(np.float32)
    noisy = clean.copy()
    # noisy_context is the full context; artifact_center is the center target.
    np.savez(
        path,
        noisy_context=noisy,
        artifact_center=artifact,
        clean_center=clean[:, context_epochs // 2],
        sfreq=np.asarray([5000.0]),
    )

    dataset = build_dataset(path=str(path), context_epochs=context_epochs, demean_input=False, demean_target=False)
    assert isinstance(dataset, ChannelWiseContextArtifactDataset)
    assert dataset.input_shape == (3, 1, 64)
    assert dataset.target_shape == (1, 64)
    assert len(dataset) == n_examples * n_channels
    assert dataset.target_type == "artifact"
    assert dataset.trigger_aligned is True

    noisy_item, target_item = dataset[0]
    assert noisy_item.shape == (3, 1, 64)
    assert target_item.shape == (1, 64)

    train, val = dataset.train_val_split(val_ratio=0.25, seed=1)
    assert len(train) + len(val) == len(dataset)
