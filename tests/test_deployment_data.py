"""The training-side packing must be the inference-side packing.

``facet.models.masterthesis.adapters`` already encodes how each family wants
its input; :mod:`facet.training.deployment_data` encodes it again for training.
Two encodings of one contract is how they drift, and a packing mismatch is
silent — the model trains happily on transposed data and predicts something
plausible-looking that is not the artifact. So the two are compared element for
element here rather than by inspection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

from facet.training.deployment_data import (  # noqa: E402
    PER_CHANNEL_PACKINGS,
    PackedDeploymentDataset,
    _pack_context,
)
from facet.training.deployment_model import PACKINGS  # noqa: E402

T, C, S, N = 7, 4, 32, 3


class _FakeBase:
    """Stands in for NPZContextArtifactDataset with target_extras set."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.ctx = rng.standard_normal((N, T, C, S)).astype(np.float32)
        clean = rng.standard_normal((N, C, S)).astype(np.float32)
        artifact = self.ctx[:, T // 2] - clean
        self.target = np.stack([artifact, clean, self.ctx[:, T // 2]], axis=1)
        self.sfreq = 1000.0
        self.target_rows = ("artifact", "clean", "noisy")

    def __len__(self) -> int:
        return N

    def __getitem__(self, i: int):
        return self.ctx[i], self.target[i]


@pytest.mark.unit
@pytest.mark.parametrize("packing", sorted(PACKINGS))
def test_packing_matches_the_inference_adapter(packing):
    """Same context in, same tensor out, as ``predict_from_context`` builds it."""
    from facet.models.masterthesis.adapters import PackingSpec, _demean

    base = _FakeBase()
    ds = PackedDeploymentDataset(base, packing=packing)
    spec = PackingSpec(
        "x",
        "x",
        "single" if packing == "b1s" else "stack",
        packing,
        "none",
        "artifact",
        multichannel=packing not in PER_CHANNEL_PACKINGS,
    )

    # The inference side's own packing arithmetic, lifted from
    # predict_from_context with demeaning disabled on both sides.
    ctx = base.ctx.astype(np.float32)
    n, t, c, s = ctx.shape
    centre = t // 2
    if spec.multichannel:
        expected = {
            "bcts": lambda: ctx.transpose(0, 2, 1, 3).reshape(n, c, t * s),
            "bcs": lambda: ctx[:, centre],
            "btcs": lambda: ctx,
        }[packing]()
    elif packing == "b1s":
        expected = ctx[:, centre].transpose(0, 1, 2).reshape(n * c, 1, s)
    else:
        stack = ctx.transpose(0, 2, 1, 3).reshape(n * c, t, s)
        expected = {
            "bt1s": lambda: stack[:, :, None, :],
            "bts": lambda: stack,
            "b1ts": lambda: stack.reshape(n * c, 1, t * s),
        }[packing]()
    expected = _demean(expected, spec.demean)

    got = np.stack([ds[i][0] for i in range(len(ds))])
    assert got.shape == expected.shape, packing
    assert np.array_equal(got, expected), packing


@pytest.mark.unit
@pytest.mark.parametrize("packing", sorted(PACKINGS))
def test_declared_target_shape_matches_what_is_returned(packing):
    ds = PackedDeploymentDataset(_FakeBase(), packing=packing)
    assert ds[0][1].shape == ds.target_shape
    assert ds.target_rows == ("artifact", "clean", "noisy")


@pytest.mark.unit
def test_target_rows_still_decompose_after_packing():
    """``noisy = clean + artifact`` has to survive the channel slice."""
    ds = PackedDeploymentDataset(_FakeBase(), packing="bt1s")
    for i in (0, 5, len(ds) - 1):
        _, target = ds[i]
        assert np.abs(target[2] - (target[0] + target[1])).max() < 1e-5


@pytest.mark.unit
def test_the_split_never_puts_two_channels_of_one_window_on_both_sides():
    """Channels of the same window share the artifact epoch and the background.

    Splitting on the flat index would let the validation loss measure
    memorisation and report it as generalisation.
    """
    ds = PackedDeploymentDataset(_FakeBase(), packing="b1s")
    train, val = ds.train_val_split(val_ratio=0.34, seed=0)
    train_windows = {i // C for i in train._indices}
    val_windows = {i // C for i in val._indices}
    assert train_windows and val_windows
    assert not (train_windows & val_windows)
    assert len(train) + len(val) == len(ds)


@pytest.mark.unit
def test_multichannel_split_is_by_example():
    ds = PackedDeploymentDataset(_FakeBase(), packing="btcs")
    train, val = ds.train_val_split(val_ratio=0.34, seed=0)
    assert not (set(train._indices) & set(val._indices))
    assert len(train) + len(val) == N


@pytest.mark.unit
def test_unknown_packing_is_rejected_at_both_entry_points():
    with pytest.raises(ValueError, match="unknown packing"):
        _pack_context(np.zeros((T, S), dtype=np.float32), "nope")
    with pytest.raises(ValueError, match="unknown packing"):
        PackedDeploymentDataset(_FakeBase(), packing="nope")
