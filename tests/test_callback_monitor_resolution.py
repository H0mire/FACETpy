"""Regression tests for callback metric-name resolution.

``TrainingState`` keys its metric dicts plainly (``"loss"``), while the JSONL log
prefixes them (``"train_loss"``/``"val_loss"``). Callbacks used to look up the
monitor name in the *unprefixed* merge only, so the documented and natural
``monitor="val_loss"`` matched nothing: checkpoint ranking and early stopping
were silently disabled and the reported best metric came out as NaN.
"""

from __future__ import annotations

import pytest

from facet.training.callbacks import EarlyStoppingCallback, _resolve_monitored
from facet.training.trainer import TrainingState


def _state(train: float | None = 0.5, val: float | None = 0.25) -> TrainingState:
    state = TrainingState()
    state.epoch = 1
    state.train_metrics = {} if train is None else {"loss": train}
    state.val_metrics = {} if val is None else {"loss": val}
    return state


@pytest.mark.unit
@pytest.mark.parametrize(
    ("monitor", "expected"),
    [
        ("val_loss", 0.25),    # prefixed — the name users see in training.jsonl
        ("train_loss", 0.5),   # prefixed
        ("loss", 0.25),        # bare: val wins, preserving the old merge order
    ],
)
def test_resolve_monitored_accepts_both_spellings(monitor, expected):
    assert _resolve_monitored(_state(), monitor) == expected


@pytest.mark.unit
def test_resolve_monitored_missing_metric_is_none():
    assert _resolve_monitored(_state(), "auroc") is None


@pytest.mark.unit
def test_resolve_monitored_falls_back_to_train_when_no_validation():
    state = _state(train=0.7, val=None)
    assert _resolve_monitored(state, "train_loss") == 0.7
    assert _resolve_monitored(state, "loss") == 0.7
    assert _resolve_monitored(state, "val_loss") is None


@pytest.mark.unit
def test_early_stopping_triggers_on_prefixed_monitor():
    """The end-to-end symptom: patience never elapsed with monitor='val_loss'."""
    cb = EarlyStoppingCallback(monitor="val_loss", mode="min", patience=2, min_delta=0.0)
    state = _state(val=1.0)
    cb.on_epoch_end(state)
    assert not state.stop_training
    for epoch in (2, 3):
        state.epoch = epoch
        state.val_metrics = {"loss": 2.0}   # worse every time
        cb.on_epoch_end(state)
    assert state.stop_training
