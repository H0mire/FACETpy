"""Explicit artifact loading and preservation of the recorded input contracts."""
from pathlib import Path

import numpy as np
import pytest
import torch

from facet.models.masterthesis.adapters import (
    FAMILY_SPECS, FamilyAdapter, load_model, predict_from_context, require_artifact,
)


def test_lfs_pointer_is_rejected_before_loading(tmp_path):
    pointer = tmp_path / 'weights.ts'
    pointer.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:' + '0' * 64 + '\nsize 12\n')
    with pytest.raises(FileNotFoundError, match='Git LFS pointer'):
        load_model('dpae', checkpoint=pointer)


def test_export_loads_outside_the_checkout(tmp_path, monkeypatch):
    model = torch.nn.Conv1d(1, 1, 1, bias=False).eval()
    with torch.no_grad():
        model.weight.fill_(0.25)
    path = tmp_path / 'model.ts'
    torch.jit.trace(model, torch.ones(1, 1, 8)).save(str(path))
    monkeypatch.chdir(tmp_path)
    loaded = load_model('dpae', checkpoint=path)
    torch.testing.assert_close(loaded(torch.ones(1, 1, 8)), torch.full((1, 1, 8), 0.25))


def test_state_dict_uses_explicit_factory_and_strict_shape(tmp_path):
    model = torch.nn.Linear(4, 2)
    path = tmp_path / 'model.pt'
    torch.save({'model_state_dict': model.state_dict()}, path)
    loaded = load_model('example', checkpoint=path, model_factory=torch.nn.Linear,
                        model_kwargs={'in_features': 4, 'out_features': 2})
    torch.testing.assert_close(loaded.weight, model.weight)
    with pytest.raises(RuntimeError, match='size mismatch'):
        load_model('example', checkpoint=path, model_factory=torch.nn.Linear,
                   model_kwargs={'in_features': 5, 'out_features': 2})


def test_single_epoch_contract_uses_only_the_centre_epoch():
    context = np.arange(3 * 7 * 2 * 16, dtype=np.float32).reshape(3, 7, 2, 16)
    changed = context.copy()
    changed[:, [0, 1, 2, 4, 5, 6]] += 1000
    model = torch.nn.Identity().eval()
    a = predict_from_context(FAMILY_SPECS['dpae'], model, context, device='cpu')
    b = predict_from_context(FAMILY_SPECS['dpae'], model, changed, device='cpu')
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, context[:, 3] - context[:, 3].mean(axis=-1, keepdims=True))


def test_adapter_does_not_search_for_a_repository_checkpoint():
    adapter = FamilyAdapter('demucs')
    with pytest.raises(ValueError, match='explicit checkpoint'):
        adapter._load_model()
