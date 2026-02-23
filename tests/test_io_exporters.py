"""Tests for EEG data exporters."""

from pathlib import Path

import pytest

from facet.core import ProcessorValidationError
from facet.io.exporters import (
    SUPPORTED_EXPORT_EXTENSIONS,
    EDFExporter,
    EEGLABExporter,
    Exporter,
    FIFExporter,
)
from facet.io.loaders import SUPPORTED_EXTENSIONS

pytestmark = pytest.mark.unit


def test_top_level_export_convenience_function(monkeypatch, sample_context, temp_dir):
    import facet

    captured = {}

    def fake_execute(self, context):
        captured["context"] = context
        captured["path"] = self.path
        captured["overwrite"] = self.overwrite
        return context

    monkeypatch.setattr(Exporter, "execute", fake_execute)

    out_path = temp_dir / "convenience.set"
    result = facet.export(sample_context, str(out_path), overwrite=False)

    assert result is sample_context
    assert captured["context"] is sample_context
    assert captured["path"] == str(out_path)
    assert captured["overwrite"] is False


def test_supported_export_extensions_cover_loader_extensions():
    for ext in SUPPORTED_EXTENSIONS:
        assert ext in SUPPORTED_EXPORT_EXTENSIONS


def test_exporter_routes_to_edf_exporter(monkeypatch, sample_context, temp_dir):
    captured = {}

    def fake_process(self, context):
        captured["path"] = self.path
        captured["overwrite"] = self.overwrite
        return context

    monkeypatch.setattr(EDFExporter, "process", fake_process)

    out_path = temp_dir / "out.edf"
    result = Exporter(path=str(out_path), overwrite=False).execute(sample_context)

    assert result is sample_context
    assert captured["path"] == str(out_path)
    assert captured["overwrite"] is False


def test_exporter_routes_fif_gz_to_fif_exporter(monkeypatch, sample_context, temp_dir):
    captured = {}

    def fake_process(self, context):
        captured["path"] = self.path
        captured["overwrite"] = self.overwrite
        return context

    monkeypatch.setattr(FIFExporter, "process", fake_process)

    out_path = temp_dir / "out.fif.gz"
    result = Exporter(path=str(out_path), overwrite=True).execute(sample_context)

    assert result is sample_context
    assert captured["path"] == str(out_path)
    assert captured["overwrite"] is True


def test_eeglab_exporter_uses_set_format(monkeypatch, sample_context, temp_dir):
    captured = {}
    raw_type = type(sample_context.get_raw())

    def fake_export(self, fname, fmt="auto", *args, **kwargs):
        captured["fname"] = fname
        captured["fmt"] = fmt
        captured["overwrite"] = kwargs.get("overwrite")

    monkeypatch.setattr(raw_type, "export", fake_export)

    out_path = temp_dir / "out.set"
    result = EEGLABExporter(path=str(out_path), overwrite=False).execute(sample_context)

    assert result is sample_context
    assert captured["fname"] == str(out_path)
    assert captured["fmt"] == "eeglab"
    assert captured["overwrite"] is False


def test_exporter_rejects_gdf_with_clear_error(sample_context, temp_dir):
    out_path = temp_dir / "out.gdf"
    with pytest.raises(ProcessorValidationError, match="GDF export is not supported"):
        Exporter(path=str(out_path)).execute(sample_context)


def test_exporter_rejects_unknown_extension(sample_context, temp_dir):
    out_path = Path(temp_dir) / "out.unknown"
    with pytest.raises(ProcessorValidationError, match="Unsupported export extension"):
        Exporter(path=str(out_path)).execute(sample_context)
