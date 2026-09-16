"""Check model coverage, print limits and unsafe or ambiguous diagram inputs."""

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def renderer():
    directory = ROOT / "tools/diagrams"
    spec = importlib.util.spec_from_file_location("model_diagram_renderer", directory / "build_model_diagrams.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


def test_every_catalog_model_has_current_printable_diagrams(renderer):
    catalog = yaml.safe_load((ROOT / "masterthesis_guide/catalog.yaml").read_text())
    for model in catalog["models"].values():
        directory = ROOT / Path(model["readme"]).parent / "diagrams"
        profile = renderer.load_profile(directory / "architecture.json")
        overview = renderer.render_overview(profile)
        assert (directory / "overview.svg").read_text() == overview
        assert (directory / "architecture.svg").read_text() == renderer.render_architecture(profile)
        element = ET.fromstring(overview)
        assert element.attrib["width"].endswith("mm")
        assert element.attrib["height"].endswith("mm")
        assert float(element.attrib["width"][:-2]) <= 190
        assert float(element.attrib["height"][:-2]) <= 128
        assert float(element.attrib["viewBox"].split()[2]) == 1000


@pytest.fixture
def profile_file(tmp_path):
    model = tmp_path / "model"
    directory = model / "diagrams"
    directory.mkdir(parents=True)
    (model / "training.py").write_text("# Implementation source\n")
    profile = {
        "schema_version": 1,
        "model_dir": "model",
        "title": "Example",
        "overview": [{"title": "Input", "description": "EEG waveform"}],
        "nodes": [{"id": "input", "title": "Input", "description": ["EEG"], "row": 0}],
        "edges": [],
        "sources": [{"path": "model/training.py", "label": "Implementation"}],
    }
    path = directory / "architecture.json"
    path.write_text(json.dumps(profile))
    return path, profile, tmp_path


def test_unknown_edge_endpoint_is_rejected(renderer, profile_file):
    path, profile, root = profile_file
    profile["edges"] = [{"source": "input", "target": "missing", "kind": "flow", "label": "Signal"}]
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="unknown node"):
        renderer.load_profile(path, root)


def test_overlapping_profile_nodes_are_rejected(renderer, profile_file):
    path, profile, root = profile_file
    profile["nodes"].append({"id": "second", "title": "Hidden node", "description": [], "row": 0})
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="overlapping"):
        renderer.load_profile(path, root)


def test_source_cannot_escape_checkout(renderer, profile_file):
    path, profile, root = profile_file
    profile["sources"][0]["path"] = "../outside.py"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="unsafe source"):
        renderer.load_profile(path, root)


def test_duplicate_profile_keys_are_rejected(renderer, profile_file):
    path, _, root = profile_file
    path.write_text('{"schema_version": 1, "schema_version": 2}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        renderer.load_profile(path, root)


def test_connectors_cannot_cross_unrelated_cards(renderer):
    boxes = {"hidden": {"x": 10, "y": 10, "w": 40, "h": 40}}
    with pytest.raises(ValueError, match="crosses node"):
        renderer.assert_route_clear([(0, 30), (60, 30)], boxes)
    renderer.assert_route_clear([(0, 0), (60, 0), (60, 60)], boxes)
