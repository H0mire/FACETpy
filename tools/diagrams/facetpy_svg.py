"""Compatibility import for the canonical, repository-local FACETpy SVG toolkit."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_asset = Path(__file__).resolve().parents[2] / ".agents/skills/facetpy-diagram/assets/facetpy_svg.py"
_spec = spec_from_file_location("_facetpy_svg_toolkit", _asset)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load FACETpy diagram toolkit: {_asset}")
_toolkit = module_from_spec(_spec)
_spec.loader.exec_module(_toolkit)
# Preserve the complete historical helper API, including private geometry helpers.
globals().update({name: value for name, value in vars(_toolkit).items() if not name.startswith("__")})
