# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Keep MNE config/cache inside the docs workspace during builds.
_mne_home = os.path.abspath('../.mne')
os.makedirs(_mne_home, exist_ok=True)
os.environ.setdefault("MNE_HOME", _mne_home)
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

sys.path.insert(0, os.path.abspath('../../src/'))

project = 'FACETpy'
copyright = '2025, FACETpy Team'
author = 'FACETpy Team'
release = '2.0.0'
version = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'myst_parser',  # For markdown support
]

# Autosummary settings
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'mne': ('https://mne.tools/stable/', None),
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# Logo is theme-dependent (light/dark) via layout template override
html_logo = '_static/logo_light_theme.png'

html_css_files = ['css/theme_logo.css']

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True
