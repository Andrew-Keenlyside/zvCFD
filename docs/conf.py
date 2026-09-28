# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#
# Mirrors zarr-vectors-py's docs/conf.py (Furo + MyST), so the two sites
# read as one ecosystem.

import os
import sys

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "zvCFD"
copyright = "2026, BRIDGE Neuroscience. Built on Zarr Vectors (zarr-vectors-py)."
author = "BRIDGE Neuroscience"
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("zvcfd")
except Exception:  # pragma: no cover - docs build without an install
    release = "0.0.0+unknown"
version = release.split("+")[0]

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_inline",
    "dollarmath",
]
myst_heading_anchors = 3

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
# Dataclass "Attributes:" sections render as :ivar: fields, so they do not
# duplicate the attribute entries autodoc makes for the same fields.
napoleon_use_ivar = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
# GPU and store modules import cupy / zarr_vectors; mock them so the docs
# build on Read the Docs without a GPU.
autodoc_mock_imports = ["cupy", "cupyx", "zarr_vectors", "pyamg", "icechunk", "kvikio"]

autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
    "zarr_vectors": ("https://zarr-vectors-py.readthedocs.io/en/latest", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "zvCFD"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#e31e72",
        "color-brand-content": "#e31e72",
        "font-stack": "'DM Sans', sans-serif",
        "font-stack--monospace": "'JetBrains Mono', monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff72c0",
        "color-brand-content": "#ff72c0",
    },
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/BRIDGE-Neuroscience/zvCFD/",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_logo = "_static/zvcfd-logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "github_user": "BRIDGE-Neuroscience",
    "github_repo": "zvCFD",
    "github_version": "main",
    "doc_path": "docs",
}

# -- copybutton ---------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
