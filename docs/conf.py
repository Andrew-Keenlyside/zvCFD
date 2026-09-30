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
GITHUB = "https://github.com/Andrew-Keenlyside/zvCFD"
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
    "source_repository": f"{GITHUB}/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": GITHUB,
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 '
                '3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37'
                '-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
                '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64'
                '-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 '
                '2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82'
                '.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95'
                '.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 '
                '0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

html_logo = "_static/zvcfd-icon.png"
html_favicon = "_static/favicon.ico"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_context = {
    "github_user": "Andrew-Keenlyside",
    "github_repo": "zvCFD",
    "github_version": "main",
    "doc_path": "docs",
}

# -- copybutton ---------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
