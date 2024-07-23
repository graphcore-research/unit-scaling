# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# Copyright (c) 2022 Read the Docs Inc. All rights reserved.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Setup based on https://example-sphinx-basic.readthedocs.io
import os
import sys
from unit_scaling._version import __version__

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "unit-scaling"
copyright = "(c) 2023 Graphcore Ltd. All rights reserved"
author = "Charlie Blake, Douglas Orr"
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",  # support for google-style docstrings
    "myst_parser",  # support for including markdown files in .rst files (e.g. readme)
    "sphinx.ext.viewcode",  # adds source code to docs
    "sphinx.ext.autosectionlabel",  # links to sections in the same document
    "sphinx.ext.mathjax",  # equations
]

autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
napoleon_google_docstring = True
napoleon_numpy_docstring = False

numfig_format = {
    "section": "Section {number}. {name}",
    "figure": "Fig. %s",
    "table": "Table %s",
    "code-block": "Listing %s",
}

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "pytorch": ("https://pytorch.org/docs/stable/", None),
}
intersphinx_disabled_domains = ["std"]

autodoc_type_aliases = {
    "BinaryConstraint": "unit_scaling.constraints.BinaryConstraint",
    "TernaryConstraint": "unit_scaling.constraints.TernaryConstraint",
    "VariadicConstraint": "unit_scaling.constraints.VariadicConstraint",
}  # make docgen output name of alias rather than definition.

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_favicon = "_static/scales.png"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
