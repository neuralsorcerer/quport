# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sphinx configuration for the QuPort documentation site."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, metadata, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "QuPort"
copyright = f"{datetime.now(UTC):%Y}, Soumyadip Sarkar"
author = "Soumyadip Sarkar"

try:
    release = version("quport")
except PackageNotFoundError:
    release = "0.1.1"
version = ".".join(release.split(".")[:2])

try:
    package_metadata = metadata("quport")
    description = package_metadata.get("Summary", "")
except PackageNotFoundError:
    description = "Research-grade multi-QPU mapping, routing, and benchmarking toolkit for Qiskit."

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_typehints = "description"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_url_schemes = ("http", "https", "mailto")

html_theme = "furo"
html_title = "QuPort Documentation"
html_static_path = ["_static"]
html_extra_path = ["_extra"]
html_css_files = ["quport.css"]
html_theme_options = {
    "source_repository": "https://github.com/neuralsorcerer/quport/",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_hide_name": False,
}
html_context = {
    "display_github": True,
    "github_user": "neuralsorcerer",
    "github_repo": "quport",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = True
html_copy_source = False
html_show_sphinx = False
html_show_copyright = True
html_baseurl = "https://neuralsorcerer.github.io/quport/"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit", None),
}

linkcheck_ignore = [
    r"https://github\.com/neuralsorcerer/quport/(actions|releases).*",
]
linkcheck_anchors = True
linkcheck_timeout = 30
linkcheck_retries = 2

nitpicky = True
nitpick_ignore = [
    ("py:class", "qiskit.circuit.quantumcircuit.QuantumCircuit"),
]
