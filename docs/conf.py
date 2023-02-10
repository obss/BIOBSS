# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))

master_doc = "index"

project = "BIOBSS"
copyright = "2022"
author = "OBSS Technology"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = True
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = True
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_sphinx = False

autodoc_type_aliases = {"Iterable": "Iterable", "ArrayLike": "ArrayLike"}

autodoc_typehints = "description"
add_module_names = False


html_theme_options = {
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": -1,
    "includehidden": True,
    "titles_only": False,
}
