# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations
import os
import sys


sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))


project = 'biobss'
copyright = '2022, me'
author = 'me'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions=['sphinx.ext.autodoc', 'sphinx.ext.todo', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax',
 'sphinx.ext.intersphinx', 'sphinx.ext.autosectionlabel', 'sphinx.ext.viewcode']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_material'
html_static_path = ['_static']

autodoc_type_aliases = {
    'Iterable': 'Iterable',
    'ArrayLike': 'ArrayLike'
}