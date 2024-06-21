#
# Copyright (c)2023. IonQ, Inc. All rights reserved.
#
import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IonQ Vision'
copyright = '2024, IonQ'
author = 'Willie Aboumrad, IonQ'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add Python source path
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

extensions = [
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.viewcode',
    'sphinx_math_dollar', 
    'sphinx.ext.mathjax'
]

# Add bibliography
bibtex_bibfiles = ['ref.bib']

# Configure doctests... Always import optimization module
doctest_global_setup = """
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..')))
from ionqvision.applications.aircraft_loading_client import * 
"""

# Configure sphinx_math_dollar to enable TeX-ing expressions wrapped in $
mathjax_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
}

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['media']
