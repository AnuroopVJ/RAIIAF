# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
<<<<<<< HEAD
project = 'raiiaf'
=======

project = 'RAIIAF'
>>>>>>> 59e07a4cc6f4af6606c7a081077a79dd15b2bc45
copyright = '2026, Anuroop V J'
author = 'Anuroop V J'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Generate autosummary stub pages for modules/classes/functions listed
# in autosummary directives (e.g., in api.rst)
autosummary_generate = True

# Reasonable defaults for autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# Napoleon to support Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
<<<<<<< HEAD
# Ensure Python can import the 'raiiaf' package. This should point to the directory
# that contains the 'raiiaf' package folder (i.e., src/raiiaf), not to the package itself.
sys.path.insert(0, os.path.abspath('../../../../'))
=======

# Ensure Python can import the 'gen5' package
# Adjust the path to point to where your package is located
sys.path.insert(0, os.path.abspath('../../../../'))
>>>>>>> 59e07a4cc6f4af6606c7a081077a79dd15b2bc45
