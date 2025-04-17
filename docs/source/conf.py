# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information

project = 'Mine-DD'
copyright = '2025, Viviani, Daza'
author = 'Eva Viviani and Angel Daza'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to view source code
    'sphinx_autodoc_typehints',  # Better support for type hints
    'myst_parser',  # Markdown support
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Enable autosummary
autosummary_generate = True

# autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
