# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../../EELSFitter/src/'))
# import pathlib

# source_dir = pathlib.Path('../../EELS_KK/pyfiles').parent

# -- Project information -----------------------------------------------------

project = 'EELSFitter'
copyright = '2025, EELSFitter developer team'
author = 'EELSFitter developer team'

# The full version, including alpha/beta/rc tags
release = '3.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosectionlabel', 'sphinx.ext.coverage', 'sphinx.ext.doctest', 'sphinx.ext.extlinks', 'sphinx.ext.ifconfig', 'sphinx.ext.intersphinx', 'sphinx.ext.mathjax', 'sphinx.ext.napoleon', 'sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autosummary', 'sphinxcontrib.bibtex', 'nbsphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Bibtex
bibtex_bibfiles = ['EELSFitter_biblio.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'

# Markdown configuration
numfig = True

# The suffix(es) of source filenames.
# you can specify multiple suffix as a list of string:
source_suffix = {".rst": "restructuredtext", ".txt": "restructuredtext", ".py": "restructuredtext"}

autosectionlabel_prefix_document = True
# autosectionlabel_maxdepth = 10
# Allow to embed rst syntax in markdown files.
enable_eval_rst = True

# The master toctree document/
master_doc = "index"
# bibtex_bibfiles = ["refs.bib"]

# The language for content autogenerated by Sphinx, Refer to documentation
# for a list of supported languages.
# This is also used if you do content translation via gettext catalogs.
# Usually you set "Language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = None

# A string to be included at the beginning of all files
# shared = pathlib.Path(__file__).absolute().parent / "shared"
# rst_prolog = "\n".join([open(x).read() for x in os.scandir(shared)])

extlinks = {}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ['css/custom.css']

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
# Thanks https://github.com/bskinn/sphobjinv
intersphinx_mapping = {"python": ("https://docs.python.org/3/", None), "scipy": ("https://docs.scipy.org/doc/scipy/reference", None), "numpy": ("https://numpy.org/doc/stable", None), "pytorch": ("https://pytorch.org/docs/stable/", None), "matplotlib": ("https://matplotlib.org/", None)}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
