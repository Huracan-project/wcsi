# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WCSI"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc options
autodoc_member_order = "bysource"
autodoc_typehints = "description"

nbsphinx_prompt_width = "88"

# Options for intersphinx.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "cartopy": ("https://cartopy.readthedocs.io/stable/", None),
    "huracanpy": ("https://huracanpy.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# myst parser (markdown)
myst_enable_extensions = [
    "colon_fence",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
