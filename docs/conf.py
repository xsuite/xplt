# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os, sys, distutils.dir_util
from xplt import __version__


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Xplt"
copyright = "2022, Philipp Niedermayer (github.com/eltos)"
author = "Philipp Niedermayer (github.com/eltos)"
github_username = "eltos"
github_repository = "xplt"
version = __version__
release = version

# Project sources
root = os.path.abspath("..")
sys.path.insert(0, root)

# Auto API
autoapi_type = "python"
autoapi_dirs = ["../xplt"]
autoapi_ignore = ["*/.ipynb_checkpoints/*"]
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autodoc_typehints = "description"
autoapi_add_toctree_entry = False


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.githubpages",
    "myst_nb",
]

myst_enable_extensions = [
    "colon_fence",
    "amsmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Example notebooks
def np_example_notebooks_init(app, *args):
    global np_example_notebooks
    np_example_notebooks = distutils.dir_util.copy_tree(
        os.path.join(root, "examples"), os.path.join(app.srcdir, "examples")
    )


def np_example_notebooks_clean(*args):
    for file in np_example_notebooks:
        os.remove(file)


nb_execution_mode = "off"


def setup(app):
    app.connect("config-inited", np_example_notebooks_init)
    app.connect("build-finished", np_example_notebooks_clean)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_theme_options = {
    "show_nav_level": 2,
    "github_url": "https://github.com/eltos/xplt",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/xplt",
            "icon": "fa-solid fa-cube",
        },
    ],
}
