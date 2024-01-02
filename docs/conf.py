# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import importlib
import inspect
import os, sys, shutil
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
    "sphinx.ext.linkcode",
    "autoapi.extension",
    "sphinx.ext.githubpages",
    "myst_nb",
    "sphinx.ext.intersphinx",
    "sphinx_codeautolink",
]

myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "amsmath",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Codeautolink and intersphinx
# > make code examples clickable, linking to the docs
codeautolink_concat_default = True
intersphinx_mapping = {
    #'python': ('https://docs.python.org/3', None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    #'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    #'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    "xsuite": ("https://xsuite.readthedocs.io/en/latest/", None),
}

# Linkcode
# > in API reference, add links to source code on GitHub
def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    try:
        # inspect source
        obj = importlib.import_module(info["module"])
        for name in info["fullname"].split("."):
            obj = getattr(obj, name)
        sourcefile = inspect.getsourcefile(obj)
        sourcecode, line = inspect.getsourcelines(obj)
        # build link
        root = "xplt" + os.path.sep
        if root not in sourcefile:
            return None  # external source
        else:
            path = (root + sourcefile.split(root)[-1]).replace(os.path.sep, "/")
            filename = f"{path}#L{line}-L{line + len(sourcecode) - 1}"
            return f"https://github.com/eltos/xplt/blob/v{version}/{filename}"
    except:
        return None


# Example notebooks
def np_example_notebooks_init(app, *args):
    shutil.copytree(
        os.path.join(root, "examples"),
        os.path.join(app.srcdir, "examples"),
    )


def np_example_notebooks_clean(app, *args):
    shutil.rmtree(os.path.join(app.srcdir, "examples"))


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
