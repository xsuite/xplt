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
copyright = "2022-2024, Philipp Niedermayer (github.com/eltos)"
author = "Philipp Niedermayer (github.com/eltos)"
github_username = "xsuite"
github_repository = "xplt"
version = __version__
release = version

# Project sources
root = os.path.abspath("..")
sys.path.insert(0, root)


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
    "matplotlib.sphinxext.roles",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


def np_example_notebooks_init(app, *args):
    shutil.copytree(
        os.path.join(root, "examples"),
        os.path.join(app.srcdir, "examples"),
    )


def np_example_notebooks_clean(app, *args):
    shutil.rmtree(os.path.join(app.srcdir, "examples"))


def setup(app):
    app.connect("config-inited", np_example_notebooks_init)
    app.connect("build-finished", np_example_notebooks_clean)
    app.connect("autoapi-skip-member", autoapi_skip_member)


# -- Auto API ----------------------------------------------------------------
# > auto-generate API documentation

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


def autoapi_skip_member(app, what, name, obj, skip, options):
    skip |= ".. deprecated::" in obj.docstring
    skip |= ":nodoc:" in obj.docstring
    skip |= not obj.docstring and what == "data"  # undocumented variables
    return skip


# -- MyST{NB} ----------------------------------------------------------------
# > use markdown and jupyter notebooks for building docs

myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
nb_render_markdown_format = "myst"
nb_execution_mode = "off"


# -- Intersphinx & Codeautolink ----------------------------------------------
# > make code examples clickable, linking to the API docs

codeautolink_concat_default = True
intersphinx_mapping = {
    #'python': ('https://docs.python.org/3', None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    #'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    #'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    "xsuite": ("https://xsuite.readthedocs.io/en/latest/", None),
}


# -- Linkcode ----------------------------------------------------------------
# > add [source] links to the API docs, linking to source code on GitHub


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
            return f"https://github.com/{github_username}/{github_repository}/blob/v{version}/{filename}"
    except:
        return None


# -- HTML theme: PyData Sphinx Theme -----------------------------------------

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "show_nav_level": 2,
    "github_url": f"https://github.com/{github_username}/{github_repository}",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/xplt",
            "icon": "fa-solid fa-cube",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "switcher": {
        "json_url": f"https://{github_username}.github.io/{github_repository}/versions.json",
        "version_match": ".".join(version.split(".")[:2]),
    },
    "check_switcher": False,  # don't check url during build
}
