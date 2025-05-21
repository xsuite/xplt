#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"


__version__ = "0.11.12"


# expose the following in global namespace
from .colors import *
from .line import *
from .particles import *
from .phasespace import *
from .properties import *
from .timestructure import *
from .twiss import *
from .util import AUTO

# allow usage of xplt.mpl.* without importing matplotlib
import matplotlib as mpl


# Deprecated, for backwards compatibility
class TimeVariationPlot(SpillQualityPlot):
    """
    .. deprecated:: 0.8
        Use :class:`xplt.SpillQualityPlot` instead.
    """

    pass


class TimeVariationScalePlot(SpillQualityTimescalePlot):
    """
    .. deprecated:: 0.8
        Use :class:`xplt.SpillQualityTimescalePlot` instead.
    """

    pass


from . import hooks as _hooks

_hooks.try_register_hooks()

import matplotlib.style as _mpl_style


def apply_style():
    """Apply xplt's matplotlib style sheet and update rcParams"""
    try:
        _mpl_style.use("xplt.xplt")
    except IOError:
        import matplotlib, shutil, os

        # for matplotlib versions prior to 3.7: copy the style file to the config directory
        dir = os.path.join(matplotlib.get_configdir(), "stylelib")
        os.makedirs(dir, exist_ok=True)
        print("Installing xplt style in", dir)
        shutil.copy(os.path.join(os.path.dirname(__file__), "xplt.mplstyle"), dir)
        matplotlib.style.reload_library()
        matplotlib.style.use("xplt")

    # import pint.formatting as _pint_formatting
    # _pint_formatting.format_default = "X"  # use explicit format instead!
