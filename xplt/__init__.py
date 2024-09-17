#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"


__version__ = "0.10.0"


# expose the following in global namespace
from .colors import *
from .line import *
from .particles import *
from .phasespace import *
from .properties import *
from .timestructure import *
from .twiss import *

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
import pint.formatting as _pint_formatting


def apply_style():
    """Apply xplt's matplotlib style sheet and update rcParams"""
    _mpl_style.use("xplt.xplt")
    # _pint_formatting.format_default = "X"  # use explicit format instead!
