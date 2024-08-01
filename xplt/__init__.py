#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"


__version__ = "0.8.0"


# expose the following in global namespace
from .colors import *
from .line import *
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

try:
    _hooks.register_matplotlib_options()
except:
    pass

try:
    _hooks.register_pint_options()
except:
    pass

import matplotlib.style as _mpl_style


def apply_style():
    """Apply xplt's matplotlib style sheet and update rcParams"""
    _mpl_style.use("xplt.xplt")
