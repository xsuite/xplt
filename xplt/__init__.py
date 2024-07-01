#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"


__version__ = "0.7.2"

# allow usage of xplt.mpl.* without importing matplotlib
import matplotlib as mpl

from .properties import register_data_property, register_derived_property
from .colors import *
from .line import KnlPlot, FloorPlot
from .phasespace import PhaseSpacePlot
from .timestructure import (
    Timeseries,
    TimePlot,
    TimeBinPlot,
    TimeFFTPlot,
    TimeIntervalPlot,
    SpillQualityPlot,
    SpillQualityTimescalePlot,
    TimeBinMetricHelper,
)
from .twiss import TwissPlot
from .util import average, normalized_coordinates, denormalized_coordinates

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


from . import hooks

try:
    hooks.register_matplotlib_options()
except:
    pass

try:
    hooks.register_pint_options()
except:
    pass


def apply_style():
    """Apply xplt's matplotlib style sheet and update rcParams"""
    mpl.style.use("xplt.xplt")
