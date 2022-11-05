#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Matplotlib setup

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-09-06"

from xplot.colors import *

import matplotlib as mpl


def register_defaults():
    """Register matplotlib options"""
    mpl.colormaps.register(cmap_petroff)
    mpl.colormaps.register(cmap_petroff_gradient)
    mpl.colormaps.register(cmap_petroff_bipolar)
    mpl.colormaps.register(cmap_petroff_cyclic)

    mpl.rcParams.update(
        {
            "figure.constrained_layout.use": True,
            "legend.fontsize": "x-small",
            "legend.title_fontsize": "small",
            "grid.color": "#DDD",
            "axes.prop_cycle": mpl.cycler(color=petroff_colors),
            # 'image.cmap': cmap_petroff_gradient,
        }
    )
