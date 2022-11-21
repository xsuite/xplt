#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Color definitions

Colors from https://arxiv.org/abs/2107.02270

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-09-06"


import matplotlib as mpl


petroff_colors = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]

#: Colormap with 10 distinct colors
cmap_petroff = mpl.colors.ListedColormap(petroff_colors, "petroff")
cmap_petroff.set_under("none")
cmap_petroff.set_over("none")

#: Gradient colormap
cmap_petroff_gradient = mpl.colors.LinearSegmentedColormap.from_list(
    "petroff_gradient", [petroff_colors[i] for i in (9, 0, 4, 2, 6, 1)]
)
cmap_petroff_gradient.set_under(petroff_colors[3])
cmap_petroff_gradient.set_over(petroff_colors[7])

#: Bipolar colormap
cmap_petroff_bipolar = mpl.colors.LinearSegmentedColormap.from_list(
    "petroff_bipolar", [petroff_colors[i] for i in (2, 6, 1, 3, 9, 0, 4)]
)
cmap_petroff_bipolar.set_under(petroff_colors[5])
cmap_petroff_bipolar.set_over(petroff_colors[8])

#: Cyclic colormap
cmap_petroff_cyclic = mpl.colors.LinearSegmentedColormap.from_list(
    "petroff_cyclic", [petroff_colors[i] for i in (3, 9, 0, 4, 2, 6, 1, 3)]
)


def make_unicoloured_cmap(color):
    """Make a linear colormap of a given color from transparent to black"""
    return mpl.colors.LinearSegmentedColormap.from_list(
        "gradient",
        (
            (0.000, (1.000, 1.000, 1.000, 0)),
            (0.400, color),
            (1.000, (0.000, 0.000, 0.000, 1)),
        ),
    )
