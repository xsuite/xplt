#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Matplotlib setup

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-09-06"

from xplt.colors import *


def register_matplotlib_options():
    """Register default options for matplotlib"""
    import matplotlib as mpl

    for cmap in (
        cmap_petroff,
        cmap_petroff_gradient,
        cmap_petroff_bipolar,
        cmap_petroff_cyclic,
    ):
        mpl.cm.register_cmap(cmap=cmap)
        cmap_r = cmap.reversed()
        cmap_r.name = cmap.name + "_r"
        mpl.cm.register_cmap(cmap=cmap_r)

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


def register_pint_options():
    """Register default options for pint"""
    import pint

    @pint.register_unit_format("l")
    def format_latex(unit, registry, **options):
        """Slightly modified latex formatter"""
        preprocessed = {
            r"\mathrm{{{}}}".format(u.replace("_", r"\_")): p for u, p in unit.items()
        }
        formatted = pint.formatter(
            preprocessed.items(),
            as_ratio=False,
            single_denominator=True,
            product_fmt=r" \cdot ",
            division_fmt=r"\frac[{}][{}]",
            power_fmt="{}^[{}]",
            parentheses_fmt=r"\left({}\right)",
            **options,
        )
        return formatted.replace("[", "{").replace("]", "}").replace("^{0.5}", "^{1/2}")

    pint.formatting.format_default = "l"
