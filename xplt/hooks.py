#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Matplotlib setup

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-09-06"

import types
import warnings
from xplt.colors import (
    cmap_petroff,
    cmap_petroff_gradient,
    cmap_petroff_bipolar,
    cmap_petroff_cyclic,
    petroff_colors,
)


def try_register_hooks():
    try:
        register_matplotlib_options()
    except Exception as e:
        warnings.warn(f"Failed to register color names with matplotlib: {e}", RuntimeWarning)
        pass

    try:
        register_pint_options()
    except Exception as e:
        warnings.warn(f"Failed to register formatters with pint: {e}", RuntimeWarning)
        pass


def register_matplotlib_options():
    """Register default options for matplotlib"""
    import matplotlib as mpl

    # register named colors
    for i, color in enumerate(petroff_colors):
        mpl.colors.get_named_colors_mapping().update({f"p{i}": color})
        mpl.colors.get_named_colors_mapping().update({f"pet{i}": color})

    # register named colormaps
    for cmap in (
        cmap_petroff,
        cmap_petroff_gradient,
        cmap_petroff_bipolar,
        cmap_petroff_cyclic,
    ):
        mpl.colormaps.register(cmap=cmap)
        cmap_r = cmap.reversed()
        cmap_r.name = cmap.name + "_r"
        mpl.colormaps.register(cmap=cmap_r)


def register_pint_options():
    """Register default options for pint"""
    import pint

    @pint.register_unit_format("X")
    def format_latex(unit, registry, **options):
        """Slightly modified latex formatter"""
        preprocessed = {
            r"\mathrm{{{}}}".format(u.replace("_", r"\_")): p for u, p in unit.items()
        }
        formatted = pint.formatter(
            preprocessed.items(),
            {},
            as_ratio=False,  # changed
            single_denominator=True,
            product_fmt=r" \cdot ",
            division_fmt=r"\frac[{}][{}]",
            power_fmt="{}^[{}]",
            parentheses_fmt=r"\left({}\right)",
            **options,
        )
        return formatted.replace("[", "{").replace("]", "}").replace("^{0.5}", "^{1/2}")


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
