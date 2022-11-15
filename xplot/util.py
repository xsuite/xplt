#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility methods for accelerator physics

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-15"


import numpy as np

from .base import get


def normalized_coordinates(x, px, twiss, xy, delta=0):
    """Convert physical to normalized coordinates

    Args:
        x: Physical x-coordinate in m
        px: Physical px-coordinate in rad
        twiss: Object or dict with twiss information in m and rad
        xy (str): Plane. Either "x" or "y".
        delta (float): Momentum deviation to account for dispersive orbit.

    Returns:
        Tuple of normalized coordinates (X, Px) in (m^(1/2), m^(1/2))
    """
    # substract (dispersive) closed orbit
    x = x - get(twiss, xy) - delta * get(twiss, "d" + xy)
    px = px - get(twiss, "p" + xy) - delta * get(twiss, "dp" + xy)
    # apply Floquet transform
    alf, bet = get(twiss, "alf" + xy), get(twiss, "bet" + xy)
    X = x / bet**0.5
    Px = alf * x / bet**0.5 + px * bet**0.5
    return X, Px


def denormalized_coordinates(X, Px, twiss, xy, delta=0):
    """Convert normalized to physical coordinates

    Args:
        X: Normalized X-coordinate in m^(1/2)
        Px: Normalized Px-coordinate in m^(1/2)
        twiss: Object or dict with local twiss information in m and rad
        xy (str): Plane. Either "x" or "y".
        delta (float): Momentum deviation to account for dispersive orbit.

    Returns:
        Tuple of physical coordinates (x, px) in (m, rad)
    """
    # apply Floquet transform
    alf, bet = get(twiss, "alf" + xy), get(twiss, "bet" + xy)
    x = X * bet**0.5
    px = -alf * X / bet**0.5 + Px / bet**0.5
    # add (dispersive) closed orbit
    x = x + get(twiss, xy) + delta * get(twiss, "d" + xy)
    px = px + get(twiss, "p" + xy) + delta * get(twiss, "dp" + xy)
    return x, px
