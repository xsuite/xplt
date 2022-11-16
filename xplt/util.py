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


def virtual_sextupole(tracker, particle_ref=None):
    """Determine virtual sextupole strength from twiss data

    The normalized strenght is defined as S = -1/2 * betx^(3/2) * k2l

    The implementation considers only normal sextupole components.

    Args:
        tracker: Tracker object with line and twiss methods
        particle_ref: Reference particle. Defaults to reference particle of tracker.

    Returns:
        Tuple (S, mu) with normalized strength in m^(-1/2) and phase in rad/2pi
    """

    # find sextupoles
    sextupoles, k2l = [], []
    for name, el in tracker.line.element_dict.items():
        if hasattr(el, "knl") and el.order >= 2 and el.knl[2]:
            sextupoles.append(name)
            k2l.append(el.knl[2])

    # twiss at sextupoles
    tw = tracker.twiss(
        method="4d",
        particle_ref=particle_ref,
        at_elements=sextupoles,
    )
    betx, mux = tw.betx, tw.mux

    # determine virtual sextupole
    Sn = -1 / 2 * betx ** (3 / 2) * k2l
    Stotal = np.sum(Sn * np.exp(3j * mux * 2 * np.pi))
    S = np.abs(Stotal)
    mu = np.angle(Stotal) / 3 / 2 / np.pi
    return S, mu
