#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility methods for accelerator physics

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-15"


import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # pandas is not required

VOID = object()


def get(obj, val, default=VOID):
    """Get value from object"""
    if pd is not None and isinstance(obj, pd.DataFrame):
        return obj[val].values
    try:
        return getattr(obj, val)
    except:
        try:
            return obj[val]
        except:
            if default is not VOID:
                return default
    raise AttributeError(f"{obj} does not provide an attribute or index '{val}'")


def defaults(kwargs, **default_style):
    """Return kwargs or defaults"""
    kwargs = kwargs or {}
    if "c" in kwargs or "color" in kwargs:
        # c and color are common aliases, remove both from default_style if either present
        default_style.pop("c", None)
        default_style.pop("color", None)
    return dict(default_style, **kwargs)


def average(*data, n=100, function=np.mean):
    """Average the data

    Applies the function to n subsequent points of the data (along last axis) to yield one point in the output

    Args:
        data (np.ndarray): the data to average over
        n (int): number of subsequent datapoints of intput to average into one point in the output. If the input size is not a multiple of n, the data will be clipped.
        function (callable, optional): averaging function to apply to last axis of input data. Defaults to np.mean
    Returns:
        averaged data
    """
    result = []
    for d in data:
        w = int(d.shape[-1] / n)
        result.append(function(d[..., : w * n].reshape(*d.shape[:-1], w, n), axis=-1))
    return result[0] if len(result) == 1 else result


# def smooth(*data, n):
#    return [scipy.signal.savgol_filter(d, n, 0) for d in data] if n else data


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
