#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility methods for accelerator physics

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-15"

import types

import numpy as np
import pandas as pd
import scipy.signal


VOID = object()


c0 = 299792458  # speed of light in m/s


def val(obj):
    """Return the value if this is an array of size 1, object otherwise"""
    if np.size(obj) == 1:
        return np.array(obj).item()
    return obj


def get(obj, value, default=VOID):
    """Get value from object

    Tries to get the value using attributes and indices,
    and handles special objects like pandas data frames.

    Args:
        obj (Any): Object to get data from
        value (str): Name of attribute, index, column etc. to get
        default (Any): Default value to return. By default an exception is raised

    Returns:
        Any: Value of the object

    Raises:
        AttributeError: if object does not provide the value and no default was specified
    """
    if isinstance(obj, pd.DataFrame):
        return val(obj[value].values)
    try:
        return val(getattr(obj, value))
    except:
        try:
            return val(obj[value])
        except:
            if default is not VOID:
                return default
    raise AttributeError(f"{obj} does not provide an attribute or index '{value}'")


def defaults(kwargs, **default_kwargs):
    """Return kwargs or default_kwargs"""
    kwargs = kwargs or {}
    if "c" in kwargs or "color" in kwargs:
        # c and color are common aliases, remove both from default_style if either present
        default_kwargs.pop("c", None)
        default_kwargs.pop("color", None)
    return dict(default_kwargs, **kwargs)


def flattened(lists):
    """Flatten a list of nested lists recursively"""
    if hasattr(lists, "__iter__"):
        return [item for sublist in lists for item in flattened(sublist)]
    return [lists]


class AttrDict(dict):
    """Dict which allows accessing values via key attributes"""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


#    def update_recursive(self, other):
#        for k, v in other.items():
#            if k in self and isinstance(self[k], collections.abc.Mapping):
#                self[k] = AttrDict(self[k])
#                self[k].update_recursive(v)
#            else:
#                self[k] = v


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


def smooth(*data, n):
    if n:
        # preserves shape but first and last n/2 values will be np.nan
        data = [scipy.signal.savgol_filter(d, n, 0, mode="constant", cval=np.nan) for d in data]
    return data[0] if len(data) == 1 else data


def normalized_coordinates(x, px, twiss, xy, delta=0):
    """Convert physical to normalized coordinates

    Args:
        x (float): Physical x-coordinate in m
        px (float): Physical px-coordinate in rad
        twiss (Any): Object or dict with twiss information in m and rad
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
        X (float): Normalized X-coordinate in m^(1/2)
        Px (float): Normalized Px-coordinate in m^(1/2)
        twiss (Any): Object or dict with local twiss information in m and rad
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


def virtual_sextupole(tracker_or_line, particle_ref=None, *, verbose=False):
    """Determine virtual sextupole strength from twiss data

    The normalized strenght is defined as S = -1/2 * betx^(3/2) * k2l

    The implementation considers only normal sextupole components.

    Args:
        tracker_or_line (xtrack.Tracker | xtrack.Line): Line or tracker with line and twiss methods
        particle_ref (xpart.Particles): Reference particle. Defaults to reference particle of tracker.
        verbose (bool): If True, print information on sextupoles

    Returns:
        Tuple (S, mu) with normalized strength in m^(-1/2) and phase in rad/2pi
    """

    line = tracker_or_line.line if hasattr(tracker_or_line, "line") else tracker_or_line

    # find sextupoles
    sextupoles, k2l = [], []
    for name, el in line.element_dict.items():
        if hasattr(el, "knl") and el.order >= 2 and el.knl[2]:
            sextupoles.append(name)
            k2l.append(el.knl[2])

    # twiss at sextupoles
    tw = tracker_or_line.twiss(
        method="4d",
        particle_ref=particle_ref,
        at_elements=sextupoles,
    )
    betx, mux = tw.betx, tw.mux

    # determine virtual sextupole
    Sn = -1 / 2 * betx ** (3 / 2) * k2l
    vectors = Sn * np.exp(3j * mux * 2 * np.pi)
    V = np.sum(vectors)
    S = np.abs(V)
    mu = np.angle(V) / 3 / 2 / np.pi

    if verbose:
        print("Sextupoles")
        df = pd.DataFrame(
            dict(
                name=sextupoles,
                k2l=k2l,
                betx=betx,
                mux=mux,
                dx=tw.dx,
                S_abs=np.abs(vectors),
                S_deg=np.rad2deg(np.angle(vectors)),
            )
        )
        info = str(df).split("\n")
        print("  " + "\n  ".join(info))
        print("  " + "-" * len(info[0]))
        print("  Virtual sextupole:", f"S = {S:g} at mu = {mu:g}")

    return S, mu


def hamiltonian_kobayashi(X, Px, S, mu, twiss, xy="x", delta=0, *, normalized=False):
    """Calculate the kobayashi hamiltonian

    Args:
        X (float): Normalized X-coordinate in m^(1/2)
        Px (float): Normalized Px-coordinate in m^(1/2)
        S (float): Normalized sextupole strength in m^(-1/2)
        mu (float): Sextupole phase in rad/2pi
        twiss (Any): Object or dict with local twiss information in m and rad
        xy (str): Plane. Either "x" or "y".
        delta (float): Momentum deviation to account for dispersive orbit.
        normalized (bool): If true, return value of hamiltonian divided by value at separatrix.

    Returns:
        Value of the hamiltonian (normalized if specified)

    """

    # Tune distance
    q = get(twiss, "q" + xy)
    q = q + delta * get(twiss, "dq" + xy)  # chromatic tune shift
    r = np.round(3 * q) / 3  # next 3rd order resonance
    d = q - r  # distance to resonance
    # h = 4 * np.pi * d / S  # size of stable striangle (inscribed circle)

    # rotate coordinates to account for phase advance
    dmu = 2 * np.pi * (mu - get(twiss, "mu" + xy))  # phase advance to virtual sextupole
    rotation = np.array([[+np.cos(dmu), np.sin(dmu)], [-np.sin(dmu), np.cos(dmu)]])
    (X, Px) = np.tensordot(rotation, (X, Px), 1)

    # calculate hamiltonian
    H = 3 * np.pi * d * (X**2 + Px**2) + S / 4 * (3 * X * Px**2 - X**3)
    if normalized:
        Hsep = (4 * np.pi * d) ** 3 / S**2
        H = H / Hsep

    return H


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
