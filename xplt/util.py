#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility methods for accelerator physics

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-15"


import types

import numpy as np
import scipy.signal
import matplotlib as mpl

try:
    import pandas as pd
except ImportError:
    # pandas is an optional dependency
    pd = None


VOID = object()


c0 = 299792458  # speed of light in m/s


def val(obj):
    """Return the value if this is an array of size 1, object otherwise"""
    if np.size(obj) == 1:
        return np.array(obj).item()
    return obj


#
def ieee_mod(values, m):
    """Return the IEEE remainder (in range -x/2 .. x/2)"""
    return np.mod(values + m / 2, m) - m / 2


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
    if pd and isinstance(obj, pd.DataFrame):
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


def defaults(kwargs, /, **default_kwargs):
    """Return keyword arguments with defaults

    Returns a union of keyword arguments, where `kwargs` take precedence over `default_kwargs`.

    Args:
        kwargs (dict | None): keyword arguments (overwrite defaults)
        default_kwargs: default keyword arguments
    """
    return dict(default_kwargs, **(kwargs or {}))


def defaults_for(alias_provider, kwargs, /, **default_kwargs):
    """Return normalized keyword arguments with defaults

    Returns a union of keyword arguments, where `kwargs` take precedence over `default_kwargs`.
    All keyword arguments are normalized beforehand via :meth:`matplotlib.cbook.normalize_kwargs`.

    Args:
        alias_provider (str | dict | class | artist): alias provider for :meth:`matplotlib.cbook.normalize_kwargs`
        kwargs (dict | None): keyword arguments (overwrite defaults)
        default_kwargs: default keyword arguments
    """
    if isinstance(alias_provider, str):
        alias_provider = dict(
            text=mpl.text.Text,
            plot=mpl.lines.Line2D,
            fill_between=mpl.patches.Polygon,
            scatter=mpl.collections.Collection,
            hexbin=mpl.collections.PolyCollection,
        )[alias_provider]
    kwargs = mpl.cbook.normalize_kwargs(kwargs, alias_provider)
    default_kwargs = mpl.cbook.normalize_kwargs(default_kwargs, alias_provider)
    return defaults(kwargs, **default_kwargs)


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


def evaluate_expression_wrapper(expression, key, data):
    """Evaluate the expression wrapper"""

    methods = dict(smooth=smooth, average=average, offset=lambda x, o: x + o)

    try:
        return eval(expression, methods, {key: data})
    except Exception as e:
        import inspect, sys

        print(f"Error evaluating exprssion `{expression}`", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        print(f"Supported methods:", file=sys.stderr)
        for k, v in methods.items():
            if k in ("self", "data") or k.startswith("_"):
                continue
            print(f"  {k}{inspect.signature(v)}", file=sys.stderr)
        raise


def binned_data(
    values, *, what=None, n=None, dv=None, v_range=None, moments=1, make_n_power_of_two=False
):
    """Get histogrammed data with equally spaced bins

    From the non-equally distributed values, a histogram or timeseries with equally
    spaced bins is derived. The parameter ``what`` determines what is returned for the timeseries.
    By default (what=None), the number of particles for each bin is returned.
    Alternatively, a particle property can be passed as array, in which case that property is averaged
    over all particles of the respective bin (or 0 if no particles are within a bin).
    It is also possible to specify the moments to return, i.e. the power to which the property is raised
    before averaging. This allows to determine mean (1st moment, default) and variance (difference between
    2nd and 1st moment) etc. To disable averaging, pass None as the moment

    Args:
        values (np.ndarray): Array of particle data, e.g. timestamps.
        n (int | None): Number of bins. Must not be used together with `dv`.
        dv (float | None): Bin width. Must not be used together with n.
        v_range (tuple[float] | None): Tuple of (min, max) values to consider.
                                       If None, the range is determined from the data.
        what (np.ndarray | None): Array of associated data or None. Must have same shape as values. See above.
        moments (int | list[int | None] | None): The moment(s) to return for associated data if what is not None. See above.
        make_n_power_of_two (bool): If true, ensure that the number of bins is a power of two by rounding up.
                                    Useful to increase performance of calculating FFTs on the timeseries data.

    Returns:
        The histogram or timeseries as tuple (v_min, dv, *counts_or_what) where
        `v_min` is the start value of the histogram or timeseries data,
        `dv` is the bin width and
        `*counts_or_what` are the values of the histogram or timeseries as an array of length n
            (if multiple moments are requested, an array is returned for each of them).
    """

    v_min = np.min(values) if v_range is None or v_range[0] is None else v_range[0]
    v_max = np.max(values) if v_range is None or v_range[1] is None else v_range[1]

    if n is not None and dv is None:
        # number of bins requested, adjust bin width accordingly
        if make_n_power_of_two:
            n = 1 << (n - 1).bit_length()
        dv = (v_max - v_min) / n
    elif n is None and dv is not None:
        # bin width requested, adjust number of bins accordingly
        n = int(np.ceil((v_max - v_min) / dv))
        if make_n_power_of_two:
            n = 1 << (n - 1).bit_length()
    else:
        raise ValueError(f"Exactly one of n or dt must be specified, but got n={n} and dt={dv}")

    # Note: The code below was optimized to run much faster than an ordinary
    # np.histogram, which quickly slows down for large datasets.
    # If you intend to change something here, make sure to benchmark it!

    # count timestamps in bins
    bins = np.floor((values - v_min) / dv).astype(int)
    # bins are i*dt <= t < (i+1)*dt where i = 0 .. n-1
    mask = (bins >= 0) & (bins < n)  # igore times outside range
    bins = bins[mask]
    # count particles per time bin
    counts = np.bincount(bins, minlength=n)[:n]

    if what is None:
        # Return particle counts
        return v_min, dv, counts

    else:
        # Return 'what' averaged
        result = [v_min, dv]
        if isinstance(moments, int) or moments is None:
            moments = [moments]
        for m in moments:
            v = np.zeros(n)
            # sum up 'what' for all the particles in each bin
            power = m if m is not None else 1
            np.add.at(v, bins, what[mask] ** power)
            if m is not None:
                # divide by particle count to get mean (default to 0)
                v[counts > 0] /= counts[counts > 0]
            result.append(v)
        return result


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
    if twiss is None:
        raise ValueError("Cannot calculate normalized coordinates when twiss parameter is None")
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


def virtual_sextupole(line, particle_ref=None, *, verbose=False):
    """Determine virtual sextupole strength from twiss data

    The normalized strenght is defined as S = -1/2 * betx^(3/2) * k2l

    The implementation considers only normal sextupole components.

    Args:
        line (xtrack.Line): Line with element_dict and twiss method
        particle_ref (xpart.Particles): Reference particle. Defaults to reference particle of tracker.
        verbose (bool): If True, print information on sextupoles

    Returns:
        Tuple (S, mu) with normalized strength in m^(-1/2) and phase in rad/2pi
    """

    # for backwards compatibility for old xsuite versions
    line = line.line if hasattr(line, "line") else line

    # find sextupoles
    sextupoles, k2l = [], []
    for name, el in line.element_dict.items():
        if hasattr(el, "knl") and el.order >= 2 and el.knl[2]:
            sextupoles.append(name)
            k2l.append(el.knl[2])

    # twiss at sextupoles
    tw = line.twiss(method="4d", particle_ref=particle_ref, at_elements=sextupoles)
    betx, mux = tw.betx, tw.mux

    # determine virtual sextupole
    Sn = -1 / 2 * betx ** (3 / 2) * k2l
    vectors = Sn * np.exp(3j * mux * 2 * np.pi)
    V = np.sum(vectors)
    S = np.abs(V)
    mu = np.angle(V) / 3 / 2 / np.pi

    if verbose:
        print("Sextupoles")
        data = dict(
            name=sextupoles,
            k2l=k2l,
            betx=betx,
            mux=mux,
            dx=tw.dx,
            S_abs=np.abs(vectors),
            S_deg=np.rad2deg(np.angle(vectors)),
        )
        head = data.keys()
        info = [[v if type(v) is str else f"{v:.5f}" for v in data[h]] for h in head]
        colw = [max(len(h), *[len(v) for v in val]) for h, val in zip(head, info)]
        info = ["  ".join([v.rjust(c) for v, c in zip(val, colw)]) for val in (head, *zip(*info))]
        print("  " + "\n  ".join(info))
        print("  " + "-" * len(info[0]))
        print("  Virtual sextupole:", f"S = {S:g} m^(-1/2) at mu = {mu:g} rad/2pi")

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
