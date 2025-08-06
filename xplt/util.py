#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility methods for accelerator physics"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-15"


import inspect
import re

import numpy as np
import scipy.signal
import matplotlib as mpl
import matplotlib.collections
import matplotlib.patches
import matplotlib.cbook
import matplotlib.text
import matplotlib.lines

try:
    import pandas as pd
except ImportError:
    # pandas is an optional dependency
    pd = None


def PUBLIC_SECTION_BEGIN():
    """Starting from here, collect objects to be included in file's `__all__` magic

    :nodoc:
    """
    g = inspect.stack()[1].frame.f_globals
    g["__private_names"] = set(g.keys()).union(["__private_names"])


def PUBLIC_SECTION_END():
    """Finish object collection and return global names for the `__all__` magic

    :nodoc:
    """
    g = inspect.stack()[1].frame.f_globals
    public = list(set(g.keys()).difference(g["__private_names"]))
    return public[:]  # copy needed for it to work with sphinx autoapi


VOID = object()
AUTO = object()


c0 = 299792458
"""Speed of light in m/s"""


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
        obj (Any): Object to get data from. Can also be a tuple of objects.
        value (str): Name of attribute, index, column etc. to get
        default (Any): Default value to return. By default an exception is raised

    Returns:
        Any: Value of the object

    Raises:
        AttributeError: if object does not provide the value and no default was specified
    """
    if pd and isinstance(obj, pd.DataFrame):
        return val(obj[value].values)
    if type(obj) == tuple:
        for o in obj:
            if (v := get(o, value, None)) is not None:
                return v
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
    # resolve aliases
    if isinstance(alias_provider, str):
        alias_provider = dict(
            text=mpl.text.Text,
            plot=mpl.lines.Line2D,
            fill_between=mpl.patches.Polygon,
            scatter=mpl.collections.Collection,
            hexbin=mpl.collections.PolyCollection,
            vlines=mpl.collections.LineCollection,
        )[alias_provider]
    if alias_provider == mpl.patches.Polygon and kwargs is not None and "c" in kwargs:
        kwargs["color"] = kwargs.pop("c")
    kwargs = mpl.cbook.normalize_kwargs(kwargs, alias_provider)
    default_kwargs = mpl.cbook.normalize_kwargs(default_kwargs, alias_provider)

    # handle custom values
    def apply_custom_to(kwargs):
        if (ls := kwargs.get("linestyle")) and isinstance(ls, str):
            if match := re.match(r"-(\d+)", ls):
                n = int(match.group(1))
                kwargs["linestyle"] = (0, [] if n == 0 else [1, 1.6] * n + [48.4 - 2.6 * n, 1.6])
            if match := re.match(r"-\.(\d+)", ls):
                n = int(match.group(1))
                kwargs["linestyle"] = (0, [1, 1.6] * (n + 1) + [6.4, 1.6])

    apply_custom_to(kwargs)
    apply_custom_to(default_kwargs)
    # apply with defaults
    return defaults(kwargs, **default_kwargs)


def flattened(lists):
    """Flatten a list of nested lists recursively"""
    if hasattr(lists, "__iter__") and not isinstance(lists, str):
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


def average(*data, n=100, function=np.mean, logspace=False, keepdim=False):
    """Average the data

    Applies the function to n subsequent points of the data (along last axis) to yield one point in the output

    Args:
        data (np.ndarray): the data to average over
        n (int | floor): number of subsequent datapoints of intput to average into one point in the output.
            If the input size is not a multiple of n, the data will be clipped.
            In case of logspace scaling, this can be a flost (usually 1 < n < 2 gives good results)
        function (function): averaging function to apply to last axis of input data. Defaults to np.mean
        logspace (bool): If true, average N subsequent points where N is adjusted along the data to yield equal window sizes on a log scale
        keepdim (bool): If true, repeat averaged data to keep the diemension of the input array

    Returns:
        averaged data
    """
    result = []
    for d in data:
        if not logspace:
            # linear space: work on every subsequent n samples
            w = int(d.shape[-1] / n)
            new = function(d[..., : w * n].reshape(*d.shape[:-1], w, n), axis=-1)
            if keepdim:
                new = np.repeat(new, n, axis=-1)
                new = np.pad(
                    new,
                    [(0, 0)] * (new.ndim - 1) + [(0, d.shape[-1] - new.shape[-1])],
                    constant_values=np.nan,
                )
        else:
            # log space: work on every subsequent n**i samples
            N = n ** np.arange(1, np.log(d.shape[-1]) / np.log(n))
            N = np.unique(np.hstack(([0], N[N < d.shape[-1]].astype("int"))))
            new = np.nan * (
                np.empty_like(d) if keepdim else np.empty([*d.shape[:-1], N.size - 1])
            )
            for i, (n0, n1) in enumerate(zip(N[:-1], N[1:])):
                new[..., slice(n0, n1) if keepdim else i] = v = function(d[..., n0:n1])

        result.append(new)

    return result[0] if len(result) == 1 else result


def smooth(*data, n):
    """Smooth the data over n consecutive bins

    Args:
        *data (np.array): The data
        n (int): Number of items to smooth over
    Returns
         (np.array | list[np.array]): The smooth array(s) with same shape as the original data
             (but first and last n/2 values will be np.nan).

    """
    if n:
        # preserves shape but first and last n/2 values will be np.nan
        data = [scipy.signal.savgol_filter(d, n, 0, mode="constant", cval=np.nan) for d in data]
    return data[0] if len(data) == 1 else data


def evaluate_expression_wrapper(expression, key, data):
    """Evaluate the expression wrapper"""

    methods = dict(np=np, smooth=smooth, average=average, offset=lambda x, o: x + o)

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
        tuple: The histogram or timeseries as tuple
            ``(v_min, dv, *counts_or_what)`` where
            ``v_min`` is the start value of the histogram or timeseries data,
            ``dv`` is the bin width and
            ``*counts_or_what`` are the values of the histogram or timeseries as an array of length n
            for each moment requested.
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
            dv = (v_max - v_min) / n  # adjust bin width to match new n
    else:
        raise ValueError(f"Exactly one of n or dt must be specified, but got n={n} and dt={dv}")

    # Note: The code below was optimized to run much faster than an ordinary np.histogram,
    # which quickly slows down for large number of bins as required for FFT calculation.
    # Benchmark: Binning of a data array of size=1000000 into 100000000 bins
    #     np.histogram(data, bins=100000000) takes 6.21 s ± 374 ms
    #     binned_data(data, n=100000000) takes 93.9 ms ± 2.8 ms
    # If you intend to change something here, make sure to benchmark it!

    # count timestamps in bins
    bins = np.floor((values - v_min) / dv).astype(int)
    # bins are i*dt <= t < (i+1)*dt where i = 0 .. n-1
    mask = (bins >= 0) & (bins < n)  # ignore times outside range
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


def element_strength(element, n):
    """Get knl strength of element"""
    knl = 0
    if knl == 0 and hasattr(element, f"k{n}") and hasattr(element, "length"):
        knl = getattr(element, f"k{n}") * element.length
    if knl == 0 and hasattr(element, "knl") and n <= element.order:
        knl = element.knl[n]
    return knl


def normalized_coordinates(x, px, twiss, xy, delta=0):
    """Convert physical to normalized coordinates

    Args:
        x (float): Physical x-coordinate in m
        px (float): Physical px-coordinate in rad
        twiss (Any): Object or dict with twiss information in m and rad
        xy (str): Plane. Either "x" or "y".
        delta (float): Momentum deviation to account for dispersive orbit.

    Returns:
        tuple: Tuple of normalized coordinates
            (X, Px) in (m^(1/2), m^(1/2))
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
        tuple: Tuple of physical coordinates
            (x, px) in (m, rad)
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
    """Determine virtual sextupole strength from the line

    The normalized strenght is defined as S = -1/2 * betx^(3/2) * k2l

    The implementation considers only normal sextupole components.

    Args:
        line (xtrack.Line): Line with element_dict and twiss method
        particle_ref (xpart.Particles): Reference particle. Defaults to reference particle of tracker.
        verbose (bool): If True, print information on sextupoles

    Returns:
        tuple: Tuple (S, mu)
            with normalized strength in m^(-1/2) and phase in rad/2pi
    """

    # for backwards compatibility for old xsuite versions
    line = line.line if hasattr(line, "line") else line

    # find sextupoles
    sextupoles, k2l = [], []
    for name, el in line.element_dict.items():
        if element_k2l := element_strength(el, 2):
            sextupoles.append(name)
            k2l.append(element_k2l)

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
        float: Value of the hamiltonian
            (possibly normalized)

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


def iter_elements(line, *, s=None, mask_nearest=False):
    """Iterate over elements in line

    Args:
        line (xtrack.Line): Line of elements.
        s (np.array | None): Optional array of s positions.
          If not None, returns an additional s-position-mask for each element.
        mask_nearest (bool): If true, zero-length elements will be masked at the nearest s-coordinate of the s array
          to avoid them being skipped if their location does not coincide exactly with any value in the s-array.
          If s is None, this has no effect.

    Yields:
        (name, element, s_from, s_to, [s_mask]): Name, object, start position,
          end position and optional position mask for each element. The mask is
          omitted if s=None (see arguments).

    """
    el_s0 = line.get_s_elements("upstream")
    el_s1 = line.get_s_elements("downstream")
    smax = line.get_length()
    for name, el, s0, s1 in zip(line.element_names, line.elements, el_s0, el_s1):
        if s0 == s1:  # thin lense located at element center
            if hasattr(el, "length"):
                s0, s1 = (s0 + s1 - el.length) / 2, (s0 + s1 + el.length) / 2
        if s is None:
            yield name, el, s0, s1
        else:
            if s0 == s1:
                if mask_nearest:
                    # zero-length element -> round to nearest s
                    mask = s == s[np.argmin(np.abs(s - s0))]
                else:
                    mask = s == s0 % smax
            elif 0 <= s0 <= smax:
                mask = (s >= s0) & (s < s1)
            else:  # handle wrap around
                mask = (s >= s0 % smax) | (s < s1 % smax)
            yield name, el, s0, s1, mask


def apertures(line, at_s=None):
    """Determine aperture

    Args:
        line (xtrack.Line): Line with element_dict
        at_s (np.ndarray | None): return apertures for these s locations in m

    """

    if at_s is None:
        s = np.unique([(s0, s1) for _, _, s0, s1 in iter_elements(line)])
    else:
        s = np.array(at_s)

    result = dict(
        s=s,
        min_x=np.full_like(s, np.nan),
        max_x=np.full_like(s, np.nan),
        min_y=np.full_like(s, np.nan),
        max_y=np.full_like(s, np.nan),
    )
    for name, el, s0, s1, mask in iter_elements(line, s=s):
        for xy in "xy":
            # find limits
            mi = getattr(el, f"min_{xy}", None)
            ma = getattr(el, f"max_{xy}", None)
            if mi is None and ma is not None:
                mi = -ma
            # handle shift
            if mi is not None:
                mi += getattr(el, f"shift_{xy}", 0)
                result[f"min_{xy}"][
                    mask & ~(result[f"min_{xy}"] > mi)
                ] = mi  # comparison inverted to handle NaN
            if ma is not None:
                ma += getattr(el, f"shift_{xy}", 0)
                result[f"max_{xy}"][mask & ~(result[f"max_{xy}"] < ma)] = ma

    if at_s is None:
        # only return s where there is an aperture
        mask = np.any(
            [np.isfinite(result[f"{m}_{xy}"]) for m in ("min", "max") for xy in "xy"], axis=0
        )
        for key in result:
            result[key] = result[key][mask]

    return AttrDict(result)
