#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Base methods for plotting

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"


import re
import types

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pint

from .util import defaults
from .units import get_property, Prop


class config:
    """Global configuration options"""

    #: Use x' and y' labels instead of px and py
    use_xprime_labels = True


class ManifoldMultipleLocator(mpl.ticker.MaxNLocator):
    def __init__(self, fixed_multiples, n=5, minor_n=None):
        """A multiple locator that chooses its base from a set of multiples to yield about n ticks

        For ranges smaller than the smallest fixed_multiple, the default MaxNLocator is used
        For ranges larger than the largest fixed_multiple, a multiple of the later is used

        Args:
            fixed_multiples (list of float): multiples to choose from
            n (int): number of ticks to produce (best effort)
            minor_n (list of float): If given, produce that many minor ticks in between each major tick. Length must match the length of fixed_multiples
        """
        super().__init__(n)
        self.fixed_multiples = fixed_multiples
        self.n = n
        self.minor_n = minor_n

    def _raw_ticks(self, vmin, vmax):
        if vmax - vmin < self.n * self.fixed_multiples[0]:
            return super()._raw_ticks(vmin, vmax)
        for step in self.fixed_multiples:
            if (vmax - vmin) / step <= self.n:
                break
        while (vmax - vmin) / step > self.n:
            step += self.fixed_multiples[-1]
        if self.minor_n is not None:
            if step in self.fixed_multiples:
                step /= self.minor_n[self.fixed_multiples.index(step)]
            else:
                step /= self.minor_n[-1]
        return np.arange(int(vmin / step) * step, vmax + step, step)


class AngleLocator(ManifoldMultipleLocator):
    def __init__(self, minor=False, deg=True):
        """A tick locator for angles

        Args:
            minor (bool): If true, return a minor locator. By default a major locator is returned.
            deg (bool): If true, locate angles is degree. If false, in radians.
        """
        multiples = (5, 15, 30, 45, 60, 90, 120, 180, 360)
        if not deg:
            multiples = list(np.deg2rad(multiples))
        subdivide = (5, 3, 3, 3, 4, 3, 4, 4, 4)
        super().__init__(multiples, 5, subdivide if minor else None)


class RadiansFormatter(mpl.ticker.Formatter):
    """A tick formatter to format angles in radians as fractions or multiples of pi"""

    def __call__(self, x, pos=None):
        if x == 0:
            return "0"
        s = "-" if x < 0 else ""
        x = abs(x)
        if x == np.pi:
            return f"${s}\\pi$"
        for n in (2, 3, 4, 6, 8, 12):
            m = round(x / (np.pi / n))
            if abs(x - m * np.pi / n) < 1e-10 and m / n != m // n:
                if m == 1:
                    m = ""
                return f"${s}{m}\\pi/{n}$"
        return f"${x/np.pi:g}\\pi$"


class XPlot:
    def __init__(
        self,
        *,
        data_units=None,
        display_units=None,
        prefix_suffix_config=None,
    ):
        """
        Base class for plotting

        Args:
            data_units (dict, optional): Units of the data. If None, the units are determined from default and user property settings.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            prefix_suffix_config (dict, optional): Prefix and suffix config for joining axes labels. A dict with prefixes and
                corresponding full names, e.g. {"p": ("px", "py", ...), ...} to join labels for px and py as $p_{x,y}$.
        """

        self._properties = {}
        if data_units:
            for name, arg in data_units.items():
                self._properties[name] = arg if isinstance(arg, Prop) else Prop(name, unit=arg)

        self._display_units = defaults(display_units, s="m", x="mm", y="mm", p="mrad")
        self._prefix_suffix_config = {}
        if not config.use_xprime_labels:
            self._prefix_suffix_config["p"] = ("px", "py")
        self._prefix_suffix_config.update(prefix_suffix_config or {})

    @classmethod
    def _parse_nested_list_string(cls, list_string, separators=",-+", subs={}):
        """Parse a separated string or nested list or a mixture of both

        Args:
            list_string (str or list): The string or nested list to parse
            separators (str): The characters that separate the elements
            subs (dict): A dictionary of substitutions to apply to the elements

        Returns:
            nested list of elements in the string
        """
        if type(list_string) is str:
            elements = []
            for element in list_string.split(separators[0]):
                element = subs.get(element, element)
                # split again in case subs contains a separator
                elements.extend(element.split(separators[0]))
        else:
            elements = list(list_string)
        if len(separators) > 1:
            for i in range(len(elements)):
                elements[i] = cls._parse_nested_list_string(elements[i], separators[1:], subs)
        return elements

    def _init_axes(self, ax, nrow=1, ncol=1, nntwins=None, grid=False, **subplots_kwargs):
        """Helper method to initialize a default manifold plot with subplots, twin axes and line plots

        Args:
            ax (matplotlib.axes.Axes or None): If given, use these axes for the plot.
                Otherwise create a new figure and axes.
            nrow (int): Number of rows in the plot.
            ncol (int): Number of columns in the plot.
            nntwins (list): List defining how many twin axes to create for each subplot.
                If None, the number of twins is automatically determined based on self.kind.
            grid (bool): If true, add a grid to the plot.
            subplots_kwargs: Additional keyword arguments to pass to matplotlib.pyplot.subplots
        """

        # Create plot axes
        if ax is None:
            fig, ax = plt.subplots(nrow, ncol, **subplots_kwargs)
            self.annotation = fig.text(
                0.005, 0.005, "", ha="left", c="gray", linespacing=1, fontsize=8
            )

        self.ax = ax
        self.fig = self.axflat[0].figure

        self.axflat_twin = []

        for i, a in enumerate(self.axflat):
            if grid:
                a.grid(grid)

            # Create twin axes
            self.axflat_twin.append([])
            for j in range(nntwins[i]):
                twin = a.twinx()
                twin._get_lines.prop_cycler = a._get_lines.prop_cycler
                if j > 0:
                    twin.spines.right.set_position(("axes", 1 + 0.2 * j))
                self.axflat_twin[i].append(twin)

    def _annotate(self, text, **kwargs):
        if not hasattr(self, "annotation"):
            return
        self.annotation.set(text=text, **kwargs)

    @property
    def axflat(self):
        """Return a flat list of all primary axes"""
        return np.array(self.ax).flatten()

    def axis_for(self, subplot=0, twin=0):
        """Return the axis for a given flat subplot index and twin index"""
        return self.axflat_twin[subplot][twin - 1] if twin else self.axflat[subplot]

    def _init_artists(self, subplots_twin_elements, create_artist):
        """Helper method to create artists for subplots and twin axes

        Args:
            subplots_twin_elements (list): A list of lists of elements to create artists for.
                The outer list is for subplots, the inner list for twin axes.
            create_artist (function): A function that creates an artist for a given element.
                It should take the indices i, j, k, the axis, and the element as arguments.
        """
        self.artists = []
        for i, ppp in enumerate(subplots_twin_elements):
            self.artists.append([])
            legend = [], []
            for j, pp in enumerate(ppp):
                a = self.axis_for(i, j)

                # format axes
                a.set(ylabel=self.label_for(*pp))
                units = np.unique([self.display_unit_for(p) for p in pp])
                if len(units) == 1:
                    if units[0] == "rad":
                        self.set_axis_ticks_angle(a.yaxis, minor=True, deg=False)
                    elif units[0] in ("deg", "°"):
                        self.set_axis_ticks_angle(a.yaxis, minor=True, deg=True)

                # create artists for traces
                self.artists[i].append([])
                for k, p in enumerate(pp):
                    artist = create_artist(i, j, k, a, p)
                    self.artists[i][j].append(artist)
                    for art in artist if hasattr(artist, "__iter__") else [artist]:
                        legend[0].append(art)
                        legend[1].append(art.get_label())

            if len(legend[0]) > 1:
                a.legend(*legend)

    def save(self, fname, **kwargs):
        """Save the figure"""
        self.fig.savefig(fname, **defaults(kwargs, dpi=300))

    def title(self, title, **kwargs):
        """Set figure title"""
        self.fig.suptitle(title, **kwargs)

    def factor_for(self, p):
        """Return factor to convert parameter into display unit"""
        quantity = pint.Quantity(self.data_unit_for(p))
        return (quantity / pint.Quantity(self.display_unit_for(p))).to("").magnitude

    def data_unit_for(self, p):
        """Return data unit for parameter"""
        return self._get_property(p).unit

    def display_unit_for(self, p):
        """Return display unit for parameter"""
        if p in self._display_units:
            return self._display_units[p]
        prefix = p[:-1] if len(p) > 1 and p[-1] in "xy" else p
        if prefix in self._display_units:
            return self._display_units[prefix]
        return self.data_unit_for(p)

    def _get_property(self, p):
        return get_property(p, self._properties)

    def _legend_label_for(self, p):
        """
        Return legend label for a single property

        Args:
            p: Property name
        """
        prop = self._get_property(p)
        return prop.description or prop.symbol

    def label_for(self, *pp, unit=True, description=True):
        """
        Return axis label for list of properties, joining where possible

        Args:
            pp: Property names
            unit: Wheather to include unit
            description: Wheather to include description
        """

        # if there are different units, treat them separately
        units = np.array([self.display_unit_for(p) for p in pp])
        if unit and np.unique(units).size > 1:
            # different units, treat parameters for each unit separately
            lines = []
            for u in np.unique(units):
                pp_with_unit = np.array(pp)[units == u]
                lines.append(self.label_for(*pp_with_unit, unit=unit))
            return "\n".join(lines)

        # reduce complexity, by combining labels with common prefix
        labels, combined = [], {}

        for p in pp:
            prop = self._get_property(p)
            label = prop.symbol
            if description and prop.description:
                label = prop.description + "   " + label

            if m := re.fullmatch("\\$(.+)_(.)\\$", label):
                pre, suf = m.groups()
                if pre not in combined:
                    combined[pre] = []
                combined[pre].append(suf)
            else:
                labels.append(label)

        for pre, ss in combined.items():
            s = ",".join(ss)
            labels.append(f"${pre}_{{{s}}}$")

        # build label
        label = ", ".join(labels)

        # add unit
        if unit:
            if units[0] == "a.u.":  # arbitrary unit
                label += f" / a.u."
            else:
                display_unit = pint.Unit(units[0])  # all have the same unit (see above)
                if display_unit != pint.Unit("1"):
                    label += f" / ${display_unit:~l}$"

        return label

    @staticmethod
    def set_axis_ticks_angle(yaxis, minor=True, deg=False):
        """Set ticks locator and formatter to display an angle

        This will set ticks at multiples or fractions of 180° or pi with appropriate labels

        Args:
            yaxis: The axis to format (ax.xaxis or ax.yaxis)
            minor (bool): If true (default), also set the minor locator
            deg (bool): If true, use angles in degree. If false (default), in radians.
        """
        yaxis.set_major_locator(AngleLocator(deg=deg))
        if minor:
            yaxis.set_minor_locator(AngleLocator(deg=deg, minor=True))
        if not deg:
            yaxis.set_major_formatter(RadiansFormatter())

    @staticmethod
    def plot_harmonics(
        ax, v, dv=0, *, n=20, scale_width=True, vertical=True, inverse=False, **plot_kwargs
    ):
        """Add vertical lines or spans indicating the location of values or spans and their harmonics

        Indicates the bands at the h-th harmonics for h = 1, 2, ..., n
        - h * (v ± dv/2)      if scale_width and not inverse (default)
        - h * v ± dv/2        if not scale_width and not inverse
        - h / (v ± dv/2)      if inverse and scale_width
        - 1 / ( v/h ± dv/2 )  if inverse and not scale_width

        Args:
            ax: Axis to plot onto.
            v (float or list of float): Value or list of values.
            dv (float or list of float, optional): Width or list of widths centered around value(s).
            n (int): Number of harmonics to plot.
            scale_width (bool, optional): Whether to scale the width for higher harmonics or keep it constant.
            vertical (bool): Plot vertical lines if true, horizontal otherweise.
            inverse (bool): If true, plot harmonics of n/(v±dv) instead of n*(v±dv). Useful to plot frequency harmonics in time domain and vice-versa.
            plot_kwargs: Keyword arguments to be passed to plotting method
        """
        if not hasattr(v, "__iter__"):
            v = [v]
        if not hasattr(dv, "__iter__"):
            dv = [dv] * len(v)
        kwargs = defaults(plot_kwargs, zorder=1.9, color="gray", lw=1)
        for i in range(1, n + 1):
            for j, (vi, dvi) in enumerate(zip(v, dv)):
                h = 1 / i if inverse else i
                vi = h * vi
                if dvi == 0:
                    method = ax.axvline if vertical else ax.axhline
                    args = np.array([vi])
                else:
                    if scale_width:
                        dvi = h * dvi
                    method = ax.axvspan if vertical else ax.axhspan
                    args = np.array([vi - dvi / 2, vi + dvi / 2])
                if inverse:
                    args = sorted(1 / args)
                method(
                    *args,
                    **defaults(kwargs, alpha=1 - np.log(1 + (np.e - 1) * (i - 1) / n)),
                )
                kwargs.pop("label", None)

    @staticmethod
    def add_scale(
        ax,
        scale,
        label=None,
        *,
        vertical=False,
        width=0.01,
        padding=0.1,
        loc="auto",
        color="k",
        fontsize="x-small",
    ):
        """Add a scale patch (a yardstick or ruler)

        Args:
            ax: The axis to add it to.
            scale: The size of the scale in data units.
            label (str, optional): A label for the scale.
            vertical (bool): If true, make a vertical one (default is a horizontal one).
            width (float): The line width of the scale in axis units.
            padding (float): The padding between the scale and the axis.
            loc (str): The location of the scale. Can be any of the usual matplotlib locations, e.g. 'auto', 'upper left', 'upper center', 'upper right', 'center left', 'center', 'center right', 'lower left', 'lower center, 'lower right'.
            color: Color for the patch.
            fontsize: Font size of the label.

        Returns:
            The artist added (an AnchoredOffsetbox).
        """
        if loc == "auto":
            loc = "upper left" if vertical else "lower right"
        w, h = scale, width
        w_trans, h_trans = ax.transData, ax.transAxes
        if vertical:  # swap dimensions
            w, h = h, w
            w_trans, h_trans = h_trans, w_trans
        aux = mpl.offsetbox.AuxTransformBox(
            mpl.transforms.blended_transform_factory(w_trans, h_trans)
        )
        aux.add_artist(plt.Rectangle((0, 0), w, h, fc=color))
        if label:
            kwa = dict(text=label, color=color, fontsize=fontsize)
            if vertical:
                aux.add_artist(plt.Text(w * 2, h / 2, ha="left", va="center", rotation=90, **kwa))
            else:
                aux.add_artist(plt.Text(w / 2, h * 1.5, va="bottom", ha="center", **kwa))
        ab = mpl.offsetbox.AnchoredOffsetbox(loc, borderpad=padding, zorder=100, frameon=False)
        ab.set_child(aux)
        ax.add_artist(ab)
        return ab


class FixedLimits:
    """Context manager for keeping axis limits fixed while plotting"""

    def __init__(self, axis):
        self.axis = axis

    def __enter__(self):
        self.limits = self.axis.get_xlim(), self.axis.get_ylim()

    def __exit__(self, *args):
        self.axis.set(xlim=self.limits[0], ylim=self.limits[1])


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
