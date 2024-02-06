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

from .util import defaults, flattened
from .properties import Property, find_property, DataProperty, arb_unit


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


class TwinFunctionLocator(mpl.ticker.Locator):
    def __init__(self, twin_locator, function_twin_to_this, function_this_to_twin, granularity=1):
        """A locator for twin axes with non-linear functional dependence

        Finds nice tick locations close to the twin, but uses a custom function to place ticks at integer multiples of granularity.
        This is useful for twin axes which share the same limits, but are formatted with values based on different functions.

        Args:
            twin_locator (mpl.ticker.Locator): The other locator to align tick locations with
            function_twin_to_this (callable): Function to calculate tick values of this axis given the tick values of the other axis.
            function_this_to_twin (callable): Function to calculate tick values of the other axis given the tick values of this axis.
            granularity (float): Base at multiples of which to locate ticks.
        """
        self.twin_locator = twin_locator
        self.twin2this = function_twin_to_this
        self.this2twin = function_this_to_twin
        self.granularity = granularity

    def __call__(self):
        """Return the locations of the ticks."""
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        twin_values = np.array(self.twin_locator.tick_values(vmin, vmax))
        this_values = self.twin2this(twin_values)
        this_values = np.round(this_values / self.granularity) * self.granularity
        twin_values = self.this2twin(this_values)
        twin_values = twin_values
        return twin_values


class TransformedLocator(mpl.ticker.Locator):
    def __init__(
        self, locator, transform=lambda x: x, inverse=lambda x: x, vmin=-np.inf, vmax=np.inf
    ):
        """A transformed locator with non-linear functional wrappers and limits

        Clips ticks to limits and then transforms the values before calling the dependent locator.

        Args:
            locator (mpl.ticker.Locator): The dependent locator to use for actual tick locating
            transform (callable): Function to transform tick values of this locator to the values of the dependent locator
            inverse (callable): Inverse of transform
            vmin (float): Optional lower limit for ticks in this locator
            vmax (float): Optional upper limit for ticks in this locator
        """
        self.locator = locator
        self.transform = transform
        self.inverse = inverse
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self):
        """Return the locations of the ticks."""
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        # clip and transform limits
        vmin = self.transform(max(vmin, self.vmin))
        vmax = self.transform(min(vmax, self.vmax))
        # locate ticks
        ticks = self.locator.tick_values(vmin, vmax)
        # inverse transform
        ticks = self.inverse(ticks)
        return ticks


class XPlot:
    def __init__(
        self,
        *,
        data_units=None,
        display_units=None,
        ax=None,
        grid=True,
        nntwins=None,
        annotation=None,
        _properties=None,
        **subplots_kwargs,
    ):
        """
        Base class for plotting

        Initialize the subplots, axes and twin axes

        Args:
            data_units (dict, optional): Units of the data. If None, the units are determined from default and user property settings.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            ax (matplotlib.axes.Axes, optional): Axes to plot onto. If None, a new figure is created.
            grid (bool): If True, show grid lines on all axes.
            nntwins (list): List defining how many twin axes to create for each subplot.
            annotation (bool | None): Whether to add an annotation or not. If None (default) add it unless `ax` is passed.
            subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.
        """

        self._default_properties = {}
        if _properties:
            self._default_properties.update(_properties)
        self._user_properties = {}
        if data_units:
            for name, arg in data_units.items():
                self._user_properties[name] = (
                    DataProperty(name, arg) if isinstance(arg, str) else arg
                )
        self._display_units = defaults(display_units, s="m", x="mm", y="mm", p="mrad")

        if annotation is None:
            annotation = ax is None

        # Create plot axes
        if ax is None:
            fig, ax = plt.subplots(**subplots_kwargs)
        self.ax = ax
        self.fig = self.axflat[0].figure
        self.axflat_twin = []

        for i, a in enumerate(self.axflat):
            if grid:
                a.grid(grid, alpha=0.5)

            # Create twin axes
            self.axflat_twin.append([])
            if nntwins is not None:
                for j in range(nntwins[i]):
                    twin = a.twinx()
                    try:
                        # hack for shared cyclers
                        # see https://github.com/matplotlib/matplotlib/issues/19479
                        twin._get_lines = a._get_lines
                    except:
                        print(
                            "Warning: failed to share cyclers, please manually set trace colors for twin axes and make sure to upvote https://github.com/matplotlib/matplotlib/issues/19479"
                        )
                        pass  # ignore
                    if j > 0:
                        twin.spines.right.set_position(("axes", 1 + 0.2 * j))
                    self.axflat_twin[i].append(twin)

        # Create annotation
        if annotation:
            self.annotation = self.fig.text(
                0.005, 0.005, "", ha="left", va="bottom", c="gray", linespacing=1, fontsize=8
            )
        else:
            self.annotation = None

    def _autoscale(self, ax, artists=[], data=[], *, reset=False, freeze=True, tight=None):
        """Autoscale axes to fit given artists

        Args:
            ax (matplotlib.axes.Axes): Axes to autoscale
            artists (iterable): Artists to consider (if any)
            data (iterable): Data points to consider (if any) in the form [(x1,y1), (x2,y2), ...]
            reset (bool): Whether to ignore any data limits already registered.
            freeze (bool): Whether to keep the updated axes limits (True) or enable automatic
                autoscaling on future draws (for all present and new artists).
            tight (str | None): Enables tight scaling without margins for "x", "y", "both" or None.
        """
        tight_x, tight_y = tight in ("x", "xy", "both"), tight in ("y", "xy", "both")
        limits = []
        data = data[:]  # make a copy so we can safely append

        # Get data limits from artists
        for art in flattened(artists):
            if hasattr(art, "get_datalim"):
                # Collections, patches, etc.
                lim = art.get_datalim(ax.transData)
                if not np.all(np.isfinite(lim)):
                    # fallback to offsets (e.g. for hexbin)
                    data.extend(art.get_offsets())
                else:
                    limits.append(lim)
            elif hasattr(art, "get_data"):
                # Line2D
                data.extend(np.transpose(art.get_data()))
            elif art is not None:
                raise NotImplementedError(f"Autoscaling not implemented for {art!r}")

        # Add limits from raw data
        for x, y in data:
            lim = mpl.transforms.Bbox.from_extents(
                np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)
            )
            if np.all(np.isfinite(lim.bounds)):
                limits.append(lim)

        # Update axes limits
        if len(limits) > 0:
            dataLim = mpl.transforms.Bbox.union(limits)

            if reset:
                ax.dataLim = dataLim
            else:
                ax.update_datalim(dataLim)  # takes previous datalim into account

        # Autoscale (on next and future draws)
        ax.autoscale(tight=tight_x, axis="x")
        ax.autoscale(tight=tight_y, axis="y")

        if freeze:
            # perform autoscale immediately and freeze limits
            ax.autoscale_view()
            ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
            ax.set_autoscale_on(False)

    def annotate(self, text, **kwargs):
        if self.annotation is not None:
            self.annotation.set(text=text, **kwargs)

    @property
    def axflat(self):
        """Return a flat list of all primary axes"""
        return flattened(self.ax)

    def axis(self, subplot=0, twin=0):
        """Return the axis for a given flat subplot index and twin index

        Args:
            subplot (int, optional): Flat subplot index
            twin (int, optional): Twin index

        Returns:
            matplotlib.axes.Axes: Axis for the given subplot and twin index
        """
        return self.axflat_twin[subplot][twin - 1] if twin else self.axflat[subplot]

    def save(self, fname, **kwargs):
        """Save the figure

        Args:
            fname (str): Filename
            kwargs: Keyword arguments passed to :meth:`matplotlib.figure.Figure.savefig`
        """
        self.fig.savefig(fname, **defaults(kwargs, dpi=300))

    def title(self, title, **kwargs):
        """Set figure title

        Args:
            title (str): Title
            kwargs: Keyword arguments passed to :meth:`matplotlib.figure.Figure.suptitle`
        """
        self.fig.suptitle(title, **kwargs)

    def factor_for(self, p):
        """Return factor to convert parameter into display unit

        Args:
            p (str): Property name

        Returns:
            float: Factor to convert parameter into display unit
        """
        return pint.Quantity(1, self.prop(p).unit).to(self.display_unit_for(p)).m

    def display_unit_for(self, p):
        """Return display unit for parameter

        Args:
            p (str): Property name

        Returns:
            str: Display unit
        """
        if p is None:
            return None

        if p in self._display_units:
            return self._display_units[p]

        if len(p) > 1 and p[-1] in "xy":
            if p[:-1] in self._display_units:
                return self._display_units[p[:-1]]

        return self.prop(p).unit  # default to data unit

    def prop(self, name):
        """Get property by key
        Args:
            name (str): Key
        Returns:
            Property: The property
        """
        prop = find_property(
            name,
            extra_user_properties=self._user_properties,
            extra_default_properties=self._default_properties,
        )
        return prop.with_property_resolver(self.prop)

    def _legend_label_for(self, p):
        """
        Return legend label for a single property

        Args:
            p (str): Property name

        Returns:
            str: Legend label
        """
        return self.prop(p).description or self._symbol_for(p)

    def _axis_label_for(self, p, description=True):
        """Return axis label for a single property

        The label is without unit, as the unit is added in `label_for` separately.

        Args:
            p (str): Property name
            description (bool): Whether to include the description (if any)

        Returns:
            str: Axis label
        """
        label = self._symbol_for(p)
        if description and (desc := self.prop(p).description):
            label = desc + "   " + label
        return label

    def _symbol_for(self, p):
        """Return symbol for a single property

        This method can be overridden by subclasses to provide custom symbols,
        such as an average over a quantity etc.

        Args:
            p (str): Property name

        Returns:
            str: Symbol
        """
        prop = self.prop(p)
        return prop.symbol

    def label_for(self, *pp, unit=True, description=True):
        """
        Return axis label for list of properties, joining where possible

        Args:
            pp: Property names
            unit (bool): Whether to include unit
            description (bool): Whether to include description

        Returns:
            str: Axis label
        """

        # filter out None
        pp = [p for p in pp if p is not None]

        if len(pp) == 0:
            return ""

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
            label = self._axis_label_for(p, description)

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
        label = ", ".join(np.unique(labels))

        # add unit
        if unit:
            append = None
            if units[0] == arb_unit:  # arbitrary unit
                append = " / " + arb_unit
            else:
                display_unit = pint.Unit(units[0])  # all have the same unit (see above)
                if display_unit != pint.Unit("1"):
                    append = f" / ${display_unit:~L}$"  # see "NIST Guide to the SI"
            if append:
                # heuristic: if labels contain expressions with + or - then add parentheses
                if re.findall(r"[-+](?![^(]*\))(?![^{]*\})", label.split("   ")[-1]):
                    label = f"({label})"
                label += append

        return label

    @staticmethod
    def _set_axis_ticks_angle(yaxis, minor=True, deg=False):
        """Set ticks locator and formatter to display an angle

        This will set ticks at multiples or fractions of 180° or pi with appropriate labels

        Args:
            yaxis (matplotlib.axis.Axis): The axis to format (ax.xaxis or ax.yaxis)
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
            ax (matplotlib.axes.Axes): Axes to plot onto.
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
                method(*args, **defaults(kwargs, alpha=1 - np.log(1 + (np.e - 1) * (i - 1) / n)))
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
            ax (matplotlib.axes.Axes): The axis to add it to.
            scale (float): The size of the scale in data units.
            label (str, optional): A label for the scale.
            vertical (bool): If true, make a vertical one (default is a horizontal one).
            width (float): The line width of the scale in axis units.
            padding (float): The padding between the scale and the axis.
            loc (str): The location of the scale. Can be any of the usual matplotlib locations, e.g. 'auto', 'upper left', 'upper center', 'upper right', 'center left', 'center', 'center right', 'lower left', 'lower center, 'lower right'.
            color (str): Color for the patch.
            fontsize (str): Font size of the label.

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


class XManifoldPlot(XPlot):
    def __init__(self, on_x, on_y, *, on_y_separators=",-+", on_y_subs={}, **kwargs):
        """
        Base class for plotting manifold plots

        A manifold plot consists of multiple subplots, axes and twin axes, all of which
        share the x-axis. The **manifold subplot specification string** ``on_y`` defines what
        is plotted on the y-axes. It should specify a property for each trace, separated by
        ``,`` for each subplot, by ``-`` for twin axes and by ``+`` for traces. For example, the
        string ``"a+b,c-d"`` specifies 2 subplots where traces a and b share the same
        y-axis on the first subplot and traces c and d have individual y-axis on the
        second subplot.

        When deriving from this class, you should call :meth:`~.base.XManifoldPlot._create_artists` during init

        Args:
            on_x (str | None): What to plot on the x-axis
            on_y (str or list): What to plot on the y-axis. See :meth:`~.base.XManifoldPlot.parse_nested_list_string`.
                                May optionally contain a post-processing function call with the property as first argument
                                such as `smooth(key)` or `smooth(key, n=10)`.
            on_y_separators (str): See :meth:`~.base.XManifoldPlot.parse_nested_list_string`
            on_y_subs (dict): See :meth:`~.base.XManifoldPlot.parse_nested_list_string`
            kwargs: Keyword arguments passed to :class:`~.base.XPlot`
        """
        if len(on_y_separators) != 3:
            raise ValueError(
                f"Exactly 3 separators required but got on_y_separators={on_y_separators!r}"
            )

        self.on_x = on_x
        self.on_y, self.on_y_expression = self.parse_nested_list_string(
            on_y, on_y_separators, on_y_subs, strip_off_methods=True
        )

        super().__init__(
            nrows=len(self.on_y),
            ncols=1,
            nntwins=[len(tw) - 1 for tw in self.on_y],
            sharex="all",
            **kwargs,
        )

        # Format axes
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                a = self.axis(i, j)
                a.set(ylabel=self.label_for(*pp))

                units = np.unique([self.display_unit_for(p) for p in pp])
                if len(units) == 1:
                    if units[0] == "rad":
                        self._set_axis_ticks_angle(a.yaxis, minor=True, deg=False)
                    elif units[0] in ("deg", "°"):
                        self._set_axis_ticks_angle(a.yaxis, minor=True, deg=True)

        self.axis(-1).set(xlabel=self.label_for(self.on_x))

    @property
    def on_y_unique(self):
        return np.unique([p for ppp in self.on_y for pp in ppp for p in pp])

    def _create_artists(self, callback):
        """Helper method to create artists for subplots and twin axes

        Args:
            callback (function): Callback function to create artists.
                Signature: (i, j, k, axis, p) -> artist
                Where i, j, k are the subplot, twin-axis, trace indices respectively;
                axis is the axis and the string p is the property to plot.
        """
        self.artists = []
        for i, ppp in enumerate(self.on_y):
            self.artists.append([])
            for j, pp in enumerate(ppp):
                self.artists[i].append([])
                a = self.axis(i, j)
                for k, p in enumerate(pp):
                    artist = callback(i, j, k, a, p)
                    self.artists[i][j].append(artist)

            self.legend(i, show="auto")

    def artist(self, name=None, subplot=None, twin=None, trace=None):
        """Return the artist either by name, or by subplot, twin axes and trace index

        Args:
            name (str, optional): Name of the property the artist is plotting
            subplot (int, optional): Flat subplot index
            twin (int, optional): Twin axis index
            trace (int, optional): Trace index

        Returns:
            matplotlib.artist.Artist: First artist that matches the given criteria
        """
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    if (
                        (p == name or name is None)
                        and (i == subplot or subplot is None)
                        and (j == twin or twin is None)
                        and (k == trace or trace is None)
                    ):
                        return self.artists[i][j][k]

    def _legend_label_for(self, p):
        """
        Return legend label for a single property

        Args:
            p (str | tuple): Property name or tuple of (subplot, twin, trace) index

        Returns:
            str: Legend label
        """
        if isinstance(p, str):
            return super()._legend_label_for(p)
        else:
            i, j, k = p
            label = super()._legend_label_for(self.on_y[i][j][k])
            if w := self.on_y_expression[i][j][k]:
                w = re.split(r"\b", w)[1]
                label += f" ({w})"
            return label

    def legend(self, subplot="all", show=True, **kwargs):
        """Add, update or remove legend for a subplot

        Args:
            subplot (Union[int, iterable, "all"]): Subplot axis index or indices
            show (Union[bool, "auto"]): If True, show the legend. If "auto", show
              legend for subplots with more than one trace, or if subplot is specified explicitly.
            kwargs: Keyword arguments passed to :meth:`matplotlib.axes.Axes.legend`

        """
        if subplot == "all":
            subplot = range(len(self.axflat))
        if isinstance(subplot, int):
            subplot = [subplot]
            if show == "auto":
                show = True  # always show legend if single subplot is specified

        for s in subplot:
            # aggregate handles and use topmost axes for legend
            handles = []
            for ax in [self.axflat[s], *self.axflat_twin[s]]:
                handles.extend(ax.get_legend_handles_labels()[0])
            if not show or (show == "auto" and len(handles) <= 1):
                if ax.get_legend():
                    ax.get_legend().remove()
            else:
                # join handles
                handle_map = {
                    h: [h] for h in handles if not hasattr(h, "_join_legend_entry_with")
                }
                for h in handles:
                    if main_handle := getattr(h, "_join_legend_entry_with", None):
                        handle_map[main_handle].append(h)
                handles = [tuple(hs) for hs in handle_map.values()]
                labels = [h.get_label() for h in handle_map]

                # show legend
                ax.legend(handles=handles, labels=labels, **kwargs)

    def autoscale(self, subplot="all", *, reset=False, freeze=True, tight=None):
        """Autoscale the axes of a subplot

        Args:
            subplot (int | iterable | str): Subplot axis index, indices or "all"
            reset (bool): Whether to ignore any data limits already registered.
            freeze (bool): Whether to keep the updated axes limits (True) or enable automatic
                autoscaling on future draws (for all present and new artists).
            tight (str | None): Enables tight scaling without margins for "x", "y", "both" or None.
        """
        kwargs = dict(reset=reset, freeze=freeze, tight=tight)

        if subplot == "all":
            subplot = range(len(self.axflat))

        for s in flattened(subplot):
            self._autoscale(self.axflat[s], flattened(self.artists[s][0]), **kwargs)
            for i, axt in enumerate(self.axflat_twin[s]):
                self._autoscale(axt, flattened(self.artists[s][i]), **kwargs)

    @staticmethod
    def parse_nested_list_string(
        list_string, separators=",-+", subs={}, *, strip_off_methods=False
    ):
        """Parse a separated string or nested list or a mixture of both




        Args:
            list_string (str or list): The string or nested list or a mixture of both to parse.
            separators (str): The characters that separate the elements. The number of characters
                determines the depth of the returned list.
            subs (dict): A dictionary of substitutions to apply to the elements during parsing.
                May introduce additional separators of equal or deeper level.
            strip_off_methods (bool): If true, each element can be a name `name` or an expression
                                      in the form `method(name, ...)`. The methods are stripped off,
                                      and returned separately.

        Returns:
            nested list of names in the string,
            nested list of expressions in the string (only if strip_off_methods is True)

        Example:
            >>> XManifoldPlot.parse_nested_list_string("a+b, c-d,fun(e,2)")
            [[['a', 'b']], [['c'], ['d']], [['fun(e,2)']]]
            >>> XManifoldPlot.parse_nested_list_string("a+b, c-d,fun(e,2)", strip_off_methods=True)
            ([[['a', 'b']], [['c'], ['d']], [['e']]],
             [[[None, None]], [[None], [None]], [['fun(e,2)']]])
        """

        def savesplit(string, sep):
            """Split the string at sep except inside parantheses"""
            return re.split(f"\\{sep}\\s*(?![^()]*\\))", string)

        if type(list_string) is str:
            elements = []
            for element in savesplit(list_string, separators[0]):
                element = subs.get(element, element)
                # split again in case subs contains a separator
                elements.extend(savesplit(element, separators[0]))
        else:
            elements = list(list_string)
        expressions = [None] * len(elements)
        if len(separators) > 1:
            for i in range(len(elements)):
                result = XManifoldPlot.parse_nested_list_string(
                    elements[i], separators[1:], subs, strip_off_methods=strip_off_methods
                )
                if strip_off_methods:
                    elements[i], expressions[i] = result
                else:
                    elements[i] = result
        elif strip_off_methods:
            for i in range(len(elements)):
                if isinstance(elements[i], str):
                    if m := re.match(r".*\(([^(),\s]+)[^()]*\)", elements[i]):
                        name, expression = m.groups()[0], m.string
                        elements[i] = name
                        expressions[i] = expression
        return (elements, expressions) if strip_off_methods else elements


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
