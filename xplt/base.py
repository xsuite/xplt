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

from .util import defaults, flattened, defaults_for
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
            function_twin_to_this (function): Function to calculate tick values of this axis given the tick values of the other axis.
            function_this_to_twin (function): Function to calculate tick values of the other axis given the tick values of this axis.
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
            transform (function): Function to transform tick values of this locator to the values of the dependent locator
            inverse (function): Inverse of transform
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
        log=False,
        nntwins=None,
        annotation=None,
        _properties=None,
        **subplots_kwargs,
    ):
        """
        Base class for plotting

        Initialize the subplots, axes and twin axes

        Args:
            data_units (dict | None): Units of the data. If None, the units are determined from default and user property settings.
            display_units (dict | None): Units to display the data in. If None, the units are determined from the data.
            ax (matplotlib.axes.Axes | None): Axes to plot onto. If None, a new figure is created.
            grid (bool): If True, show grid lines on all axes.
            log (bool | str): If True, `"xy"`, `"x"` or `"y"`, make the respective axis/axes log-scaled
            nntwins (list | None): List defining how many twin axes to create for each subplot.
            annotation (bool | None): Whether to add an annotation or not. If None (default) add it unless `ax` is passed.
            subplots_kwargs: Keyword arguments passed to :func:`matplotlib.pyplot.subplots` command when a new figure is created.
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
            for xy in "xy":
                if log is True or (isinstance(log, str) and xy in log.lower()):
                    a.set(**{f"{xy}scale": "log"})

            # Create twin axes
            self.axflat_twin.append([])
            if nntwins is not None:
                for j in range(nntwins[i]):
                    twin = a.twinx()
                    if log is True or (isinstance(log, str) and "y" in log.lower()):
                        twin.set(yscale="log")
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

    def _autoscale(
        self, ax, axis="xy", *, artists=None, data=None, reset=False, freeze=True, tight=None
    ):
        """Autoscale axes to fit given artists

        If neither artists nor data is specified, consider all artists associated with the axis.

        Args:
            ax (matplotlib.axes.Axes): Axes to autoscale
            axis (str | None | bool): Whether and on which axes to perform autoscaling.
                One of `"x"`, `"y"`, `"xy"`, `False` or `None`. If `None`, decide based on :meth:`matplotlib.axes.Axes.get_autoscalex_on` and :meth:`matplotlib.axes.Axes.get_autoscaley_on`.
                For backwards compatibility, the following aliases are also supported: `"both"`, `True`, `""`.
            artists (iterable): Artists to consider (if any)
            data (iterable): Data points to consider (if any) in the form [(x1,y1), (x2,y2), ...]
            reset (bool): Whether to ignore any data limits already registered.
            freeze (bool): Whether to keep the updated axes limits (True) or enable automatic
                autoscaling on future draws (for all present and new artists).
            tight (str | None | bool): Enables tight scaling without margins for axis.
                Any of `"x"`, `"y"`, `"xy"` or `"both"`, `""` or `None`.

        Returns:
            str: The axes autoscaled (`""`, `"x"`, `"y"` or `"xy"`)
        """
        if axis is None:
            axis = "x" * ax.get_autoscalex_on() + "y" * ax.get_autoscaley_on()
        axis = {False: "", True: "xy", "both": "xy"}.get(axis, axis)
        if not axis:
            return ""
        tight_x = ("x" in tight or tight == "both") if isinstance(tight, str) else tight
        tight_y = ("y" in tight or tight == "both") if isinstance(tight, str) else tight

        if artists is None and data is None:
            # use limits from all artists associated with the axis   https://stackoverflow.com/a/71966295
            # unlike ax.relim(), we also handle collections
            artists = ax.lines + ax.collections + ax.patches + ax.images

        data = [] if data is None else data[:]  # make a copy so we can safely append
        limits = []

        # Get data limits from artists
        drawn = False
        for art in flattened(artists) if artists is not None else []:
            if hasattr(art, "get_datalim"):
                # Collections, patches, etc.
                lim = art.get_datalim(ax.transData)
                if not np.all(np.isfinite(lim)):
                    # fallback to offsets (e.g. for hexbin)
                    data.extend(art.get_offsets())
                else:
                    limits.append(lim)

            elif hasattr(art, "get_data"):
                # e.g. Line2D
                data.extend(np.transpose(art.get_data()))

            elif isinstance(art, mpl.artist.Artist):
                # Any other artist (e.g. Text), but requires to draw figure
                if not drawn:
                    self.fig.canvas.draw()
                    drawn = True
                lim = art.get_window_extent().transformed(ax.transData.inverted())
                limits.append(lim)

            elif art is not None:
                raise NotImplementedError(f"Autoscaling not implemented for {art!r}")

        # Add limits from raw data
        if len(data) > 0:
            x, y = [a[np.isfinite(a)] for a in np.transpose(data)]
            limits.append(mpl.transforms.Bbox.from_extents(x.min(), y.min(), x.max(), y.max()))

        # Update axes limits
        if len(limits) > 0:
            dataLim = mpl.transforms.Bbox.union(limits)

            if reset:
                ax.dataLim = dataLim
            else:
                ax.update_datalim(dataLim)  # takes previous datalim into account

        # Autoscale (on next and future draws)
        if "x" in axis:
            if tight_x is False and ax.margins()[0] == 0:
                ax.margins(x=0.05)  # restore default margins
            ax.autoscale(axis="x", tight=tight_x)
        if "y" in axis:
            if tight_y is False and ax.margins()[1] == 0:
                ax.margins(y=0.05)  # restore default margins
            ax.autoscale(axis="y", tight=tight_y)

        if freeze:
            # perform autoscale immediately and freeze limits
            ax.autoscale_view()
            ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
            ax.set_autoscale_on(False)

        return axis

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
            subplot (int): Flat subplot index
            twin (int): Twin index

        Returns:
            matplotlib.axes.Axes: Axis for the given subplot and twin index
        """
        return self.axflat_twin[subplot][twin - 1] if twin else self.axflat[subplot]

    def axes(self, subplots="all", twins="all"):
        """Generator yielding axes for the given flat subplot and twin indices

        Args:
            subplots (int | list[int] | str): Flat subplot indices or ``"all"``
            twins (int | list[int] | str | bool): Twin index or indices or ``"all"``.
                ``True`` as alias for ``"all"`` and ``False`` as alias for ``0`` are also supported.

        Yields:
            matplotlib.axes.Axes: Iterator over the selected axes where
        """
        subplots = [subplots] if isinstance(subplots, int) else subplots
        twins = [twins] if isinstance(twins, int) else {False: [0], True: "all"}.get(twins, twins)
        for s in range(len(self.axflat)):
            if subplots != "all" and s not in subplots:
                continue
            for t in range(len(self.axflat_twin[s]) + 1):
                if twins != "all" and t not in twins:
                    continue
                yield self.axflat_twin[s][t - 1] if t else self.axflat[s]

    def save(self, fname, **kwargs):
        """Save the figure with sensible default options

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
                    append = f" / ${display_unit:~X}$"  # see "NIST Guide to the SI"
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

        Indicates the bands at the h-th harmonics (h = 1, 2, ..., n) at
            - h * (v ± dv/2)      if scale_width and not inverse (default)
            - h * v ± dv/2        if not scale_width and not inverse
            - h / (v ± dv/2)      if inverse and scale_width
            - 1 / ( v/h ± dv/2 )  if inverse and not scale_width

        Args:
            ax (matplotlib.axes.Axes): Axes to plot onto.
            v (float | list[float]): Value or list of values.
            dv (float | list[float]): Width or list of widths centered around value(s).
            n (int): Number of harmonics to plot.
            scale_width (bool): Whether to scale the width for higher harmonics or keep it constant.
            vertical (bool): Plot vertical lines if true, horizontal otherweise.
            inverse (bool): If true, plot harmonics of n/(v±dv) instead of n*(v±dv).
                Useful to plot frequency harmonics in time domain and vice-versa.
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
            label (str | None): A label for the scale.
            vertical (bool): If true, make a vertical one (default is a horizontal one).
            width (float): The line width of the scale in axis units.
            padding (float): The padding between the scale and the axis.
            loc (str): The location of the scale. Can be any of the usual matplotlib locations,
                e.g. ``"auto"``, ``"upper left"``, ``"upper center"``, ``"upper right"``,
                ``"center left"``, ``"center"``, ``"center right"``, ``"lower left"``,
                ``"lower center"`` or ``"lower right"``.
            color (str | tuple): Color for the patch.
            fontsize (str): Font size of the label.

        Returns:
            matplotlib.offsetbox.AnchoredOffsetbox: The artist added
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
        string ``"a+b,c-d"`` specifies 2 subplots (a+b and c-d) where on the first subplot the
        traces a and b share the same y-axis and on the second subplot traces c and d have individual y-axis.

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
        self._on_y_separators = on_y_separators
        self._on_y_subs = on_y_subs
        self._artists = {}

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

    def _create_artists(self, callback, dataset_id=None):
        """Helper method to create artists for subplots and twin axes

        Args:
            callback (callable[int, int, int, matplotlib.axes.Axes, str]): Callback function to create artists.
                Signature: (i, j, k, axis, p) -> artist
                Where i, j, k are the subplot, twin-axis, trace indices respectively;
                axis is the axis and the string p is the property to plot.
            dataset_id (str | None): The dataset identifier if this plot represents multiple datasets
        """

        if dataset_id in self._artists:
            raise ValueError(f"Dataset identifier `{dataset_id}` already exists")
        if dataset_id is not None and not isinstance(dataset_id, str):
            raise ValueError(f"The dataset identifier must be a str, but got {type(dataset_id)}")

        self._artists[dataset_id] = []
        for i, ppp in enumerate(self.on_y):
            self._artists[dataset_id].append([])
            for j, pp in enumerate(ppp):
                self._artists[dataset_id][i].append([])
                a = self.axis(i, j)
                for k, p in enumerate(pp):
                    artist = callback(i, j, k, a, p)
                    self._artists[dataset_id][i][j].append(artist)

        self.legend(show="auto")

    @property
    def artists(self):
        """Convenient access to artist

        The index can be an (optional) dataset identifier, followed by the subplot, axis and trace index.
        If the dataset identifier is missing, traces are concatenated for all datasets
        Examples: `self.artists[i][j][k]`, `self.artists[i,j,k]`, `self.artists[dataset_id,i,j,k]`
        """

        ALL = object()

        class ArtistIndexHelper:
            """Helper for convenient and backwards-compatible indexing of artists"""

            def __init__(s, dataset=ALL, *indices):
                s.dataset = dataset
                s.indices = indices

            def __iter__(s):
                for i in range(len(s)):
                    yield s[i]

            def __len__(s):
                d = s.dataset if s.dataset is not ALL else list(self._artists)[0]
                if len(s.indices) == 0:
                    return len(self._artists[d])
                elif len(s.indices) == 1:
                    (i,) = s.indices
                    return len(self._artists[d][i])
                elif len(s.indices) == 2:
                    i, j = s.indices
                    if s.dataset is not ALL:
                        return len(self._artists[d][i][j])
                    else:
                        return sum([len(self._artists[d][i][j]) for d in self._artists])

            def __getitem__(s, item):
                if isinstance(item, tuple):
                    return s[item[0]][item[1:]] if len(item) > 1 else s[item[0]]

                elif item is None or isinstance(item, str) or item == ALL:  # select dataset
                    if len(s.indices) > 0:
                        raise IndexError("only the first index may be a string or None")
                    return ArtistIndexHelper(dataset=item)

                elif isinstance(item, int):  # return index for all datasets
                    if len(s.indices) < 2:
                        if item < 0 or item >= len(s):
                            raise IndexError(f"list index {item} out of range 0..{len(s)}")
                        return ArtistIndexHelper(s.dataset, *s.indices, item)
                    else:
                        i, j, k = *s.indices, item
                        if s.dataset is not ALL:
                            return self._artists[s.dataset][i][j][k]
                        else:
                            for d in self._artists:
                                if 0 <= k < len(self._artists[d][i][j]):
                                    return self._artists[d][i][j][k]
                                else:
                                    k -= len(self._artists[d][i][j])
                            raise IndexError(f"list index {item} out of range")

                else:
                    raise IndexError(f"Index must be an int, string, tuple or None. Got {item}.")

            def __setitem__(s, item, value):
                if isinstance(item, tuple):
                    if len(item) > 1:
                        s[item[0]][item[1:]] = value
                    else:
                        s[item[0]] = value

                elif len(s.indices) < 2:
                    raise TypeError("Only single elements may be assigned")

                elif isinstance(item, int):
                    if item < 0 or item >= len(s):
                        raise IndexError(f"list index {item} out of range 0..{len(s)}")

                    i, j, k = *s.indices, item
                    if s.dataset is not ALL:
                        self._artists[s.dataset][i][j][k] = value
                    else:
                        for d in self._artists:
                            if 0 <= k < len(self._artists[d][i][j]):
                                self._artists[d][i][j][k] = value
                                break
                            else:
                                k -= len(self._artists[d][i][j])
                        else:
                            raise IndexError(f"list index {item} out of range")

                else:
                    raise IndexError(f"Index must be an int, string, tuple or None. Got {item}.")

            def __repr__(s):
                info = f"ArtistIndexHelper(dataset={'ALL' if s.dataset is ALL else s.dataset}"
                for i, label in enumerate(("subplot", "axis", "trace")):
                    if len(s.indices) > i:
                        info += f", {label}={s.indices[i]}"
                return info + ")"

        return ArtistIndexHelper()

    def artist(self, name=None, subplot=None, twin=None, trace=None, *, dataset_id=None):
        """Return the artist either by name, or by subplot, twin axes and trace index

        Args:
            name (str | None): Name of the property the artist is plotting
            subplot (int | None): Flat subplot index
            twin (int | None): Twin axis index
            trace (int | None): Trace index
            dataset_id (str | None): The dataset identifier if this plot represents multiple datasets

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
                        return self.artists[dataset_id][i][j][k]

    def axis(self, subplot_or_name=0, twin=0, *, subplot=None, name=None):
        """Return the axis for a given flat subplot index and twin index, or an artist name

        Either get axis by index ``axis(subplot, twin)`` or by property name ``axis(name)``.

        Args:
            subplot_or_name (int | str): Flat subplot index or property name
            subplot (int): Flat subplot index
            twin (int): Twin index
            name (str | None): Name of the property the axis is plotting. If this is not None, subplot and twin are ignored.

        Returns:
            matplotlib.axes.Axes: Axis for the given subplot and twin index
        """
        if isinstance(subplot_or_name, str):
            name = subplot_or_name
        elif subplot is None:
            subplot = subplot_or_name
        if name is not None:
            if subs := self._on_y_subs.get(name):
                for n in subs.split(self._on_y_separators[2]):
                    try:
                        return self.axis(name=n)
                    except ValueError:
                        continue
            else:
                for i, ppp in enumerate(self.on_y):
                    for j, pp in enumerate(ppp):
                        for k, p in enumerate(pp):
                            if p == name:
                                return self.axis(i, j)
            raise ValueError(f"Property {name} not found")
        return super().axis(subplot, twin)

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
            subplot (int | list | str): Subplot axis index or indices
            show (bool | str): If True, show the legend. If "auto", show
                legend for subplots with more than one trace, or if subplot is specified explicitly.
            kwargs: Keyword arguments passed to :meth:`matplotlib.axes.Axes.legend`

        """
        if subplot == "all":
            subplot = range(len(self.axflat))
        if isinstance(subplot, int):
            subplot = [subplot]
            if show == "auto":
                show = True  # always show legend if single subplot is specified

        set_kwargs = {k: kwargs.pop(k) for k in ("in_layout",) if k in kwargs}

        for s in subplot:
            # aggregate handles and use topmost axes for legend
            handles = []
            for ax in self.axes(s):
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
                if len(handles) > 0:
                    legend = ax.legend(handles=handles, labels=labels, **kwargs)
                    legend.set(**set_kwargs)

    def autoscale(self, subplot="all", *, axis="xy", reset=False, freeze=True, tight=None):
        """Autoscale the axes of a subplot

        Args:
            subplot (int | iterable | str): Subplot axis index, indices or "all"
            axis (str | None | bool): Whether and on which axes to perform autoscaling.
                One of `"x"`, `"y"`, `"xy"`, `False` or `None`. If `None`, decide based on :meth:`matplotlib.axes.Axes.get_autoscalex_on` and :meth:`matplotlib.axes.Axes.get_autoscaley_on`.
                For backwards compatibility, the following aliases are also supported: `"both"`, `True`, `""`.
            reset (bool): Whether to ignore any data limits already registered.
            freeze (bool): Whether to keep the updated axes limits (True) or enable automatic
                autoscaling on future draws (for all present and new artists).
            tight (str | None): Enables tight scaling without margins for the specified dimension.
                May be ``"x"``, ``"y"``, ``"both"`` or ``None``.
        """
        kwargs = dict(axis=axis, reset=reset, freeze=freeze, tight=tight)

        if subplot == "all":
            subplot = range(len(self.axflat))

        for s in flattened(subplot):
            self._autoscale(self.axflat[s], **kwargs)
            for i, axt in enumerate(self.axflat_twin[s]):
                self._autoscale(axt, **kwargs)

    def axline(self, kind, val, **kwargs):
        """Plot a vertical or horizontal line for a given coordinate

        Args:
            kind (str): property at which to place the line (e.g. "s", "x", "betx", etc.)
            val (float): Value of property.
            kwargs: See :meth:`~.base.XManifoldPlot.axspan`.

        """
        self.axspan(kind, val, None, **kwargs)

    def axspan(
        self,
        kind,
        val,
        val_to=None,
        *,
        subplots="all",
        annotation=None,
        annotation_loc=None,
        annotation_kwargs=None,
        **kwargs,
    ):
        """Plot a vertical or horizontal span (or line) for a given coordinate

        Args:
            kind (str): property at which to place the line (e.g. "s", "x", "betx", etc.).
            val (float): Value of property.
            val_to (float | None): Second value of property to plot a span. If this is `None`, plot a line instead of a span.
            subplots (list[int]): Subplots to plot line onto. Defaults to all with matching coordinates.
            annotation (string | None): Optional text annotation for the line or span. Use this to place
                text on the axes. To put text in the legend, use `label=...`.
            annotation_loc (float, string): Location of annotation along line or span as fraction between 0 and 1,
                or a string ``"lower"``, ``"bottom"``, ``left"``, ``"upper"``, ``"top"``, ``"right"``, ``center``.
                For vertical lines or spans extending over multiple subplots, 0 is the bottom of the lowermost
                and 1 the top of the uppermost subplot.
            annotation_kwargs (dict | None): Arguments for :meth:`matplotlib.axes.Axes.text`.
            kwargs: Arguments passed to :meth:`matplotlib.axes.Axes.axvspan` or :meth:`matplotlib.axes.Axes.axhspan`
                (or :meth:`matplotlib.axes.Axes.axvline` or :meth:`matplotlib.axes.Axes.axhline` if `val_to` is `None`)

        """
        if subplots == "all":
            subplots = list(range(len(self.on_y)))

        if isinstance(annotation_loc, str):
            if annotation_loc in ("lower", "bottom", "left"):
                annotation_loc = 0
            elif annotation_loc in ("upper", "top", "right"):
                annotation_loc = 1
            elif annotation_loc in ("center", "centre"):
                annotation_loc = 0.5
            else:
                raise ValueError(f"Invalid annotation location: {annotation_loc}")

        if val_to is None:  # only a line
            kwargs = defaults_for("plot", kwargs, color="k", lw=1, zorder=1.9)
        else:  # a span
            kwargs = defaults_for(
                "fill_between", kwargs, color="lightgray", zorder=1.9, lw=0, alpha=0.6
            )

        val = val * self.factor_for(kind)
        if val_to is not None:
            val_to = val_to * self.factor_for(kind)

        def plot_line_with_annotation(hor, loc, with_annotation=True):
            if val_to is None:
                line = (a.axhline if hor else a.axvline)(val, **kwargs)
                color = line.get_color()
            else:
                line = (a.axhspan if hor else a.axvspan)(val, val_to, **kwargs)
                color = line.get_facecolor()
            if annotation and with_annotation:
                align = int(np.clip(round(2 * loc), 0, 2))  # 0, 1 or 2
                text_kwargs = defaults_for(
                    "text",
                    annotation_kwargs,
                    fontsize="xx-small",
                    c=color,  # RGBA tuple
                    alpha=1,  # overwrites color[3]
                    zorder=line.get_zorder(),
                    ha=["left", "center", "right"][align] if hor else "right",
                    va="bottom" if hor else ["bottom", "center", "top"][align],
                    rotation=0 if hor else 90,
                )
                a.text(
                    *(val, loc)[:: -1 if hor else 1],
                    s=f" {annotation} ",
                    transform=mpl.transforms.blended_transform_factory(
                        *(a.transData, a.transAxes)[:: -1 if hor else 1]
                    ),
                    **text_kwargs,
                )

        for i, ppp in enumerate(self.on_y):
            if i not in subplots:
                continue

            for j, pp in enumerate(ppp):
                a = self.axis(i, j)
                if kind == self.on_x:  # vertical line or span
                    loc = annotation_loc if annotation_loc is not None else 1
                    # rescale from global to subplot coordinates:
                    loc = (max(subplots) - min(subplots) + 1) * loc - (max(subplots) - i)
                    plot_line_with_annotation(hor=False, loc=loc, with_annotation=0 <= loc <= 1)
                    break  # skip twin axes

                else:
                    # horizontal line or span
                    for k, p in enumerate(pp):
                        if p == kind:  # axis found
                            loc = annotation_loc if annotation_loc is not None else 0
                            plot_line_with_annotation(hor=True, loc=loc)

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
                in the form `method(name, ...)`. The methods are stripped off, and returned separately.

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
