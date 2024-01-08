#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting twiss

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"

import types

from .util import defaults, defaults_for
from .base import XManifoldPlot
from .line import KnlPlot


class TwissPlot(XManifoldPlot):
    """A plot for twiss parameters and closed orbit"""

    def __init__(self, twiss=None, kind="bet-dx,x+y", *, line=None, line_kwargs={}, **kwargs):
        """

        Args:
            twiss (Any): Dictionary with twiss information
            kind (str | list): Defines the properties to plot.
                This is a manifold subplot specification string like ``"bet-dx,x+y"``, see :class:`~.base.XManifoldPlot` for details.
                In addition, abbreviations for x-y-parameter pairs are supported (e.g. 'bet' for 'betx+bety').
            line (xtrack.Line): Line of elements. If given, adds a line plot to the top.
            line_kwargs (dict): Keyword arguments passed to line plot.
            kwargs: See :class:`~.base.XPlot` for additional arguments


        """
        subs = {p: f"{p}x+{p}y" for p in "alf,bet,gam,mu,d,dp,q,dq".split(",")}

        if line:
            kind = self.parse_nested_list_string(kind, subs=subs)
            kind = [[[None]], *kind]

        super().__init__(
            on_x="s",
            on_y=kind,
            on_y_subs=subs,
            display_units=defaults(kwargs.pop("display_units", None), bet="m", d="m"),
            **kwargs,
        )

        # create plot elements
        def create_artists(i, j, k, a, p):
            if line and i == 0:
                return None  # skip line plot placeholder
            return a.plot([], [], label=self._legend_label_for((i, j, k)))[0]

        self._create_artists(create_artists)

        # Create line plot
        self.lineplot = None
        if line:
            self.lineplot = KnlPlot(line, ax=self.axis(0), **line_kwargs)
            self.lineplot.ax.set(xlabel=None)

        # set data
        if twiss is not None:
            self.update(twiss, autoscale=True)

    def update(self, twiss, *, autoscale=False, line=None):
        """
        Update the twiss data this plot shows

        Args:
            twiss (Any): Dictionary with twiss information
            autoscale (bool): Whether or not to perform autoscaling on all axes.
            line (xtack.Line): Line of elements.

        Returns:
            changed artists
        """
        s = self.prop("s").values(twiss, unit=self.display_unit_for("s"))
        changed = []
        for i, ppp in enumerate(self.on_y):
            if self.lineplot is not None and i == 0:
                continue  # skip line plot
            for j, pp in enumerate(ppp):
                a = self.axis(i, j)
                for k, p in enumerate(pp):
                    v = self.prop(p).values(twiss, unit=self.display_unit_for(p))
                    self.artists[i][j][k].set_data((s, v))
                    changed.append(self.artists[i][j][k])
                if autoscale:
                    a.relim()
                    a.autoscale()
                    a.set(xlim=(min(s), max(s)))

        if line:
            changed.append(self.lineplot.update(line, autoscale=autoscale))

        return changed

    def axline(self, kind, val, subplots="all", **kwargs):
        """Plot a vertical or horizontal line for a given coordinate

        Args:
            kind (str): property at which to place the line (e.g. "s", "x", "betx", etc.)
            val (float): Value of property
            subplots (list of int): Subplots to plot line onto. Defaults to all with matching coordinates.
            kwargs: Arguments for axvline or axhline

        """
        self.axspan(kind, val, None, subplots=subplots, **kwargs)

    def axspan(self, kind, val, val_to=None, subplots="all", **kwargs):
        """Plot a vertical or horizontal span (or line) for a given coordinate

        Args:
            kind (str): property at which to place the line (e.g. "s", "x", "betx", etc.).
            val (float): Value of property.
            val_to (float, optional): Second value of property to plot a span. If this is None, plot a line instead of a span.
            subplots (list of int): Subplots to plot line onto. Defaults to all with matching coordinates.
            kwargs: Arguments for axvspan or axhspan (or axvline or axhline if val_to is None)

        """

        if val_to is None:  # only a line
            kwargs = defaults_for("plot", kwargs, color="k", zorder=1.9)
        else:  # a span
            kwargs = defaults_for(
                "fill_between", kwargs, color="lightgray", zorder=1.9, lw=0, alpha=0.6
            )

        if kind == "s":
            # vertical span or line on all axes
            for a in self.axflat:
                if val_to is None:  # only a line
                    a.axvline(val * self.factor_for("s"), **kwargs)
                else:
                    a.axvspan(val * self.factor_for("s"), val_to * self.factor_for("s"), **kwargs)

        else:
            # horizontal span or line
            for i, p_subplot in enumerate(self.on_y):
                if subplots != "all" and i not in subplots:
                    continue

                for j, p_axis in enumerate(p_subplot):
                    a = self.axis(i, j)
                    for k, p in enumerate(p_axis):
                        if p == kind:  # axis found
                            if val_to is None:  # only a line
                                a.axhline(val * self.factor_for(p), **kwargs)
                            else:
                                a.axhspan(
                                    val * self.factor_for("s"),
                                    val_to * self.factor_for("s"),
                                    **kwargs,
                                )


## Restrict star imports to local namespace
__all__ = [
    name
    for name, thing in globals().items()
    if not (name.startswith("_") or isinstance(thing, types.ModuleType))
]
