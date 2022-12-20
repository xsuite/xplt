#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting twiss

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"

import types

from .util import defaults
from .base import XPlot
from .line import KnlPlot


class TwissPlot(XPlot):
    def __init__(
        self,
        twiss=None,
        kind="bet-dx,x+y",
        *,
        ax=None,
        line=None,
        line_kwargs=dict(),
        data_units=None,
        display_units=None,
        **subplots_kwargs,
    ):
        """
        A plot for twiss parameters and closed orbit

        Args:
            kind: Defines the properties to plot.
                     This can be a nested list or a separated string or a mixture of lists and strings where
                     the first list level (or separator ``,``) determines the subplots,
                     the second list level (or separator ``-``) determines any twinx-axes,
                     and the third list level (or separator ``+``) determines plots on the same axis.
                     In addition, abbreviations for x-y-parameter pairs are supported (e.g. 'bet' for 'betx+bety').

                     Examples:
                      - ``'bet-dx'``: single subplot with 'betx' and 'bety' on the left and 'dx' on the right axis
                      - ``[[['betx', 'bety'], ['dx']]]``: same as above
                      - ``'betx+alf,mu'``: two suplots the first with 'betx', 'alfx' and 'alfy' and the second with 'mux' and 'muy'
                      - ``[[['betx', 'alfx', 'alfy']], [['mux', 'muy']]]``: same as above

            twiss: Dictionary with twiss information
            ax: A list of axes to plot onto, length must match the number of subplots and optional line plot. If None, a new figure is created.
                If required, twinx-axes will be added automatically.
            line: Line of elements. If given, adds a line plot to the top.
            line_kwargs: Keyword arguments passed to line plot.
            data_units (dict, optional): Units of the data. If None, the units are determined from the data.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(
            data_units=data_units,
            display_units=defaults(display_units, bet="m", d="m"),
            prefix_suffix_config={
                p: (p + "x", p + "y") for p in "alf,bet,gam,mu,d,dp,q,dq".split(",")
            },
        )

        # parse kind string
        subs = {p: "+".join(s) for p, s in self._prefix_suffix_config.items()}
        self.kind = self._parse_nested_list_string(kind, ",-+", subs)

        # initialize figure with n subplots
        n, nntwin = len(self.kind), [len(tw) - 1 for tw in self.kind]
        if line:
            n += 1
            nntwin = [0] + nntwin
        self._init_axes(ax, n, 1, nntwin, sharex="col", **subplots_kwargs)

        # Create line plot
        self.lineplot = None
        if line:
            self.lineplot = KnlPlot(line, ax=self.axis_for(0), **line_kwargs)
            self.lineplot.ax.set(xlabel=None)

        # Format plot axes
        self.axis_for(-1).set(xlabel=self.label_for("s"))

        # create plot elements
        def create_artists(i, j, k, a, p):
            return a.plot([], [], label=self.label_for(p, unit=False))[0]

        self._init_artists([[], *self.kind] if line else self.kind, create_artists)

        # set data
        if twiss is not None:
            self.update(twiss, autoscale=True)

    def update(self, twiss, *, autoscale=False, line=None):
        """
        Update the twiss data this plot shows

        :param twiss: Dictionary with twiss information.
        :param line: Line of elements.
        :param autoscale: Whether or not to perform autoscaling on all axes.
        :return: changed artists.
        """
        s = self.factor_for("s")
        changed = []
        for i, ppp in enumerate(self.kind):
            if self.lineplot is not None:
                i += 1  # skip line plot
            for j, pp in enumerate(ppp):
                a = self.axis_for(i, j)
                for k, p in enumerate(pp):
                    f = self.factor_for(p)
                    self.artists[i][j][k].set_data((s * twiss["s"], f * twiss[p]))
                    changed.append(self.artists[i][j][k])
                if autoscale:
                    a.relim()
                    a.autoscale()
                    a.set(xlim=(s * min(twiss["s"]), s * max(twiss["s"])))

        if line:
            self.lineplot.update(line, autoscale=autoscale)

        return changed

    def _texify_label(self, label, suffixes=()):
        label = {
            "alf": "\\alpha",
            "bet": "\\beta",
            "gam": "\\gamma",
            "mu": "\\mu",
            "d": "D",
            "dzeta": "D_\\zeta",
            "ptau": "p_\\tau",
        }.get(label, label)
        return super()._texify_label(label, suffixes)

    def axline(self, kind, val, subplots="all", **kwargs):
        """Plot a vertical or horizontal line for a given coordinate

        Args:
            kind (str): property at which to place the line (e.g. "s", "x", "betx", etc.)
            val (float): Value of property
            subplots (list of int): Subplots to plot line onto. Defaults to all with matching coordinates.
            kwargs: Arguments for axvline or axhline

        """
        return self.axspan(kind, val, None, subplots=subplots, **kwargs)

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
            kwargs = defaults(kwargs, color="k", zorder=1.9)
        else:  # a span
            kwargs = defaults(kwargs, color="lightgray", zorder=1.9, lw=0, alpha=0.6)

        if kind == "s":
            # vertical span or line on all axes
            for a in self.axflat:
                if val_to is None:  # only a line
                    a.axvline(val * self.factor_for("s"), **kwargs)
                else:
                    a.axvspan(
                        val * self.factor_for("s"),
                        val_to * self.factor_for("s"),
                        **kwargs,
                    )

        else:
            # horizontal span or line
            for i, p_subplot in enumerate(self.kind):
                if subplots != "all" and i not in subplots:
                    continue

                for j, p_axis in enumerate(p_subplot):
                    a = self.axis_for(i, j)
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
