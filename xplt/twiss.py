#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting twiss

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from .base import Xplot, style
from .line import KnlPlot


class TwissPlot(Xplot):
    def __init__(
        self,
        twiss=None,
        kind="bet-dx,x+y",
        *,
        ax=None,
        line=None,
        line_kwargs=dict(),
        display_units=None,
        **subplots_kwargs,
    ):
        """
        A plot for twiss parameters and closed orbit

        :param kind: Defines the properties to plot.
                     This can be a nested list or a separated string or a mixture of lists and strings where
                     the first list level (or separator ``,``) determines the subplots,
                     the second list level (or separator ``-``) determines any twinx-axes,
                     and the third list level (or separator ``+``) determines plots.
                     In addition, abbreviations for x-y-parameter pairs are supported (e.g. 'bet' for 'betx+bety').

                     Examples:
                      - ``'bet-dx'``: single subplot with 'betx' and 'bety' on the left and 'dx' on the right axis
                      - ``[[['betx', 'bety'], ['dx']]]``: same as above
                      - ``'betx+alf,mu'``: two suplots the first with 'betx', 'alfx' and 'alfy' and the second with 'mux' and 'muy'
                      - ``[[['betx', 'alfx', 'alfy']], [['mux', 'muy']]]``: same as above

        :param twiss: Dictionary with twiss information
        :param ax: A list of axes to plot onto, length must match the number of subplots and optional line plot. If None, a new figure is created.
                   If required, twinx-axes will be added automatically.
        :param line: Line of elements. If given, adds a line plot to the top.
        :param line_kwargs: Keyword arguments passed to line plot.
        :param display_units: Dictionary with units for parameters. Supports prefix notation, e.g. 'bet' for 'betx' and 'bety'.
        :param subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(
            display_units=dict(
                dict(s="m", x="mm", y="mm", p="mrad", bet="m", d="m"),
                **(display_units or {}),
            )
        )

        # parse kind string
        if type(kind) is str:
            kind = kind.split(",")
        kind = list(kind)
        for i in range(len(kind)):
            if type(kind[i]) is str:
                kind[i] = kind[i].split("-")
            for j in range(len(kind[i])):
                if type(kind[i][j]) is str:
                    kind[i][j] = kind[i][j].split("+")
                k = 0
                while k < len(kind[i][j]):
                    if kind[i][j][k] in ("", "p", "alf", "bet", "gam", "mu", "d", "dp"):
                        kind[i][j].insert(k + 1, kind[i][j][k] + "y")
                        kind[i][j][k] += "x"
                        k += 1
                    k += 1
        self.kind = kind

        # Create plot axes
        if ax is None:
            _, ax = plt.subplots(
                len(self.kind) + (1 if line else 0), sharex="col", **subplots_kwargs
            )
        if not hasattr(ax, "__iter__"):
            ax = [ax]
        self.fig, self.ax = ax[0].figure, ax
        # Create line plot
        if line:
            self.lineplot = KnlPlot(line, ax=self.ax[0], **line_kwargs)
            self.lineplot.ax.set(xlabel=None)
            self.ax = self.ax[1:]
        self.ax_twin = [[] for a in self.ax]  # twinx-axes are created below
        # Format plot axes
        for a in self.ax:
            a.grid()
        self.ax[-1].set(xlabel="s / " + self.display_unit_for("s"))

        # create plot elements
        self.artists = []
        for i, ppp in enumerate(self.kind):
            self.artists.append([])
            legend = [], []

            for j, pp in enumerate(ppp):
                # axes
                if j == 0:
                    a = self.ax[i]
                else:  # create twinx-axes if required
                    a = self.ax[i].twinx()
                    a._get_lines.prop_cycler = self.ax[i]._get_lines.prop_cycler
                    self.ax_twin[i].append(a)
                a.set(ylabel=self.label_for(*pp))

                # create artists for traces
                self.artists[i].append([])
                for k, p in enumerate(pp):
                    (artist,) = a.plot([], [], label=self.label_for(p, unit=False))
                    self.artists[i][j].append(artist)
                    legend[0].append(artist)
                    legend[1].append(artist.get_label())

            if len(legend[0]) > 1:
                a.legend(*legend)

        # set data
        if twiss:
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
            for j, pp in enumerate(ppp):
                a = self.ax[i] if j == 0 else self.ax_twin[i][j - 1]
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
            kwargs = style(kwargs, color="k", zorder=1.9)
        else:  # a span
            kwargs = style(kwargs, color="lightgray", zorder=1.9, lw=0, alpha=0.6)

        if kind == "s":
            # vertical span or line on all axes
            for a in self.ax:
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
                    a = self.ax[i] if j == 0 else self.ax_twin[i][j - 1]
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
