#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting lines

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

from .base import XsuitePlot


class KnlPlot(XsuitePlot):
    """
    A plot for knl values along line
    """

    def __init__(
        self,
        line=None,
        *,
        knl=None,
        ax=None,
        filled=True,
        display_units=None,
        resolution=1000,
        **subplots_kwargs,
    ):
        """
        A plot for knl values along line

        :param line: Line of elements.
        :param knl: List of orders n to plot knl values for. If None, automatically determine from line.
        :param ax: An axes to plot onto. If None, a new figure is created.
        :param filled: If True, make a filled plot instead of a line plot.
        :param display_units: Dictionary with units for parameters. Supports prefix notation, e.g. 'bet' for 'betx' and 'bety'.
        :param resolution: Number of points to use for plot.
        :param subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(display_units=display_units)

        if knl is None:
            if line is None:
                raise ValueError("Either line or N parameter must not be None")
            knl = range(max([e.order for e in line.elements if hasattr(e, "order")]) + 1)
        self.knl = knl
        self.S = np.linspace(0, line.get_length(), resolution)
        self.filled = filled

        # Create plot
        if ax is None:
            _, ax = plt.subplots(**subplots_kwargs)
        self.fig, self.ax = ax.figure, ax
        self.ax.set(xlabel=self.label_for("s"), ylabel="$k_nl$")
        self.ax.grid()

        # create plot elements
        self.artists = []
        for n in self.knl:
            if self.filled:
                artist = self.ax.fill_between(
                    self.S, np.zeros_like(self.S), alpha=0.5, label=self.label_for(f"k{n}l"), zorder=3
                )
            else:
                (artist,) = self.ax.plot([], [], alpha=0.5, label=self.label_for(f"k{n}l"))
            self.artists.append(artist)
        self.ax.plot(self.S, np.zeros_like(self.S), "k-", lw=1)
        self.ax.legend(ncol=5)

        # set data
        if line:
            self.update(line, autoscale=True)

    def update(self, line, autoscale=False):
        """
        Update the line data this plot shows

        :param line: Line of elements.
        :param autoscale: Whether or not to perform autoscaling on all axes
        :return: changed artists
        """
        # compute knl as function of s
        KNL = np.zeros((len(self.knl), self.S.size))
        for s, name, el in zip(line.get_s_elements(), line.element_names, line.elements):
            if hasattr(el, "knl"):
                for i, n in enumerate(self.knl):
                    if n <= el.order:
                        mask = (s <= self.S) & (self.S < s + el.length)
                        KNL[i, mask] += el.knl[n]

        # plot
        s = self.factor_for("s")
        changed = []
        for n, art, knl in zip(self.knl, self.artists, KNL):
            f = self.factor_for(f"k{n}l")
            if self.filled:
                art.get_paths()[0].vertices[1 : 1 + knl.size, 1] = f * knl  # TODO shape 1003 ?
            else:
                art.set_data((s * self.S, f * knl))
            changed.append(art)
        if autoscale:
            if self.filled:  # At present, relim does not support collection instances.

                self.ax.update_datalim(
                    mpl.transforms.Bbox.union([a.get_datalim(self.ax.transData) for a in self.artists])
                )
            else:
                self.ax.relim()
            self.ax.autoscale()

        return changed
