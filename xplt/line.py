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

from .base import XPlot


def iter_elements(line):
    """Iterate over elements in line

    Yields (name, element, s_from, s_to) with

    """
    el_s0 = line.get_s_elements("upstream")
    el_s1 = line.get_s_elements("downstream")
    for name, el, s0, s1 in zip(line.element_names, line.elements, el_s0, el_s1):
        if s0 == s1:  # thin lense located at element center
            if hasattr(el, "length"):
                s0, s1 = (s0 + s1 - el.length) / 2, (s0 + s1 + el.length) / 2
        yield name, el, s0, s1


class KnlPlot(XPlot):
    def __init__(
        self,
        line=None,
        *,
        knl=None,
        ax=None,
        filled=True,
        display_units=None,
        resolution=1000,
        line_length=None,
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
        :param line_length: Length of line (only required if line is None).
        :param subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.

        """
        super().__init__(display_units=display_units)

        if knl is None:
            if line is None:
                raise ValueError("Either line or knl parameter must not be None")
            knl = range(max([e.order for e in line.elements if hasattr(e, "order")]) + 1)
        self.knl = knl
        if line is None and line_length is None:
            raise ValueError("Either line or line_length parameter must not be None")
        self.S = np.linspace(0, line_length or line.get_length(), resolution)
        self.filled = filled

        # Create plot
        if ax is None:
            _, ax = plt.subplots(**subplots_kwargs)
        self.fig, self.ax = ax.figure, ax
        self.ax.set(
            xlabel=self.label_for("s"),
            ylabel="$k_nl$",
            xlim=(self.S.min(), self.S.max()),
        )
        self.ax.grid()

        # create plot elements
        self.artists = []
        for n in self.knl:
            if self.filled:
                artist = self.ax.fill_between(
                    self.S,
                    np.zeros_like(self.S),
                    alpha=0.5,
                    label=self.label_for(f"k{n}l"),
                    zorder=3,
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
        Smax = line.get_length()
        for name, el, s0, s1 in iter_elements(line):
            if hasattr(el, "knl"):
                for i, n in enumerate(self.knl):
                    if n <= el.order:
                        if 0 <= s0 <= Smax:
                            mask = (self.S >= s0) & (self.S < s1)
                        else:
                            # handle wrap around
                            mask = (self.S >= s0 % Smax) | (self.S < s1 % Smax)
                        KNL[i, mask] += el.knl[n]

        # plot
        s = self.factor_for("s")
        changed = []
        for n, art, knl in zip(self.knl, self.artists, KNL):
            f = self.factor_for(f"k{n}l")
            if self.filled:
                art.get_paths()[0].vertices[1 : 1 + knl.size, 1] = f * knl
            else:
                art.set_data((s * self.S, f * knl))
            changed.append(art)
        if autoscale:
            if self.filled:  # At present, relim does not support collection instances.
                self.ax.update_datalim(
                    mpl.transforms.Bbox.union(
                        [a.get_datalim(self.ax.transData) for a in self.artists]
                    )
                )
            else:
                self.ax.relim()
            self.ax.autoscale()

        return changed
