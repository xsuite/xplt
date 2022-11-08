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

from . import util


class TwissPlot:
    def __init__(
        self,
        twiss=None,
        kind="bet-dx,x+y",
        *,
        ax=None,
    ):
        """
        A plot for twiss parameters and closed orbit

        :param kind: Defines the properties to plot.
                     This can be a nested list or a separated string or a mixture of lists and strings where
                     the first list level (or separator ',') determines the subplots,
                     the second list level (or separator '-') determines any twinx-axes,
                     and the third list level (or separator '+') determines plots.
                     In addition, abbreviations for x-y-parameter pairs are supported (e.g. 'bet' for 'betx+bety').
                     Examples:
                         - 'bet-dx': single subplot with 'betx' and 'bety' on the left and 'dx' on the right axis
                         - [[['betx', 'bety'], ['dx']]]: same as above
                         - 'betx+alf,mu': two suplots the first with 'betx', 'alfx' and 'alfy' and the second with 'mux' and 'muy'
                         - [[['betx', 'alfx', 'alfy']], [['mux', 'muy']]]: same as above

        :param twiss: Dictionary with twiss information
        :param ax: A list of axes to plot onto, length must match the number of subplots. If None, a new figure is created.
                   If required, twinx-axes will be added automatically.

        """

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

        # display units (prefix notation supported)
        self._display_units = dict(s="m", x="mm", y="mm", p="mrad", bet="m", d="m")

        # Create plot axes
        if ax is None:
            _, ax = plt.subplots(len(self.kind), sharex="col")
        if not hasattr(ax, "__iter__"):
            ax = [ax]
        self.fig, self.ax = ax[0].figure, ax
        self.ax_twin = [[] for a in self.ax]  # twinx-axes are created below
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
                    a.grid()
                else:  # create twinx-axes if required
                    a = self.ax[i].twinx()
                    a._get_lines.prop_cycler = self.ax[i]._get_lines.prop_cycler
                    self.ax_twin[i].append(a)
                a.set(ylabel=self.label_for(*pp))

                # create artists for traces
                self.artists[i].append([])
                for k, p in enumerate(pp):
                    (artist,) = a.plot([], [])
                    self.artists[i][j].append(artist)
                    legend[0].append(artist)
                    legend[1].append(self.label_for(p, unit=False))

            if legend[0]:
                a.legend(*legend)

        # set data
        if twiss:
            self.update(twiss, autoscale=True)

    def update(self, twiss, autoscale=False):
        """
        Update the twiss data this plot shows

        :param twiss: Dictionary with twiss information
        :param autoscale: Whether or not to perform autoscaling on all axes
        :return: changed artists
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

        return changed

    def factor_for(self, p):
        """Return factor to convert parameter into display unit"""
        return util.factor_for(p, self.display_unit_for(p))

    def display_unit_for(self, p):
        """Return display unit for parameter"""
        prefix = p[:-1] if len(p) > 1 and p[-1] in "xy" else p
        return self._display_units.get(p, self._display_units.get(prefix, util.data_unit(p)))

    def label_for(self, *pp, unit=True):
        """
        Return label for list of twiss parameters, joining where possible

        :param pp: Parameter names
        :param unit: Wheather to include unit
        """
        texmap = {
            "alf": "\\alpha",
            "bet": "\\beta",
            "gam": "\\gamma",
            "mu": "\\mu",
            "d": "D",
            "px": "x'",
        }

        def split(p):
            if p[-1] in "xy":
                return p[:-1], p[-1]
            return p, ""

        # split pre- and suffix
        prefix, _ = split(pp[0])
        display_unit = self.display_unit_for(pp[0])
        suffix = []
        for p in pp:
            pre, suf = split(p)
            suffix.append(suf)
            if pre != prefix or self.display_unit_for(p) != display_unit:
                # no common prefix or different units, treat separately!
                return " ,  ".join([self.label_for(p) for p in pp])
        suffix = ",".join(suffix)

        # build label
        label = "$"
        if prefix:
            label += texmap.get(prefix, prefix)
            if suffix:
                label += "_{" + suffix + "}"
        else:
            label += suffix
        label += "$"
        if unit and display_unit and display_unit != 1:
            label += " / " + str(display_unit)
        return label
