#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting phase space distributions

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-09-06"


import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

from .base import Xplot, get, style, FixedLimits
from .util import normalized_coordinates, denormalized_coordinates

pairwise = np.c_


class PhaseSpacePlot(Xplot):
    def __init__(
        self,
        particles=None,
        kind=None,
        plot="auto",
        *,
        ax=None,
        mask=None,
        masks=None,
        display_units=None,
        twiss=None,
        color=None,
        cmap="magma_r" or "Blues",
        projections="auto",
        projections_kwargs=None,
        mean=False,
        mean_kwargs=None,
        std=False,
        std_kwargs=None,
        percentiles=None,
        percentile_kwargs=None,
        grid=None,
        titles="auto",
        **subplots_kwargs,
    ):
        """
        A plot for phase space distributions

        :param particles: A dictionary with particle information
        :param kind: Defines the properties to plot.
                     This can be a nested list or a separated string or a mixture of lists and strings where
                     the first list level (or separator ``,``) determines the subplots,
                     and the second list level (or separator ``-``) determines coordinate pairs.
                     In addition, abbreviations for x-y-parameter pairs are supported (e.g. 'x' for 'x-px').
                     For normalized coordinates, use uppercase letters (e.g. 'X' for 'X-Px').

                     Examples:
                      - ``'x'``: single subplot with x-px phase space
                      - ``[['x', 'px']]``: same as above
                      - ``'x,x-y'``: two suplots the first with x-px and the second with x-y phase space
                      - ``[['x', 'px'], ['x', 'y']]``: same as above

        :param plot: Defines the type of plot. Can be 'auto', 'scatter' or 'hist'. Default is 'auto' for which the plot type is chosen automatically based on the number of particles.
        :param ax: A list of axes to plot onto, length must match the number of subplots. If None, a new figure is created.
        :param mask: An index mask to select particles to plot. If None, all particles are plotted.
        :param masks: List of masks for each subplot.
        :param display_units: Dictionary with units for parameters.
        :param twiss: Object holding twiss parameters (alfx, alfy, betx and bety) for calculation of normalized coordinates.
        :param color: Color of the scatter plot. If None, the color is determined by the color cycle.
        :param cmap: Colormap to use for the hist plot.
        :param projections: Add histogrammed projections onto axis. Can be True, False, "x", "y", "auto" or a list of these for each subplot
        :param projections_kwargs: Additional kwargs for histogram projection (step plot)
        :param mean: Whether to indicate mean of distribution with a cross marker. Boolean or list of booleans for each subplot.
        :param mean_kwargs: Additional kwargs for mean cross
        :param std: Whether to indicate standard deviation of distribution with an ellipse. Boolean or list of booleans for each subplot.
        :param std_kwargs: Additional kwargs for std ellipses.
        :param percentiles: List of percentiles (in percent) to indicate in the distribution with ellipses. Can also be a list of lists for each subplot.
        :param percentile_kwargs: Additional kwargs for percentile ellipses.
        :param grid: Tuple (ncol, nrow) for subplot layout. If None, the layout is determined automatically.
        :param titles: List of titles for each subplot or 'auto' to automatically set titles based on plot kind.
        :param subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.


        """
        super().__init__(display_units=display_units)

        # parse kind string by splitting at commas and dashes and replacing abbreviations
        abbreviations = dict(
            x=("x", "px"),
            y=("y", "py"),
            z=("zeta", "delta"),
            X=("X", "Px"),
            Y=("Y", "Py"),
        )
        if kind is None:
            kind = "x,y,z" if twiss is None else "X,Y,z"
        if isinstance(kind, str):
            kind = kind.split(",")
        kind = list(kind)
        for i in range(len(kind)):
            if isinstance(kind[i], str):
                kind[i] = kind[i].split("-")
                # replace abbreviations elements by corresponding tuple
                if len(kind[i]) == 1 and kind[i][0] in abbreviations:
                    kind[i] = abbreviations[kind[i][0]]
                if len(kind[i]) != 2:
                    raise ValueError(
                        "Kind must only contain exactly two coordinates per subplot, "
                        f"but got {kind[i]}"
                    )
        self.kind = kind
        n = len(self.kind)

        # sanitize parameters
        if not hasattr(mean, "__iter__"):
            mean = n * [mean]
        if not hasattr(std, "__iter__"):
            std = n * [std]
        if percentiles is None or not hasattr(percentiles[0], "__iter__"):
            percentiles = n * [percentiles]
        if isinstance(projections, str) or not hasattr(projections, "__iter__"):
            projections = n * [projections]

        if len(mean) != n:
            raise ValueError(f"mean must be a boolean or a list of length {n}")
        if len(std) != n:
            raise ValueError(f"std must be a boolean or a list of length {n}")
        if len(percentiles) != n:
            raise ValueError(f"percentiles list must be flat or of length {n}")
        if grid and (len(grid) != 2 or grid[0] * grid[1] < n):
            raise ValueError(f"grid must be a tuple (n, m) with n*m >= {n}")
        if plot not in ["auto", "scatter", "hist"]:
            raise ValueError("plot must be 'auto', 'scatter' or 'hist'")

        self.plot = plot
        self.percentiles = percentiles
        self.projections = projections
        self.twiss = twiss

        # Create plot axes
        if ax is None:
            if grid:
                ncol, nrow = grid
            else:
                nrow = int(np.sqrt(n))
                while n % nrow != 0:
                    nrow -= 1
                ncol = n // nrow

            kwargs = style(subplots_kwargs, figsize=(4 * ncol, 4 * nrow))
            _, ax = plt.subplots(nrow, ncol, **kwargs)
        if not hasattr(ax, "__iter__"):
            ax = [ax]
        self.ax = ax
        self.fig = self.axflat[0].figure
        if len(self.axflat) < n:
            raise ValueError(f"Need {n} axes but got only {len(self.axflat)}")

        # Create distribution plots
        self.artists_scatter = [None] * n
        self.artists_hexbin = [()] * n
        self.artists_mean = [None] * n
        self.artists_std = [None] * n
        self.artists_percentiles = [()] * n
        self.ax_twin = [{} for i in range(n)]
        self.artists_twin = [{} for i in range(n)]
        self.artists_hamiltonian = [{} for i in range(n)]

        for i, ((a, b), ax) in enumerate(zip(self.kind, self.axflat)):

            # 2D phase space distribution
            ##############################

            self.artists_scatter[i] = ax.scatter([], [], s=4, color=color, lw=0)
            self._hxkw = dict(cmap=cmap, rasterized=True)

            # 2D mean indicator
            if mean[i]:
                kwargs = style(mean_kwargs, color="k", marker="+", ms=8, zorder=100)
                (self.artists_mean[i],) = ax.plot([], [], **kwargs)

            # 2D std ellipses
            if std[i]:
                kwargs = style(std_kwargs, color="k", lw=1, ls="-", zorder=100)
                self.artists_std[i] = Ellipse([0, 0], 0, 0, fill=False, **kwargs)
                ax.add_artist(self.artists_std[i])

            # 2D percentile ellipses
            if percentiles[i]:
                self.artists_percentiles[i] = []
                for j, _ in enumerate(self.percentiles[i]):
                    kwargs = style(
                        percentile_kwargs,
                        color="k",
                        lw=1,
                        ls=(0, [5, 5] + [1, 5] * j),
                        zorder=100,
                    )
                    artist = Ellipse([0, 0], 0, 0, fill=False, **kwargs)
                    ax.add_artist(artist)
                    self.artists_percentiles[i].append(artist)

            # Axes title and labels
            if titles == "auto":
                title = self.title_for(a, b)
            elif titles is None:
                title = None
            else:
                title = titles[i]
            ax.set(title=title, xlabel=self.label_for(a), ylabel=self.label_for(b))
            ax.grid(alpha=0.5)

            # 1D histogram projections
            ###########################

            if self.projections[i]:
                kwargs = style(projections_kwargs, color="k", alpha=0.3, lw=1)
                for xy, yx in zip("xy", "yx"):
                    if self.projections[i] != yx:
                        # Create twin xy axis and artists
                        twin = ax.twinx() if xy == "x" else ax.twiny()
                        twin.set(**{f"{yx}ticks": [], f"{yx}lim": (0, 0.2)})
                        self.ax_twin[i][xy] = twin
                        (hist,) = twin.step([], [], **kwargs)
                        self.artists_twin[i][xy] = hist

        # set data
        if particles is not None:
            self.update(particles, mask=mask, masks=masks, autoscale=True)

    def update(self, particles, *, mask=None, masks=None, autoscale=False):
        """
        Update the data this plot shows

        :param particles: A dictionary with particle information
        :param mask: An index mask to select particles to plot. If None, all particles are plotted.
        :param masks: List of masks for each subplot.
        :param autoscale: Whether or not to perform autoscaling on all axes.
        :return: changed artists.
        """
        if masks is None:
            masks = [mask] * len(self.kind)
        elif mask is not None:
            raise ValueError("Only one of mask and masks can be set.")
        if len(masks) != len(self.kind):
            raise ValueError(f"masks must be a list of length {len(self.kind)}")

        for i, ((a, b), ax) in enumerate(zip(self.kind, self.axflat)):
            ax.autoscale(autoscale)
            # coordinates
            x = self.factor_for(a) * self._masked(particles, a, masks[i])
            y = self.factor_for(b) * self._masked(particles, b, masks[i])

            # statistics
            XY = np.array((x, y))
            XY0 = np.mean(XY, axis=1)
            UV = XY - XY0[:, np.newaxis]  # centered coordinates
            evals, evecs = np.linalg.eig(np.cov(UV))  # eigenvalues and -vectors

            # 2D phase space distribution
            ##############################

            plot = self.plot
            if plot == "auto":
                plot = "scatter" if len(x) <= 1000 else "hist"
            # scatter plot
            self.artists_scatter[i].set_visible(plot == "scatter")
            if plot == "scatter":
                self.artists_scatter[i].set_offsets(pairwise[x, y])
            # hexbin plot
            # remove old hexbin and create a new one (no update method)
            for artist in self.artists_hexbin[i]:
                artist.remove()
            self.artists_hexbin[i] = []
            if plot == "hist":
                # twice to mitigate https://stackoverflow.com/q/17354095
                hexbin_bg = ax.hexbin(x, y, mincnt=1, **self._hxkw)
                hexbin_fg = ax.hexbin(x, y, mincnt=1, edgecolors="none", **self._hxkw)
                self.artists_hexbin[i] = [hexbin_bg, hexbin_fg]

            # 2D mean indicator
            if self.artists_mean[i]:
                self.artists_mean[i].set_data(XY0)

            # 2D std indicator
            if self.artists_std[i]:
                w, h = 2 * np.sqrt(evals)
                self.artists_std[i].set(
                    center=XY0,
                    width=w,
                    height=h,
                    angle=np.degrees(np.arctan2(*evecs[1])),
                )

            # 2D percentile indicator
            if self.artists_percentiles[i]:
                # normalize distribution using eigenvalues and -vectors
                NN = np.dot(evecs.T, UV) / np.sqrt(evals)[:, np.newaxis]
                for j, p in enumerate(self.percentiles[i]):
                    # percentile in normalized distribution
                    e = np.percentile(np.sum(NN**2, axis=0), p) ** 0.5
                    w, h = 2 * e * np.sqrt(evals)
                    self.artists_percentiles[i][j].set(
                        center=XY0,
                        width=w,
                        height=h,
                        angle=np.degrees(np.arctan2(*evecs[1])),
                    )

            # 1D histogram projections
            ###########################

            project = self.projections[i]
            if project == "auto":
                project = len(x) >= 200

            for xy, v in zip("xy", (x, y)):
                if xy in self.artists_twin[i]:
                    hist = self.artists_twin[i][xy]
                    hist.set_visible(project)
                    if project:
                        # 1D histogram
                        counts, edges = np.histogram(v, bins=101)
                        counts = counts / len(v)
                        steps = (
                            np.append(edges, edges[-1]),
                            np.concatenate(([0], counts, [0])),
                        )
                        hist.set_data(steps if xy == "x" else steps[::-1])

            if autoscale:
                # ax.relim()  # At present, relim does not support collection instances.
                if plot == "scatter":
                    artists = [self.artists_scatter[i]]
                elif plot == "hist":
                    artists = self.artists_hexbin[i]
                else:
                    artists = []
                ax.update_datalim(
                    mpl.transforms.Bbox.union(
                        [a.get_datalim(ax.transData) for a in artists]
                    )
                )
                ax.autoscale()

    @property
    def axflat(self):
        return np.array(self.ax).flatten()

    def _masked(self, particles, prop, mask=None):
        """Get masked particle property"""

        if prop in ("X", "Px", "Y", "Py"):
            # normalized coordinates
            xy = prop.lower()[-1]
            coords = [get(particles, p) for p in (xy, "p" + xy)]
            delta = get(particles, "delta")
            X, Px = normalized_coordinates(*coords, self.twiss, xy, delta=delta)
            v = X if prop.lower() == xy else Px

        else:
            v = get(particles, prop)

        if mask is not None:
            v = v[mask]

        return np.array(v).flatten()

    def title_for(self, a, b):
        """
        Plot title for a given pair (a,b) of properties
        """
        titles = {  # Order of a,b does not matter
            "x-px": "Horizontal phase space",
            "y-py": "Vertical phase space",
            "zeta-delta": "Longitudinal phase space",
            "X-Px": "Normalized horizontal phase space",
            "Y-Py": "Normalized vertical phase space",
            "x-y": "Transverse profile",
        }
        return titles.get(f"{a}-{b}", titles.get(f"{b}-{a}", f"{a}-{b} phase space"))

    def axline(self, kind, val, also_on_normalized=True, subplots="all", **kwargs):
        """Plot a vertical or horizontal line for a given coordinate

        Args:
            kind (str): Phase space coordinate
            val (float): Value of phase space coordinate
            also_on_normalized (bool, optional): If true, also plot line for related (de-)normalized phase space coordinate
            subplots (list of int): Subplots to plot line onto. Defaults to all with matching coordinates.
            kwargs: Arguments for axvline or axhline

        """

        kwargs = style(kwargs, c="k")

        for i, (ab, ax) in enumerate(zip(self.kind, self.axflat)):
            if subplots != "all" and i not in subplots:
                continue

            for a, line in zip(ab, (ax.axvline, ax.axhline)):
                if a == kind:
                    # same axis found, draw line
                    line(val * self.factor_for(kind), **kwargs)

                elif self.twiss is not None and kind.lower() == a.lower():
                    # related (de-)normalized axis found, transform and draw line
                    xy = kind.lower()[-1]
                    if kind in "x,y":  # x,y -> X,Y
                        v = normalized_coordinates(val, 0, self.twiss, xy)[0]
                    elif kind in "X,Y":  # X,Y -> x,y
                        v = denormalized_coordinates(val, 0, self.twiss, xy)[0]
                    elif kind in "px,py":  # px,py -> Px,Py
                        v = normalized_coordinates(0, val, self.twiss, xy)[1]
                    elif kind in "Px,Py":  # Px,Py -> px,py
                        v = denormalized_coordinates(0, val, self.twiss, xy)[1]
                    else:
                        continue
                    line(v * self.factor_for(a), **kwargs)

    def plot_hamiltonian_kobayashi(
        self,
        subplot,
        S,
        mu,
        *,
        q=None,
        extend=1,
        autoscale=True,
        separatrix=True,
        separatrix_kwargs=None,
        equipotentials=True,
        equipotentials_kwargs=None,
    ):
        """Plot separatrix and equipotential lines of kobayashi hamiltonian

        Args:
            subplot (int): Index of subplot
            S (float): Virtual sextupole strength in m^(-1/2)
            mu (float): Virtual sextupole phase in rad/2pi
            q (float, optional): Tune. Defaults to tune from twiss.
            extend (float, optional): Extend of separatrix and equipotential lines. If > 1 they are drawn beyond the the stable region.
            autoscale (bool, optional): Whether to autoscale axis or not
            separatrix (bool, optional): Plot separatrix. Defaults to True.
            separatrix_kwargs (dict, optional): Keyword arguments for separatrix line plot.
            equipotentials (bool, optional): Plot equipotential lines. Defaults to True.
            equipotentials_kwargs (dict, optional): Keyword arguments for equipotential line contour plot.
        """

        # TODO: provide option to plot for different particle delta (dispersion shifts and chromaticity shrinks the triangle)

        ax = self.axflat[subplot]
        a, b = self.kind[subplot]
        config = {
            "x-px": ("x", False),
            "y-py": ("y", False),
            "X-Px": ("x", True),
            "Y-Py": ("y", True),
        }
        if f"{a}-{b}" in config:
            xy, normalized = config[f"{a}-{b}"]
            swap = False
        elif f"{b}-{a}" in config:
            xy, normalized = config[f"{b}-{a}"]
            swap = True
        else:
            raise TypeError(f"Invalid plot kind {a}-{b} for kobayashi hamiltonian")

        # twiss parameters at plot location
        if self.twiss is None:
            raise ValueError("No twiss parameters provided during plot creation")
        alfx, betx, mux, x, px = [
            get(self.twiss, pre + xy).squeeze() for pre in ["alf", "bet", "mu", "", "p"]
        ]
        if q is None:
            q = get(self.twiss, "q" + xy).squeeze()

        # distance d to nearby 3rd order resonance r
        r = round(3 * q) / 3
        d = q - r
        # size of stable triangle
        h = 4 * np.pi * d / S
        # phase advance from virtual sextupole to here
        dmu = 2 * np.pi * (mux - mu)
        rotation = np.array([[+np.cos(dmu), np.sin(dmu)], [-np.sin(dmu), np.cos(dmu)]])

        def transform(XY):
            XY = np.tensordot(rotation, XY, 1)
            # match plot settings
            if not normalized:
                XY = np.array(denormalized_coordinates(*XY, self.twiss, xy))
            if swap:
                XY = XY[::-1]
            XY[0] *= self.factor_for(a)
            XY[1] *= self.factor_for(b)
            return XY

        lim = ax.dataLim
        with FixedLimits(ax):

            # plot separatrix
            if separatrix:
                kwarg = style(separatrix_kwargs, color="r", ls="--", label="Separatrix")

                t = extend
                for X, Y in (
                    [(-2, -2), (-2 * t, 2 * t)],
                    [(1 - 3 * t, 1 + 3 * t), (1 + t, 1 - t)],
                    [(1 - 3 * t, 1 + 3 * t), (-1 - t, -1 + t)],
                ):
                    X = h * np.array(X) / 2
                    Y = h * 3**0.5 * np.array(Y) / 2
                    ax.plot(*transform((X, Y)), **kwarg)

            # plot equipotential lines
            if equipotentials:
                kwargs = style(
                    equipotentials_kwargs, colors="lightgray", linewidths=1, alpha=0.5
                )

                # X = h * np.linspace(-5, 5, 500)
                X = extend * h * np.linspace(-1, 2, 500)
                Y = extend * h * 3**0.5 * np.linspace(-1, 1, 500)
                X, Y = np.meshgrid(X, Y)
                H = (3 * h * (X**2 + Y**2) + 3 * X * Y**2 - X**3) / h**3 / 4
                levels = np.linspace(0, extend, int(10 * extend)) ** 2

                ax.contour(*transform((X, Y)), H, levels=levels, **kwargs)

                ax.grid(False)

        # autoscale to separatrix
        if autoscale:
            edges = np.array(
                [(2 * h, -h, -h, 2 * h), (0, -(3**0.5) * h, 3**0.5 * h, 0)]
            )
            x, y = transform(edges)
            ax.dataLim = lim.union(
                [mpl.transforms.Bbox([[min(x), min(y)], [max(x), max(y)]])]
            )
            ax.autoscale()
