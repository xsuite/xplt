#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting phase space distributions

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-09-06"


import matplotlib as mpl
import numpy as np
from matplotlib.patches import Ellipse

from .base import XParticlePlot, get, defaults, FixedLimits
from .util import normalized_coordinates, denormalized_coordinates

pairwise = np.c_


class PhaseSpacePlot(XParticlePlot):
    def __init__(
        self,
        particles=None,
        kind=None,
        plot="auto",
        *,
        scatter_kwargs=None,
        hist_kwargs=None,
        ax=None,
        mask=None,
        masks=None,
        display_units=None,
        data_units=None,
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
        wrap_zeta=None,
        beta=None,
        frev=None,
        circumference=None,
        animated=False,
        **subplots_kwargs,
    ):
        """
        A plot for phase space distributions

        Args:
            particles: A dictionary with particle information
            kind: Defines the properties to plot.
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

            plot: Defines the type of plot. Can be 'auto', 'scatter' or 'hist'. Default is 'auto' for which the plot type is chosen automatically based on the number of particles.
            scatter_kwargs: Additional kwargs for scatter plot
            hist_kwargs: Additional kwargs for hexbin histogram plot
            ax: A list of axes to plot onto, length must match the number of subplots. If None, a new figure is created.
            mask: An index mask to select particles to plot. If None, all particles are plotted.
            masks: List of masks for each subplot.
            display_units (dict, optional): Units to display the data in. If None, the units are determined from the data.
            data_units (dict, optional): Units of the data. If None, the units are determined from the data.
            twiss (dict, optional): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
            color (str or list of str): Properties defining the color of points for the scatter plot(s). Implies plot='scatter'. Pass a list of properties to use different values for each subplot
            cmap: Colormap to use for the hist plot.
            projections: Add histogrammed projections onto axis. Can be True, False, "x", "y", "auto" or a list of these for each subplot
            projections_kwargs: Additional kwargs for histogram projection (step plot)
            mean: Whether to indicate mean of distribution with a cross marker. Boolean or list of booleans for each subplot.
            mean_kwargs: Additional kwargs for mean cross
            std: Whether to indicate standard deviation of distribution with an ellipse. Boolean or list of booleans for each subplot.
            std_kwargs: Additional kwargs for std ellipses.
            percentiles: List of percentiles (in percent) to indicate in the distribution with ellipses. Can also be a list of lists for each subplot.
            percentile_kwargs: Additional kwargs for percentile ellipses.
            grid: Tuple (ncol, nrow) for subplot layout. If None, the layout is determined automatically.
            titles: List of titles for each subplot or 'auto' to automatically set titles based on plot kind.
            wrap_zeta: If set, wrap the zeta-coordinate plotted at the machine circumference. Either pass the circumference directly or set this to True to use the circumference from twiss.
            beta (float, optional): Relativistic beta of particles. Defaults to particles.beta0.
            frev (float, optional): Revolution frequency of circular line for calculation of particle time.
            circumference (float, optional): Path length of circular line if frev is not given.
            animated: If True, improve plotting performance for creating an animation.
            subplots_kwargs: Keyword arguments passed to matplotlib.pyplot.subplots command when a new figure is created.


        """
        super().__init__(
            data_units=data_units,
            display_units=display_units,
            twiss=twiss,
            beta=beta,
            frev=frev,
            circumference=circumference,
            wrap_zeta=wrap_zeta,
        )

        # parse kind string by splitting at commas and dashes and replacing abbreviations
        abbreviations = dict(
            x="x-px",
            y="y-py",
            z="zeta-delta",
            X="X-Px",
            Y="Y-Py",
        )
        if kind is None:
            kind = "x,y,z" if twiss is None else "X,Y,z"
        self.kind = self._parse_nested_list_string(
            self._parse_nested_list_string(kind, ",", abbreviations),
            ",-",  # no abbreviations on second level!
        )
        if np.any([len(k) != 2 for k in self.kind]):
            raise ValueError("Kind must only contain exactly two coordinates per subplot")
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
        if color is not None:
            if plot not in ["auto", "scatter"]:
                raise ValueError(f"Setting color requires plot='scatter', but plot was {plot}")
            plot = "scatter"
            if isinstance(color, str):
                color = np.resize(color.split(","), n)
        else:
            color = n * [None]

        self.plot = plot
        self.color = color
        self.percentiles = percentiles
        self.projections = projections
        self.twiss = twiss
        self.wrap_zeta = wrap_zeta

        # Create plot axes

        # initialize figure with n subplots
        if grid:
            ncol, nrow = grid
        else:
            nrow = int(np.sqrt(n))
            while n % nrow != 0:
                nrow -= 1
            ncol = n // nrow
        nntwins = [0 for _ in self.kind]
        kwargs = defaults(subplots_kwargs, figsize=(4 * ncol, 4 * nrow))
        self._init_axes(ax, nrow, ncol, nntwins, grid, **kwargs)
        if len(self.axflat) < n:
            raise ValueError(f"Need {n} axes but got only {len(self.axflat)}")

        # Create distribution plots
        self.artists_scatter = [None] * n
        self.artists_hexbin = [()] * n
        self.artists_mean = [None] * n
        self.artists_std = [None] * n
        self.artists_percentiles = [()] * n
        self.ax_twin = [{} for _ in range(n)]
        self.artists_twin = [{} for _ in range(n)]
        self.artists_hamiltonian = [{} for _ in range(n)]

        for i, ((a, b), c, ax) in enumerate(zip(self.kind, self.color, self.axflat)):

            # 2D phase space distribution
            ##############################

            # scatter plot
            kwargs = defaults(scatter_kwargs, s=4, cmap=cmap, lw=0, animated=animated)
            scatter_cmap = kwargs.pop("cmap")  # bypass UserWarning: ignored
            self.artists_scatter[i] = ax.scatter([], [], **kwargs)
            self.artists_scatter[i].cmap = mpl.colormaps[scatter_cmap]
            if c is not None and (np.any(self.color != c) or i == n - 1):
                self.fig.colorbar(self.artists_scatter[i], ax=ax, label=self.label_for(c))

            # hexbin histogram
            self._hxkw = defaults(hist_kwargs, cmap=cmap, rasterized=True, animated=animated)

            # 2D mean indicator
            if mean[i]:
                kwargs = defaults(mean_kwargs, color="k", marker="+", ms=8, zorder=100)
                (self.artists_mean[i],) = ax.plot([], [], **kwargs, animated=animated)

            # 2D std ellipses
            if std[i]:
                kwargs = defaults(std_kwargs, color="k", lw=1, ls="-", zorder=100)
                self.artists_std[i] = Ellipse(
                    [0, 0], 0, 0, fill=False, **kwargs, animated=animated
                )
                ax.add_artist(self.artists_std[i])

            # 2D percentile ellipses
            if percentiles[i]:
                self.artists_percentiles[i] = []
                for j, _ in enumerate(self.percentiles[i]):
                    kwargs = defaults(
                        percentile_kwargs,
                        color="k",
                        lw=1,
                        ls=(0, [5, 5] + [1, 5] * j),
                        zorder=100,
                    )
                    artist = Ellipse([0, 0], 0, 0, fill=False, **kwargs, animated=animated)
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
                kwargs = defaults(projections_kwargs, color="k", alpha=0.3, lw=1)
                for xy, yx in zip("xy", "yx"):
                    if self.projections[i] != yx:
                        # Create twin xy axis and artists
                        twin = ax.twinx() if xy == "x" else ax.twiny()
                        twin.set(**{f"{yx}ticks": [], f"{yx}lim": (0, 0.2)})
                        self.ax_twin[i][xy] = twin
                        (hist,) = twin.step([], [], **kwargs, animated=animated)
                        self.artists_twin[i][xy] = hist

        # set data
        if particles is not None:
            self.update(particles, mask=mask, masks=masks, autoscale=True)

    def update(self, particles, *, mask=None, masks=None, autoscale=False):
        """
        Update the data this plot shows

        Args:
            particles: A dictionary with particle information
            mask: An index mask to select particles to plot. If None, all particles are plotted.
            masks: List of masks for each subplot.
            autoscale: Whether or not to perform autoscaling on all axes.

        Returns:
            List of changed artists.
        """
        if masks is None:
            masks = [mask] * len(self.kind)
        elif mask is not None:
            raise ValueError("Only one of mask and masks can be set.")
        if len(masks) != len(self.kind):
            raise ValueError(f"masks must be a list of length {len(self.kind)}")

        changed_artists = []

        for i, ((a, b), c, ax) in enumerate(zip(self.kind, self.color, self.axflat)):
            ax.autoscale(autoscale)

            # coordinates
            x = self.factor_for(a) * self._get_masked(particles, a, masks[i])
            y = self.factor_for(b) * self._get_masked(particles, b, masks[i])

            # statistics
            XY = np.array((x, y))
            XY0 = np.mean(XY, axis=1)
            UV = XY - XY0[:, np.newaxis]  # centered coordinates
            if self.artists_std[i] or self.artists_percentiles[i]:
                evals, evecs = np.linalg.eig(np.cov(UV))  # eigenvalues and -vectors

            # 2D phase space distribution
            ##############################

            plot = self.plot
            if plot == "auto":
                plot = "scatter" if len(x) <= 1000 else "hist"

            # scatter plot
            scatter = self.artists_scatter[i]
            if plot == "scatter":
                scatter.set_visible(True)
                scatter.set_offsets(pairwise[x, y])
                if c is not None:
                    v = self.factor_for(c) * self._get_masked(particles, c, masks[i])
                    if autoscale:
                        # scatter.set_clim(np.min(v), np.max(v)) # does not always work (bug?)
                        scatter.norm = mpl.colors.Normalize(np.min(v), np.max(v))
                    cm = mpl.cm.ScalarMappable(norm=scatter.norm, cmap=scatter.cmap)
                    scatter.set_facecolor(cm.to_rgba(v))

                changed_artists.append(scatter)
            elif scatter.get_visible():
                scatter.set_visible(False)
                changed_artists.append(scatter)

            # hexbin plot
            # remove old hexbin and create a new one (no update method exists)
            for artist in self.artists_hexbin[i]:
                changed_artists.append(artist)
                artist.remove()
            self.artists_hexbin[i] = []
            if plot == "hist":
                # twice to mitigate https://stackoverflow.com/q/17354095
                hexbin_bg = ax.hexbin(x, y, mincnt=1, **self._hxkw)
                hexbin_fg = ax.hexbin(x, y, mincnt=1, edgecolors="none", **self._hxkw)
                self.artists_hexbin[i] = [hexbin_bg, hexbin_fg]
                changed_artists.extend(self.artists_hexbin[i])

            # 2D mean indicator
            if self.artists_mean[i]:
                self.artists_mean[i].set_data(XY0)
                changed_artists.append(self.artists_mean[i])

            # 2D std indicator
            if self.artists_std[i]:
                w, h = 2 * np.sqrt(evals)
                self.artists_std[i].set(
                    center=XY0,
                    width=w,
                    height=h,
                    angle=np.degrees(np.arctan2(*evecs[1])),
                )
                changed_artists.append(self.artists_std[i])

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
                    changed_artists.append(self.artists_percentiles[i][j])

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
                        # counts = np.bincount((101 * (v - np.min(v)) / (np.max(v) - np.min(v))).astype(int))[:101] / len(v)
                        # edges = np.linspace(np.min(v), np.max(v), 102)

                        steps = (
                            np.append(edges, edges[-1]),
                            np.concatenate(([0], counts, [0])),
                        )
                        hist.set_data(steps if xy == "x" else steps[::-1])
                        changed_artists.append(hist)

            if autoscale:
                # ax.relim()  # At present, relim does not support collection instances.
                if plot == "scatter":
                    artists = [self.artists_scatter[i]]
                elif plot == "hist":
                    artists = self.artists_hexbin[i]
                else:
                    artists = []
                ax.update_datalim(
                    mpl.transforms.Bbox.union([a.get_datalim(ax.transData) for a in artists])
                )
                ax.autoscale(True)

        return changed_artists

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

        kwargs = defaults(kwargs, color="k")

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
        subplots,
        S,
        mu,
        *,
        delta=0,
        extend=1,
        autoscale=1.1,
        separatrix=True,
        separatrix_kwargs=None,
        equipotentials=True,
        equipotentials_kwargs=None,
    ):
        """Plot separatrix and equipotential lines of kobayashi hamiltonian

        Args:
            subplots (int or list of int): Index of subplot(s)
            S (float): Virtual sextupole strength in m^(-1/2)
            mu (float): Virtual sextupole phase in rad/2pi
            delta (float, optional): Momentum offset.
            extend (float, optional): Extend of separatrix and equipotential lines. If > 1 they are drawn beyond the the stable region.
            autoscale (bool or float, optional): Whether to autoscale axis or not (bool), or the extend to consider for autoscaling
            separatrix (bool, optional): Plot separatrix. Defaults to True.
            separatrix_kwargs (dict, optional): Keyword arguments for separatrix line plot.
            equipotentials (bool, optional): Plot equipotential lines. Defaults to True.
            equipotentials_kwargs (dict, optional): Keyword arguments for equipotential line contour plot.
        """
        if not hasattr(subplots, "__iter__"):
            subplots = [subplots]

        for subplot in subplots:
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
            q = get(self.twiss, "q" + xy).squeeze()
            q = q + delta * get(self.twiss, "dq" + xy).squeeze()

            # distance d to nearby 3rd order resonance r
            r = round(3 * q) / 3
            d = q - r
            # size of stable triangle
            h = 4 * np.pi * d / S
            # phase advance from virtual sextupole to here
            dmu = 2 * np.pi * (mux - mu)
            rotation = np.array([[+np.cos(dmu), np.sin(dmu)], [-np.sin(dmu), np.cos(dmu)]])

            def transform(XY):
                """Transform normalized phase space coordinates into coordinates of plot"""
                # rotate so as to account for phase advance
                XY = np.tensordot(rotation, XY, 1)
                # match plot settings
                if not normalized:
                    XY = np.array(denormalized_coordinates(*XY, self.twiss, xy, delta))
                if swap:
                    XY = XY[::-1]
                XY[0] *= self.factor_for(a)
                XY[1] *= self.factor_for(b)
                return XY

            lim = ax.dataLim
            with FixedLimits(ax):

                # plot separatrix
                if separatrix:
                    kwarg = defaults(separatrix_kwargs, color="r", ls="--", label="Separatrix")

                    t = extend
                    for X, Y in (  # triangle edges
                        [(-2, -2), (-2 * t, 2 * t)],
                        [(1 - 3 * t, 1 + 3 * t), (1 + t, 1 - t)],
                        [(1 - 3 * t, 1 + 3 * t), (-1 - t, -1 + t)],
                    ):
                        X = h * np.array(X) / 2
                        Y = h * 3**0.5 * np.array(Y) / 2
                        ax.plot(*transform((X, Y)), **kwarg)
                        kwarg.pop("label", None)

                # plot equipotential lines
                if equipotentials:
                    kwargs = defaults(
                        equipotentials_kwargs,
                        colors="lightgray",
                        linewidths=1,
                        alpha=0.5,
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
                t = 1 if autoscale is True else autoscale
                X = np.array((-2, -2, 1 - 3 * t, 1 + 3 * t, 1 - 3 * t, 1 + 3 * t)) / 2
                Y = np.array((-2 * t, 2 * t, 1 + t, 1 - t, -1 - t, -1 + t)) * 3**0.5 / 2
                x, y = transform((h * X, h * Y))
                ax.dataLim = lim.union(
                    [mpl.transforms.Bbox([[min(x), min(y)], [max(x), max(y)]])]
                )
                ax.autoscale()
