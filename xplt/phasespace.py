#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting phase space distributions

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-09-06"


from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .util import *
from .base import XPlot, XManifoldPlot, AngleLocator, RadiansFormatter
from .particles import ParticlePlotMixin

pairwise = np.c_


PUBLIC_SECTION_BEGIN()


class PhaseSpacePlot(XPlot, ParticlePlotMixin):
    """A plot for phase space distributions"""

    def __init__(
        self,
        particles=None,
        kind=None,
        plot="auto",
        *,
        scatter_kwargs=None,
        hist_kwargs=None,
        mask=None,
        masks=None,
        color=None,
        cmap="magma_r" or "Blues",
        cbar_loc="auto",
        projections="auto",
        projections_kwargs=None,
        mean=False,
        mean_kwargs=None,
        std=False,
        std_kwargs=None,
        percentiles=None,
        percentile_kwargs=None,
        nrows=None,
        ncols=None,
        titles="auto",
        animated=False,
        twiss=None,
        **kwargs,
    ):
        """

        Args:
            particles (Any): A dictionary with particle information
            kind (str | list): Defines the properties to plot.
                This can be a nested list or a separated string or a mixture of lists and strings where
                the first list level (or separator ``,``) determines the subplots,
                and the second list level (or separator ``-``) determines coordinate pairs.
                In addition, abbreviations for x-y-parameter pairs are supported (e.g. 'x' for 'x-px').
                For normalized coordinates, use uppercase letters (e.g. 'X' for 'X-Px').
            plot (str): Defines the type of plot. Can be ``"auto"``, ``"scatter"`` or ``"hist"``. Default is ``"auto"``
                for which the plot type is chosen automatically based on the number of particles.
            scatter_kwargs (dict): Additional kwargs for scatter plot, see :meth:`matplotlib.axes.Axes.scatter`.
            hist_kwargs (dist): Additional kwargs for 2D histogram plot, see :meth:`matplotlib.axes.Axes.hexbin`.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            masks (list[mask]): List of masks for each subplot.
            color (str | list[str]): Properties defining the color of points for the scatter plot(s). Implies plot='scatter'.
                Pass a list of properties to use different values for each subplot
            cmap (str): Colormap to use for the hist plot.
            cbar_loc (str): Location of the colorbar, such as 'auto', 'right', 'inside upper right', etc.
                Use None to disable colorbar.
            projections (bool | str | list): Add histogrammed projections onto axis. Can be ``True``, ``False``, ``"x"``,
                ``"y"``, ``"auto"`` or a list of these for each subplot.
            projections_kwargs (dict): Additional kwargs for histogram projection, see :meth:`matplotlib.axes.Axes.step`.
            mean (bool | list): Whether to indicate mean of distribution with a cross marker. Boolean or list of booleans for each subplot.
            mean_kwargs (dict): Additional kwargs for marker, see :meth:`matplotlib.axes.Axes.plot`.
            std (bool | list): Whether to indicate standard deviation of distribution with an ellipse. Boolean or list of booleans for each subplot.
            std_kwargs (dict): Additional kwargs for std ellipses, see :class:`matplotlib.patches.Ellipse`.
            percentiles (list): List of percentiles (in percent) to indicate in the distribution with ellipses.
                Can also be a list of lists for each subplot.
            percentile_kwargs (dict): Additional kwargs for percentile ellipses, see :class:`matplotlib.patches.Ellipse`.
            ncols (int | None): Number of columns in subplot layout. If None, the layout is determined automatically.
            nrows (int | None): Number of columns in subplot layout. If None, the layout is determined automatically.
            titles (list[str]): List of titles for each subplot or 'auto' to automatically set titles based on plot kind.
            animated (bool): If True, improve plotting performance for creating an animation.
            twiss (dict | None): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments


        """

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
        self.kind = XManifoldPlot.parse_nested_list_string(
            XManifoldPlot.parse_nested_list_string(kind, ",", abbreviations),
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
            raise ValueError(f"mean must be a boolean or a list of length {n}, but got {mean!r}")
        if len(std) != n:
            raise ValueError(f"std must be a boolean or a list of length {n}, but got {std!r}")
        if len(percentiles) != n:
            raise ValueError(
                f"percentiles list must be flat or of length {n}, but got {percentiles!r}"
            )
        if plot not in ["auto", "scatter", "hist"]:
            raise ValueError(f"plot must be 'auto', 'scatter' or 'hist', but got {plot!r}")
        if color is not None:
            if plot not in ["auto", "scatter"]:
                raise ValueError(f"Setting color requires plot='scatter', but got plot={plot!r}")
            plot = "scatter"
            if isinstance(color, str):
                color = np.resize(color.split(","), n)
            if cbar_loc == "auto":
                cbar_loc = (
                    "right"
                    if len(color) <= 1 or np.all(color == color[0])
                    else "inside upper right"
                )
        else:
            color = n * [None]

        self.plot = plot
        self.color = color
        self.percentiles = percentiles
        self.projections = projections

        # Create plot axes

        # initialize figure with n subplots
        if ncols is None:
            if nrows is None:
                nrows = int(np.sqrt(n))
                while n % nrows != 0:
                    nrows -= 1
            ncols = n // nrows
        if ncols * nrows != n:
            raise ValueError(
                f"Layout with {ncols} columns and {nrows} rows conflicts with {n} plots!"
            )
        kwargs = defaults(kwargs, figsize=(4 * ncols, 4 * nrows))

        kwargs = self._init_particle_mixin(twiss=twiss, **kwargs)
        super().__init__(nrows=nrows, ncols=ncols, **kwargs)

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
            # we handle autoscaling manually
            ax.set_autoscale_on(False)

            # 2D phase space distribution
            ##############################

            # scatter plot
            kwargs = defaults_for(
                "scatter", scatter_kwargs, s=4, cmap=cmap, lw=0, animated=animated
            )
            scatter_cmap = kwargs.pop("cmap")  # bypass UserWarning: ignored
            vmin, vmax = kwargs.pop("vmin", None), kwargs.pop("vmax", None)
            self.artists_scatter[i] = ax.scatter([], [], **kwargs)
            # TODO: if color is None, use ax.plot([], [], marker=".", ls="") since it is faster
            self.artists_scatter[i].cmap = (
                mpl.colormaps[scatter_cmap] if isinstance(scatter_cmap, str) else scatter_cmap
            )
            self.artists_scatter[i].vmin_vmax = vmin, vmax
            # add colorbar
            if c is not None and (np.any(self.color != c) or i == n - 1):
                cbargs = dict(label=self.label_for(c))
                if self.display_unit_for(c) == "rad":
                    cbargs.update(ticks=AngleLocator(deg=False), format=RadiansFormatter())
                if cbar_loc and cbar_loc.startswith("inside "):
                    # colorbar inside plot
                    axins = inset_axes(
                        ax, width="50%", height="3%", loc=cbar_loc.lstrip("inside ")
                    )
                    self.fig.colorbar(
                        self.artists_scatter[i], **cbargs, cax=axins, orientation="horizontal"
                    )
                    axins.tick_params(labelsize=8)
                    axins.xaxis.offsetText.set_fontsize(8)
                    axins.xaxis.label.set_size(8)
                elif cbar_loc:
                    self.fig.colorbar(self.artists_scatter[i], **cbargs, ax=ax, location=cbar_loc)

            # hexbin histogram
            self._hxkw = defaults_for(
                "hexbin", hist_kwargs, mincnt=1, cmap=cmap, rasterized=True, animated=animated
            )

            # 2D mean indicator
            if mean[i]:
                kwargs = defaults_for(
                    "plot", mean_kwargs, color="k", marker="+", ms=8, zorder=100
                )
                (self.artists_mean[i],) = ax.plot([], [], **kwargs, animated=animated)

            # 2D std ellipses
            if std[i]:
                kwargs = defaults_for(
                    mpl.patches.Ellipse,
                    std_kwargs,
                    fill=False,
                    color="k",
                    lw=1,
                    ls="-",
                    zorder=100,
                )
                self.artists_std[i] = Ellipse((0, 0), 0, 0, **kwargs, animated=animated)
                ax.add_artist(self.artists_std[i])

            # 2D percentile ellipses
            if percentiles[i]:
                self.artists_percentiles[i] = []
                for j, _ in enumerate(self.percentiles[i]):
                    kwargs = defaults_for(
                        mpl.patches.Ellipse,
                        percentile_kwargs,
                        fill=False,
                        color="k",
                        lw=1,
                        ls=(0, [5, 5] + [1, 5] * j),
                        zorder=100,
                    )
                    artist = Ellipse((0, 0), 0, 0, **kwargs, animated=animated)
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

            # Set angle ticks
            for ab, axis in ((a, ax.xaxis), (b, ax.yaxis)):
                if self.display_unit_for(ab) == "rad":
                    self._set_axis_ticks_angle(axis, minor=True, deg=False)
                elif self.display_unit_for(a) in ("deg", "°"):
                    self._set_axis_ticks_angle(axis, minor=True, deg=True)

            # 1D histogram projections
            ###########################

            if self.projections[i]:
                kwargs = defaults_for("plot", projections_kwargs, color="k", alpha=0.3, lw=1)
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
            particles (Any): A dictionary with particle information
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            masks (list): List of masks for each subplot.
            autoscale (bool): Whether or not to perform autoscaling on all axes.

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
            # coordinates
            x = self.prop(a).values(particles, masks[i], unit=self.display_unit_for(a))
            y = self.prop(b).values(particles, masks[i], unit=self.display_unit_for(b))

            # statistics
            XY = np.array((x, y))

            XY0 = np.mean(XY, axis=1)
            UV = XY - XY0[:, np.newaxis]  # centered coordinates

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
                    v = self.prop(c).values(particles, masks[i], unit=self.display_unit_for(c))
                    if autoscale:
                        # scatter.set_clim(np.min(v), np.max(v)) # sometimes leaves behind black dots (bug?)
                        # scatter.norm = mpl.colors.Normalize(np.min(v), np.max(v)) # works, but resets colorbar locator/formatter
                        # scatter.norm.autoscale(v) # also sometimes leaves behind black dots (bug?)

                        if scatter.colorbar is not None:
                            locator, formatter = (
                                scatter.colorbar.locator,
                                scatter.colorbar.formatter,
                            )
                        vmin, vmax = scatter.vmin_vmax
                        scatter.norm = mpl.colors.Normalize(vmin or np.min(v), vmax or np.max(v))
                        if scatter.colorbar is not None:
                            scatter.colorbar.locator, scatter.colorbar.formatter = (
                                locator,
                                formatter,
                            )

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
                hexbin_bg = ax.hexbin(x, y, edgecolors="face", lw=0.1, **self._hxkw)
                hexbin_fg = ax.hexbin(x, y, edgecolors="none", lw=0.0, **self._hxkw)
                self.artists_hexbin[i] = [hexbin_bg, hexbin_fg]
                changed_artists.extend(self.artists_hexbin[i])

            # 2D mean indicator (cross)
            if self.artists_mean[i]:
                self.artists_mean[i].set_data(XY0.reshape([2, 1]))
                changed_artists.append(self.artists_mean[i])

            # 2D size indicator (ellipses)
            if UV.shape[1] > 1 and (self.artists_std[i] or self.artists_percentiles[i]):
                evals, evecs = np.linalg.eig(np.cov(UV))  # eigenvalues and -vectors

                # 2D std indicator
                if self.artists_std[i]:
                    w, h = 2 * np.sqrt(evals)
                    self.artists_std[i].set(
                        center=XY0, width=w, height=h, angle=np.degrees(np.arctan2(*evecs[1]))
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
                            center=XY0, width=w, height=h, angle=np.degrees(np.arctan2(*evecs[1]))
                        )
                        changed_artists.append(self.artists_percentiles[i][j])

            # Autoscale
            if autoscale:
                if plot == "scatter":
                    self._autoscale(ax, artists=[self.artists_scatter[i]])
                elif plot == "hist":
                    self._autoscale(ax, artists=self.artists_hexbin[i])

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
                        rng = getattr(ax, f"get_{xy}lim")()
                        counts, edges = np.histogram(v, bins=101, range=rng)
                        counts = counts / len(v)

                        steps = (np.append(edges, edges[-1]), np.concatenate(([0], counts, [0])))
                        hist.set_data(steps if xy == "x" else steps[::-1])
                        changed_artists.append(hist)

        return changed_artists

    def title_for(self, a, b):
        """
        Plot title for a given pair (a,b) of properties
        """
        titles = {  # Order of a,b does not matter
            "x-px": "Horizontal phase space",
            "y-py": "Vertical phase space",
            "zeta-delta": "Longitudinal phase space",
            "zeta_wrapped-delta": "Longitudinal phase space",
            "X-Px": "Normalized horizontal phase space",
            "Y-Py": "Normalized vertical phase space",
            "x-y": "Transverse profile",
        }
        title = titles.get(f"{a}-{b}", titles.get(f"{b}-{a}"))
        if title is None:
            style = dict(unit=False, description=False)
            title = f"{self.label_for(a, **style)}-{self.label_for(b, **style)} phase space"
        return title

    def axline(self, kind, val, *, subplots="all", also_on_normalized=False, delta=0, **kwargs):
        """Plot a vertical or horizontal line for a given coordinate

        Args:
            kind (str): Phase space coordinate
            val (float): Value of phase space coordinate
            subplots (list of int): Subplots to plot line onto. Defaults to all with matching coordinates.
            also_on_normalized (bool): If true, also plot line for related (de-)normalized phase space coordinates.
            delta (float): The momentrum error used to convert to (de-)normalized  phase space coordinates.
            kwargs: Arguments for axvline or axhline

        """

        kwargs = defaults_for("plot", kwargs, color="k")

        for i, (ab, ax) in enumerate(zip(self.kind, self.axflat)):
            if subplots != "all" and i not in subplots:
                continue

            for a, line in zip(ab, (ax.axvline, ax.axhline)):
                if a == kind:
                    # same axis found, draw line
                    line(val * self.factor_for(kind), **kwargs)

                elif also_on_normalized and self.twiss is not None and kind.lower() == a.lower():
                    # related (de-)normalized axis found, transform and draw line
                    xy = kind.lower()[-1]
                    if kind in "x,y":  # x,y -> X,Y
                        v = normalized_coordinates(val, 0, self.twiss, xy, delta)[0]
                    elif kind in "X,Y":  # X,Y -> x,y
                        v = denormalized_coordinates(val, 0, self.twiss, xy, delta)[0]
                    elif kind in "px,py":  # px,py -> Px,Py
                        v = normalized_coordinates(0, val, self.twiss, xy, delta)[1]
                    elif kind in "Px,Py":  # Px,Py -> px,py
                        v = denormalized_coordinates(0, val, self.twiss, xy, delta)[1]
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
            delta (float): Momentum offset.
            extend (float): Extend of separatrix and equipotential lines. If > 1 they are drawn beyond the the stable region.
            autoscale (bool | float): Whether to autoscale axis or not (bool), or the extend to consider for autoscaling
            separatrix (bool): Plot separatrix. Defaults to True.
            separatrix_kwargs (dict | None): Keyword arguments for separatrix line plot.
            equipotentials (bool): Plot equipotential lines. Defaults to True.
            equipotentials_kwargs (dict | None): Keyword arguments for equipotential line contour plot.
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
            elif not np.isscalar(self.twiss["betx"]) and len(self.twiss["betx"]) != 1:
                raise ValueError(
                    f"Twiss table has {len(self.twiss)} entries, expected exactly 1. "
                    "Did you forget to specify at_elements with a single element during twiss?"
                )
            alfx, betx, mux, x, px = [
                get(self.twiss, pre + xy) for pre in ["alf", "bet", "mu", "", "p"]
            ]
            q = get(self.twiss, "q" + xy)
            q = q + delta * get(self.twiss, "dq" + xy)

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

            # plot separatrix
            if separatrix:
                kwarg = defaults_for(
                    "plot", separatrix_kwargs, color="r", ls="--", label="Separatrix"
                )

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
                    equipotentials_kwargs, colors="lightgray", linewidths=1, alpha=0.5
                )

                # Hamiltonian normalized to value at separatrix
                X = extend * h * np.linspace(-1, 2, 500)
                Y = extend * h * 3**0.5 * np.linspace(-1, 1, 500)
                X, Y = np.meshgrid(X, Y)
                H = (3 * h * (X**2 + Y**2) + 3 * X * Y**2 - X**3) / h**3 / 4
                levels = np.linspace(0, min(extend, 1), int(10 * min(extend, 1))) ** 2
                if extend > 1:
                    levels2 = np.linspace(1, extend, int(5 * (extend - 1)))[1:] ** 2
                    levels = np.hstack([levels, levels2])

                ax.contour(*transform((X, Y)), H, levels=levels, **kwargs)

                ax.grid(False)

            # autoscale to separatrix
            if autoscale:
                t = 1 if autoscale is True else autoscale
                X = np.array((-2, -2, 1 - 3 * t, 1 + 3 * t, 1 - 3 * t, 1 + 3 * t)) / 2
                Y = np.array((-2 * t, 2 * t, 1 + t, 1 - t, -1 - t, -1 + t)) * 3**0.5 / 2
                x, y = transform((h * X, h * Y))
                self._autoscale(ax, data=np.transpose((x, y)), reset=True)


__all__ = PUBLIC_SECTION_END()
