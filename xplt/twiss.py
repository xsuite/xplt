#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Methods for plotting twiss"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-11-08"

from matplotlib.lines import Line2D

from .util import *
from .base import XManifoldPlot
from .line import KnlPlot
from .properties import Property, DataProperty, DerivedProperty, _has_pint

from matplotlib.collections import PolyCollection
from numpy.testing import assert_equal

PUBLIC_SECTION_BEGIN()


class TwissPlot(XManifoldPlot):
    """A plot for twiss parameters and closed orbit"""

    def __init__(self, twiss=None, kind="bet-dx,x+y", *, line=None, line_kwargs={}, **kwargs):
        """

        Args:
            twiss (Any): Dictionary or table with twiss information (or tuple thereof)
            kind (str | list): Defines the properties to plot.
                This is a manifold subplot specification string like ``"bet-dx,x+y"``, see :class:`~.base.XManifoldPlot` for details.
                In addition, abbreviations for x-y-parameter pairs are supported (e.g. 'bet' for 'betx+bety').
            line (xtrack.Line): Line of elements. If given, adds a line plot to the top.
            line_kwargs (dict): Keyword arguments passed to line plot.
            kwargs: See :class:`~.base.XPlot` for additional arguments


        """
        subs = {p: f"{p}x+{p}y" for p in "alf,bet,gam,mu,d,dp,q,dq".split(",")}
        subs["sigma"] = "sigma_x+sigma_y"

        if line:
            kind = self.parse_nested_list_string(kind, subs=subs)
            kind = [[[None]], *kind]

        kwargs["_properties"] = defaults(
            kwargs.get("_properties"),
            # twiss table data
            betx=DataProperty("betx", "m", "$\\beta_x$"),  # Horizontal twiss beta-function
            bety=DataProperty("bety", "m", "$\\beta_y$"),  # Vertical twiss beta-function
            alfx=DataProperty("alfx", "1", "$\\alpha_x$"),  # Horizontal twiss alpha-function
            alfy=DataProperty("alfy", "1", "$\\alpha_y$"),  # Vertical twiss alpha-function
            gamx=DataProperty("gamx", "1/m", "$\\gamma_x$"),  # Horizontal twiss gamma-function
            gamy=DataProperty("gamy", "1/m", "$\\gamma_y$"),  # Vertical twiss gamma-function
            mux=DataProperty("mux", "1", "$\\mu_x$"),  # Horizontal phase advance
            muy=DataProperty("muy", "1", "$\\mu_y$"),  # Vertical phase advance
            dx=DataProperty("dx", "m", "$D_x$"),  # Horizontal dispersion
            dy=DataProperty("dy", "m", "$D_y$"),  # Vertical dispersion
            dpx=DataProperty("dpx", "1", "$D_{x'}$"),  # Horizontal dispersion of px
            dpy=DataProperty("dpy", "1", "$D_{y'}$"),  # Vertical dispersion of py
            # twiss beam covariance
            sigma_x=DataProperty("sigma_x", "m", "$\\sigma_x$"),  # Horizontal beam size
            sigma_px=DataProperty("sigma_px", "1", "$\\sigma_{x'}$"),  # Horizontal beam spread
            sigma_y=DataProperty("sigma_y", "m", "$\\sigma_y$"),  # Vertical beam size
            sigma_py=DataProperty("sigma_py", "1", "$\\sigma_{y'}$"),  # Vertical beam spread
            sigma_zeta=DataProperty(
                "sigma_zeta", "m", "$\\sigma_\\zeta$"
            ),  # Longitudinal beam size
            sigma_pzeta=DataProperty(
                "sigma_pzeta", "1", "$\\sigma_{p_\\zeta}$"
            ),  # Longitudinal beam spread
            # derived
            envelope_x=DerivedProperty(
                "$x\\pm\\sigma_x$", "m", lambda x, sigma_x: (x + sigma_x, x - sigma_x)
            ),
            envelope3_x=DerivedProperty(
                "$x\\pm3\\sigma_x$", "m", lambda x, sigma_x: (x + 3 * sigma_x, x - 3 * sigma_x)
            ),
            envelope_y=DerivedProperty(
                "$y\\pm\\sigma_y$", "m", lambda y, sigma_y: (y + sigma_y, y - sigma_y)
            ),
            envelope3_y=DerivedProperty(
                "$y\\pm3\\sigma_y$", "m", lambda y, sigma_y: (y + 3 * sigma_y, y - 3 * sigma_y)
            ),
            # apertures
            min_x=DataProperty("min_x", "m", "$x_\\mathrm{min}$"),
            max_x=DataProperty("max_x", "m", "$x_\\mathrm{max}$"),
            min_y=DataProperty("min_y", "m", "$y_\\mathrm{min}$"),
            max_y=DataProperty("max_y", "m", "$y_\\mathrm{max}$"),
        )

        display_units = kwargs.pop("display_units", None)
        if _has_pint():
            display_units = defaults(
                display_units,
                bet="m",
                d="m",
                sigma_="mm",
                envelope_="mm",
                envelope3_="mm",
            )

        super().__init__(
            on_x="s",
            on_y=kind,
            on_y_subs=subs,
            display_units=display_units,
            **kwargs,
        )

        # create plot elements
        def create_artists(i, j, k, a, p):
            if line and i == 0:
                return None  # skip line plot placeholder
            label = self._legend_label_for((i, j, k))
            if p.startswith("envelope"):
                kwargs = dict(alpha=0.5, label=label, lw=0, zorder=1.6)
                artist = a.fill_between([], [], **kwargs)
                artist._constructor_kwargs = defaults_for(
                    "fill_between", kwargs, color=artist.get_facecolor()
                )
                return artist
            else:
                return a.plot([], [], label=label)[0]

        self._create_artists(create_artists)

        # Create line plot
        self.lineplot = None
        if line:
            self.lineplot = KnlPlot(line, ax=self.axis(0), **line_kwargs)
            self.lineplot.ax.set(xlabel=None)

        # set data
        if twiss is not None:
            self.update(twiss)

    def update(self, twiss, *, autoscale=None, line=None):
        """
        Update the twiss data this plot shows

        Args:
            twiss (Any): Dictionary or table with twiss information (or tuple thereof)
            autoscale (str | None | bool): Whether and on which axes to perform autoscaling.
                One of `"x"`, `"y"`, `"xy"`, `False` or `None`. If `None`, decide based on :meth:`matplotlib.axes.Axes.get_autoscalex_on` and :meth:`matplotlib.axes.Axes.get_autoscaley_on`.
            line (xtack.Line): Line of elements.

        Returns:
            changed artists
        """

        if type(twiss) == tuple and len(twiss) > 1:
            s = get(twiss[0], "s")
            for tw in twiss[1:]:
                assert_equal(
                    get(tw, "s"), s, "Got multiple twiss data but for different s coordinates"
                )

        changed = []

        if line:
            changed.append(self.lineplot.update(line, autoscale=autoscale))

        s = self.prop("s").values(twiss, unit=self.display_unit_for("s"))
        for i, ppp in enumerate(self.on_y):
            if self.lineplot is not None and i == 0:
                continue  # skip line plot
            for j, pp in enumerate(ppp):
                a = self.axis(i, j)
                for k, p in enumerate(pp):
                    art = self.artists[i][j][k]
                    v = self.prop(p).values(twiss, unit=self.display_unit_for(p))
                    if v.ndim == 2 and isinstance(art, PolyCollection):
                        kwargs = art._constructor_kwargs
                        changed.append(art)
                        art.remove()
                        self.artists[i][j][k] = art = a.fill_between(s, *v, **kwargs)
                        art._constructor_kwargs = kwargs
                    elif v.ndim == 1 and isinstance(art, Line2D):
                        art.set_data((s, v))
                    else:
                        raise ValueError(
                            f"Data for {p} has unexpected number of dimensions: {v.ndim}"
                        )
                    changed.append(art)

                self._autoscale(a, autoscale, tight="x")

        return changed

    def plot_apertures(self, data, which=None, **kwargs):
        """Plot apertures onto the relevant axes

        Args:
            data (Any): The aperture data: a dict, table or something which provides arrays for the keys "s", "min_x",
              "max_x", "min_y", "max_y". Use :meth:`~.util.apertures` to get such data from a :class:`xtrack.Line`.
            which (str | None | list | set): Name(s) of properties for which to plot apertures on the corresponding axis.
              If None (default), plot for all relevant properties (x, y, sigma, envelope)
            kwargs: Keyword arguments passed to :meth:`matplotlib.axes.Axes.vlines`
        """

        if which is None:
            props = "{}", "sigma_{}", "envelope_{}", "envelope3_{}"
            which = {p.format(xy) for xy in "xy" for p in props}
            which.intersection_update(self.on_y_unique)

        kwargs = defaults_for("vlines", kwargs, colors="k", lw=1, zorder=999)

        axes = {}
        for p in which:
            if "x" in p and "y" not in p:
                axes[self.axis(p)] = (p, "x")
            elif "y" in p and "x" not in p:
                axes[self.axis(p)] = (p, "y")
            else:
                raise ValueError(
                    f"Apertures not supported for property {p}. Cannot identify if {p} is x or y plane."
                )

        s = self.prop("s").values(data, unit=self.display_unit_for("s"))
        for ax, (p, xy) in axes.items():
            axlim = ax.get_ylim()
            for key, lim in zip((f"min_{xy}", f"max_{xy}"), axlim):
                v = self.prop(key).values(data, unit=self.display_unit_for(p))
                visible = (axlim[0] < v) & (v < axlim[1])
                if np.any(visible):
                    ax.vlines(s[visible], v[visible], lim, **kwargs)


__all__ = PUBLIC_SECTION_END()
