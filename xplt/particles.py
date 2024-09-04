#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Methods for plotting phase space distributions

"""

__author__ = "Philipp Niedermayer"
__contact__ = "eltos@outlook.de"
__date__ = "2022-12-07"


from .util import *
from .base import XManifoldPlot
from .properties import Property, DerivedProperty, find_property


class ParticlePlotMixin:
    """Mixin for plotting of particle data

    .. automethod:: _init_particle_mixin
    """

    def _init_particle_mixin(
        self, *, twiss=None, beta=None, frev=None, circumference=None, **kwargs
    ):
        r"""Initializes the mixin by providing associated information

        For a given particle object with inherent properties like ``x``, ``y``, ``px``, ``py``, ``zeta``, ``delta``, etc.
        this mixin allows determination of the following derived properties:

        - Wrapped longitudinal coordinate: ``zeta_wrapped``
           |  This is the ``zeta`` coordinate wrapped to be in range (-circumference/2; circumference/2)
        - Normalized coordinates: ``X``, ``Y``, ``Px``, ``Py``
           |  :math:`X = x/\sqrt{\beta_x} = \sqrt{2J_x} \cos(\Theta_x)`
           |  :math:`P_x = (\alpha_x x + \beta_x p_x)/\sqrt{\beta_x} = -\sqrt{2J_x} \sin(\Theta_x)`
        - Action-angle coordinates: ``Jx``, ``Jy``, ``Θx``, ``Θy``
           |  :math:`J_x = (X^2+P_x^2)/2`
           |  :math:`\Theta_x = -\mathrm{atan2}(P_x, X)`
        - Particle arrival time: ``t``
           |  t = at_turn / frev - zeta / beta / c0

        Prefix notation is also available, i.e. ``P`` for ``Px+Py`` or ``Θ`` for ``Θx+Θy``


        Args:
            twiss (dict | None): Twiss parameters (alfx, alfy, betx and bety) to use for conversion to normalized phase space coordinates.
            beta (float | None): Relativistic beta of particles. Defaults to `beta0` property of particles.
            frev (float | None): Revolution frequency of circular line for calculation of particle time.
            circumference (float | None): Path length of circular line if frev is not given.
            kwargs: Keyword arguments for :class:`~.base.XPlot`

        Returns:
            Updated keyword arguments for :class:`~.base.XPlot` constructor.

        """
        self.twiss = twiss
        self._beta = val(beta)
        self._frev = val(frev)
        self._circumference = val(circumference)

        # Particle specific properties
        # fmt: off
        self._derived_particle_properties = dict(
            zeta_wrapped=DerivedProperty("$\\zeta$", "m", lambda zeta: ieee_mod(zeta, self.circumference)),
            X=DerivedProperty("$X$", "m^(1/2)", lambda x, px, delta: normalized_coordinates(x, px, self.twiss, "x", delta)[0]),
            Y=DerivedProperty("$Y$", "m^(1/2)", lambda y, py, delta: normalized_coordinates(y, py, self.twiss, "y", delta)[0]),
            Px=DerivedProperty("$X'$", "m^(1/2)", lambda x, px, delta: normalized_coordinates(x, px, self.twiss, "x", delta)[1]),
            Py=DerivedProperty("$Y'$", "m^(1/2)", lambda y, py, delta: normalized_coordinates(y, py, self.twiss, "y", delta)[1]),
            Jx=DerivedProperty("$J_x$", "m", lambda X, Px: (X**2 + Px**2) / 2),
            Jy=DerivedProperty("$J_y$", "m", lambda Y, Py: (Y**2 + Py**2) / 2),
            Θx=DerivedProperty("$Θ_x$", "rad", lambda X, Px: -np.arctan2(Px, X)),
            Θy=DerivedProperty("$Θ_y$", "rad", lambda Y, Py: -np.arctan2(Py, Y)),
            t=DerivedProperty("$t$", "s", lambda _data, at_turn, zeta: self._particle_time(at_turn, zeta, _data))
        )
        # fmt: on

        # Update kwargs with particle specific settings
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"), **self._derived_particle_properties
        )
        kwargs["display_units"] = defaults(
            kwargs.get("display_units"),
            X="mm^(1/2)",
            Y="mm^(1/2)",
            P="mm^(1/2)",
            J="mm",  # Action
            Θ="rad",  # Angle
        )

        return kwargs

    @property
    def circumference(self):
        """Circumference of circular accelerator"""
        if self._circumference is not None:
            return self._circumference
        if self.twiss is not None:
            return get(self.twiss, "circumference")

    def beta(self, particles=None):
        """Get reference relativistic beta as float"""
        if self._beta is not None:
            return self._beta
        if self.circumference is not None:
            if self._frev is not None:
                return self._frev * self.circumference / c0
            if self.twiss is not None:
                return self.circumference / get(self.twiss, "T_rev0") / c0
        if particles is not None:
            try:
                beta = get(particles, "beta0")
                if np.size(beta) > 1:
                    mean_beta = np.mean(beta)
                    if not np.allclose(beta, mean_beta):
                        raise ValueError(
                            "Particle beta0 is not constant. Please specify beta in constructor!"
                        )
                    beta = mean_beta
                return beta
            except:
                pass

    def frev(self, particles=None):
        """Get reference revolution frequency"""
        if self._frev is not None:
            return self._frev
        if self.twiss is not None:
            return 1 / get(self.twiss, "T_rev0")
        beta = self.beta(particles)
        if beta is not None and self.circumference is not None:
            return beta * c0 / self.circumference

    def _particle_time(self, turn, zeta, particles=None):
        """Particle arrival time (t = at_turn / frev - zeta / beta / c0)"""

        # use time directly (if available)
        if particles is not None:
            try:
                return get(particles, "t")
            except AttributeError:
                pass

        # determine time from longitudinal coordinates
        beta = self.beta(particles)
        if beta is None:
            raise ValueError(
                "Particle arrival time can not be determined "
                "because all of the following are unknown: "
                "beta, (frev and circumference), twiss. "
                "To resolve this error, pass either to the plot constructor "
                "or specify particle.beta0."
            )

        time = -zeta / beta / c0  # zeta>0 means early; zeta<0 means late
        if np.any(turn != 0):
            frev = self.frev(particles)
            if frev is None:
                raise ValueError(
                    "Particle arrival time can not be determined while at_turn > 0 "
                    "because all of the following are unknown: "
                    "frev, twiss, (beta and circumference). "
                    "To resolve this error, pass either to the plot constructor "
                    "and/or specify particle.beta0."
                )
            time = time + turn / frev

        return np.array(time)

    def get_property(self, name):
        # Note: this method is not used by the library, but it's handy for standalone use
        """Public method to get a particle property by key

        Args:
            name (str): Key
        Returns:
            Property: The property
        """
        prop = find_property(name, extra_default_properties=self._derived_particle_properties)
        return prop.with_property_resolver(self.get_property)


class ParticleHistogramPlotMixin:
    """Mixin for plotting of histogram particle data

    .. automethod:: _init_particle_histogram_mixin
    """

    def _init_particle_histogram_mixin(self, **kwargs):
        r"""Initializes the mixin by providing associated information

        Args:
            kwargs: Keyword arguments for :class:`~.base.XPlot`

        Returns:
            Updated keyword arguments for :class:`~.base.XPlot` constructor.

        """

        self._histogram_particle_properties = dict(
            count=Property("$N$", "1", description="Particles per bin"),
            cumulative=Property("$N$", "1", description="Particles (cumulative)"),
            rate=Property("$\\dot{N}$", "1/s", description="Particle rate"),
            charge=Property("$Q$", find_property("q").unit, description="Charge per bin"),
            current=Property("$I$", f"({find_property('q').unit})/s", description="Current"),
        )

        # Update kwargs with particle specific settings
        kwargs["_properties"] = defaults(
            kwargs.get("_properties"), **self._histogram_particle_properties
        )
        kwargs["display_units"] = defaults(
            kwargs.get("display_units"),
            current="nA",
        )

        return kwargs

    def _count_based(self, key):
        return key in self._histogram_particle_properties


PUBLIC_SECTION_BEGIN()


class ParticleHistogramPlot(XManifoldPlot, ParticlePlotMixin, ParticleHistogramPlotMixin):
    """A 1D histogram plot for any particle property

    See also:
        :class:`~.phasespace.PhaseSpacePlot` for a 2D histogram
    """

    def __init__(
        self,
        property,
        particles=None,
        kind="count",
        *,
        bin_width=None,
        bin_count=None,
        range=None,
        relative=False,
        moment=1,
        mask=None,
        plot_kwargs=None,
        add_default_dataset=True,
        **kwargs,
    ):
        """

        The main purpose is to plot particle distributions,
        but the `kind` keyword also accepts particle properties
        in which case the property is averaged over all particles falling into the bin.

        Useful to plot beam profiles over `x`, `y` etc., bunch shapes over `zeta`, spill structures over `t`, ...


        Args:
            property (str): The property to bin.
            particles (Any): Particles data to plot.
            kind (str | list): Defines the type of the histogram. Can be 'count' (default), 'cumulative', 'charge',
                or a particle property to use as weights when computing the histogram.
                This is a manifold subplot specification string like ``"x,y"``, see :class:`~.base.XManifoldPlot` for details.
            bin_width (float): Bin width (in data units of `property`) if `bin_count` is None.
            bin_count (int): Number of bins if `bin_width` is None.
            range (tuple[float] | None): The lower and upper range of the bins. Values outside the range are ignored.
                If not provided, range is simply (min, max) of the data.
            relative (bool): If True, plot relative numbers normalized to total count/charge/rate/....
                If `kind` is a particle property, this has no effect.
            moment (int): The moment(s) to plot if kind is a particle property.
                Allows to get the mean (1st moment, default), variance (difference between 2nd and 1st moment) etc.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.plot`.
            add_default_dataset (bool): Whether to add a default dataset.
                Use :meth:`~.particles.ParticleHistogramPlot.add_dataset` to manually add datasets.
            kwargs: See :class:`~.particles.ParticlePlotMixin`, :class:`~.particles.ParticleHistogramPlotMixin`
                and :class:`~.base.XManifoldPlot` for additional arguments

        """

        if bin_width is None and bin_count is None:
            bin_count = 100
        if bin_width is not None and bin_count is not None:
            raise ValueError("Only one of bin_width or bin_count may be specified.")
        self.bin_width = bin_width
        self.bin_count = bin_count
        self.range = range
        self.relative = relative
        self.moment = moment  # required by `self._symbol_for`, so set it before calling super
        self._actual_bin_width = {}

        kwargs = self._init_particle_mixin(**kwargs)
        kwargs = self._init_particle_histogram_mixin(**kwargs)

        super().__init__(on_x=property, on_y=kind, **kwargs)

        # Format plot axes
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                count_based = np.all([self._count_based(p) for p in pp])
                if count_based and relative:
                    self.axis(i, j).yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

        if add_default_dataset:
            self.add_dataset(None, particles=particles, mask=mask, plot_kwargs=plot_kwargs)

    def add_dataset(self, id, *, plot_kwargs=None, **kwargs):
        """Create artists for a new dataset to the plot and optionally update their values

        Args:
            id (str): An arbitrary dataset identifier unique for this plot
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.plot`.
            **kwargs: Arguments passed to :meth:`~.particles.ParticleHistogramPlot.update`.
        """

        # Create plot elements
        def create_artist(i, j, k, ax, p):
            kwargs = defaults_for(
                "plot", plot_kwargs, lw=1, label=self._legend_label_for((i, j, k))
            )
            if self._count_based(p) and p != "cumulative":
                kwargs = defaults_for("plot", kwargs, drawstyle="steps-pre")
            return ax.plot([], [], **kwargs)[0]

        self._create_artists(create_artist, dataset_id=id)

        # set data
        if kwargs.get("particles") is not None:
            self.update(**kwargs, dataset_id=id)

    def _symbol_for(self, p):
        symbol = super()._symbol_for(p)
        if p != self.on_x and not self._count_based(p):
            # it is averaged
            power = self.moment if self.moment > 1 else ""
            symbol = f"$\\langle${symbol}$^{{{power}}}\\rangle$"
        return symbol

    def _histogram(self, p, particles, mask):

        values = self.prop(self.on_x).values(particles, mask)
        if p in ("current", "charge"):  # Note: these are also "count_based" i.e. use moments=None
            what = self.prop("q").values(particles, mask)
        elif self._count_based(p):
            what = None
        else:
            what = self.prop(p).values(particles, mask)

        v_min, dv, hist = binned_data(
            values,
            what=what,
            n=self.bin_count,
            dv=self.bin_width,
            v_range=self.range,
            moments=None if self._count_based(p) else self.moment,
        )
        edges = v_min + dv * np.arange(hist.size + 1)

        return hist, edges

    def update(self, particles, mask=None, *, autoscale=None, dataset_id=None):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (str | None | bool): Whether and on which axes to perform autoscaling.
                One of `"x"`, `"y"`, `"xy"`, `False` or `None`. If `None`, decide based on :meth:`matplotlib.axes.Axes.get_autoscalex_on` and :meth:`matplotlib.axes.Axes.get_autoscaley_on`.
            dataset_id (str | None): The dataset identifier to update if this plot represents multiple datasets

        Returns:
            list: Changed artists
        """

        prop_x = self.prop(self.on_x)
        if self._count_based(self.on_x):
            raise ValueError(f"Binning property cannot be `{self.on_x}`")

        # update plots
        changed = []
        dv = None
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    art = self.artists[dataset_id, i, j, k]

                    # compute histogram
                    hist, edges = self._histogram(p, particles, mask)
                    hist = hist.astype(np.float64)

                    if self._count_based(p) and self.relative:
                        hist /= np.sum(hist)

                    if p in ("rate", "current"):
                        factor = pint.Quantity(1, prop_x.unit).to("s").m
                        hist /= factor * np.diff(edges)

                    # post-processing expression wrappers
                    if wrap := self.on_y_expression[i][j][k]:
                        hist = evaluate_expression_wrapper(wrap, p, hist)

                    # determine actual bin width for annotation
                    if dv is None:
                        dv = np.mean(np.diff(edges))
                    elif dv != np.mean(np.diff(edges)):
                        dv = False  # different traces have different bin widths
                    elif np.abs(np.std(np.diff(edges)) / np.mean(np.diff(edges))) > 1e-5:
                        dv = False  # trace has variable bin width

                    # display units
                    hist *= self.factor_for(p)
                    edges *= self.factor_for(self.on_x)

                    # update plot
                    if p == "cumulative":
                        # steps open after last bin
                        hist = np.concatenate(([0], np.cumsum(hist)))
                    elif self._count_based(p):
                        # steps go back to zero after last bin
                        edges = np.append(edges, edges[-1])
                        hist = np.concatenate(([0], hist, [0]))
                    else:
                        edges = (edges[1:] + edges[:-1]) / 2

                    art.set_data((edges, hist))
                    changed.append(art)

                # autoscale
                a = self.axis(i, j)
                scaled = self._autoscale(a, autoscale)  # , tight="x"
                if "y" in scaled:
                    if np.any([self._count_based(p) for p in pp]):
                        a.set_ylim(0, None, auto=None)

        # annotation
        self._actual_bin_width[dataset_id] = dv
        dv = np.unique(list(self._actual_bin_width.values()))
        if len(dv) == 1:
            x = prop_x.symbol.strip("$")
            self.annotate(f"$\\Delta {{{x}}}_\\mathrm{{bin}} = {fmt(dv[0], prop_x.unit)}$")
        else:
            self.annotate("")

        return changed


class ParticlesPlot(XManifoldPlot, ParticlePlotMixin):
    """A plot of particle properties as function of another property"""

    def __init__(
        self,
        particles=None,
        kind="x+y",
        as_function_of="at_turn",
        *,
        mask=None,
        plot_kwargs=None,
        sort_by=None,
        add_default_dataset=True,
        **kwargs,
    ):
        """

        Args:
            particles (Any): Particles data to plot.
            kind (str | list): Defines the properties to plot.
                 This can be a separated string or a nested list or a mixture of both where
                 the first list level (or separator ``,``) determines the subplots,
                 the second list level (or separator ``-``) determines any twinx-axes,
                 and the third list level (or separator ``+``) determines plots on the same axis.
            as_function_of (str): The property to plot as function of.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            plot_kwargs (dict): Keyword arguments passed to the plot function.
            sort_by (str | None): Sort the data by this property. Default is to sort by the `as_function_of` property.
            add_default_dataset (bool): Whether to add a default dataset.
                Use :meth:`~.particles.ParticlesPlot.add_dataset` to manually add datasets.
            kwargs: See :class:`~.particles.ParticlePlotMixin` and :class:`~.base.XPlot` for additional arguments

        """
        kwargs = self._init_particle_mixin(**kwargs)
        kwargs["display_units"] = defaults(kwargs.get("display_units"), bet="m", d="m")
        super().__init__(
            on_x=as_function_of, on_y=kind, on_y_subs={"J": "Jx+Jy", "Θ": "Θx+Θy"}, **kwargs
        )

        # parse kind string
        self.sort_by = sort_by

        if add_default_dataset:
            self.add_dataset(None, particles=particles, mask=mask, plot_kwargs=plot_kwargs)

    def add_dataset(self, id, *, plot_kwargs=None, **kwargs):
        """Create artists for a new dataset to the plot and optionally update their values

        Args:
            id (str): An arbitrary dataset identifier unique for this plot
            plot_kwargs (dict): Keyword arguments passed to the plot function, see :meth:`matplotlib.axes.Axes.plot`.
            **kwargs: Arguments passed to :meth:`~.particles.ParticleHistogramPlot.update`.
        """

        # create plot elements
        def create_artists(i, j, k, a, p):
            kwargs = defaults_for(
                "plot", plot_kwargs, marker=".", ls="", label=self._legend_label_for((i, j, k))
            )
            return a.plot([], [], **kwargs)[0]

        self._create_artists(create_artists, dataset_id=id)

        # set data
        if kwargs.get("particles") is not None:
            self.update(**kwargs, dataset_id=id)

    def update(self, particles, mask=None, *, autoscale=None, dataset_id=None):
        """Update plot with new data

        Args:
            particles (Any): Particles data to plot.
            mask (Any): An index mask to select particles to plot. If None, all particles are plotted.
            autoscale (str | None | bool): Whether and on which axes to perform autoscaling.
                One of `"x"`, `"y"`, `"xy"`, `False` or `None`. If `None`, decide based on :meth:`matplotlib.axes.Axes.get_autoscalex_on` and :meth:`matplotlib.axes.Axes.get_autoscaley_on`.
            dataset_id (str | None): The dataset identifier to update if this plot represents multiple datasets

        Returns:
            List of artists that have been updated.
        """

        xdata = self.prop(self.on_x).values(
            particles, mask, unit=self.display_unit_for(self.on_x)
        )
        order = np.argsort(
            xdata if self.sort_by is None else self.prop(self.sort_by).values(particles, mask)
        )
        xdata = xdata[order]

        changed = []
        for i, ppp in enumerate(self.on_y):
            for j, pp in enumerate(ppp):
                for k, p in enumerate(pp):
                    art = self.artists[dataset_id, i, j, k]
                    values = self.prop(p).values(particles, mask, unit=self.display_unit_for(p))
                    values = values[order]
                    art.set_data((xdata, values))
                    changed.append(art)

                a = self.axis(i, j)
                self._autoscale(a, autoscale)

        return changed


__all__ = PUBLIC_SECTION_END()
